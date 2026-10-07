from __future__ import annotations

import argparse
import torch

from common.data import load_yaml, repo_path

import pandas as pd
import torch.nn.functional as F
from torch.optim import AdamW

from common.data import prompt_messages, read_jsonl, render_prompt
from common.generation import response_token_logprobs
from common.logging_utils import load_json, save_json, set_seed
from common.metrics import masked_mean
from common.models import clear_gpu, load_policy, load_tokenizer, trainable_parameters
from task2_ppo.continue_train import disable_dropout, run_ppo
from task2_ppo.evaluate import evaluate_policy
from task2_ppo.evaluate import load_evaluation_bundle as load_ppo_evaluation_bundle
from task2_ppo.ppo import affected_token_fraction, compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def rebuild_cached_batch(cfg,rows,tokenizer):
    """Re-tokenize the cached prompts/responses and rebuild rewards, advantages and returns."""
    prompt_lookup={} #prompt_id -> prompt row
    for row in read_jsonl(cfg["paths"]["rl_prompt_train"]):
        prompt_lookup[row["prompt_id"]]=row #training pool
    for row in read_jsonl(cfg["paths"]["rl_prompt_eval"]):
        prompt_lookup[row["prompt_id"]]=row #held-out pool (the cache prompts come from here)
    items=[] #one entry per usable cached rollout
    skipped=[] #rows we could not line up exactly
    for row in rows:
        prompt_row=prompt_lookup.get(row["prompt_id"]) #find the prompt
        if prompt_row is None:
            skipped.append({"source_index":row["source_index"],"reason":"prompt_id not in prompt pools"}) #cannot rebuild
            continue
        messages=prompt_messages(prompt_row) #chat messages
        prompt_ids=tokenizer(render_prompt(tokenizer,messages),truncation=True,max_length=int(cfg["max_prompt_length"]))["input_ids"] #exactly what batch_generate fed the policy (256 cap, cut from the right), so log-probs line up with the cache
        response_ids=tokenizer(row["response"],add_special_tokens=False)["input_ids"] #response text back to tokens
        if row["terminated_with_eos"]==True:
            response_ids=response_ids+[tokenizer.eos_token_id] #generation mask keeps the eos token, decode dropped it
        if len(response_ids)!=len(row["old_logprobs"]):
            skipped.append({"source_index":row["source_index"],"reason":"retokenized length "+str(len(response_ids))+" != cached "+str(len(row["old_logprobs"]))}) #would misalign log-probs
            continue
        items.append({"row":row,"prompt_ids":prompt_ids,"response_ids":response_ids}) #usable

    steps=0 #longest cached response
    for item in items:
        if len(item["response_ids"])>steps:
            steps=len(item["response_ids"]) #new longest
    count=len(items) #usable rollouts
    old_logp=torch.zeros(count,steps) #cached log pi_old
    ref_logp=torch.zeros(count,steps) #cached log pi_ref
    values=torch.zeros(count,steps) #cached V(s_t)
    mask=torch.zeros(count,steps) #valid tokens
    task_reward=torch.zeros(count) #cached effective terminal reward (eos penalty already in)
    for i in range(count):
        row=items[i]["row"] #cached rollout
        length=len(items[i]["response_ids"]) #its length
        old_logp[i,:length]=row["old_logprobs"].float() #fill
        ref_logp[i,:length]=row["ref_logprobs"].float()
        values[i,:length]=row["values"].float()
        mask[i,:length]=1.0 #valid
        task_reward[i]=float(row["effective_terminal_reward"]) #same reward the training loop would use
    rewards=shaped_rewards(task_reward,old_logp,ref_logp,mask,float(cfg["kl_beta"])) #kl-shaped token rewards
    advantages,returns=compute_gae(rewards,values,mask,float(cfg["gamma"]),float(cfg["gae_lambda"])) #gae + returns
    advantages=normalize_advantages(advantages,mask) #same whitening as the training loop
    return {"items":items,"skipped":skipped,"old_logp":old_logp,"ref_logp":ref_logp,"mask":mask,"advantages":advantages,"returns":returns}


def current_logprobs(policy,items,steps,device,with_grad=False):
    """Log-probs of the cached responses under the current policy, one rollout at a time (no padding)."""
    rows_logp=[] #one [1, steps] tensor per rollout
    for item in items:
        ids=item["prompt_ids"]+item["response_ids"] #prompt then response
        sequence=torch.tensor([ids],device=device) #[1, length]
        attention_mask=torch.ones_like(sequence) #no padding, batch of one
        response=torch.tensor([item["response_ids"]],device=device) #[1, response length]
        if with_grad==True:
            logp,logits=response_token_logprobs(policy,sequence,attention_mask,len(item["prompt_ids"]),response) #with gradient
        else:
            with torch.no_grad():
                logp,logits=response_token_logprobs(policy,sequence,attention_mask,len(item["prompt_ids"]),response) #no gradient
        del logits #not needed
        rows_logp.append(F.pad(logp.float(),(0,steps-logp.shape[1]))) #right pad to the batch width
    return rows_logp


def geometry(new_logp,batch,eps):
    """Clipped surrogate, unclipped surrogate, clip fraction and affected fraction at one epsilon."""
    old_logp=batch["old_logp"] #cached pi_old
    advantages=batch["advantages"] #whitened gae
    mask=batch["mask"] #valid tokens
    loss,ratio,clip_fraction=ppo_policy_loss(new_logp,old_logp,advantages,mask,eps) #fixed objective
    affected=affected_token_fraction(ratio,advantages,mask,eps) #clipping actually bites
    unclipped=masked_mean(ratio*advantages,mask) #plain importance-weighted surrogate
    log_ratio=(new_logp-old_logp)*mask #0 on padding
    return {
        "epsilon":eps, #setting
        "clipped_surrogate":float(-loss.item()), #L_clip (loss is its negative)
        "unclipped_surrogate":float(unclipped.item()), #for comparison
        "clip_fraction":float(clip_fraction.item()), #manual: ratio outside [1-eps,1+eps]
        "affected_fraction":float(affected.item()), #manual: affected-token fraction
        "ratio_mean":float(masked_mean(ratio,mask).item()), #average ratio
        "max_abs_log_ratio":float(log_ratio.abs().max().item()), #largest single-token move
    }


def cached_batch_study(cfg,rows,results_dir):
    """Part 1: what each epsilon does to the same fixed batch."""
    set_seed(int(cfg["seed"])) #fixed
    tokenizer=load_tokenizer(cfg["base_model"]) #policy tokenizer
    batch=rebuild_cached_batch(cfg,rows,tokenizer) #tensors on cpu
    items=batch["items"] #usable rollouts
    print("usable cached rollouts:",len(items),"skipped:",len(batch["skipped"])) #should be all 32
    steps=batch["mask"].shape[1] #batch width
    policy=load_policy(cfg,adapter_path=cfg["paths"]["ppo_midpoint_policy"],trainable=True) #supplied midpoint
    disable_dropout(policy) #same as the training loop
    device=next(policy.parameters()).device #cuda
    for key in ["old_logp","ref_logp","mask","advantages","returns"]:
        batch[key]=batch[key].to(device) #move tensors next to the model
    initial_state={} #midpoint lora weights, restored before every epsilon
    for name,parameter in policy.named_parameters():
        if parameter.requires_grad==True:
            initial_state[name]=parameter.detach().clone() #copy

    #a) the midpoint policy itself against the cached pi_old
    midpoint_logp=torch.cat(current_logprobs(policy,items,steps,device),dim=0) #[n, steps]
    results=[] #one row per (stage, epsilon)
    for eps in cfg["clip_values"]:
        row=geometry(midpoint_logp,batch,float(eps)) #ratio here is pi_midpoint / cached pi_old
        row["stage"]="midpoint_vs_cached_old" #before any update
        results.append(row) #add

    #b) the same batch after ppo_epochs optimizer steps at each epsilon, always from the midpoint
    total_tokens=float(batch["mask"].sum().item()) #for a batch-level masked mean
    for eps in cfg["clip_values"]:
        eps=float(eps) #setting
        for name,parameter in policy.named_parameters():
            if name in initial_state:
                parameter.data.copy_(initial_state[name]) #back to the midpoint
        optimizer=AdamW(trainable_parameters(policy),lr=float(cfg["policy_learning_rate"])) #fresh optimizer per epsilon
        for epoch in range(int(cfg["ppo_epochs"])):
            optimizer.zero_grad() #clear
            for i in range(len(items)):
                row_logp=current_logprobs(policy,[items[i]],steps,device,with_grad=True)[0] #[1, steps]
                row_mask=batch["mask"][i:i+1] #this rollout's mask
                row_loss,_,_=ppo_policy_loss(row_logp,batch["old_logp"][i:i+1],batch["advantages"][i:i+1],row_mask,eps) #masked mean over this row
                weight=float(row_mask.sum().item())/total_tokens #turns per-row means into one batch-level masked mean
                (row_loss*weight).backward() #accumulate, one rollout in memory at a time
            torch.nn.utils.clip_grad_norm_(trainable_parameters(policy),float(cfg["max_grad_norm"])) #same clipping as training
            optimizer.step() #one step per epoch over the whole cached batch
        updated_logp=torch.cat(current_logprobs(policy,items,steps,device),dim=0) #after the update
        row=geometry(updated_logp,batch,eps) #what clipping now does
        row["stage"]="after_"+str(cfg["ppo_epochs"])+"_steps" #after the update
        results.append(row) #add

    table=pd.DataFrame(results) #comparison table
    table.to_csv(results_dir/"clipping_cached_batch.csv",index=False) #for the report
    save_json(results_dir/"clipping_cached_batch.json",{"results":results,"skipped":batch["skipped"],"usable_rollouts":len(items),"kl_beta":float(cfg["kl_beta"])}) #json copy
    print(table.to_string()) #quick look
    del policy #drop the model


def short_forks(config_path,cfg,results_dir,skip_train,limit):
    """Part 2: matched short continuations from the same midpoint, one per epsilon."""
    fork_rows=[] #one row per epsilon
    for eps in cfg["clip_values"]:
        eps=float(eps) #setting
        run_name="clip_"+str(eps) #clip_0.05, clip_0.2, clip_0.5
        output="outputs/task2_ppo/"+run_name #adapter folder (git ignored)
        if skip_train==False:
            run_ppo(config_path,output,int(cfg["fork_updates"]),eps,None,run_name) #only epsilon changes, kl_beta stays 0.10
            clear_gpu() #free training models
        bundle=load_ppo_evaluation_bundle(config_path,output) #same loader for every fork
        summary=evaluate_policy(bundle,output,run_name,8,limit) #same held-out protocol
        del bundle #drop models
        clear_gpu() #free memory
        train_summary=load_json(results_dir/run_name/"train_summary.json") #budget, time
        log=read_jsonl(results_dir/run_name/"train_log.jsonl") #per-update diagnostics
        approx_kls=[] #per-update drift from pi_old
        clip_fractions=[] #per-update clip fraction
        grad_norms=[] #per-update policy grad norm
        clipped_updates=0 #updates where the grad norm hit max_grad_norm
        for record in log:
            approx_kls.append(record["approx_kl_old_new"]) #collect
            clip_fractions.append(record["clip_fraction"])
            grad_norms.append(record["policy_grad_norm"])
            if record["policy_grad_norm"]>float(cfg["max_grad_norm"]):
                clipped_updates=clipped_updates+1 #count
        fork_rows.append({
            "condition":run_name, #fork
            "epsilon":eps, #setting
            "updates":train_summary["updates"], #matched update budget
            "generated_tokens":train_summary["total_generated_tokens"], #token budget actually used
            "heldout_reward":summary["reward_mean"], #learned reward on held-out prompts
            "heldout_effective_reward":summary["effective_reward_mean"], #with eos penalty
            "heldout_kl":summary["kl_from_reference"], #drift from the reference
            "heldout_entropy":summary["entropy"], #diversity diagnostic
            "heldout_length_mean":summary["response_length"]["mean"], #generated tokens
            "heldout_truncation_rate":summary["truncation_rate"], #hit 768
            #stability statistic (defined once): drift from pi_old inside one update, measured after the last ppo epoch
            "stability_mean_approx_kl_old_new":float(pd.Series(approx_kls).mean()), #average per-update drift
            "stability_max_approx_kl_old_new":float(pd.Series(approx_kls).abs().max()), #worst update
            "train_clip_fraction_mean":float(pd.Series(clip_fractions).mean()), #how often ratios left the band
            "train_grad_norm_std":float(pd.Series(grad_norms).std()), #how jumpy the gradients were
            "updates_with_grad_clipped":clipped_updates, #grad norm above 1.0
        })
    table=pd.DataFrame(fork_rows) #fork comparison
    table.to_csv(results_dir/"clipping_forks.csv",index=False) #for the report
    save_json(results_dir/"clipping_forks.json",fork_rows) #json copy
    print(table.to_string()) #quick look


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--skip-cached",action="store_true") #only run the forks
    ap.add_argument("--skip-forks",action="store_true") #only run the cached-batch measurement
    ap.add_argument("--skip-train",action="store_true") #forks already trained, only evaluate them
    ap.add_argument("--limit",type=int) #same eval prefix for every fork if you need to save time
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    print("Cached PPO rollouts:", len(rows))
    print("Required epsilon values:", cfg["clip_values"])
    print("Cache keys:", sorted(rows[0].keys()))
    results_dir=repo_path(cfg["results_dir"]) #results/task2_ppo/
    if args.skip_cached==False:
        cached_batch_study(cfg,rows,results_dir) #part 1: geometry on the fixed batch
        clear_gpu() #free the policy
    if args.skip_forks==False:
        short_forks(args.config,cfg,results_dir,args.skip_train,args.limit) #part 2: matched short continuations


if __name__ == "__main__":
    main()
