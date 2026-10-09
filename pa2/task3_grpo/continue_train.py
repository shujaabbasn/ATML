from __future__ import annotations

import argparse

from torch.optim import AdamW

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import set_seed
from common.models import load_policy, load_reward_model, load_tokenizer, trainable_parameters

import random

import torch

from common.data import prompt_messages
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, wall_timer
from common.metrics import masked_mean, sample_entropy, sampled_kl
from common.models import reference_mode
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences


def disable_dropout(model):
    """Set every dropout to 0 so pi_old and pi_theta are the same function before the step (same as task 2)."""
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):
            module.p=0.0 #lora dropout would otherwise make the ratio != 1 with identical weights


def rows_logprobs(policy,sequences,attention_mask,prompt_width,response_ids):
    """Log-probs of every completion, one row at a time and without gradient (keeps t4 memory low)."""
    rows=[] #one [1, steps] tensor per completion
    with torch.no_grad():
        for row in range(sequences.shape[0]):
            logp,logits=response_token_logprobs(policy,sequences[row:row+1],attention_mask[row:row+1],prompt_width,response_ids[row:row+1]) #log pi(a_t|s_t) for this completion
            del logits #only need the gathered log-probs
            rows.append(logp.float()) #fp32
    return torch.cat(rows,dim=0) #[K, steps]


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo", run_name: str = "standard"):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    policy=bundle["policy"] #grpo midpoint lora, trainable
    reward_model=bundle["reward_model"] #same frozen reward model as ppo
    reward_tokenizer=bundle["reward_tokenizer"] #base tokenizer
    tokenizer=bundle["tokenizer"] #policy tokenizer, left padding
    prompt_rows=bundle["prompt_rows"] #same ultrafeedback prompt pool as ppo
    optimizer=bundle["optimizer"] #adamw, lr 5e-6
    updates=int(cfg["updates"]) #20 standard, 8 for the normalization forks
    prompts_per_update=int(cfg["prompts_per_update"]) #config: 1
    group_size=int(cfg["num_generations"]) #K=4 completions per prompt
    policy_epochs=int(cfg["policy_epochs"]) #config: 1 step per rollout
    eps=float(cfg["clip_epsilon"]) #0.20
    beta=float(cfg["kl_beta"]) #0.10
    max_completion_length=int(cfg["max_completion_length"]) #512, also the dr_grpo constant
    max_grad_norm=float(cfg["max_grad_norm"]) #1.0
    generation_cfg=cfg["generation"] #temperature 0.7, top_p 0.9, sampling
    device=next(policy.parameters()).device #cuda
    results_dir=repo_path(cfg["results_dir"])/run_name #results/task3_grpo/<run_name>/
    results_dir.mkdir(parents=True,exist_ok=True) #make it
    log_path=results_dir/"train_log.jsonl" #one line per update
    if log_path.exists():
        log_path.unlink() #fresh log for this run

    disable_dropout(policy) #ratio exactly 1 at theta=theta_old
    order=list(range(len(prompt_rows))) #0..n-1
    random.Random(int(cfg["seed"])).shuffle(order) #same prompt sequence for the standard run and both forks

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats() #peak vram for this run only
    elapsed=wall_timer() #wall clock from here
    total_generated_tokens=0 #manual: compare forks at equal generated-token budgets

    for update in range(updates):
        #1) this update's prompts, each repeated K times
        prompts=[] #K copies of each prompt's messages
        group_list=[] #which prompt each completion belongs to
        prompt_ids=[] #fixed ids for the log
        for k in range(prompts_per_update):
            row=prompt_rows[order[(update*prompts_per_update+k)%len(order)]] #next prompt in the fixed order
            prompt_ids.append(row.get("prompt_id")) #record it
            for copy in range(group_size):
                prompts.append(prompt_messages(row)) #same prompt K times
                group_list.append(k) #group id = which prompt
        group_ids=torch.tensor(group_list,device=device) #[K*prompts]

        #2) sample K completions per prompt with the current policy (pi_old)
        generated=batch_generate(
            policy,
            tokenizer,
            prompts,
            max_prompt_length=int(cfg["max_prompt_length"]), #256
            max_new_tokens=max_completion_length, #512
            temperature=float(generation_cfg["temperature"]),
            top_p=float(generation_cfg["top_p"]),
            do_sample=bool(generation_cfg["do_sample"]),
        )
        sequences=generated["sequences"].clone() #clone: inference-mode tensors cannot go through backward
        attention_mask=generated["attention_mask"].clone() #prompt mask + ones on the response
        response_ids=generated["response_ids"].clone() #sampled tokens
        token_mask=generated["response_mask"].clone().float() #valid tokens up to and including eos
        prompt_width=generated["prompt_width"] #where the completions start
        loss_mask=token_mask #tokens that enter the loss
        if bool(cfg["mask_truncated_completions"])==True:
            loss_mask=mask_truncated_sequences(token_mask,generated["truncated"]) #config: completions that hit 512 get an all-zero mask

        #3) rewards and group-relative advantages
        with torch.no_grad():
            rewards=score_reward_pairs(reward_model,reward_tokenizer,prompts,generated["responses"],max_length=1280).to(device).float() #same reward model and 1280 limit as task 2
        advantages=group_relative_advantages(rewards,group_ids) #fixed: (r - group mean)/(group std + eps), one per completion
        group_stds=[] #within-group reward std per prompt
        uninformative=0 #groups where every completion got the same reward
        for k in range(prompts_per_update):
            std=rewards[group_ids==k].std(unbiased=False) #same std as the helper
            group_stds.append(float(std.item())) #record
            if float(std.item())<=1e-6:
                uninformative=uninformative+1 #helper tolerance: advantage is all zeros, no learning signal

        #4) reference log-probs, no gradient
        with reference_mode(policy):
            ref_logp=rows_logprobs(policy,sequences,attention_mask,prompt_width,response_ids) #log pi_ref, adapter off

        #5) policy step(s)
        for epoch in range(policy_epochs):
            current_logp=rows_logprobs(policy,sequences,attention_mask,prompt_width,response_ids) #log pi_theta before this step
            if epoch==0:
                old_logp=current_logp.clone() #pi_old = the policy that sampled the completions
            #loss value with nothing attached to the graph, for the log
            loss,stats=grpo_policy_loss(current_logp,old_logp,advantages,loss_mask,ref_logp,eps,beta,loss_type,max_completion_length)
            if torch.isfinite(loss)==False:
                raise RuntimeError("grpo loss is not finite at update "+str(update+1)) #fail loudly
            #gradient one completion at a time: put a grad-carrying row k into the detached full batch,
            #call the same grpo_policy_loss, backward. the loss is a sum over rows, so adding the K backward
            #passes gives exactly the full-batch gradient, with only one row of activations in memory
            optimizer.zero_grad() #clear
            for row in range(sequences.shape[0]):
                if float(loss_mask[row].sum().item())==0.0:
                    continue #masked (truncated) completion contributes nothing to the loss
                row_logp,row_logits=response_token_logprobs(policy,sequences[row:row+1],attention_mask[row:row+1],prompt_width,response_ids[row:row+1]) #with gradient
                del row_logits #not needed
                pieces=[] #full [K, steps] tensor, only row k carries gradient
                for other in range(sequences.shape[0]):
                    if other==row:
                        pieces.append(row_logp.float()) #this completion, with gradient
                    else:
                        pieces.append(current_logp[other:other+1]) #others, detached
                row_loss,_=grpo_policy_loss(torch.cat(pieces,dim=0),old_logp,advantages,loss_mask,ref_logp,eps,beta,loss_type,max_completion_length) #same objective
                row_loss.backward() #gradient flows only through this completion
            grad_norm=torch.nn.utils.clip_grad_norm_(trainable_parameters(policy),max_grad_norm) #norm before clipping
            optimizer.step() #one policy update

        lengths=token_mask.sum(-1) #tokens per completion
        total_generated_tokens=total_generated_tokens+int(lengths.sum().item()) #running budget
        completions=[] #per completion detail, used by compare_normalization for the length-conditioned analysis
        for row in range(sequences.shape[0]):
            completions.append({
                "group":int(group_list[row]), #which prompt
                "tokens":int(lengths[row].item()), #completion length
                "reward":float(rewards[row].item()), #learned reward
                "advantage":float(advantages[row].item()), #group-relative advantage
                "truncated":bool(generated["truncated"][row]), #hit 512
                "in_loss":bool(float(loss_mask[row].sum().item())>0.0), #false if masked out
            })
        masked_count=0 #completions removed from the loss
        for completion in completions:
            if completion["in_loss"]==False:
                masked_count=masked_count+1 #count
        record={
            "run_name":run_name, #standard / norm_grpo / norm_dr_grpo
            "loss_type":loss_type, #grpo or dr_grpo
            "update":update+1, #1-based
            "prompt_ids":prompt_ids, #which prompt(s)
            "reward_mean":float(rewards.mean().item()), #mean learned reward of the K completions
            "group_reward_std_mean":float(torch.tensor(group_stds).mean().item()), #manual: mean within-group reward std
            "uninformative_fraction":uninformative/prompts_per_update, #manual: fraction of zero-std groups
            "kl_from_reference":float(sampled_kl(old_logp,ref_logp,token_mask).item()), #same sampled estimator as tasks 1-2
            "kl_k3":float(stats["sampled_kl"].item()), #the k3 estimate inside the loss (lecture 12)
            "policy_loss":float(loss.item()), #full objective incl. beta*kl
            "policy_term":float(stats["policy_term"].item()), #clipped part only
            "grad_norm":float(grad_norm), #before clipping
            "entropy":float(sample_entropy(old_logp,token_mask).item()), #mean token entropy (sampled)
            "clip_fraction":float(stats["clip_fraction"].item()), #ratio outside the band
            "response_length":float(lengths.float().mean().item()), #mean completion tokens
            "truncated":int(sum(generated["truncated"])), #completions that hit 512
            "masked_from_loss":masked_count, #completions with an all-zero loss mask
            "completions":completions, #per completion detail
            "generated_tokens_so_far":total_generated_tokens, #budget check
            "seconds":elapsed(), #wall clock so far
        }
        append_jsonl(log_path,record) #one json line per update
        print("update",update+1,"reward",round(record["reward_mean"],3),"group_std",round(record["group_reward_std_mean"],3),"kl",round(record["kl_from_reference"],4),"len",record["response_length"]) #progress

    policy.save_pretrained(str(out)) #grpo lora (task 4 reads outputs/task3_grpo/standard)
    tokenizer.save_pretrained(str(out)) #self-contained adapter folder

    peak_vram=None #none on cpu
    if torch.cuda.is_available():
        peak_vram=torch.cuda.max_memory_allocated()/2**30 #GiB, manual wants this for the standard run
    summary={
        "run_name":run_name, #condition
        "loss_type":loss_type, #grpo or dr_grpo
        "updates":updates, #update budget
        "prompts_per_update":prompts_per_update, #1
        "num_generations":group_size, #K=4
        "clip_epsilon":eps, #0.20
        "kl_beta":beta, #0.10
        "max_completion_length":max_completion_length, #512
        "mask_truncated_completions":bool(cfg["mask_truncated_completions"]), #true
        "seed":int(cfg["seed"]), #6304
        "total_generated_tokens":total_generated_tokens, #token budget actually used
        "wall_clock_seconds":elapsed(), #manual: wall-clock time
        "peak_vram_gib":peak_vram, #manual: peak vram
        "output":str(out), #adapter folder
    }
    save_json(results_dir/"train_summary.json",summary) #small json for the report
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()
