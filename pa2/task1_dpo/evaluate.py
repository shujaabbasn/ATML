from __future__ import annotations

import argparse

from common.data import load_yaml, read_jsonl
from common.models import load_policy, load_reward_model, load_tokenizer

import torch

from common.data import encode_prompt_response, pad_batch, preference_responses, prompt_messages, prompt_messages_from_preference, repo_path, write_jsonl
from common.evaluation import generation_metrics, pick_examples
from common.generation import response_sequence_logprobs
from common.logging_utils import save_json, set_seed
from common.metrics import safe_corr
from common.models import reference_mode
from task1_dpo.dpo import dpo_loss


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["dpo_standard_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def pair_metrics(policy,tokenizer,rows,beta,max_length,batch_size):
    """Held-out dpo loss and preference accuracy, m_theta from the manual for every pair."""
    device=next(policy.parameters()).device #wherever load_policy put the model
    records=[] #one line per pair
    skipped=0 #pairs whose prompt alone does not fit max_length
    for start in range(0,len(rows),batch_size):
        chunk=rows[start:start+batch_size] #fixed order
        chosen=[] #encoded (ids, response_mask) for y+
        rejected=[] #encoded for y-
        kept=[] #rows that encoded fine
        for row in chunk:
            prompt=prompt_messages_from_preference(row) #chat messages without the final assistant turn
            chosen_text,rejected_text=preference_responses(row) #y+ and y- strings
            try:
                chosen_encoded=encode_prompt_response(tokenizer,prompt,chosen_text,max_length) #same encoder as training
                rejected_encoded=encode_prompt_response(tokenizer,prompt,rejected_text,max_length) #same
            except ValueError:
                skipped=skipped+1 #prompt too long, reported in the summary
                continue
            chosen.append(chosen_encoded) #keep
            rejected.append(rejected_encoded) #keep
            kept.append(row) #keep
        if len(kept)==0:
            continue #whole chunk skipped
        chosen_batch=pad_batch(tokenizer,chosen) #left padded tensors
        rejected_batch=pad_batch(tokenizer,rejected) #same
        for key in chosen_batch:
            chosen_batch[key]=chosen_batch[key].to(device) #to gpu
        for key in rejected_batch:
            rejected_batch[key]=rejected_batch[key].to(device) #to gpu
        with torch.no_grad():
            policy_chosen,_,_=response_sequence_logprobs(policy,chosen_batch) #summed over response tokens only
            policy_rejected,_,_=response_sequence_logprobs(policy,rejected_batch)
            with reference_mode(policy):
                ref_chosen,_,_=response_sequence_logprobs(policy,chosen_batch) #adapter off = reference
                ref_rejected,_,_=response_sequence_logprobs(policy,rejected_batch)
        for i in range(len(kept)):
            row=kept[i] #the pair
            policy_margin=float(policy_chosen[i].item())-float(policy_rejected[i].item()) #log pi(y+) - log pi(y-)
            ref_margin=float(ref_chosen[i].item())-float(ref_rejected[i].item()) #same under the reference
            records.append({
                "prompt_id":row["prompt_id"], #fixed id
                "length_stratum":row.get("length_stratum","all"), #only the length file has strata
                "length_difference":row.get("length_difference"), #chosen tokens - rejected tokens, length file only
                "chosen_response_tokens":float(chosen_batch["response_mask"][i].sum().item()), #after encoding
                "rejected_response_tokens":float(rejected_batch["response_mask"][i].sum().item()),
                "policy_chosen_logp":float(policy_chosen[i].item()), #raw pieces, so anything can be recomputed
                "policy_rejected_logp":float(policy_rejected[i].item()),
                "ref_chosen_logp":float(ref_chosen[i].item()),
                "ref_rejected_logp":float(ref_rejected[i].item()),
                "margin":policy_margin-ref_margin, #manual m_theta, >0 means the pair is ranked correctly
            })
    return records,skipped


def summarise_pairs(records,beta):
    """Loss/accuracy over a list of pair records, through the same dpo_loss used in training."""
    if len(records)==0:
        return {"n_pairs":0} #nothing to score
    policy_chosen=[] #rebuild the four tensors
    policy_rejected=[]
    ref_chosen=[]
    ref_rejected=[]
    margins=[] #m_theta per pair
    length_differences=[] #only in the length file
    for record in records:
        policy_chosen.append(record["policy_chosen_logp"]) #collect
        policy_rejected.append(record["policy_rejected_logp"])
        ref_chosen.append(record["ref_chosen_logp"])
        ref_rejected.append(record["ref_rejected_logp"])
        margins.append(record["margin"])
        if record["length_difference"] is not None:
            length_differences.append(record["length_difference"]) #chosen minus rejected length
    loss,stats=dpo_loss(torch.tensor(policy_chosen),torch.tensor(policy_rejected),torch.tensor(ref_chosen),torch.tensor(ref_rejected),beta) #same objective as training
    correct=0 #pairs with m_theta>0
    for margin in margins:
        if margin>0:
            correct=correct+1 #manual: accuracy is the fraction with m>0
    summary={
        "n_pairs":len(records), #pairs scored
        "dpo_loss":float(loss.item()), #held-out loss at this beta
        "preference_accuracy":correct/len(records), #manual definition, counted by hand
        "preference_accuracy_check":float(stats["preference_accuracy"].item()), #same thing from dpo_loss, should match
        "mean_margin":float(torch.tensor(margins).mean().item()), #average m_theta
    }
    if len(length_differences)==len(margins):
        summary["corr_length_difference_margin"]=safe_corr(length_differences,margins) #does the policy prefer the longer side?
    return summary


def evaluate_adapter(bundle,adapter,name,beta=None,batch_size=8):
    """Full task 1 evaluation protocol for one adapter, results go to results/task1_dpo/<name>/.

    bundle comes from load_evaluation_bundle: cfg, held-out pairs, tokenizer, policy, reward model.
    """
    cfg=bundle["cfg"] #merged config
    tokenizer=bundle["tokenizer"] #left padded tokenizer
    policy=bundle["policy"] #adapter loaded, eval mode
    reward_model,reward_tokenizer=bundle["reward"] #frozen course reward model
    if beta is None:
        beta=float(cfg["beta"]) #standard beta 0.10
    max_length=int(cfg["max_sequence_length"]) #768, same as training
    max_new_tokens=int(cfg["max_generation_tokens"]) #256
    out_dir=repo_path(cfg["results_dir"])/name #results/task1_dpo/<name>/
    set_seed(int(cfg["seed"])) #same sampling stream for every condition

    #1) held-out preference pairs (standard eval file)
    standard_records,standard_skipped=pair_metrics(policy,tokenizer,bundle["rows"],beta,max_length,2) #300 pairs, batch 2 like training: 8 pairs x 768 tokens x 151k vocab logits would not fit on a t4
    standard_summary=summarise_pairs(standard_records,beta) #loss, accuracy, margin
    standard_summary["skipped"]=standard_skipped #prompt too long

    #2) length-stratified pairs, split by stratum (needed for step 3, cheap so every model gets it)
    stratified_rows=read_jsonl(cfg["paths"]["dpo_length_eval"]) #82 per stratum
    stratified_records,stratified_skipped=pair_metrics(policy,tokenizer,stratified_rows,beta,max_length,2) #same batch 2
    groups={} #stratum -> its records
    for record in stratified_records:
        stratum=record["length_stratum"] #preferred_longer / length_matched / rejected_longer
        if stratum not in groups:
            groups[stratum]=[] #first time we see it
        groups[stratum].append(record) #add
    by_stratum={} #stratum -> summary
    for stratum in groups:
        by_stratum[stratum]=summarise_pairs(groups[stratum],beta) #per-stratum accuracy and loss
    by_stratum["all"]=summarise_pairs(stratified_records,beta) #whole stratified set
    by_stratum["skipped"]=stratified_skipped #prompt too long

    #3) generations on the held-out prompts: reward, kl, length
    scored_ids={} #prompt_ids that fit max_length (the pairs actually scored above)
    for record in standard_records:
        scored_ids[record["prompt_id"]]=True #mark
    heldout_prompts=[] #prompt_id + messages, fixed order
    for row in bundle["rows"]:
        if row["prompt_id"] in scored_ids:
            heldout_prompts.append({"prompt_id":row["prompt_id"],"messages":prompt_messages_from_preference(row)}) #same prompts as the scored pairs, long ones would be cut by batch_generate
    generation_summary,generation_records=generation_metrics(
        policy,tokenizer,reward_model,reward_tokenizer,heldout_prompts,cfg,
        max_new_tokens=max_new_tokens,
        max_prompt_length=max_length, #big enough that dpo prompts are not cut (prompts_truncated reports it)
        batch_size=batch_size,
        reward_max_length=1280, #prompt (<768) + response (<=256) + template tokens can pass 1024, 1280 = ppo.yaml reward_max_length so nothing gets cut
    )

    #4) the common word-limit prompt set (10 prompts tracked in the repo)
    word_rows=read_jsonl(cfg["paths"]["word_limit_prompts"]) #fixed file
    word_prompts=[] #prompt_id + messages
    for row in word_rows:
        word_prompts.append({"prompt_id":row["prompt_id"],"messages":prompt_messages(row)}) #messages field
    word_summary,word_records=generation_metrics(
        policy,tokenizer,reward_model,reward_tokenizer,word_prompts,cfg,
        max_new_tokens=max_new_tokens,
        max_prompt_length=max_length,
        batch_size=batch_size,
        reward_max_length=1280, #same as above
    )

    summary={
        "name":name, #condition name
        "adapter":str(adapter), #which checkpoint
        "beta":beta, #used for the held-out dpo loss
        "heldout_pairs":standard_summary, #loss / accuracy on the standard eval file
        "length_strata":by_stratum, #per-stratum accuracy
        "heldout_generation":generation_summary, #reward, kl, length on held-out prompts
        "word_limit_prompts":word_summary, #length and word-limit compliance on the common set
    }
    save_json(out_dir/"eval_summary.json",summary) #main numbers for the report
    write_jsonl(out_dir/"heldout_pairs.jsonl",standard_records) #per pair margins
    write_jsonl(out_dir/"stratified_pairs.jsonl",stratified_records) #per pair margins with strata
    write_jsonl(out_dir/"heldout_generations.jsonl",generation_records) #every generated response
    write_jsonl(out_dir/"word_limit_generations.jsonl",word_records) #every word-limit response
    save_json(out_dir/"examples.json",pick_examples(generation_records)) #candidates for the qualitative part
    return summary


def summary_row(summary,budget):
    """One flat row for the comparison tables (beta ablation, length study)."""
    pairs=summary["heldout_pairs"] #shortcut
    generation=summary["heldout_generation"] #shortcut
    words=summary["word_limit_prompts"] #shortcut
    strata=summary["length_strata"] #shortcut
    row={
        "condition":summary["name"], #run name
        "budget":budget, #full_epoch or short_600, manual wants this visible
        "beta":summary["beta"], #dpo beta
        "heldout_dpo_loss":pairs.get("dpo_loss"), #loss at this beta
        "heldout_preference_accuracy":pairs.get("preference_accuracy"), #fraction m>0
        "heldout_mean_margin":pairs.get("mean_margin"), #average m
        "kl_from_reference":generation.get("kl_from_reference"), #sampled kl
        "reward_mean":generation.get("reward_mean"), #reward-model score
        "reward_std":generation.get("reward_std"),
        "length_mean":generation.get("response_length",{}).get("mean"), #generated tokens
        "length_std":generation.get("response_length",{}).get("std"),
        "length_iqr":generation.get("response_length",{}).get("iqr"),
        "truncation_rate":generation.get("truncation_rate"), #hit the 256 cap
        "word_limit_compliance":words.get("word_limit_compliance"), #common prompt set
        "word_prompt_length_mean":words.get("response_length",{}).get("mean"), #length on the common set
    }
    for stratum in ["preferred_longer","length_matched","rejected_longer"]:
        if stratum in strata:
            row["accuracy_"+stratum]=strata[stratum].get("preference_accuracy") #per-stratum accuracy
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--beta",type=float) #held-out dpo loss at this beta, defaults to the config beta
    ap.add_argument("--batch-size",type=int,default=8) #generation batch, does not change results
    args = ap.parse_args()
    bundle = load_evaluation_bundle(args.config, args.adapter)
    summary=evaluate_adapter(bundle,args.adapter,args.name,args.beta,args.batch_size) #runs everything and saves
    print(summary["heldout_pairs"]) #quick look in the terminal
    print(summary["heldout_generation"])


if __name__ == "__main__":
    main()
