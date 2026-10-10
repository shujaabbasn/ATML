from __future__ import annotations

import argparse

from common.data import load_yaml, read_jsonl
from common.models import load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward

import time

import numpy as np

from common.data import repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import save_json, set_seed
from common.models import clear_gpu
from task5_feedback.rlvr import extract_designated_final


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def generate_policy_responses(cfg,tokenizer,rows,policy_name,batch_size=8):
    """One deterministic (greedy) response per problem, same prompt order and cap for every policy."""
    model=load_frozen_policy(cfg,policy_name) #sft = base model, rlvr / rlaif = supplied frozen adapters
    records=[] #one per problem
    for start in range(0,len(rows),batch_size):
        chunk=rows[start:start+batch_size] #fixed order
        prompts=[] #chat messages, they already ask for '#### <number>'
        for row in chunk:
            prompts.append(row["messages"]) #same instruction for every policy
        generated=batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=512, #gsm/svamp prompts are short, nothing gets cut
            max_new_tokens=int(cfg["math_max_new_tokens"]), #512 from feedback.yaml
            temperature=0.0,
            top_p=1.0,
            do_sample=False, #greedy: one deterministic answer per policy, like task 4
        )
        for i in range(len(chunk)):
            row=chunk[i] #the problem
            response=generated["responses"][i] #decoded answer
            records.append({
                "source_index":row["source_index"], #fixed id
                "policy":policy_name, #sft / rlvr / rlaif
                "question":row["question"], #problem text, also what the judge sees
                "gold_final":str(row["gold_final"]), #gold answer
                "response":response, #model output
                "predicted_final":extract_designated_final(response), #none if there is no '#### <number>'
                "exact_reward":exact_reward(response,str(row["gold_final"])), #rlvr verifier: 1 if the designated final matches
                "response_tokens":int(generated["response_lengths"][i]), #generated tokens, padding excluded
                "truncated":bool(generated["truncated"][i]), #hit the 512 cap
            })
    del model #drop the policy
    clear_gpu() #free memory for the next one
    return records


def policy_summary(records):
    """Exact accuracy, format compliance and length for one policy."""
    rewards=[] #exact verifier rewards
    formatted=0 #responses with a parsable '#### <number>'
    lengths=[] #response tokens
    truncated=0 #hit the cap
    for record in records:
        rewards.append(record["exact_reward"]) #collect
        lengths.append(record["response_tokens"]) #collect
        if record["predicted_final"] is not None:
            formatted=formatted+1 #manual: format compliance ignores correctness
        if record["truncated"]==True:
            truncated=truncated+1 #count
    return {
        "n":len(records), #problems
        "exact_accuracy":float(np.mean(rewards)), #manual: exact-answer accuracy
        "format_compliance":formatted/len(records), #manual: required final format parsed
        "length_mean":float(np.mean(lengths)), #manual: mean response tokens
        "length_std":float(np.std(lengths)), #manual: dispersion
        "length_median":float(np.median(lengths)), #middle response
        "truncation_rate":truncated/len(records), #fraction that hit the cap
    }


def judge_against_sft(judge,policy_records,sft_records):
    """Pairwise ai preference of a policy against sft on the same problem, plus verifier-judge agreement."""
    comparisons=[] #one per problem
    for k in range(len(policy_records)):
        mine=policy_records[k] #policy answer
        base=sft_records[k] #sft answer to the same problem
        preference=judge.compare(mine["question"],mine["response"],base["response"]) #given: A = policy better, B = sft better, TIE
        score=0.5 #tie counts 0.5 (manual)
        if preference=="A":
            score=1.0 #policy wins
        elif preference=="B":
            score=0.0 #policy loses
        comparisons.append({
            "source_index":mine["source_index"], #problem id
            "judge_preference":preference, #A / B / TIE
            "win_score":score, #1 / 0.5 / 0
            "policy_exact_reward":mine["exact_reward"], #verifier on the policy answer
            "sft_exact_reward":base["exact_reward"], #verifier on the sft answer
        })
    scores=[] #win scores
    ties=0 #judge ties
    decisive=0 #pairs where the verifier separates the two answers
    agree=0 #judge prefers the verifier-correct answer
    disagree=0 #judge prefers the verifier-wrong answer
    decisive_ties=0 #judge ties although the verifier separates them
    same_reward=0 #pairs the verifier treats identically
    same_reward_ties=0 #judge also ties on those
    for comparison in comparisons:
        scores.append(comparison["win_score"]) #collect
        if comparison["judge_preference"]=="TIE":
            ties=ties+1 #count
        if comparison["policy_exact_reward"]!=comparison["sft_exact_reward"]:
            decisive=decisive+1 #verifier says one is right and one is wrong
            correct_side="A" #policy is the correct one
            if comparison["sft_exact_reward"]>comparison["policy_exact_reward"]:
                correct_side="B" #sft is the correct one
            if comparison["judge_preference"]==correct_side:
                agree=agree+1 #judge agrees with the verifier
            elif comparison["judge_preference"]=="TIE":
                decisive_ties=decisive_ties+1 #judge cannot separate them
            else:
                disagree=disagree+1 #judge prefers the wrong answer
        else:
            same_reward=same_reward+1 #both right or both wrong
            if comparison["judge_preference"]=="TIE":
                same_reward_ties=same_reward_ties+1 #judge also sees no difference
    summary={
        "n":len(comparisons), #problems compared
        "ai_win_rate_vs_sft":float(np.mean(scores)), #manual: win=1, tie=0.5, loss=0
        "judge_ties":ties, #manual: report ties separately
        "verifier_decisive_pairs":decisive, #pairs where exactly one answer is correct
        "judge_agrees_with_verifier":agree, #judge picks the correct one
        "judge_disagrees_with_verifier":disagree, #judge picks the wrong one
        "judge_ties_on_decisive":decisive_ties, #judge says tie
        "verifier_judge_agreement":None, #filled below
        "verifier_tied_pairs":same_reward, #verifier gives both the same reward
        "judge_ties_on_verifier_tied":same_reward_ties, #judge also ties
        "judge_separates_verifier_tied":same_reward-same_reward_ties, #extra distinctions the judge makes (manual RQ1)
    }
    if decisive>0:
        summary["verifier_judge_agreement"]=agree/decisive #fraction of verifier-decisive pairs where the judge agrees
    return summary,comparisons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--batch-size",type=int,default=8) #generation batch, does not change greedy answers
    args = ap.parse_args()
    cfg, rows, tokenizer = load_math_evaluation(args.config, args.dataset)
    print("Rows:", len(rows))
    print("Policies:", list(policy_specs(cfg)))
    print("Exact verifier available as task5_feedback.rlvr.exact_reward")
    print("Pairwise judge available as task5_feedback.rlaif.PairwiseAIJudge")
    outdir=repo_path(cfg["results_dir"])/"task5_feedback"/args.dataset #results/task5_feedback/gsm or /transfer
    outdir.mkdir(parents=True,exist_ok=True) #make it

    #1) generate with every policy (judge not loaded yet, so only one model is on the gpu)
    set_seed(int(cfg["seed"])) #fixed
    records={} #policy -> list of records
    generation_seconds={} #policy -> wall clock
    for policy_name in policy_specs(cfg):
        start=time.perf_counter() #timer
        records[policy_name]=generate_policy_responses(cfg,tokenizer,rows,policy_name,args.batch_size) #same problems, same order
        generation_seconds[policy_name]=time.perf_counter()-start #seconds
        write_jsonl(outdir/("generated_"+policy_name+".jsonl"),records[policy_name]) #every answer
        print(policy_name,"accuracy",round(policy_summary(records[policy_name])["exact_accuracy"],3)) #progress

    #2) exact verifier summaries (cheap: a regex per answer)
    summary={"dataset":args.dataset,"policies":{},"pairwise_vs_sft":{},"cost":{}} #everything for this dataset
    for policy_name in records:
        summary["policies"][policy_name]=policy_summary(records[policy_name]) #accuracy, format, length

    #3) fixed pairwise ai judge: rlvr vs sft and rlaif vs sft on the same problems
    judge=PairwiseAIJudge(cfg,repo_path(cfg["results_dir"])/"task5_feedback"/"judge_cache.json") #given judge with a/b balancing and cache
    for policy_name in ["rlvr","rlaif"]:
        start=time.perf_counter() #timer
        pair_summary,comparisons=judge_against_sft(judge,records[policy_name],records["sft"]) #win rate + verifier agreement
        pair_summary["judge_seconds"]=time.perf_counter()-start #inference cost of the ai feedback
        summary["pairwise_vs_sft"][policy_name]=pair_summary #save
        write_jsonl(outdir/("judge_"+policy_name+"_vs_sft.jsonl"),comparisons) #every judge decision
        print(policy_name,"vs sft win rate",round(pair_summary["ai_win_rate_vs_sft"],3)) #progress
    summary["cost"]={"generation_seconds":generation_seconds,"judge_calls":2*len(rows)} #for the cost discussion
    save_json(outdir/"metrics.json",summary) #machine-readable results
    print(summary["policies"]) #quick look


if __name__ == "__main__":
    main()
