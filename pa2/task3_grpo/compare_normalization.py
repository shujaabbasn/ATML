from __future__ import annotations

import argparse
from common.data import load_yaml

import numpy as np
import pandas as pd

from common.data import read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from common.metrics import safe_corr
from common.models import clear_gpu
from task3_grpo.continue_train import run_grpo
from task3_grpo.evaluate import evaluate_policy, load_evaluation_bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--skip-train",action="store_true") #forks already trained, only evaluate/analyse
    ap.add_argument("--limit",type=int) #same eval prefix for both forks if you need to save time
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Fork updates:", cfg["fork_updates"])
    print("Compare loss_type='grpo' vs loss_type='dr_grpo' from the identical supplied midpoint.")
    results_dir=repo_path(cfg["results_dir"]) #results/task3_grpo/
    max_completion_length=int(cfg["max_completion_length"]) #512, the dr_grpo constant
    summary_rows=[] #one row per loss type
    length_rows=[] #length-conditioned statistics per loss type and advantage sign

    for loss_type in ["grpo","dr_grpo"]:
        run_name="norm_"+loss_type #norm_grpo, norm_dr_grpo
        output="outputs/task3_grpo/"+run_name #adapter folder (git ignored)
        if args.skip_train==False:
            run_grpo(args.config,output,int(cfg["fork_updates"]),loss_type,run_name) #same midpoint, prompts, seed, reward, kl, eps; only the normalization changes
            clear_gpu() #free training models
        bundle=load_evaluation_bundle(args.config,output) #same loader for both
        summary=evaluate_policy(bundle,output,run_name,8,args.limit) #same held-out protocol
        del bundle #drop models
        clear_gpu() #free memory
        train_summary=load_json(results_dir/run_name/"train_summary.json") #budget
        log=read_jsonl(results_dir/run_name/"train_log.jsonl") #per update, with per-completion detail

        #length-conditioned statistic: how much gradient each completion gets per token and in total.
        #grpo divides a completion's token sum by its own length T, so per-token weight = |A|/T and total = |A|
        #dr_grpo divides by the constant 512, so per-token weight = |A|/512 and total = |A|*T/512
        completions=[] #every completion that entered the loss
        grad_norms=[] #per update
        lengths_by_update=[] #per update mean length
        for record in log:
            grad_norms.append(record["grad_norm"]) #collect
            lengths_by_update.append(record["response_length"]) #collect
            for completion in record["completions"]:
                if completion["in_loss"]==True:
                    if completion["tokens"]>0:
                        completions.append(completion) #only completions that got a gradient
        for sign in ["positive","negative"]:
            tokens=[] #lengths
            per_token=[] #per-token gradient weight
            totals=[] #total sequence weight
            for completion in completions:
                keep=False #does this completion match the sign
                if sign=="positive":
                    if completion["advantage"]>0:
                        keep=True #better than its group
                else:
                    if completion["advantage"]<0:
                        keep=True #worse than its group
                if keep==True:
                    length=completion["tokens"] #T
                    size=abs(completion["advantage"]) #|A|
                    tokens.append(length) #record
                    if loss_type=="grpo":
                        per_token.append(size/length) #|A|/T
                        totals.append(size) #|A|
                    else:
                        per_token.append(size/max_completion_length) #|A|/L_max
                        totals.append(size*length/max_completion_length) #|A|*T/L_max
            long_share=None #share of total weight on completions longer than the median
            if len(tokens)>0:
                median=float(np.median(tokens)) #split point
                long_total=0.0 #weight on long completions
                for index in range(len(tokens)):
                    if tokens[index]>median:
                        long_total=long_total+totals[index] #add
                if sum(totals)>0:
                    long_share=long_total/sum(totals) #fraction of the gradient mass on long completions
            mean_length=None #stays none if no completion has this sign
            if len(tokens)>0:
                mean_length=float(sum(tokens)/len(tokens)) #average length of these completions
            length_rows.append({
                "loss_type":loss_type, #grpo or dr_grpo
                "advantage_sign":sign, #positive or negative advantage
                "completions":len(tokens), #how many
                "corr_length_per_token_weight":safe_corr(tokens,per_token), #negative for grpo: short completions get bigger per-token updates
                "long_completion_weight_share":long_share, #share of total update mass on above-median-length completions
                "mean_length":mean_length, #average length of these completions
            })
        summary_rows.append({
            "condition":run_name, #fork
            "loss_type":loss_type, #grpo or dr_grpo
            "updates":train_summary["updates"], #matched update budget
            "generated_tokens":train_summary["total_generated_tokens"], #token budget actually used
            "heldout_reward":summary["reward_mean"], #learned reward
            "heldout_kl":summary["kl_from_reference"], #drift
            "heldout_length_mean":summary["response_length"]["mean"], #generated tokens
            "heldout_length_std":summary["response_length"]["std"], #spread
            "heldout_truncation_rate":summary["truncation_rate"], #hit 768
            "train_length_first":lengths_by_update[0], #mean completion length at update 1
            "train_length_last":lengths_by_update[-1], #mean completion length at the last update
            "train_grad_norm_mean":float(np.mean(grad_norms)), #average gradient norm
        })

    summary_table=pd.DataFrame(summary_rows) #held-out comparison
    length_table=pd.DataFrame(length_rows) #length-conditioned statistics
    summary_table.to_csv(results_dir/"normalization.csv",index=False) #for the report
    length_table.to_csv(results_dir/"normalization_length_weights.csv",index=False) #for the report
    save_json(results_dir/"normalization.json",{"heldout":summary_rows,"length_weights":length_rows}) #json copy
    print(summary_table.to_string()) #quick look
    print(length_table.to_string())


if __name__ == "__main__":
    main()
