from __future__ import annotations

import argparse
from common.data import load_yaml, read_jsonl

import pandas as pd

from common.data import repo_path
from common.logging_utils import load_json, save_json
from common.models import clear_gpu
from task1_dpo.evaluate import evaluate_adapter, load_evaluation_bundle, summary_row
from task1_dpo.train import run_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--skip-train",action="store_true") #reuse the length-balanced adapter if it already exists
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    balanced = read_jsonl(cfg["paths"]["dpo_length_train"])
    stratified = read_jsonl(cfg["paths"]["dpo_length_eval"])
    print("Length-balanced train rows:", len(balanced))
    print("Length-stratified eval rows:", len(stratified))
    results_dir=repo_path(cfg["results_dir"]) #results/task1_dpo/

    #1) train the length-balanced model from the original initialization, same settings as standard except the data
    if args.skip_train==False:
        run_training(args.config,"length_balanced",cfg["paths"]["dpo_length_train"],cfg["length_output"]) #full epoch, beta 0.10
        clear_gpu() #free memory before evaluation

    #2) evaluate both models with the same protocol (stratified pairs, held-out generations, word-limit set)
    summaries={} #condition -> eval summary
    standard_summary_path=results_dir/"standard"/"eval_summary.json" #from task1_dpo.evaluate --name standard
    if standard_summary_path.exists():
        summaries["standard"]=load_json(standard_summary_path) #already evaluated, same protocol
    else:
        bundle=load_evaluation_bundle(args.config,cfg["standard_output"]) #standard adapter
        summaries["standard"]=evaluate_adapter(bundle,cfg["standard_output"],"standard") #evaluate it now
        del bundle #drop models
        clear_gpu() #free memory
    bundle=load_evaluation_bundle(args.config,cfg["length_output"]) #length-balanced adapter
    summaries["length_balanced"]=evaluate_adapter(bundle,cfg["length_output"],"length_balanced") #same protocol
    del bundle #drop models
    clear_gpu() #free memory

    #3) per-stratum table: one row per (condition, stratum)
    strata_rows=[] #long format so it plots easily
    for condition in summaries:
        strata=summaries[condition]["length_strata"] #per-stratum summaries
        for stratum in ["preferred_longer","length_matched","rejected_longer","all"]:
            if stratum in strata:
                stratum_summary=strata[stratum] #numbers for this stratum
                strata_rows.append({
                    "condition":condition, #standard or length_balanced
                    "stratum":stratum, #which length relation
                    "n_pairs":stratum_summary.get("n_pairs"), #82 each
                    "preference_accuracy":stratum_summary.get("preference_accuracy"), #fraction m>0
                    "dpo_loss":stratum_summary.get("dpo_loss"), #held-out loss
                    "mean_margin":stratum_summary.get("mean_margin"), #average m
                    "corr_length_difference_margin":stratum_summary.get("corr_length_difference_margin"), #length sensitivity
                })
    strata_table=pd.DataFrame(strata_rows) #table
    strata_table.to_csv(results_dir/"length_strata.csv",index=False) #for the report

    #4) generated length and word-limit compliance side by side
    comparison_rows=[] #one row per condition
    for condition in summaries:
        comparison_rows.append(summary_row(summaries[condition],"full_epoch")) #both are one full epoch
    comparison_table=pd.DataFrame(comparison_rows) #table
    comparison_table.to_csv(results_dir/"length_comparison.csv",index=False) #for the report
    save_json(results_dir/"length_analysis.json",{"strata":strata_rows,"comparison":comparison_rows}) #json copy
    print(strata_table.to_string()) #quick look
    print(comparison_table.to_string())


if __name__ == "__main__":
    main()
