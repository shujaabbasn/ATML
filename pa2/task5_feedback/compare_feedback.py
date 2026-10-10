from __future__ import annotations

import argparse
from common.data import load_yaml

import pandas as pd

from common.data import repo_path
from common.logging_utils import load_json, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    base=repo_path(cfg["results_dir"])/"task5_feedback" #results/task5_feedback/
    gsm=load_json(base/"gsm"/"metrics.json") #in-domain
    transfer=load_json(base/"transfer"/"metrics.json") #out-of-domain
    diagnostics=load_json(base/"diagnostics"/"diagnostics_summary.json") #controlled set

    #1) in-domain vs transfer per policy, with the drop (manual: drop from the in-domain metric)
    rows=[] #one row per policy
    for policy_name in ["sft","rlvr","rlaif"]:
        in_domain=gsm["policies"][policy_name] #gsm numbers
        out_domain=transfer["policies"][policy_name] #svamp numbers
        row={
            "policy":policy_name, #sft / rlvr / rlaif
            "gsm_exact_accuracy":in_domain["exact_accuracy"], #in-domain accuracy
            "svamp_exact_accuracy":out_domain["exact_accuracy"], #transfer accuracy
            "accuracy_drop":in_domain["exact_accuracy"]-out_domain["exact_accuracy"], #in-domain minus transfer
            "gsm_format_compliance":in_domain["format_compliance"], #'#### <number>' parsed
            "svamp_format_compliance":out_domain["format_compliance"],
            "gsm_length_mean":in_domain["length_mean"], #tokens
            "svamp_length_mean":out_domain["length_mean"],
            "gsm_ai_win_rate_vs_sft":None, #sft is the reference itself
            "svamp_ai_win_rate_vs_sft":None,
            "win_rate_drop":None,
        }
        if policy_name!="sft":
            row["gsm_ai_win_rate_vs_sft"]=gsm["pairwise_vs_sft"][policy_name]["ai_win_rate_vs_sft"] #in-domain ai preference
            row["svamp_ai_win_rate_vs_sft"]=transfer["pairwise_vs_sft"][policy_name]["ai_win_rate_vs_sft"] #transfer ai preference
            row["win_rate_drop"]=row["gsm_ai_win_rate_vs_sft"]-row["svamp_ai_win_rate_vs_sft"] #drop
            row["gsm_verifier_judge_agreement"]=gsm["pairwise_vs_sft"][policy_name]["verifier_judge_agreement"] #judge agrees with the verifier when it separates the answers
            row["svamp_verifier_judge_agreement"]=transfer["pairwise_vs_sft"][policy_name]["verifier_judge_agreement"]
        rows.append(row) #add
    table=pd.DataFrame(rows) #comparison table
    table.to_csv(base/"feedback_comparison.csv",index=False) #for the report

    #2) cost: the verifier is a regex, the ai judge is a 3b model call per comparison
    cost={
        "gsm_judge_seconds":gsm["pairwise_vs_sft"]["rlvr"]["judge_seconds"]+gsm["pairwise_vs_sft"]["rlaif"]["judge_seconds"], #in-domain ai feedback time
        "gsm_judge_calls":gsm["cost"]["judge_calls"],
        "transfer_judge_seconds":transfer["pairwise_vs_sft"]["rlvr"]["judge_seconds"]+transfer["pairwise_vs_sft"]["rlaif"]["judge_seconds"],
        "transfer_judge_calls":transfer["cost"]["judge_calls"],
        "diagnostics_judge_seconds":diagnostics["judge_seconds"], #200 comparisons
    }
    save_json(base/"feedback_comparison.json",{"policies":rows,"sensitivities":diagnostics["sensitivities"],"controlled_pairs":diagnostics["controlled_pairs"],"cost":cost}) #everything in one file
    print(table.to_string()) #quick look
    print("sensitivities:",diagnostics["sensitivities"])
    print("cost:",cost)


if __name__ == "__main__":
    main()
