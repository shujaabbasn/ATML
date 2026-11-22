"""Paired 95% bootstrap intervals for the adaptive-vs-uniform differences (P3-P2, Q3-Q2) and the summary table.
Perplexity: resample the same scored tokens for both models. HellaSwag: resample the same examples.

python -m task1_compression.compare
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from common.utils import load_yaml, load_json, read_jsonl, save_json, repo_path, bootstrap_ppl_diff, bootstrap_mean_ci


PAIRS=[("p3_owl_s50","p2_wanda_s50"),("p3_owl_s75","p2_wanda_s75"),("q3_mixed_match_int4","q2_awq_int4"),("q3_mixed_match_int3","q2_awq_int3"),("nm24_wanda_s50","p2_wanda_s50")]


def hellaswag_column(folder,column):
    rows=read_jsonl(folder/"hellaswag_scores.jsonl")
    values=[]
    for row in rows:
        values.append(row[column])
    return np.array(values,dtype=np.float64)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/task1.yaml")
    args=parser.parse_args()
    config=load_yaml(args.config)
    out=repo_path(config["results_dir"])
    repeats=config["eval"]["bootstrap"]
    seed=config["seed"]
    comparisons=[]
    for adaptive,uniform in PAIRS:
        a=out/adaptive
        u=out/uniform
        if (a/"metrics.json").exists()==False or (u/"metrics.json").exists()==False:
            print("skip",adaptive,uniform,"(not run yet)")
            continue
        nll_a=np.load(a/"token_nll.npz")
        nll_u=np.load(u/"token_nll.npz")
        entry={"adaptive":adaptive,"uniform":uniform}
        for suite in ["wikitext","c4_val"]:
            entry[f"{suite}_ppl_diff"]=bootstrap_ppl_diff(nll_a[suite],nll_u[suite],nll_a[suite+"_lengths"],repeats,seed)
        for column in ["acc","acc_norm"]:
            difference=hellaswag_column(a,column)-hellaswag_column(u,column) #paired per example
            entry[f"hellaswag_{column}_diff"]=bootstrap_mean_ci(difference,repeats,seed)
        comparisons.append(entry)
    save_json(out/"paired_bootstrap.json",comparisons)
    #one row per condition, every number comes from the metrics files
    rows=[]
    for folder in sorted(out.iterdir()):
        if (folder/"metrics.json").exists()==False:
            continue
        m=load_json(folder/"metrics.json")
        row={"condition":m["condition"],"wikitext2_ppl":m["wikitext2_ppl"]["ppl"],"c4_val_ppl":m["c4_val_ppl"]["ppl"],"hellaswag_acc":m["hellaswag"]["acc"],"hellaswag_acc_norm":m["hellaswag"]["acc_norm"],
            "achieved_sparsity":m["achieved_sparsity_eligible"],"eligible_packed_bytes":m["storage"]["eligible_packed_bytes"],"total_packed_bytes":m["storage"]["total_packed_bytes"],
            "bits_per_eligible_weight":m["storage"]["bits_per_eligible_weight"],"compression_ratio_total":m["storage"]["compression_ratio_total"]}
        if "resources" in m:
            for key in m["resources"]:
                row[key+"_mean_ms"]=m["resources"][key]["mean_ms"] #prefill_512, prefill_2048, decode_64
                row[key+"_peak_mb"]=m["resources"][key]["peak_mb"]
                if "per_token_ms" in m["resources"][key]:
                    row["decode_ms_per_token"]=m["resources"][key]["per_token_ms"]
        rows.append(row)
    if len(rows)>0:
        columns=[]
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
        with open(out/"summary.csv","w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    print("wrote",out/"paired_bootstrap.json","and summary.csv")


if __name__=="__main__":
    main()
