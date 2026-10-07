from __future__ import annotations

import argparse
from common.data import load_yaml

import pandas as pd

from common.data import read_jsonl, repo_path
from common.logging_utils import save_json
from common.models import clear_gpu
from task2_ppo.continue_train import run_ppo
from task2_ppo.evaluate import evaluate_policy, load_evaluation_bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--skip-train",action="store_true") #forks already trained, only evaluate them
    ap.add_argument("--limit",type=int) #same eval prefix for every fork if you need to save time
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("KL beta conditions:", cfg["kl_values"])
    print("Fork update budget:", cfg["fork_updates"])
    results_dir=repo_path(cfg["results_dir"]) #results/task2_ppo/
    summary_rows=[] #one row per kl beta (held-out numbers)
    trajectory_rows=[] #one row per (kl beta, update) for the "what changes first" question

    for kl_beta in cfg["kl_values"]:
        kl_beta=float(kl_beta) #0.0, 0.1, 0.2
        run_name="kl_"+str(kl_beta) #kl_0.0, kl_0.1, kl_0.2
        output="outputs/task2_ppo/"+run_name #adapter folder (git ignored)
        if args.skip_train==False:
            run_ppo(args.config,output,int(cfg["fork_updates"]),None,kl_beta,run_name) #same midpoint, prompts, seed; only kl_beta changes (eps stays 0.20)
            clear_gpu() #free training models
        bundle=load_evaluation_bundle(args.config,output) #same loader for every fork
        summary=evaluate_policy(bundle,output,run_name,8,args.limit) #same held-out protocol
        del bundle #drop models
        clear_gpu() #free memory
        summary_rows.append({
            "condition":run_name, #fork
            "kl_beta":kl_beta, #setting
            "heldout_reward":summary["reward_mean"], #learned reward
            "heldout_effective_reward":summary["effective_reward_mean"], #with eos penalty
            "heldout_kl":summary["kl_from_reference"], #policy drift
            "heldout_entropy":summary["entropy"], #diversity diagnostic
            "heldout_length_mean":summary["response_length"]["mean"], #generated tokens
            "heldout_length_std":summary["response_length"]["std"], #spread
            "heldout_truncation_rate":summary["truncation_rate"], #hit 768
            "heldout_reward_length_corr":summary["reward_length_corr"], #is extra reward just extra length?
        })
        for record in read_jsonl(results_dir/run_name/"train_log.jsonl"):
            trajectory_rows.append({
                "condition":run_name, #fork
                "kl_beta":kl_beta, #setting
                "update":record["update"], #1..8
                "reward_raw":record["reward_raw"], #rollout reward
                "kl_from_reference":record["kl_from_reference"], #rollout drift
                "entropy":record["entropy"], #rollout entropy
                "response_length":record["response_length"], #rollout length
                "clip_fraction":record["clip_fraction"], #for context
            })

    summary_table=pd.DataFrame(summary_rows) #held-out comparison
    trajectory_table=pd.DataFrame(trajectory_rows) #per-update curves
    summary_table.to_csv(results_dir/"kl_ablation.csv",index=False) #for the report
    trajectory_table.to_csv(results_dir/"kl_trajectories.csv",index=False) #for the plots
    save_json(results_dir/"kl_ablation.json",{"heldout":summary_rows,"trajectories":trajectory_rows}) #json copy
    print(summary_table.to_string()) #quick look


if __name__ == "__main__":
    main()
