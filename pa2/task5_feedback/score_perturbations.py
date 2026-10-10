from __future__ import annotations

import argparse
from collections import defaultdict

from common.data import load_yaml, read_jsonl
from task5_feedback.rlvr import exact_reward
from task5_feedback.rlaif import PairwiseAIJudge

import time

import numpy as np
import pandas as pd

from common.data import repo_path, write_jsonl
from common.logging_utils import save_json

#variant order inside each problem; clean_correct first so every pair below is (clean, other)
VARIANT_ORDER=["clean_correct","corrupt_reasoning_correct_final","good_reasoning_wrong_final","persuasive_filler_correct","gold_distractor_wrong_final"]
#controlled pairs: (better, worse, what is perturbed). better is the diagnostically better response
PAIRS=[
    ("clean_correct","corrupt_reasoning_correct_final","reasoning"), #same correct final, reasoning corrupted -> S_reason
    ("clean_correct","good_reasoning_wrong_final","outcome"), #reasoning held, final changed -> S_outcome
    ("clean_correct","gold_distractor_wrong_final","outcome_distractor"), #gold shown as a rejected candidate, wrong final -> S_outcome
    ("clean_correct","persuasive_filler_correct","filler"), #same correct answer plus persuasive filler -> verbosity test
]

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    print("Diagnostic problems:", len(groups))
    print("Variants/problem:", sorted(EXPECTED_VARIANTS))
    print("Use exact_reward(...) for RLVR and PairwiseAIJudge(...) for RLAIF.")
    outdir=repo_path(cfg["results_dir"])/"task5_feedback"/"diagnostics" #results/task5_feedback/diagnostics
    outdir.mkdir(parents=True,exist_ok=True) #make it
    judge=PairwiseAIJudge(cfg,repo_path(cfg["results_dir"])/"task5_feedback"/"judge_cache.json") #given judge

    #1) score every variant with both mechanisms
    scored=[] #one row per (problem, variant)
    rewards={} #(problem, variant, mechanism) -> reward
    verifier_matches_expected=0 #sanity check against the staff's expected_exact_reward
    start=time.perf_counter() #judge cost
    for problem_id in groups:
        variants=groups[problem_id] #the five responses for this problem
        question=variants["clean_correct"]["question"] #same problem text for all five
        responses=[] #in VARIANT_ORDER
        for variant in VARIANT_ORDER:
            responses.append(variants[variant]["response"]) #collect
        ai_rewards=judge.group_rewards(question,responses) #given: rlaif reward = (wins + 0.5 ties)/(K-1) over all 10 pairs
        for index in range(len(VARIANT_ORDER)):
            variant=VARIANT_ORDER[index] #which perturbation
            row=variants[variant] #the response
            verifier=exact_reward(row["response"],str(row["gold_final"])) #rlvr reward
            if int(verifier)==int(row["expected_exact_reward"]):
                verifier_matches_expected=verifier_matches_expected+1 #parser behaves as the staff expected
            rewards[(problem_id,variant,"rlvr")]=verifier #store
            rewards[(problem_id,variant,"rlaif")]=float(ai_rewards[index]) #store
            scored.append({"problem_id":problem_id,"variant_type":variant,"rlvr_reward":verifier,"rlaif_reward":float(ai_rewards[index]),"expected_exact_reward":int(row["expected_exact_reward"])}) #record
    judge_seconds=time.perf_counter()-start #inference cost of ai feedback on this set
    write_jsonl(outdir/"variant_rewards.jsonl",scored) #every reward

    #2) mean reward per variant (all five categories, both mechanisms)
    variant_rows=[] #table
    for variant in VARIANT_ORDER:
        rlvr_values=[] #verifier rewards for this variant
        rlaif_values=[] #ai rewards for this variant
        for problem_id in groups:
            rlvr_values.append(rewards[(problem_id,variant,"rlvr")]) #collect
            rlaif_values.append(rewards[(problem_id,variant,"rlaif")]) #collect
        variant_rows.append({"variant_type":variant,"n":len(rlvr_values),"rlvr_mean_reward":float(np.mean(rlvr_values)),"rlaif_mean_reward":float(np.mean(rlaif_values))}) #row

    #3) controlled pairs: better / tie / wrong preference for each mechanism (manual: S = Pr[R(better) > R(worse)])
    pair_rows=[] #table
    for better,worse,perturbed in PAIRS:
        for mechanism in ["rlvr","rlaif"]:
            better_count=0 #R(better) > R(worse)
            tie_count=0 #equal rewards
            wrong_count=0 #R(better) < R(worse)
            for problem_id in groups:
                difference=rewards[(problem_id,better,mechanism)]-rewards[(problem_id,worse,mechanism)] #reward gap
                if difference>1e-9:
                    better_count=better_count+1 #prefers the diagnostically better response
                elif difference<-1e-9:
                    wrong_count=wrong_count+1 #wrong preference
                else:
                    tie_count=tie_count+1 #cannot separate them
            n=len(groups) #20 problems
            pair_rows.append({"pair":better+" vs "+worse,"perturbation":perturbed,"mechanism":mechanism,"n":n,"better_rate":better_count/n,"tie_rate":tie_count/n,"wrong_preference_rate":wrong_count/n}) #row

    #4) the two sensitivities from the manual
    sensitivities={} #mechanism -> S_reason, S_outcome
    for mechanism in ["rlvr","rlaif"]:
        reason_better=0 #R(clean) > R(corrupt reasoning)
        outcome_better=0 #R(correct final) > R(wrong final), both outcome-changing pairs pooled
        outcome_pairs=0 #how many outcome pairs
        for problem_id in groups:
            if rewards[(problem_id,"clean_correct",mechanism)]>rewards[(problem_id,"corrupt_reasoning_correct_final",mechanism)]+1e-9:
                reason_better=reason_better+1 #reasoning noticed
            for wrong_variant in ["good_reasoning_wrong_final","gold_distractor_wrong_final"]:
                outcome_pairs=outcome_pairs+1 #count
                if rewards[(problem_id,"clean_correct",mechanism)]>rewards[(problem_id,wrong_variant,mechanism)]+1e-9:
                    outcome_better=outcome_better+1 #outcome noticed
        sensitivities[mechanism]={"S_reason":reason_better/len(groups),"S_outcome":outcome_better/outcome_pairs,"outcome_pairs":outcome_pairs} #manual definitions

    #5) direct head-to-head judge choices for the same pairs (already in the cache, no extra cost)
    head_to_head=[] #one row per pair type
    for better,worse,perturbed in PAIRS:
        counts={"better":0,"tie":0,"worse":0} #judge choice
        for problem_id in groups:
            variants=groups[problem_id] #responses
            choice=judge.compare(variants[better]["question"],variants[better]["response"],variants[worse]["response"]) #A = better response preferred
            if choice=="A":
                counts["better"]=counts["better"]+1 #prefers the better one
            elif choice=="B":
                counts["worse"]=counts["worse"]+1 #prefers the worse one
            else:
                counts["tie"]=counts["tie"]+1 #tie
        head_to_head.append({"pair":better+" vs "+worse,"perturbation":perturbed,"judge_prefers_better":counts["better"],"judge_ties":counts["tie"],"judge_prefers_worse":counts["worse"]}) #row

    summary={
        "problems":len(groups), #20
        "variant_mean_rewards":variant_rows, #all five categories
        "controlled_pairs":pair_rows, #better / tie / wrong rates
        "sensitivities":sensitivities, #S_reason and S_outcome per mechanism
        "judge_head_to_head":head_to_head, #direct a/b choices
        "verifier_matches_expected_exact_reward":verifier_matches_expected, #should be 100
        "judge_seconds":judge_seconds, #ai feedback cost on this set
        "judge_calls_per_problem":10, #5 choose 2 comparisons
    }
    save_json(outdir/"diagnostics_summary.json",summary) #machine-readable
    pd.DataFrame(pair_rows).to_csv(outdir/"controlled_pairs.csv",index=False) #for the report
    pd.DataFrame(variant_rows).to_csv(outdir/"variant_rewards_mean.csv",index=False) #for the report
    print(pd.DataFrame(pair_rows).to_string()) #quick look
    print("sensitivities:",sensitivities)
    print("verifier matches expected_exact_reward:",verifier_matches_expected,"/",len(scored))


if __name__ == "__main__":
    main()
