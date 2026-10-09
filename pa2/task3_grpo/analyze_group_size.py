from __future__ import annotations

import argparse
from collections import defaultdict
import numpy as np

from common.data import load_yaml, read_jsonl

import pandas as pd
import torch

from common.data import repo_path
from common.logging_utils import save_json
from task3_grpo.grpo import group_relative_advantages


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int):
    """Return K-sized groups while keeping total cached completions fixed.

    Students should decide and document exactly how prompts/completions are partitioned for the
    requested equal-generation comparison.
    """
    #partition rule (stated once): every prompt keeps its first 8 cached completions in generation_index order,
    #split into 8/K consecutive groups. K=8 -> 1 group per prompt, K=4 -> 2, K=2 -> 4. every K uses the same
    #24x8 completions, so the total generation budget is identical and only the group size changes
    groups=[] #list of groups, each a dict with its prompt and rows
    for prompt_key in by_prompt:
        rows=by_prompt[prompt_key][:8] #first 8 completions (already sorted by generation_index)
        for start in range(0,8,k):
            groups.append({"prompt":prompt_key,"rows":rows[start:start+k]}) #one group of k completions
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    by_prompt = load_k8_cache(cfg["group_cache"])
    print("Cached prompts:", len(by_prompt))
    print("Group sizes to analyze:", cfg["group_sizes"])
    first = next(iter(by_prompt.values()))
    print("Cache row keys:", sorted(first[0].keys()))
    results_dir=repo_path(cfg["results_dir"]) #results/task3_grpo/
    results_dir.mkdir(parents=True,exist_ok=True) #make it
    tolerance=1e-6 #eps of the released group_relative_advantages: a group is informative if its std is above this

    #difficulty bins (stated once): rank prompts by the mean reward of all 8 cached completions and cut into
    #three equal terciles. lowest mean reward = "hard", middle = "medium", highest = "easy"
    prompt_means=[] #(mean reward, prompt key)
    for prompt_key in by_prompt:
        total=0.0 #sum of the 8 rewards
        for row in by_prompt[prompt_key][:8]:
            total=total+float(row["reward"]) #add
        prompt_means.append((total/8,prompt_key)) #mean reward of this prompt
    prompt_means.sort() #lowest mean first
    difficulty={} #prompt key -> bin name
    third=len(prompt_means)/3 #24 prompts -> 8 per bin
    for position in range(len(prompt_means)):
        prompt_key=prompt_means[position][1] #this prompt
        if position<third:
            difficulty[prompt_key]="hard" #lowest third of mean reward
        elif position<2*third:
            difficulty[prompt_key]="medium" #middle third
        else:
            difficulty[prompt_key]="easy" #highest third

    table_rows=[] #one row per (K, bin)
    for k in cfg["group_sizes"]:
        k=int(k) #2, 4, 8
        groups=regroup_equal_generation_budget(by_prompt,k) #equal total generations
        rewards=[] #every reward, in group order
        group_ids=[] #group index for every reward
        group_bins=[] #difficulty bin for every group
        for index in range(len(groups)):
            group_bins.append(difficulty[groups[index]["prompt"]]) #bin of the group's prompt
            for row in groups[index]["rows"]:
                rewards.append(float(row["reward"])) #reward
                group_ids.append(index) #which group
        reward_tensor=torch.tensor(rewards) #[192]
        id_tensor=torch.tensor(group_ids) #[192]
        advantages=group_relative_advantages(reward_tensor,id_tensor) #fixed helper, normalized relative signal
        for bin_name in ["all","hard","medium","easy"]:
            stds=[] #within-group reward std of each group in this bin
            informative=0 #groups with std above tolerance
            centered=[] #r - group mean (relative signal before dividing by std)
            normalized=[] #advantage after dividing by std
            for index in range(len(groups)):
                if bin_name!="all":
                    if group_bins[index]!=bin_name:
                        continue #group not in this bin
                members=id_tensor==index #this group's completions
                group_rewards=reward_tensor[members] #their rewards
                std=float(group_rewards.std(unbiased=False).item()) #same std as the helper
                stds.append(std) #record
                if std>tolerance:
                    informative=informative+1 #this group gives a nonzero learning signal
                for value in (group_rewards-group_rewards.mean()).tolist():
                    centered.append(value) #centered reward
                for value in advantages[members].tolist():
                    normalized.append(value) #normalized advantage
            table_rows.append({
                "K":k, #group size
                "bin":bin_name, #difficulty bin or all
                "groups":len(stds), #how many groups
                "completions":len(centered), #how many completions (equal budget across K)
                "informative_rate":informative/len(stds), #manual: fraction of groups with nonzero std
                "mean_within_group_std":float(torch.tensor(stds).mean().item()), #manual: mean within-group reward std
                "var_centered_signal":float(torch.tensor(centered).var(unbiased=False).item()), #variance of r - mu_r
                "var_normalized_advantage":float(torch.tensor(normalized).var(unbiased=False).item()), #manual: variance of the group-relative signal
            })

    clipped=0 #cached completions that hit the cache cap
    for prompt_key in by_prompt:
        for row in by_prompt[prompt_key][:8]:
            if bool(row.get("clipped_at_max",False))==True:
                clipped=clipped+1 #count
    table=pd.DataFrame(table_rows) #k x bin table
    table.to_csv(results_dir/"group_size.csv",index=False) #for the report
    bins_out=[] #prompt -> mean reward -> bin, so the binning is auditable
    for mean_reward,prompt_key in prompt_means:
        bins_out.append({"source_index":prompt_key,"mean_reward":mean_reward,"bin":difficulty[prompt_key]}) #record
    save_json(results_dir/"group_size.json",{"table":table_rows,"bins":bins_out,"tolerance":tolerance,"cached_completions_clipped_at_cap":clipped,"partition_rule":"first 8 completions per prompt in generation_index order, split into 8/K consecutive groups"}) #json copy
    print(table.to_string()) #quick look


if __name__ == "__main__":
    main()
