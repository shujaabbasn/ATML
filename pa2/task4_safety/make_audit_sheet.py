from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    src = outdir / "generated_sft.jsonl"
    if not src.exists():
        raise FileNotFoundError("Generate/save SFT responses first: " + str(src))
    ids = fixed_audit_ids(read_jsonl(src), int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    pd.DataFrame({"xstest_id": ids, "manual_label": [""] * len(ids)}).to_csv(outdir / "manual_audit_ids.csv", index=False)
    print("Wrote fixed audit IDs:", outdir / "manual_audit_ids.csv")
    print("Join these IDs to each policy's generated responses and label without viewing AI labels first.")
    #the join: one row per (audit id, policy) with the prompt and response but NO judge label, so labelling stays blind
    sheet_rows=[] #rows of the labelling sheet
    for policy_name in ["sft","dpo","ppo","grpo"]:
        by_id={} #xstest_id -> generated record
        for row in read_jsonl(outdir/("generated_"+policy_name+".jsonl")):
            by_id[int(row["xstest_id"])]=row #index by id
        for xstest_id in ids:
            row=by_id[int(xstest_id)] #this policy's response to this prompt
            sheet_rows.append({
                "xstest_id":int(xstest_id), #fixed audit id
                "policy":policy_name, #which model wrote the response
                "benchmark_class":row["benchmark_class"], #SAFE or UNSAFE prompt
                "type":row["type"], #xstest category
                "prompt":row["prompt"], #user prompt
                "response":row["response"], #model response
                "manual_label":"", #fill with SAFE_ANSWER / JUSTIFIED_REFUSAL / UNSAFE_COMPLIANCE / OVER_REFUSAL / AMBIGUOUS
            })
    sheet=pd.DataFrame(sheet_rows).sort_values(["xstest_id","policy"]) #the four responses to one prompt sit together
    sheet.to_csv(outdir/"manual_audit_sheet.csv",index=False) #label this file by hand
    print("Wrote blind labelling sheet:",outdir/"manual_audit_sheet.csv","rows:",len(sheet)) #60 ids x 4 policies


if __name__ == "__main__":
    main()
