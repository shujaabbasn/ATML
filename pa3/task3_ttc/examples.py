from __future__ import annotations

import argparse

from common.utils import load_yaml, read_jsonl, repo_path, save_json
from task3_ttc.offline_eval import load_prepared
from task3_ttc.selection import group_candidates, rollout_scores, select_all


#step 6 representative cases, each picked by a rule stated here (not by looking at outcomes first):
#  selection disagreement  first 3 held-out questions (permutation order) where, at N = 32 in order 0, self-consistency and the
#                          quality-aware rule (self-certainty + RWC for math, VisConf + RWC for mathvista) pick different answers
#  revision                first 2 live questions with a wrong-to-correct and first 2 with a correct-to-wrong refinement
#  premature stopping      first 3 held-out stops per controller that ended early, were wrong, and fixed-32 got right

EXCERPT=300 #characters kept from the end of a response


def disagreements(cfg,dataset,count=3):
    prepared=load_prepared(cfg,dataset)
    by_id={}
    for q in prepared["questions"]:
        by_id[q["sample_id"]]=q
    if dataset=="math":
        quality_method="self_certainty_rwc"
    else:
        quality_method="visconf_rwc"
    picked=[]
    for sample_id in prepared["splits"]["eval"]:
        q=by_id[sample_id]
        candidates=prepared["orders"][sample_id][0] #order 0, all 32
        chosen=select_all(q,candidates,rollout_scores(q),dataset)
        a=chosen["self_consistency"]
        b=chosen[quality_method]
        if q["eq"][a,b]==False:
            support={}
            for group in group_candidates(candidates,q["eq"]):
                support[q["keys"][group[0]]]=len(group)
            picked.append({"dataset":dataset,"sample_id":sample_id,"question":q["question_text"][:EXCERPT],"gold":q["gold"],
                "self_consistency":{"answer":q["keys"][a],"correct":bool(q["correct"][a])},
                quality_method:{"answer":q["keys"][b],"correct":bool(q["correct"][b]),"rollout_tail":q["texts"][b][-EXCERPT:]},
                "group_support":support})
            if len(picked)==count:
                break
    return picked


def revisions(cfg,count=2):
    path=repo_path(cfg["results_dir"])/"live"/"generations.jsonl"
    if path.exists()==False:
        return []
    picked=[]
    for kind,before,after in [("wrong_to_correct",False,True),("correct_to_wrong",True,False)]:
        found=0
        for row in read_jsonl(path):
            if row["correct"]["initial"]==before and row["correct"]["refinement"]==after:
                picked.append({"kind":kind,"sample_id":row["sample_id"],"gold":row["gold"],"initial_answer":row["answers"]["initial"],
                    "revised_answer":row["answers"]["refinement"],"initial_tail":row["parallel"][0]["response"][-EXCERPT:],"revision_tail":row["revision"]["response"][-EXCERPT:]})
                found=found+1
                if found==count:
                    break
    return picked


def premature_stops(cfg,dataset,count=3):
    path=repo_path(cfg["results_dir"])/"adaptive"/dataset/"eval_stops.jsonl"
    picked=[]
    for family in ["agreement","quality"]:
        found=0
        for row in read_jsonl(path):
            if row["family"]==family and row["stop_n"]<32 and row["correct"]==0.0 and row["fixed32_correct"]==1.0:
                picked.append(row)
                found=found+1
                if found==count:
                    break
    return picked


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task3.yaml")
    args=ap.parse_args()
    cfg=load_yaml(args.config)
    examples={"selection_disagreement":disagreements(cfg,"math")+disagreements(cfg,"mathvista"),"revision":revisions(cfg),
        "premature_stopping":premature_stops(cfg,"math")+premature_stops(cfg,"mathvista")}
    save_json(repo_path(cfg["results_dir"])/"examples.json",examples)
    for kind in examples:
        print(kind,len(examples[kind]))


if __name__=="__main__":
    main()
