from __future__ import annotations

import argparse

from common.utils import load_yaml, read_jsonl, repo_path, save_json


#step 6 aligned complementary-image examples, each picked by a rule stated here (not by looking at outcomes first):
#  follows_image   first test pair per question type where both correct-image answers are right and both swapped answers follow the partner
#  same_answer     first test pair per question type where the model gives one identical answer to both images
#order is the saved test order, same question, references and predictions shown for both sides

CONDITIONS=["correct","complementary_swap"]


def load_pairs(folder,suite):
    """pair id -> side -> condition -> saved row."""
    pairs={}
    for condition in CONDITIONS:
        for row in read_jsonl(folder/(suite+"_"+condition+".jsonl")):
            if row["pair_id"] not in pairs:
                pairs[row["pair_id"]]={}
            if row["side"] not in pairs[row["pair_id"]]:
                pairs[row["pair_id"]][row["side"]]={}
            pairs[row["pair_id"]][row["side"]][condition]=row
    return pairs


def follows_image(pair):
    for side in pair:
        correct=pair[side]["correct"]
        swap=pair[side]["complementary_swap"]
        if correct["normalized"]!=correct["canonical"]:
            return False
        if swap["normalized"]!=swap["partner_canonical"]:
            return False
    return True


def same_answer(pair):
    answers=[]
    for side in pair:
        answers.append(pair[side]["correct"]["normalized"])
    return answers[0]==answers[1]


def describe(pair_id,pair,rule):
    sides=[]
    for side in sorted(pair):
        correct=pair[side]["correct"]
        swap=pair[side]["complementary_swap"]
        sides.append({"image":correct["image"],"reference":correct["reference"],"prediction_correct_image":correct["prediction"],
            "prediction_swapped_image":swap["prediction"],"score_correct_image":correct["score"],"nll_correct_image":correct["nll"],"nll_swapped_image":swap["nll"]})
    first=pair[sorted(pair)[0]]["correct"]
    return {"rule":rule,"pair_id":pair_id,"qtype":first["qtype"],"question":first["question"],"sides":sides}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task2.yaml")
    args=ap.parse_args()
    cfg=load_yaml(args.config)
    results_dir=repo_path(cfg["results_dir"])
    folder=results_dir/"tokens196"/"final_eval" #the main run, the 49 token run is optional
    picked={}
    for suite in ["grounding_test","vqa_test"]:
        pairs=load_pairs(folder,suite)
        picked[suite]=[]
        qtypes=[]
        for pair_id in pairs:
            qtype=pairs[pair_id][sorted(pairs[pair_id])[0]]["correct"]["qtype"]
            if qtype not in qtypes:
                qtypes.append(qtype)
        for qtype in qtypes:
            for rule,check in [("follows_image",follows_image),("same_answer",same_answer)]:
                for pair_id in pairs:
                    pair=pairs[pair_id]
                    if pair[sorted(pair)[0]]["correct"]["qtype"]==qtype and check(pair)==True:
                        picked[suite].append(describe(pair_id,pair,rule))
                        break
    save_json(results_dir/"examples.json",picked)
    for suite in picked:
        print(suite,len(picked[suite]))


if __name__=="__main__":
    main()
