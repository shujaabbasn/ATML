from __future__ import annotations

import argparse
import re
import urllib.request

import numpy as np
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

from common.utils import load_yaml, repo_path, save_json, write_jsonl


#builds everything task 1 reads before training:
#  data/task1/train.jsonl          first 40,000 eligible metamathqa examples, source order
#  data/task1/probe_ids.json       32 gsm8k test + 32 hellaswag validation ids (seed 6304)
#  data/task1/qualitative.jsonl    the 24 prompts of first_plot_questions.yaml


def normalize_text(text):
    """Text normalization for the gsm8k overlap check: lowercase, keep letters/digits only, single spaces."""
    text=text.lower()
    text=re.sub(r"[^a-z0-9]+"," ",text) #punctuation, latex symbols and newlines become spaces
    return text.strip()


def prompt_ids(tokenizer,question):
    """Chat prompt with thinking disabled, ends with the assistant header and the empty think block."""
    messages=[{"role":"user","content":question}]
    return tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)


def response_ids(tokenizer,response):
    """Supervised part: the supplied response plus the end-of-turn token <|im_end|>."""
    return tokenizer(response+"<|im_end|>",add_special_tokens=False)["input_ids"]


def build_train_set(cfg,tokenizer):
    gsm_test=load_dataset(cfg["data"]["gsm8k"],"main",split="test")
    gsm_questions=set() #normalized gsm8k test questions
    for row in gsm_test:
        gsm_questions.add(normalize_text(row["question"]))

    metamath=load_dataset(cfg["data"]["metamath"],split="train") #source order, never shuffled
    target=int(cfg["data"]["train_examples"]) #40,000
    max_length=int(cfg["data"]["max_length"]) #2,048
    kept=[]
    excluded={"empty":0,"too_long":0,"gsm8k_test_overlap":0} #why each skipped example was skipped
    for index in range(len(metamath)):
        row=metamath[index]
        query=row["query"]
        response=row["response"]
        original=row["original_question"]
        if query is None or response is None or query.strip()=="" or response.strip()=="":
            excluded["empty"]=excluded["empty"]+1
            continue
        if normalize_text(query) in gsm_questions or (original is not None and normalize_text(original) in gsm_questions):
            excluded["gsm8k_test_overlap"]=excluded["gsm8k_test_overlap"]+1 #manual: query/original_question matching a gsm8k test question
            continue
        n_prompt=len(prompt_ids(tokenizer,query))
        n_response=len(response_ids(tokenizer,response))
        if n_prompt+n_response>max_length:
            excluded["too_long"]=excluded["too_long"]+1 #formatted length above 2,048, we drop it instead of truncating
            continue
        kept.append({"source_index":index,"type":row["type"],"query":query,"response":response,"prompt_tokens":n_prompt,"response_tokens":n_response})
        if len(kept)==target:
            break
    summary={"kept":len(kept),"last_source_index":kept[-1]["source_index"],"scanned":kept[-1]["source_index"]+1,"excluded":excluded}
    return kept,summary


def build_probe_ids(cfg):
    """32 gsm8k test + 32 hellaswag validation indices, drawn with seed 6304 before any model is evaluated."""
    rng=np.random.default_rng(int(cfg["seed"]))
    gsm_size=len(load_dataset(cfg["data"]["gsm8k"],"main",split="test")) #1,319
    hellaswag_size=len(load_dataset(cfg["data"]["hellaswag"],split="validation")) #10,042
    gsm_ids=rng.choice(gsm_size,size=int(cfg["data"]["probe_gsm8k"]),replace=False)
    hellaswag_ids=rng.choice(hellaswag_size,size=int(cfg["data"]["probe_hellaswag"]),replace=False)
    gsm_list=[]
    for i in gsm_ids:
        gsm_list.append(int(i))
    hellaswag_list=[]
    for i in hellaswag_ids:
        hellaswag_list.append(int(i))
    return {"seed":int(cfg["seed"]),"gsm8k_test":sorted(gsm_list),"hellaswag_validation":sorted(hellaswag_list)}


def build_qualitative(cfg):
    """The 24 prompts: 8 questions x (plain, json, template), in the file's order."""
    path=repo_path(cfg["data"]["qualitative_yaml"])
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        urllib.request.urlretrieve(cfg["data"]["qualitative_yaml_url"],path) #link from the manual
    entries=yaml.safe_load(path.read_text(encoding="utf-8"))
    rows=[]
    for entry in entries:
        question_id=entry["id"]
        if question_id.endswith("_json"):
            prompt_format="json"
            base_id=question_id[:-len("_json")]
        elif question_id.endswith("_template"):
            prompt_format="template"
            base_id=question_id[:-len("_template")]
        else:
            prompt_format="plain"
            base_id=question_id
        messages=[]
        if entry.get("system") is not None:
            messages.append({"role":"system","content":entry["system"]}) #json format asks for json in the system prompt
        messages.append({"role":"user","content":entry["paraphrases"][0]}) #every entry has one paraphrase
        rows.append({"id":question_id,"question":base_id,"format":prompt_format,"messages":messages})
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task1.yaml")
    args=ap.parse_args()
    cfg=load_yaml(args.config)
    tokenizer=AutoTokenizer.from_pretrained(cfg["base_model"])

    probe=build_probe_ids(cfg) #drawn first, before anything is trained or evaluated
    save_json("data/task1/probe_ids.json",probe)

    qualitative=build_qualitative(cfg)
    write_jsonl("data/task1/qualitative.jsonl",qualitative)

    kept,summary=build_train_set(cfg,tokenizer)
    write_jsonl("data/task1/train.jsonl",kept)
    summary["probe_counts"]=[len(probe["gsm8k_test"]),len(probe["hellaswag_validation"])]
    summary["qualitative_prompts"]=len(qualitative)
    save_json("results/task1/data_summary.json",summary)
    print(summary)


if __name__=="__main__":
    main()
