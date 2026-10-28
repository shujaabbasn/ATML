from __future__ import annotations

import argparse
import csv

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.utils import load_yaml, read_jsonl, repo_path, save_json, set_seed, write_jsonl
from task1_finetune.evaluate import condition_path


#manual step 7: one greedy response per qualitative prompt, thinking disabled, 512-new-token cap
#  --name <condition>  generates the 24 responses of one condition
#  --collect           groups all 168 responses by question and format, and writes the annotation sheet

CONDITIONS=["baseline","full_ft_mid","full_ft","full_ft_l2sp","lora4","lora16","oplora16"]


def generate(cfg,name):
    set_seed(int(cfg["seed"]))
    device="cuda"
    tokenizer=AutoTokenizer.from_pretrained(cfg["base_model"])
    model=AutoModelForCausalLM.from_pretrained(condition_path(cfg,name),torch_dtype=torch.float16).to(device) #fp16, eval mode = dropout off
    model.eval()
    max_new_tokens=int(cfg["eval"]["qualitative_max_new_tokens"]) #512
    rows=[]
    for prompt in read_jsonl("data/task1/qualitative.jsonl"):
        text=tokenizer.apply_chat_template(prompt["messages"],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=tokenizer(text,return_tensors="pt",add_special_tokens=False).to(device) #one prompt at a time, no padding
        with torch.no_grad():
            output=model.generate(**ids,max_new_tokens=max_new_tokens,do_sample=False,pad_token_id=tokenizer.pad_token_id) #greedy
        new_tokens=output[0,ids["input_ids"].shape[1]:]
        ended=False #stopped on <|im_end|> or eos by itself
        for token in new_tokens.tolist():
            if token in (tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids("<|im_end|>")):
                ended=True
        rows.append({
            "condition":name,
            "id":prompt["id"],
            "question":prompt["question"],
            "format":prompt["format"],
            "response":tokenizer.decode(new_tokens,skip_special_tokens=True),
            "new_tokens":int(new_tokens.shape[0]),
            "truncated":bool(ended==False and new_tokens.shape[0]>=max_new_tokens), #hit the 512 cap
        })
        print(name,prompt["id"],rows[-1]["new_tokens"],rows[-1]["truncated"])
    write_jsonl(repo_path(cfg["results_dir"])/"qualitative"/(name+".jsonl"),rows)


def collect(cfg):
    folder=repo_path(cfg["results_dir"])/"qualitative"
    prompts=read_jsonl("data/task1/qualitative.jsonl")
    by_key={} #(id, condition) -> row
    for name in CONDITIONS:
        for row in read_jsonl(folder/(name+".jsonl")):
            by_key[(row["id"],name)]=row
    grouped=[] #question -> format -> condition -> response
    order=[] #underlying questions in the file's plain-question order
    for prompt in prompts:
        if prompt["question"] not in order:
            order.append(prompt["question"])
    for question in order:
        entry={"question":question,"formats":{}}
        for prompt in prompts:
            if prompt["question"]==question:
                entry["formats"][prompt["format"]]={"messages":prompt["messages"],"responses":{}}
                for name in CONDITIONS:
                    row=by_key[(prompt["id"],name)]
                    entry["formats"][prompt["format"]]["responses"][name]={"response":row["response"],"new_tokens":row["new_tokens"],"truncated":row["truncated"]}
        grouped.append(entry)
    save_json(folder/"all_responses_grouped.json",grouped) #all 168

    #annotation sheet: first and last underlying question, 3 formats x 7 conditions = 42 responses
    annotated=[order[0],order[-1]]
    with open(folder/"annotation_sheet.csv","w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["question","format","condition","truncated","new_tokens","format_compliance","coherence_relevance","repetition","unnecessary_math_framing","inappropriate_content","excerpt","notes","response"])
        for question in annotated:
            for prompt in prompts:
                if prompt["question"]==question:
                    for name in CONDITIONS:
                        row=by_key[(prompt["id"],name)]
                        writer.writerow([question,prompt["format"],name,row["truncated"],row["new_tokens"],"","","","","","","",row["response"]]) #blank columns are filled by hand
    print("annotation rows:",len(annotated)*3*len(CONDITIONS))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task1.yaml")
    ap.add_argument("--name",choices=CONDITIONS)
    ap.add_argument("--collect",action="store_true")
    args=ap.parse_args()
    cfg=load_yaml(args.config)
    if args.collect:
        collect(cfg)
    else:
        generate(cfg,args.name)


if __name__=="__main__":
    main()
