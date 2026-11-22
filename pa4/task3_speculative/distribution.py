"""Step 5: compare every draft with the teacher on the SAME frozen prefixes.
For each prefix h: forward KL(p||q), reverse KL(q||p), TV = 1/2 sum|p-q|, expected one-token acceptance 1-TV,
and greedy top-1 agreement, all at temperature 1 over the full vocabulary.

python -m task3_speculative.distribution
Banks: staff teacher + original_draft, our exact_spec_original_gamma4, and each trained draft's own policy bank.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from common.utils import load_yaml, save_json, read_jsonl, write_jsonl, repo_path, data_path, get_device
from task3_speculative.common import load_causal, load_tokenizer, chat_ids, test_prompts


def trained_drafts(config):
    names=[]
    for method in config["distill"]["methods"]:
        for seed in config["distill"]["seeds"]:
            name=f"{method}_s{seed}"
            if (repo_path(config["results_dir"])/"drafts"/name/"merged").exists():
                names.append(name)
    return names


def load_banks(config,drafts):
    records=read_jsonl(data_path(config["data"]["prefix_banks"])) #teacher and original_draft
    ours=repo_path(config["results_dir"])/"banks"/"exact_spec_original_gamma4.jsonl.gz"
    if ours.exists():
        records.extend(read_jsonl(ours))
    for name in drafts:
        path=repo_path(config["results_dir"])/"banks"/f"{name}.jsonl.gz"
        if path.exists():
            records.extend(read_jsonl(path))
    grouped={} #(bank, prompt_id) -> records
    for record in records:
        key=(record["bank"],record["prompt_id"])
        if key not in grouped:
            grouped[key]=[]
        grouped[key].append(record)
    return grouped


def metrics(teacher_logits,draft_logits):
    p_log=F.log_softmax(teacher_logits.float(),dim=-1)
    q_log=F.log_softmax(draft_logits.float(),dim=-1)
    p=p_log.exp()
    q=q_log.exp()
    forward=(p*(p_log-q_log)).sum(dim=-1)
    reverse=(q*(q_log-p_log)).sum(dim=-1)
    tv=0.5*(p-q).abs().sum(dim=-1)
    top1=(p.argmax(dim=-1)==q.argmax(dim=-1)).float()
    return forward,reverse,tv,top1,p.argmax(dim=-1),q.argmax(dim=-1)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/task3.yaml")
    args=parser.parse_args()
    config=load_yaml(args.config)
    device=get_device()
    tokenizer=load_tokenizer(config)
    teacher=load_causal(config["target"])
    names=["original"]+trained_drafts(config)
    drafts={"original":load_causal(config["draft"])}
    for name in names[1:]:
        drafts[name]=load_causal(str(repo_path(config["results_dir"])/"drafts"/name/"merged"))
    grouped=load_banks(config,names)
    prompts={}
    for item in test_prompts(config):
        prompts[item["id"]]=item
    rows=[]
    keys=sorted(grouped.keys())
    for index in range(len(keys)):
        bank,prompt_id=keys[index]
        records=grouped[(bank,prompt_id)]
        longest=records[0]
        for record in records:
            if len(record["response_prefix_token_ids"])>len(longest["response_prefix_token_ids"]):
                longest=record
        prompt_ids=chat_ids(tokenizer,prompts[prompt_id]["prompt"])
        ids=torch.tensor([prompt_ids+longest["response_prefix_token_ids"]],device=device)
        #one forward per (bank, prompt): every saved prefix is a prefix of the longest one
        index_of=[]
        for record in records:
            if record["response_prefix_token_ids"]!=longest["response_prefix_token_ids"][:len(record["response_prefix_token_ids"])]:
                raise RuntimeError("bank prefixes of one prompt are not nested")
            index_of.append(len(prompt_ids)+record["position"]-2) #logit that predicts response token `position`
        index_tensor=torch.tensor(index_of,device=device)
        teacher_logits=teacher(input_ids=ids).logits[0].index_select(0,index_tensor)
        for name in names:
            draft_logits=drafts[name](input_ids=ids).logits[0].index_select(0,index_tensor)
            forward,reverse,tv,top1,teacher_top,draft_top=metrics(teacher_logits,draft_logits)
            for j in range(len(records)):
                rows.append({"bank":bank,"prompt_id":prompt_id,"domain":prompts[prompt_id]["domain"],"subset":prompts[prompt_id]["subset"],"position":records[j]["position"],"draft":name,
                    "forward_kl":float(forward[j]),"reverse_kl":float(reverse[j]),"tv":float(tv[j]),"accept":float(1-tv[j]),"top1":float(top1[j]),
                    "teacher_top_token":int(teacher_top[j]),"draft_top_token":int(draft_top[j]),"prefix_tail":records[j]["response_prefix_token_ids"][-8:]})
        if index%50==0:
            print("bank prompt",index,"/",len(keys),flush=True)
    out_dir=repo_path(config["results_dir"])/"distribution"
    write_jsonl(out_dir/"per_prefix.jsonl.gz",rows)
    table={}
    for row in rows:
        for scope in ["all",row["subset"],position_bin(row["position"])]: #math / familiar / shifted, and response-position bins
            key=f"{row['draft']}|{row['bank']}|{scope}"
            if key not in table:
                table[key]={"forward_kl":[],"reverse_kl":[],"tv":[],"accept":[],"top1":[]}
            for metric in table[key]:
                table[key][metric].append(row[metric])
    summary={}
    for key in table:
        summary[key]={"prefixes":len(table[key]["tv"])}
        for metric in table[key]:
            summary[key][metric]=float(np.mean(table[key][metric])) #token-weighted: every prefix counts once
    save_json(out_dir/"summary.json",summary)
    #training-seed dispersion: mean and std over the seeds of each method, per bank and scope
    dispersion={}
    for key in summary:
        draft,bank,scope=key.split("|")
        method=draft.rsplit("_s",1)[0]
        if draft=="original":
            continue
        group=f"{method}|{bank}|{scope}"
        if group not in dispersion:
            dispersion[group]={"forward_kl":[],"tv":[],"accept":[],"seeds":[]}
        dispersion[group]["forward_kl"].append(summary[key]["forward_kl"])
        dispersion[group]["tv"].append(summary[key]["tv"])
        dispersion[group]["accept"].append(summary[key]["accept"])
        dispersion[group]["seeds"].append(draft)
    for group in dispersion:
        for metric in ["forward_kl","tv","accept"]:
            values=np.array(dispersion[group][metric])
            dispersion[group][metric]={"mean":float(values.mean()),"std":float(values.std()),"per_seed":values.tolist()}
    save_json(out_dir/"seed_dispersion.json",dispersion)
    #aligned disagreements: for each draft, the 10 common-bank prefixes with the largest TV, same prefix for every draft
    disagreements=[]
    for name in names:
        mine=[]
        for row in rows:
            if row["draft"]==name and row["bank"]=="teacher":
                mine.append(row)
        mine.sort(key=tv_of,reverse=True) #largest disagreement first
        for row in mine[:10]:
            aligned={"draft":name,"prompt_id":row["prompt_id"],"position":row["position"],"prefix_tail_text":tokenizer.decode(row["prefix_tail"]),
                "teacher_top":tokenizer.decode([row["teacher_top_token"]]),"others":{}}
            for other in rows:
                if other["bank"]=="teacher" and other["prompt_id"]==row["prompt_id"] and other["position"]==row["position"]:
                    aligned["others"][other["draft"]]={"tv":other["tv"],"top":tokenizer.decode([other["draft_top_token"]])}
            disagreements.append(aligned)
    write_jsonl(out_dir/"aligned_disagreements.jsonl",disagreements)
    print("wrote",out_dir/"summary.json")


def tv_of(row):
    return row["tv"]


def position_bin(position):
    """Manual diagnostic bins: short positions [1,16], [17,32], [33,64]; long-response bank 65-192."""
    if position<=16:
        return "pos1-16"
    elif position<=32:
        return "pos17-32"
    elif position<=64:
        return "pos33-64"
    else:
        return "pos65-192"


if __name__=="__main__":
    main()
