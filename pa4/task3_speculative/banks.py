"""Prefix banks we have to make ourselves (the staff script only makes teacher and original_draft):
  exact_spec_original_gamma4: our exact speculative sampler, target 1.5B + original 0.5B draft, gamma 4
  <draft name>: each trained student's own policy prefixes (student sampled alone)
Positions are drawn with the staff rule (8 from 1-16, 8 from 17-32, 16 from 33-64, 16 from 65-192).

python -m task3_speculative.banks --bank exact_spec
python -m task3_speculative.banks --bank student --name off_policy_s6304
python -m task3_speculative.banks --bank consistency      exact-spec bank vs staff teacher bank
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from scipy import stats as scipy_stats

from common.utils import load_yaml, save_json, read_jsonl, write_jsonl, repo_path, data_path, get_device
from task3_speculative.common import load_causal, load_tokenizer, eos_ids, chat_ids, test_prompts
from task3_speculative.sampler import speculative_generate, plain_generate, make_generator


def bank_positions(prompt_id,which,length):
    """Copy of the staff position rule so every bank samples positions the same way."""
    rng=random.Random(f"PA4-banks-v1:6304:{prompt_id}:{which}")
    positions=[]
    for low,high,count in [(1,16,8),(17,32,8),(33,64,16),(65,192,16)]:
        candidates=list(range(low,min(high,length)+1))
        positions.extend(sorted(rng.sample(candidates,min(count,len(candidates)))))
    return positions


def bank_records(prompt,which,sequence,seed):
    out=[]
    for position in bank_positions(prompt["id"],which,len(sequence)):
        out.append({"prompt_id":prompt["id"],"domain":prompt["domain"],"bank":which,"position":position,"prompt":prompt["prompt"],
            "response_prefix_token_ids":sequence[:position-1],"source_length":len(sequence),"source_generation_seed":seed})
    return out


def draft_dir(config,name):
    return repo_path(config["results_dir"])/"drafts"/name/"merged"


@torch.no_grad()
def make_bank(config,kind,name):
    device=get_device()
    tokenizer=load_tokenizer(config)
    target=None
    if kind=="exact_spec":
        target=load_causal(config["target"])
        draft=load_causal(config["draft"])
        which="exact_spec_original_gamma4"
    else:
        draft=load_causal(str(draft_dir(config,name)))
        which=name
    eos_list=eos_ids(tokenizer)
    prompts=test_prompts(config)
    records=[]
    for j in range(len(prompts)):
        prompt_ids=chat_ids(tokenizer,prompts[j]["prompt"])
        seed=6304+10007*j+777 #ours: a different seed from the staff teacher bank, same law for exact_spec
        generator=make_generator(device,seed)
        if kind=="exact_spec":
            sequence,stats=speculative_generate(target,draft,prompt_ids,config["evaluation"]["bank_gamma"],config["evaluation"]["bank_cap"],eos_list,generator)
        else:
            sequence,stats=plain_generate(draft,prompt_ids,config["evaluation"]["bank_cap"],eos_list,generator)
        records.extend(bank_records(prompts[j],which,sequence,seed))
        if (j+1)%20==0:
            print(which,j+1,"/",len(prompts),flush=True)
    write_jsonl(repo_path(config["results_dir"])/"banks"/f"{which}.jsonl.gz",records)


def longest_prefixes(records,bank):
    """prompt_id -> (longest saved prefix, source_length) for one bank."""
    out={}
    for record in records:
        if record["bank"]!=bank:
            continue
        key=record["prompt_id"]
        if key not in out or len(record["response_prefix_token_ids"])>len(out[key][0]):
            out[key]=(record["response_prefix_token_ids"],record["source_length"])
    return out


@torch.no_grad()
def consistency(config):
    """Teacher bank and exact-spec bank should be two samples of the same law. Compare per prompt:
    response length (two-sample KS) and the target's mean per-token NLL of the saved prefix (paired Wilcoxon)."""
    device=get_device()
    tokenizer=load_tokenizer(config)
    target=load_causal(config["target"])
    staff=read_jsonl(data_path(config["data"]["prefix_banks"]))
    ours=read_jsonl(repo_path(config["results_dir"])/"banks"/"exact_spec_original_gamma4.jsonl.gz")
    teacher=longest_prefixes(staff,"teacher")
    original=longest_prefixes(staff,"original_draft")
    spec=longest_prefixes(ours,"exact_spec_original_gamma4")
    prompts={}
    for item in test_prompts(config):
        prompts[item["id"]]=item

    def mean_nll(prompt_id,prefix):
        prompt_ids=chat_ids(tokenizer,prompts[prompt_id]["prompt"])
        ids=torch.tensor([prompt_ids+prefix],device=device)
        logits=target(input_ids=ids).logits[0].float()
        log_probs=torch.log_softmax(logits[len(prompt_ids)-1:-1],dim=-1)
        picked=log_probs.gather(1,torch.tensor(prefix,device=device).unsqueeze(1))
        return float(-picked.mean())

    rows=[]
    for prompt_id in sorted(spec.keys()):
        if prompt_id not in teacher:
            continue
        rows.append({"prompt_id":prompt_id,"domain":prompts[prompt_id]["domain"],"teacher_length":teacher[prompt_id][1],"spec_length":spec[prompt_id][1],"original_length":original[prompt_id][1],
            "teacher_nll":mean_nll(prompt_id,teacher[prompt_id][0]),"spec_nll":mean_nll(prompt_id,spec[prompt_id][0]),"original_nll":mean_nll(prompt_id,original[prompt_id][0])})
    out={"prompts":len(rows),"rows":rows}
    for other in ["spec","original"]:
        a_length=[]
        b_length=[]
        a_nll=[]
        b_nll=[]
        for row in rows:
            a_length.append(row["teacher_length"])
            b_length.append(row[other+"_length"])
            a_nll.append(row["teacher_nll"])
            b_nll.append(row[other+"_nll"])
        ks=scipy_stats.ks_2samp(a_length,b_length)
        wilcoxon=scipy_stats.wilcoxon(a_nll,b_nll)
        out[f"teacher_vs_{other}"]={"length_ks_statistic":float(ks.statistic),"length_ks_p":float(ks.pvalue),"nll_wilcoxon_p":float(wilcoxon.pvalue),
            "mean_teacher_nll":float(np.mean(a_nll)),"mean_other_nll":float(np.mean(b_nll)),"mean_teacher_length":float(np.mean(a_length)),"mean_other_length":float(np.mean(b_length))}
    out["reading"]="exact_spec should NOT differ from teacher (large p-values); original_draft is a different law and is the positive control"
    save_json(repo_path(config["results_dir"])/"banks"/"consistency.json",out)
    print(out["teacher_vs_spec"],out["teacher_vs_original"])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/task3.yaml")
    parser.add_argument("--bank",required=True,choices=["exact_spec","student","consistency"])
    parser.add_argument("--name",default=None)
    args=parser.parse_args()
    config=load_yaml(args.config)
    if args.bank=="consistency":
        consistency(config)
    else:
        make_bank(config,args.bank,args.name)


if __name__=="__main__":
    main()
