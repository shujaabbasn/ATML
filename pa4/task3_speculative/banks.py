"""Prefix banks we have to make ourselves (the staff script only makes teacher and original_draft):
  exact_spec_original_gamma4: our exact speculative sampler, target 1.5B + original 0.5B draft, gamma 4
  target_plain: the target sampled alone by our plain sampler (temperature 1, top-p 1, nothing else)
  <draft name>: each trained student's own policy prefixes (student sampled alone)
Positions are drawn with the staff rule (8 from 1-16, 8 from 17-32, 16 from 33-64, 16 from 65-192).

Why target_plain: the staff script calls model.generate(temperature=1, top_p=1, top_k=0) but never sets
repetition_penalty, so generate takes 1.1 from Qwen's generation_config.json. The staff teacher bank is therefore
not the plain temperature-1 target law that exact speculative sampling reproduces, and the consistency control
(manual: "at matched prompts and sampling settings") needs a target bank sampled with the same settings as ours.

python -m task3_speculative.banks --bank exact_spec
python -m task3_speculative.banks --bank target_plain
python -m task3_speculative.banks --bank student --name off_policy_s6304
python -m task3_speculative.banks --bank consistency      exact-spec bank vs target_plain (and the staff banks)
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
        offset=777 #ours: a different seed from every other bank, same law as target_plain
    elif kind=="target_plain":
        draft=load_causal(config["target"]) #the target sampled alone, plain_generate below
        which="target_plain"
        offset=555
    else:
        draft=load_causal(str(draft_dir(config,name)))
        which=name
        offset=777
    eos_list=eos_ids(tokenizer)
    prompts=test_prompts(config)
    records=[]
    for j in range(len(prompts)):
        prompt_ids=chat_ids(tokenizer,prompts[j]["prompt"])
        seed=6304+10007*j+offset
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


def repeat_fraction(prompt_ids,prefix):
    """Share of response tokens that already appeared earlier in the prompt or response.
    repetition_penalty divides the logits of exactly these tokens, so a penalized bank shows a lower value."""
    seen=set(prompt_ids)
    repeats=0
    for token in prefix:
        if token in seen:
            repeats=repeats+1
        seen.add(token)
    return repeats/max(len(prefix),1)


@torch.no_grad()
def consistency(config):
    """Manual: banks of the target and of correct exact speculative sampling sample the same target-prefix law
    at matched prompts and sampling settings. Reference = target_plain (our plain sampler, temperature 1).
    Per prompt, on the longest saved prefix: response length (two-sample KS), the target's mean per-token NLL and the
    repeated-token fraction (paired Wilcoxon over prompts).
      exact_spec      should NOT differ (the consistency check)
      staff teacher   sampled with Qwen's default repetition_penalty 1.1 (see the module docstring)
      original_draft  a different model, the positive control that must differ"""
    device=get_device()
    tokenizer=load_tokenizer(config)
    target=load_causal(config["target"])
    staff=read_jsonl(data_path(config["data"]["prefix_banks"]))
    ours=read_jsonl(repo_path(config["results_dir"])/"banks"/"exact_spec_original_gamma4.jsonl.gz")
    ours.extend(read_jsonl(repo_path(config["results_dir"])/"banks"/"target_plain.jsonl.gz"))
    banks={"target_plain":longest_prefixes(ours,"target_plain"),"spec":longest_prefixes(ours,"exact_spec_original_gamma4"),
        "teacher":longest_prefixes(staff,"teacher"),"original":longest_prefixes(staff,"original_draft")}
    prompts={}
    for item in test_prompts(config):
        prompts[item["id"]]=item

    def mean_nll(prompt_ids,prefix):
        if len(prefix)==0:
            return None #response was one token long, nothing saved before it
        ids=torch.tensor([prompt_ids+prefix],device=device)
        logits=target(input_ids=ids).logits[0].float()
        log_probs=torch.log_softmax(logits[len(prompt_ids)-1:-1],dim=-1)
        picked=log_probs.gather(1,torch.tensor(prefix,device=device).unsqueeze(1))
        return float(-picked.mean())

    rows=[]
    for prompt_id in sorted(banks["target_plain"].keys()):
        complete=True
        for bank in banks:
            if prompt_id not in banks[bank]:
                complete=False
        if complete==False:
            continue
        prompt_ids=chat_ids(tokenizer,prompts[prompt_id]["prompt"])
        row={"prompt_id":prompt_id,"domain":prompts[prompt_id]["domain"]}
        for bank in banks:
            prefix,length=banks[bank][prompt_id]
            row[bank+"_length"]=length
            row[bank+"_nll"]=mean_nll(prompt_ids,prefix)
            row[bank+"_repeat"]=repeat_fraction(prompt_ids,prefix)
        rows.append(row)
    out={"prompts":len(rows),"reference":"target_plain","rows":rows}
    for other in ["spec","teacher","original"]:
        a_length=[]
        b_length=[]
        a_nll=[]
        b_nll=[]
        a_repeat=[]
        b_repeat=[]
        for row in rows:
            a_length.append(row["target_plain_length"])
            b_length.append(row[other+"_length"])
            a_repeat.append(row["target_plain_repeat"])
            b_repeat.append(row[other+"_repeat"])
            if row["target_plain_nll"] is not None and row[other+"_nll"] is not None:
                a_nll.append(row["target_plain_nll"])
                b_nll.append(row[other+"_nll"])
        ks=scipy_stats.ks_2samp(a_length,b_length)
        nll_test=scipy_stats.wilcoxon(a_nll,b_nll)
        repeat_test=scipy_stats.wilcoxon(a_repeat,b_repeat)
        out[f"target_plain_vs_{other}"]={"length_ks_statistic":float(ks.statistic),"length_ks_p":float(ks.pvalue),
            "mean_reference_length":float(np.mean(a_length)),"mean_other_length":float(np.mean(b_length)),
            "nll_wilcoxon_p":float(nll_test.pvalue),"mean_reference_nll":float(np.mean(a_nll)),"mean_other_nll":float(np.mean(b_nll)),"nll_pairs":len(a_nll),
            "repeat_wilcoxon_p":float(repeat_test.pvalue),"mean_reference_repeat":float(np.mean(a_repeat)),"mean_other_repeat":float(np.mean(b_repeat))}
    out["reading"]="spec should NOT differ from target_plain (large p-values); teacher was sampled with repetition_penalty 1.1 (lower repeat fraction expected); original_draft is the positive control"
    save_json(repo_path(config["results_dir"])/"banks"/"consistency.json",out)
    for other in ["spec","teacher","original"]:
        print(other,out[f"target_plain_vs_{other}"],flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/task3.yaml")
    parser.add_argument("--bank",required=True,choices=["exact_spec","target_plain","student","consistency"])
    parser.add_argument("--name",default=None)
    args=parser.parse_args()
    config=load_yaml(args.config)
    if args.bank=="consistency":
        consistency(config)
    else:
        make_bank(config,args.bank,args.name)


if __name__=="__main__":
    main()
