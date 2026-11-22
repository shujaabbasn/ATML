"""Shared evaluation for every task 1 condition: two perplexity suites, hellaswag, latency and memory.
Manual: every later comparison uses this same baseline and evaluation code."""
from __future__ import annotations

import math
import re

import numpy as np
import torch
import torch.nn.functional as F

from common.utils import read_jsonl, data_path, time_call, summarize_times, reset_peak, peak_mb, sync


@torch.no_grad()
def perplexity(model,windows,batch_size,pad_id):
    """windows = staff locked records. Each record scores token_ids[a:b] with a,b=score_target_offsets.
    The first token of each window is never a target (manual: exclude it from the denominator).
    Returns ppl, total scored tokens and the per-token nll array (kept for the paired bootstrap)."""
    device=next(model.parameters()).device
    all_nll=[] #one array per window, in file order
    for start in range(0,len(windows),batch_size):
        batch=windows[start:start+batch_size]
        longest=0
        for record in batch:
            if len(record["token_ids"])>longest:
                longest=len(record["token_ids"])
        ids=torch.full((len(batch),longest),pad_id,dtype=torch.long) #right padding
        mask=torch.zeros((len(batch),longest),dtype=torch.long)
        for row in range(len(batch)):
            tokens=batch[row]["token_ids"]
            ids[row,:len(tokens)]=torch.tensor(tokens)
            mask[row,:len(tokens)]=1
        ids=ids.to(device)
        mask=mask.to(device)
        logits=model(input_ids=ids,attention_mask=mask).logits.float() #fp32 for the log softmax
        log_probs=F.log_softmax(logits,dim=-1)
        for row in range(len(batch)):
            first=batch[row]["score_target_offsets"][0] #always 1
            last=batch[row]["score_target_offsets"][1] #exclusive
            targets=ids[row,first:last]
            predicted=log_probs[row,first-1:last-1] #logit at t predicts token t+1
            token_nll=-predicted.gather(1,targets.unsqueeze(1)).squeeze(1)
            if len(token_nll)!=batch[row]["scored_tokens"]:
                raise RuntimeError("scored token count does not match the staff record")
            all_nll.append(token_nll.cpu().numpy().astype(np.float32))
    nll=np.concatenate(all_nll)
    return {"ppl":float(math.exp(nll.astype(np.float64).mean())),"scored_tokens":int(len(nll)),"mean_nll":float(nll.mean())},nll


def hellaswag_preprocess(text):
    """Same cleaning as lm-evaluation-harness hellaswag (we record this template in the results)."""
    text=text.strip()
    text=text.replace(" [title]",". ")
    text=re.sub("\\[.*?\\]","",text) #drop wikihow tags like [header]
    text=text.replace("  "," ")
    return text


def hellaswag_items(records,count):
    """Turns the first `count` locked records into (context, endings, gold)."""
    items=[]
    for record in records[:count]:
        example=record["example"]
        context=example["ctx_a"]+" "+example["ctx_b"].capitalize()
        query=hellaswag_preprocess(example["activity_label"]+": "+context)
        endings=[]
        for ending in example["endings"]:
            endings.append(hellaswag_preprocess(ending))
        items.append({"order":record["order"],"source_row_index":record["source_row_index"],"context":query,"endings":endings,"label":int(example["label"])})
    return items


HELLASWAG_TEMPLATE="context = activity_label + ': ' + ctx_a + ' ' + ctx_b.capitalize(); continuation = ' ' + ending; lm-eval preprocess; score = sum of continuation token log-probs; acc_norm divides by len(ending) in characters"


@torch.no_grad()
def hellaswag(model,tokenizer,items,batch_size,pad_id):
    """Zero-shot scoring: sum of log p(ending tokens | context) for the 4 endings.
    acc uses the raw sum, acc_norm divides by the ending's character length (manual: not by token count)."""
    device=next(model.parameters()).device
    pairs=[] #(item index, ending index, context ids, ending ids)
    for i in range(len(items)):
        context_ids=tokenizer(items[i]["context"],add_special_tokens=False)["input_ids"]
        for j in range(4):
            ending_ids=tokenizer(" "+items[i]["endings"][j],add_special_tokens=False)["input_ids"]
            pairs.append((i,j,context_ids,ending_ids))
    scores=np.zeros((len(items),4),dtype=np.float64)
    for start in range(0,len(pairs),batch_size):
        batch=pairs[start:start+batch_size]
        longest=0
        for pair in batch:
            length=len(pair[2])+len(pair[3])
            if length>longest:
                longest=length
        ids=torch.full((len(batch),longest),pad_id,dtype=torch.long)
        mask=torch.zeros((len(batch),longest),dtype=torch.long)
        for row in range(len(batch)):
            full=batch[row][2]+batch[row][3] #context then ending
            ids[row,:len(full)]=torch.tensor(full)
            mask[row,:len(full)]=1
        logits=model(input_ids=ids.to(device),attention_mask=mask.to(device)).logits.float()
        log_probs=F.log_softmax(logits,dim=-1)
        for row in range(len(batch)):
            i,j,context_ids,ending_ids=batch[row]
            first=len(context_ids) #first ending token position
            last=first+len(ending_ids)
            targets=ids[row,first:last].to(device)
            predicted=log_probs[row,first-1:last-1]
            scores[i,j]=float(predicted.gather(1,targets.unsqueeze(1)).sum())
    per_example=[] #kept for paired bootstrap and the qualitative example
    correct=0
    correct_norm=0
    for i in range(len(items)):
        normalized=[]
        for j in range(4):
            normalized.append(scores[i,j]/len(items[i]["endings"][j]))
        pick=int(np.argmax(scores[i]))
        pick_norm=int(np.argmax(normalized))
        hit=0
        if pick==items[i]["label"]:
            hit=1
        hit_norm=0
        if pick_norm==items[i]["label"]:
            hit_norm=1
        correct=correct+hit
        correct_norm=correct_norm+hit_norm
        per_example.append({"order":items[i]["order"],"label":items[i]["label"],"scores":scores[i].tolist(),"normalized":normalized,"pick":pick,"pick_norm":pick_norm,"acc":hit,"acc_norm":hit_norm})
    summary={"acc":correct/len(items),"acc_norm":correct_norm/len(items),"examples":len(items)}
    return summary,per_example


@torch.no_grad()
def resources(model,token_pool,lengths,decode_tokens,warmup,repeats):
    """Prefill latency for each prompt length, 64-token greedy decode latency and peak allocated memory.
    Fake-quantized or masked weights still run dense fp16 matmuls, so these numbers are expected to barely move."""
    device=next(model.parameters()).device
    out={}
    for length in lengths:
        ids=torch.tensor(token_pool[:length],device=device).unsqueeze(0) #batch 1

        def prefill():
            model(input_ids=ids,use_cache=False)

        reset_peak()
        times=time_call(prefill,warmup,repeats)
        out[f"prefill_{length}"]=summarize_times(times)
        out[f"prefill_{length}"]["peak_mb"]=peak_mb()
    prompt=torch.tensor(token_pool[:512],device=device).unsqueeze(0) #decode after a 512-token prompt (ours)

    def decode():
        output=model(input_ids=prompt,use_cache=True)
        cache=output.past_key_values
        next_token=output.logits[:,-1:].argmax(dim=-1)
        for step in range(decode_tokens):
            output=model(input_ids=next_token,past_key_values=cache,use_cache=True)
            cache=output.past_key_values
            next_token=output.logits[:,-1:].argmax(dim=-1) #greedy

    def prefill_only():
        model(input_ids=prompt,use_cache=True)

    reset_peak()
    total=time_call(decode,warmup,repeats)
    peak=peak_mb()
    base=time_call(prefill_only,warmup,repeats)
    decode_ms=[]
    for i in range(len(total)):
        decode_ms.append(total[i]-base[i]) #remove the prompt prefill so only the 64 steps remain
    out["decode_64"]=summarize_times(decode_ms)
    out["decode_64"]["per_token_ms"]=out["decode_64"]["mean_ms"]/decode_tokens
    out["decode_64"]["peak_mb"]=peak
    sync()
    return out


def load_eval_data(config):
    data={}
    data["wikitext"]=read_jsonl(data_path(config["data"]["wikitext"]))
    data["c4_val"]=read_jsonl(data_path(config["data"]["c4_val"]))
    data["hellaswag"]=read_jsonl(data_path(config["data"]["hellaswag"]))
    data["calibration"]=read_jsonl(data_path(config["data"]["calibration"]))
    data["development"]=read_jsonl(data_path(config["data"]["development"]))
    return data
