"""Part C: KV-cache memory with Qwen2.5-1.5B-Instruct (fp16) on the 180 staff retrieval cases.

python -m task2_attention.kv_cache --step c1        full-cache vs no-cache check
python -m task2_attention.kv_cache --step eval      reference + the five compressed caches on all cases (C2-C5 quality)
python -m task2_attention.kv_cache --step timing    packing/unpacking, conversion and decode timing (C5 systems)
python -m task2_attention.kv_cache --step summary   tables and paired bootstrap

How a case runs (same for every cache):
  1. tokenize the chat-template prompt and split it at QUESTION_BEGIN: context tokens | question suffix tokens
  2. prefill the context in fp16 -> the pre-question cache (this is the point where budgets are measured)
  3. compress that context cache (or not, for the reference)
  4. feed the question suffix on top of the compressed cache, then greedy-decode the answer
  New tokens always get their true absolute positions, so RoPE stays right after eviction.
  The question-aware score comes from a separate pass of the question over the FULL cache; that pass is thrown
  away afterwards, so the question's own KV entries never carry information from evicted tokens.
Gold evidence offsets are only used for evaluation (retained-evidence fraction), never for selection.
"""
from __future__ import annotations

import argparse
import math
import re

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from common.utils import load_yaml, save_json, load_json, read_jsonl, write_jsonl, repo_path, data_path, set_seed, get_device, time_call, summarize_times, reset_peak, peak_mb, allocated_mb, bootstrap_mean_ci, package_versions


METHODS=["full","sink_recent","question_aware","int4_tokenwise","int4_kivi","hybrid"]


#---------------- prompt handling ----------------

def build_case(tokenizer,case):
    """Chat-template prompt split at QUESTION_BEGIN, plus the token indices that overlap the gold evidence span."""
    text=tokenizer.apply_chat_template([{"role":"user","content":case["user_text"]}],tokenize=False,add_generation_prompt=True)
    encoded=tokenizer(text,return_offsets_mapping=True,add_special_tokens=False)
    ids=encoded["input_ids"]
    offsets=encoded["offset_mapping"]
    user_start=text.index(case["user_text"]) #where the user message starts inside the template
    split_char=user_start+case["user_text"].index("QUESTION_BEGIN")
    evidence_start=user_start+case["evidence_char_start"]
    evidence_end=user_start+case["evidence_char_end"]
    context_length=None
    evidence=[]
    for i in range(len(ids)):
        start,end=offsets[i]
        if context_length is None and start>=split_char:
            context_length=i #first token of the question suffix
        if end>evidence_start and start<evidence_end:
            evidence.append(i) #token overlaps the evidence characters
    distractor_codes=re.findall(r"access code for OTHER-\S+ is (\d{5})",case["user_text"])
    gold_ids=tokenizer(case["code"],add_special_tokens=False)["input_ids"]
    return {"id":case["id"],"ids":ids,"context_length":context_length,"total_length":len(ids),"evidence":evidence,"gold_ids":gold_ids,"code":case["code"],"distractor_codes":distractor_codes,
        "target_tokens":case["target_tokens"],"position":case["position"],"distractors":case["distractors"]}


#---------------- cache helpers ----------------

def cache_from_tensors(keys,values):
    cache=DynamicCache()
    for layer in range(len(keys)):
        cache.update(keys[layer],values[layer],layer)
    return cache


def cache_tensors(cache):
    keys=[]
    values=[]
    for layer in cache.layers:
        keys.append(layer.keys)
        values.append(layer.values)
    return keys,values


def clone_cache(cache):
    keys,values=cache_tensors(cache)
    new_keys=[]
    new_values=[]
    for layer in range(len(keys)):
        new_keys.append(keys[layer].clone())
        new_values.append(values[layer].clone())
    return cache_from_tensors(new_keys,new_values)


def cache_bytes(cache):
    """Bytes the cache tensors really occupy on the gpu (fake-quantized entries are still fp16)."""
    total=0
    keys,values=cache_tensors(cache)
    for layer in range(len(keys)):
        total=total+keys[layer].numel()*keys[layer].element_size()+values[layer].numel()*values[layer].element_size()
    return total


#---------------- 4-bit group quantization (fp16 scale + fp16 zero per group of 64) ----------------

def fake_quant_last(x,bits):
    """Asymmetric quantization along the last axis (one group): zero = min, scale = (max-min)/(2^b-1), both
    rounded to fp16 because that is how they are stored. Returns the dequantized tensor in x's dtype."""
    levels=2**bits-1
    x32=x.float()
    low=x32.amin(dim=-1,keepdim=True).half().float()
    high=x32.amax(dim=-1,keepdim=True)
    scale=((high-low)/levels).clamp(min=1e-6).half().float()
    q=torch.clamp(torch.round((x32-low)/scale),0,levels)
    return (q*scale+low).to(x.dtype)


def quant_tokenwise(x,bits,group):
    """x (B,H,T,d): each token's d channels split into groups of 64."""
    b,h,t,d=x.shape
    grouped=x.reshape(b,h,t,d//group,group)
    return fake_quant_last(grouped,bits).reshape(b,h,t,d)


def quant_channelwise(x,bits,group):
    """x (B,H,T,d) with T a multiple of 64: each channel's 64 consecutive tokens form a group (KIVI keys)."""
    b,h,t,d=x.shape
    blocks=x.reshape(b,h,t//group,group,d).transpose(-1,-2) #(B,H,blocks,d,64)
    out=fake_quant_last(blocks,bits).transpose(-1,-2)
    return out.reshape(b,h,t,d)


def packed_int4_bytes(config,length,layers,kv_heads,head_dim):
    """Manual formula M_q(L) = 2 n_l H_kv d_h [ floor(L/64)(64 b/8 + 4) + 2 (L mod 64) ] (bytes)."""
    group=config["group"]
    full_groups=length//group
    tail=length%group
    return 2*layers*kv_heads*head_dim*(full_groups*(group*config["bits"]//8+4)+2*tail)


def pack_nibbles(codes):
    """Real packing for the C5 timing: two 4-bit codes per byte."""
    flat=codes.reshape(-1)
    if flat.numel()%2==1:
        flat=torch.cat([flat,flat.new_zeros(1)])
    low=flat[0::2]
    high=flat[1::2]
    return (low|(high<<4)).to(torch.uint8)


def unpack_nibbles(packed,count):
    low=packed&15
    high=packed>>4
    out=torch.stack([low,high],dim=1).reshape(-1)
    return out[:count]


def quantize_codes(x,bits):
    """Integer codes + fp16 scale/zero along the last axis (the packed representation)."""
    levels=2**bits-1
    x32=x.float()
    low=x32.amin(dim=-1,keepdim=True)
    scale=((x32.amax(dim=-1,keepdim=True)-low)/levels).clamp(min=1e-6)
    q=torch.clamp(torch.round((x32-low)/scale),0,levels).to(torch.uint8)
    return q,scale.half(),low.half()


#---------------- the compressed caches ----------------

class Context:
    """Everything one case needs for compression: the fp16 context cache, the question-aware scores and sizes."""

    def __init__(self,cache,scores,config,model_config,context_length):
        self.cache=cache
        self.scores=scores #per layer (kv_heads, L) or None
        self.config=config
        self.layers=model_config.num_hidden_layers
        self.kv_heads=model_config.num_key_value_heads
        self.head_dim=model_config.hidden_size//model_config.num_attention_heads
        self.length=context_length
        self.budget=packed_int4_bytes(config,context_length,self.layers,self.kv_heads,self.head_dim) #matched int4 budget B


def gather_positions(keys,values,keep):
    """keep (H_kv, N) sorted positions per kv head -> (1,H_kv,N,d) tensors."""
    index=keep.unsqueeze(0).unsqueeze(-1).expand(1,keep.shape[0],keep.shape[1],keys.shape[-1])
    return torch.gather(keys,2,index),torch.gather(values,2,index)


def eviction_keep(context,layer,rule,count):
    """Positions kept by one eviction rule for one layer, (H_kv, count). Both rules always keep the
    8 sink tokens and the 64 most recent tokens (manual safeguard)."""
    config=context.config
    length=context.length
    sinks=list(range(config["sinks"]))
    recent=list(range(length-config["recent"],length))
    must=sorted(set(sinks+recent))
    keep_rows=[]
    for head in range(context.kv_heads):
        if rule=="sink_recent":
            rest=count-len(must)
            middle_end=length-config["recent"]
            chosen=list(range(middle_end-rest,middle_end)) #most recent of what is left
        else:
            score=context.scores[layer][head].clone()
            score[must]=float("-inf") #already kept
            rest=count-len(must)
            chosen=torch.topk(score,rest).indices.tolist() #highest question attention
        row=sorted(set(must+chosen))
        keep_rows.append(row)
    return torch.tensor(keep_rows,device=context.cache.layers[0].keys.device)


def compress_eviction(context,rule):
    """C2: fp16 tokens only, the largest count whose bytes fit the int4 budget. Each kept token costs K and V in fp16
    (4 d_h bytes) plus its int32 position index, because the selection differs per layer and kv head."""
    per_token=4*context.head_dim+context.config["index_bytes"]
    count=int(context.budget//(context.layers*context.kv_heads*per_token))
    count=min(count,context.length)
    keys,values=cache_tensors(context.cache)
    new_keys=[]
    new_values=[]
    kept=[]
    for layer in range(context.layers):
        keep=eviction_keep(context,layer,rule,count)
        k,v=gather_positions(keys[layer],values[layer],keep)
        new_keys.append(k)
        new_values.append(v)
        kept.append(keep.cpu())
    stored=context.layers*context.kv_heads*per_token*count
    info={"fp16_tokens_per_head":int(count),"lowbit_tokens_per_head":0,"packed_bytes":int(stored),"index_bytes":int(context.config["index_bytes"]*count*context.layers*context.kv_heads)}
    return cache_from_tensors(new_keys,new_values),kept,None,info


def compress_quant(context,layout):
    """C3: every context token kept, complete 64-token blocks in int4, the trailing L mod 64 tokens fp16."""
    config=context.config
    group=config["group"]
    keys,values=cache_tensors(context.cache)
    upto=(context.length//group)*group
    new_keys=[]
    new_values=[]
    k_error=0.0
    k_norm=0.0
    v_error=0.0
    v_norm=0.0
    for layer in range(context.layers):
        k=keys[layer].clone()
        v=values[layer].clone()
        if upto>0:
            if layout=="int4_tokenwise":
                k[:,:,:upto]=quant_tokenwise(k[:,:,:upto],config["bits"],group)
            else:
                k[:,:,:upto]=quant_channelwise(k[:,:,:upto],config["bits"],group) #kivi: keys per channel
            v[:,:,:upto]=quant_tokenwise(v[:,:,:upto],config["bits"],group) #values per token in both layouts
        k_error=k_error+float(((k.float()-keys[layer].float())**2).sum())
        k_norm=k_norm+float((keys[layer].float()**2).sum())
        v_error=v_error+float(((v.float()-values[layer].float())**2).sum())
        v_norm=v_norm+float((values[layer].float()**2).sum())
        new_keys.append(k)
        new_values.append(v)
    info={"fp16_tokens_per_head":context.length-upto,"lowbit_tokens_per_head":upto,"packed_bytes":int(context.budget),
        "k_relative_error":math.sqrt(k_error/max(k_norm,1e-12)),"v_relative_error":math.sqrt(v_error/max(v_norm,1e-12))} #C3: K and V reconstruction separately
    state={"layout":layout,"quantized_upto":upto} #residual rule state for later tokens
    return cache_from_tensors(new_keys,new_values),None,state,info


def residual_quantize(cache,state,config):
    """Keep the C3 rule for tokens added after the pre-question point: once 64 unquantized tokens have piled up
    after the last quantized block, that block is quantized too."""
    group=config["group"]
    length=cache.get_seq_length()
    while length-state["quantized_upto"]>=group:
        a=state["quantized_upto"]
        b=a+group
        for layer in cache.layers:
            if state["layout"]=="int4_tokenwise":
                layer.keys[:,:,a:b]=quant_tokenwise(layer.keys[:,:,a:b],config["bits"],group)
            else:
                layer.keys[:,:,a:b]=quant_channelwise(layer.keys[:,:,a:b],config["bits"],group)
            layer.values[:,:,a:b]=quant_tokenwise(layer.values[:,:,a:b],config["bits"],group)
        state["quantized_upto"]=b


def hybrid_plan(context):
    """C4 sizes per (layer, kv head), in bytes, metadata included:
      fp16 token: K and V, 2*d*2 bytes, + an int32 position index (the int2 positions are the complement, so no index)
      int2 group of 64 tokens: keys per channel d*(64*2/8+4) + values per token 64*(d/64)*(64*2/8+4) = 2*d*(16+4)
    Every context token stays in the cache. The 8 sinks and 64 recent tokens are fp16; of the rest, the lowest-ranked
    go to int2 in complete 64-token groups and everything else (top-ranked + the incomplete residual) stays fp16.
    G = the smallest number of int2 groups whose total fits the INT4 budget, which is the largest legal fp16 count."""
    config=context.config
    d=context.head_dim
    group=config["group"]
    bits=config["hybrid_bits"]
    per_head_budget=context.budget/(context.layers*context.kv_heads)
    fp16_cost=4*d+config["index_bytes"]
    group_cost=2*d*(group*bits//8+4)
    base=config["sinks"]+config["recent"]
    max_groups=(context.length-base)//group
    chosen=None
    for groups in range(max_groups+1):
        cost=(context.length-groups*group)*fp16_cost+groups*group_cost
        if cost<=per_head_budget:
            chosen=groups
            break
    if chosen is None:
        raise RuntimeError("hybrid cannot fit the INT4 budget even with every non-safeguard token in INT2")
    fp16_tokens=context.length-chosen*group
    used=fp16_tokens*fp16_cost+chosen*group_cost
    return {"fp16_tokens":fp16_tokens,"extra_fp16":fp16_tokens-base,"lowbit_groups":chosen,"lowbit_tokens":chosen*group,"bits":bits,
        "bytes_per_head":used,"budget_per_head":per_head_budget,"index_bytes_per_head":fp16_tokens*config["index_bytes"]}


def compress_hybrid(context):
    """C4: sinks + recent + the top question-aware tokens in fp16, the lowest-ranked tokens in complete 64-token INT2
    groups (grouped in position order; keys per channel, values per token). Nothing is evicted."""
    config=context.config
    plan=hybrid_plan(context)
    keys,values=cache_tensors(context.cache)
    length=context.length
    must=sorted(set(list(range(config["sinks"]))+list(range(length-config["recent"],length))))
    new_keys=[]
    new_values=[]
    kept=[]
    fp16_sets=[]
    for layer in range(context.layers):
        layer_keys=[]
        layer_values=[]
        layer_keep=[]
        layer_fp16=[]
        for head in range(context.kv_heads):
            score=context.scores[layer][head].clone()
            score[must]=float("-inf")
            ranked=torch.argsort(score,descending=True).tolist() #the must tokens end up last, they are dropped below
            ranked=ranked[:length-len(must)]
            extra=ranked[:plan["extra_fp16"]]
            quantized=sorted(ranked[plan["extra_fp16"]:])
            fp16_positions=sorted(set(must+extra))
            k_fp=keys[layer][:,head:head+1,fp16_positions]
            v_fp=values[layer][:,head:head+1,fp16_positions]
            if len(quantized)>0:
                k_q=quant_channelwise(keys[layer][:,head:head+1,quantized],plan["bits"],config["group"])
                v_q=quant_tokenwise(values[layer][:,head:head+1,quantized],plan["bits"],config["group"])
                layer_keys.append(torch.cat([k_q,k_fp],dim=2)) #order inside the cache does not matter, rope is already in the keys
                layer_values.append(torch.cat([v_q,v_fp],dim=2))
            else:
                layer_keys.append(k_fp)
                layer_values.append(v_fp)
            layer_keep.append(sorted(quantized+fp16_positions))
            layer_fp16.append(fp16_positions)
        new_keys.append(torch.cat(layer_keys,dim=1))
        new_values.append(torch.cat(layer_values,dim=1))
        kept.append(torch.tensor(layer_keep))
        fp16_sets.append(layer_fp16)
    info={"fp16_tokens_per_head":plan["fp16_tokens"],"lowbit_tokens_per_head":plan["lowbit_tokens"],"packed_bytes":int(plan["bytes_per_head"]*context.layers*context.kv_heads),
        "index_bytes":int(plan["index_bytes_per_head"]*context.layers*context.kv_heads),"plan":plan}
    return cache_from_tensors(new_keys,new_values),kept,{"fp16":fp16_sets},info


def compress(context,method):
    if method=="full":
        info={"fp16_tokens_per_head":context.length,"lowbit_tokens_per_head":0,"packed_bytes":int(4*context.layers*context.kv_heads*context.head_dim*context.length)}
        return clone_cache(context.cache),None,None,info
    elif method=="sink_recent" or method=="question_aware":
        return compress_eviction(context,method)
    elif method=="int4_tokenwise" or method=="int4_kivi":
        return compress_quant(context,method)
    else:
        return compress_hybrid(context)


#---------------- model passes ----------------

@torch.no_grad()
def question_scores(model,cache,question_ids,start_position):
    """Question-aware importance: run the question suffix over a copy of the full context cache, grab every layer's
    query projection, apply RoPE at the question's positions, and take softmax attention over the context keys.
    Summed over question tokens and over the query heads that share each kv head (GQA), kept per layer and kv head."""
    config=model.config
    heads=config.num_attention_heads
    kv_heads=config.num_key_value_heads
    head_dim=config.hidden_size//heads
    captured={}
    handles=[]
    for index in range(len(model.model.layers)):
        def hook(module,args,output,index=index):
            captured[index]=output.detach() #(1, Tq, heads*d), before rope
        handles.append(model.model.layers[index].self_attn.q_proj.register_forward_hook(hook))
    keys,values=cache_tensors(cache)
    context_length=keys[0].shape[2]
    work=clone_cache(cache)
    positions=torch.arange(start_position,start_position+question_ids.shape[1],device=question_ids.device).unsqueeze(0)
    try:
        model(input_ids=question_ids,past_key_values=work,position_ids=positions,use_cache=True)
    finally:
        for handle in handles:
            handle.remove()
    scores=[]
    for index in range(len(model.model.layers)):
        q=captured[index]
        q=q.view(1,q.shape[1],heads,head_dim).transpose(1,2) #(1,H,Tq,d)
        cos,sin=model.model.rotary_emb(q,positions)
        q,unused=apply_rotary_pos_emb(q,q,cos,sin)
        k=keys[index][:,:,:context_length].repeat_interleave(heads//kv_heads,dim=1) #(1,H,L,d)
        attention=torch.softmax((q.float()@k.float().transpose(-1,-2))/math.sqrt(head_dim),dim=-1) #over context keys only
        per_head=attention.sum(dim=2)[0] #(H,L)
        grouped=per_head.view(kv_heads,heads//kv_heads,context_length).sum(dim=1) #(H_kv,L)
        scores.append(grouped)
    del work
    return scores


@torch.no_grad()
def run_on_cache(model,cache,question_ids,gold_ids,start_position,max_new,eos_ids,quant_state,config):
    """Question suffix on the given cache, then (a) teacher-forced gold log-prob on a copy, (b) greedy answer."""
    device=question_ids.device
    positions=torch.arange(start_position,start_position+question_ids.shape[1],device=device).unsqueeze(0)
    output=model(input_ids=question_ids,past_key_values=cache,position_ids=positions,use_cache=True)
    if quant_state is not None:
        residual_quantize(cache,quant_state,config)
    first_logits=output.logits[0,-1].float()
    next_position=start_position+question_ids.shape[1]
    #teacher forcing of the gold code on a copy of the cache
    forced=clone_cache(cache)
    gold=torch.tensor([gold_ids],device=device)
    gold_positions=torch.arange(next_position,next_position+gold.shape[1],device=device).unsqueeze(0)
    forced_output=model(input_ids=gold,past_key_values=forced,position_ids=gold_positions,use_cache=True)
    all_logits=torch.cat([first_logits.unsqueeze(0),forced_output.logits[0,:-1].float()],dim=0) #predicts each gold token
    log_probs=F.log_softmax(all_logits,dim=-1)
    gold_logprob=float(log_probs.gather(1,gold[0].unsqueeze(1)).sum())
    del forced
    #greedy answer
    generated=[]
    logits=first_logits
    position=next_position
    for step in range(max_new):
        token=int(logits.argmax())
        if token in eos_ids:
            break
        generated.append(token)
        step_output=model(input_ids=torch.tensor([[token]],device=device),past_key_values=cache,position_ids=torch.tensor([[position]],device=device),use_cache=True)
        if quant_state is not None:
            residual_quantize(cache,quant_state,config)
        logits=step_output.logits[0,-1].float()
        position=position+1
    return {"generated":generated,"gold_log_probs":all_logits,"gold_logprob":gold_logprob}


def classify_answer(text,case):
    found=re.findall(r"\d{5}",text)
    if len(found)==0:
        return False,"no five-digit code"
    answer=found[0]
    if answer==case["code"]:
        return True,"gold"
    elif answer in case["distractor_codes"]:
        return False,"distractor code"
    else:
        return False,"other code"


def kl_rows(reference_logits,logits):
    """KL(reference || compressed) of the next-token distributions, per gold position."""
    p=F.log_softmax(reference_logits,dim=-1)
    q=F.log_softmax(logits,dim=-1)
    return (p.exp()*(p-q)).sum(dim=-1)


def evidence_kept(kept,fp16_sets,evidence):
    """Share of evidence tokens still in the cache (any precision / fp16 only), averaged over layers and kv heads."""
    if kept is None:
        return 1.0,None
    evidence_set=set(evidence)
    total=0.0
    total_fp16=0.0
    count=0
    for layer in range(len(kept)):
        for head in range(kept[layer].shape[0]):
            row=set(kept[layer][head].tolist())
            total=total+len(row&evidence_set)/max(len(evidence_set),1)
            if fp16_sets is not None:
                total_fp16=total_fp16+len(set(fp16_sets[layer][head])&evidence_set)/max(len(evidence_set),1)
            count=count+1
    if fp16_sets is None:
        return total/count,total/count
    return total/count,total_fp16/count


#---------------- steps ----------------

def load_everything(full_config):
    config=full_config["kv"]
    device=get_device()
    tokenizer=AutoTokenizer.from_pretrained(config["model"])
    if device.type=="cuda":
        dtype=torch.float16
    else:
        dtype=torch.float32
    model=AutoModelForCausalLM.from_pretrained(config["model"],dtype=dtype,attn_implementation="sdpa").to(device).eval()
    cases=read_jsonl(data_path("data/"+config["cases"]))
    built=[]
    for case in cases:
        built.append(build_case(tokenizer,case))
    if full_config.get("limit") is not None:
        built=built[::len(built)//full_config["limit"]][:full_config["limit"]] #smoke tests only, spread over the cells
    eos_ids=[tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids("<|im_end|>"),tokenizer.convert_tokens_to_ids("<|endoftext|>")]
    return config,device,tokenizer,model,built,eos_ids


@torch.no_grad()
def step_c1(full_config,out_dir):
    """C1: (a) logits of prompt+gold with one uncached forward vs. cached prefill + incremental tokens,
    (b) greedy tokens with the cache vs. recomputing the whole sequence every step, (c) all logits finite."""
    config,device,tokenizer,model,cases,eos_ids=load_everything(full_config)
    rows=[]
    for case in cases[:config["c1_cases"]]:
        ids=torch.tensor([case["ids"]+case["gold_ids"]],device=device)
        prompt_length=case["total_length"]
        uncached=model(input_ids=ids,use_cache=False).logits[0,prompt_length-1:-1].float() #logits predicting each gold token
        output=model(input_ids=ids[:,:prompt_length],use_cache=True)
        cache=output.past_key_values
        cached=[output.logits[0,-1].float()]
        for j in range(len(case["gold_ids"])-1):
            step=model(input_ids=ids[:,prompt_length+j:prompt_length+j+1],past_key_values=cache,use_cache=True)
            cached.append(step.logits[0,-1].float())
        cached=torch.stack(cached)
        #greedy with cache vs recompute
        output=model(input_ids=ids[:,:prompt_length],use_cache=True)
        cache=output.past_key_values
        token=int(output.logits[0,-1].argmax())
        with_cache=[token]
        sequence=case["ids"]+[token]
        recomputed=[token]
        for step_index in range(config["max_new_tokens"]-1):
            step=model(input_ids=torch.tensor([[with_cache[-1]]],device=device),past_key_values=cache,use_cache=True)
            with_cache.append(int(step.logits[0,-1].argmax()))
            full=model(input_ids=torch.tensor([case["ids"]+recomputed],device=device),use_cache=False)
            recomputed.append(int(full.logits[0,-1].argmax()))
        text=tokenizer.decode(with_cache,skip_special_tokens=True)
        correct,kind=classify_answer(text,case)
        rows.append({"id":case["id"],"encoded_length":case["total_length"],"context_length":case["context_length"],"max_abs_logit_diff":float((uncached-cached).abs().max()),
            "argmax_agree":bool((uncached.argmax(-1)==cached.argmax(-1)).all()),"greedy_tokens_equal":with_cache==recomputed,"finite":bool(torch.isfinite(uncached).all() and torch.isfinite(cached).all()),"full_cache_correct":correct,"answer":text})
    agree=0
    equal=0
    finite=0
    correct=0
    worst=0.0
    for row in rows:
        agree=agree+int(row["argmax_agree"])
        equal=equal+int(row["greedy_tokens_equal"])
        finite=finite+int(row["finite"])
        correct=correct+int(row["full_cache_correct"])
        worst=max(worst,row["max_abs_logit_diff"])
    save_json(out_dir/"c1_reference_check.json",{"cases":len(rows),"argmax_agree":agree,"greedy_equal":equal,"finite":finite,"full_cache_accuracy":correct/len(rows),"worst_max_abs_logit_diff":worst,"rows":rows,"versions":package_versions()})
    print("c1 cases",len(rows),"greedy equal",equal,"accuracy",correct/len(rows))


@torch.no_grad()
def prepare_context(model,case,device,config):
    ids=torch.tensor([case["ids"]],device=device)
    context_ids=ids[:,:case["context_length"]]
    question_ids=ids[:,case["context_length"]:]
    output=model(input_ids=context_ids,use_cache=True)
    cache=output.past_key_values
    scores=question_scores(model,cache,question_ids,case["context_length"])
    return Context(cache,scores,config,model.config,case["context_length"]),question_ids


@torch.no_grad()
def step_eval(full_config,out_dir):
    config,device,tokenizer,model,cases,eos_ids=load_everything(full_config)
    rows=[]
    examples=[]
    for case_index in range(len(cases)):
        case=cases[case_index]
        context,question_ids=prepare_context(model,case,device,config)
        reference_logits=None
        for method in METHODS:
            cache,kept,state,info=compress(context,method)
            quant_state=None
            if method=="int4_tokenwise" or method=="int4_kivi":
                quant_state=state
            result=run_on_cache(model,cache,question_ids,case["gold_ids"],case["context_length"],config["max_new_tokens"],eos_ids,quant_state,config)
            if method=="full":
                reference_logits=result["gold_log_probs"]
            kl=kl_rows(reference_logits,result["gold_log_probs"])
            text=tokenizer.decode(result["generated"],skip_special_tokens=True)
            correct,kind=classify_answer(text,case)
            fp16_sets=None
            if state is not None and "fp16" in state:
                fp16_sets=state["fp16"]
            kept_any,kept_fp16=evidence_kept(kept,fp16_sets,case["evidence"])
            row={"id":case["id"],"method":method,"target_tokens":case["target_tokens"],"position":case["position"],"distractors":case["distractors"],
                "context_length":case["context_length"],"encoded_length":case["total_length"],"budget_bytes":int(context.budget),
                "correct":correct,"answer_kind":kind,"answer":text,"gold_logprob":result["gold_logprob"],"kl_first":float(kl[0]),"kl_mean_gold":float(kl.mean()),
                "evidence_retained":kept_any,"evidence_retained_fp16":kept_fp16}
            row.update(info)
            full_fp16=4*context.layers*context.kv_heads*context.head_dim*context.length
            row["fraction_of_full_fp16_bytes"]=info["packed_bytes"]/full_fp16 #manual: realized fraction for every prompt
            shared_tokens=question_ids.shape[1]+len(result["generated"])
            row["shared_decode_fp16_bytes"]=4*context.layers*context.kv_heads*context.head_dim*shared_tokens #question suffix + generated tokens, same for every method
            row["total_live_cache_bytes"]=info["packed_bytes"]+row["shared_decode_fp16_bytes"]
            if "plan" in row:
                del row["plan"]
            rows.append(row)
            if case_index%30==0 and kept is not None:
                example_keep=kept[len(kept)//2][0].tolist() #middle layer, kv head 0, for the qualitative table
                example={"id":case["id"],"method":method,"layer":len(kept)//2,"kv_head":0,"kept_positions":example_keep,"evidence_positions":case["evidence"],"answer":text}
                if fp16_sets is not None:
                    example["fp16_positions"]=fp16_sets[len(kept)//2][0] #hybrid: which kept tokens stayed fp16
                examples.append(example)
            del cache
        del context
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if case_index%10==0:
            print("case",case_index,case["id"],flush=True)
    write_jsonl(out_dir/"c_eval_rows.jsonl",rows)
    write_jsonl(out_dir/"c_masks_examples.jsonl",examples)


def step_timing(full_config,out_dir):
    """C5 systems: prefill, question scoring, conversion (selection or quantization) and one decode step on each
    cache at batch 1, then the packed encode/decode benchmark at batch 1 and 8, both target lengths."""
    config,device,tokenizer,model,cases,eos_ids=load_everything(full_config)
    rows=[]
    packed_rows=[]
    for target in [2048,4096]:
        chosen=[]
        for case in cases:
            if case["target_tokens"]==target and len(chosen)<config["timing_cases_per_length"]:
                chosen.append(case)
        for case in chosen:
            ids=torch.tensor([case["ids"]],device=device)
            context_ids=ids[:,:case["context_length"]]

            def prefill():
                with torch.no_grad():
                    model(input_ids=context_ids,use_cache=True)

            prefill_times=summarize_times(time_call(prefill,config["warmup"],config["repeats"]))
            context,question_ids=prepare_context(model,case,device,config)

            def scoring():
                with torch.no_grad():
                    question_scores(model,context.cache,question_ids,case["context_length"])

            scoring_times=summarize_times(time_call(scoring,config["warmup"],config["repeats"]))
            for method in METHODS:
                def convert():
                    with torch.no_grad():
                        compress(context,method)

                convert_times=summarize_times(time_call(convert,config["warmup"],config["repeats"]))
                cache,kept,state,info=compress(context,method)
                position=case["context_length"]
                with torch.no_grad():
                    positions=torch.arange(position,position+question_ids.shape[1],device=device).unsqueeze(0)
                    model(input_ids=question_ids,past_key_values=cache,position_ids=positions,use_cache=True)
                base_length=cache.get_seq_length()
                token=torch.tensor([[13]],device=device)
                step_position=torch.tensor([[position+question_ids.shape[1]]],device=device)

                def decode_step():
                    with torch.no_grad():
                        model(input_ids=token,past_key_values=cache,position_ids=step_position,use_cache=True)
                        cache.crop(base_length) #undo the step so every repeat sees the same cache

                reset_peak()
                decode_times=summarize_times(time_call(decode_step,config["warmup"],config["repeats"]))
                row={"id":case["id"],"target":target,"context_length":case["context_length"],"method":method,"prefill":prefill_times,"convert":convert_times,"decode_step":decode_times,
                    "allocated_cache_bytes":cache_bytes(cache),"packed_bytes_formula":info["packed_bytes"],"decode_peak_mb":peak_mb()}
                if method=="question_aware" or method=="hybrid":
                    row["question_scoring"]=scoring_times
                rows.append(row)
                del cache
                print("timing",target,case["id"],method,flush=True)
            for batch in config["timing_batches"]:
                for layout in ["int4_tokenwise","int4_kivi"]:
                    try:
                        entry=packed_benchmark(model,context,config,layout,batch,device)
                        entry["status"]="ok"
                    except torch.cuda.OutOfMemoryError:
                        entry={"layout":layout,"batch":batch,"context_length":context.length,"status":"OOM"}
                    entry["id"]=case["id"]
                    entry["target"]=target
                    packed_rows.append(entry)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print("packed",target,batch,layout,entry["status"],flush=True)
    save_json(out_dir/"c5_timing.json",{"rows":rows,"packed":packed_rows,"versions":package_versions()})


def packed_benchmark(model,context,config,layout,batch,device):
    """C5 packed-cache systems cost at one batch size: really pack every complete 64-token block of the context cache
    (tiled to `batch` sequences) into nibbles + fp16 scale/zero, unpack + dequantize it, and time one decode step
    (a) on the fp16 cache and (b) after unpacking the packed cache into a fresh fp16 cache. Same shapes in both."""
    keys,values=cache_tensors(context.cache)
    group=config["group"]
    upto=(context.length//group)*group
    batched_keys=[]
    batched_values=[]
    for layer in range(len(keys)):
        batched_keys.append(keys[layer].expand(batch,-1,-1,-1).contiguous())
        batched_values.append(values[layer].expand(batch,-1,-1,-1).contiguous())

    def to_groups(x,channel):
        b,h,t,d=x.shape
        if channel==True:
            return x[:,:,:upto].reshape(b,h,upto//group,group,d).transpose(-1,-2) #(B,H,blocks,d,64) keys per channel
        return x[:,:,:upto].reshape(b,h,upto,d//group,group) #(B,H,T,d/64,64) per token

    def from_groups(x,channel,shape):
        if channel==True:
            return x.transpose(-1,-2).reshape(shape)
        return x.reshape(shape)

    def pack_all():
        packed=[]
        for layer in range(len(batched_keys)):
            for tensor,channel in [(batched_keys[layer],layout=="int4_kivi"),(batched_values[layer],False)]:
                q,scale,zero=quantize_codes(to_groups(tensor,channel),config["bits"])
                tail=tensor[:,:,upto:].clone() #fp16 residual, L mod 64 tokens
                packed.append((pack_nibbles(q),scale,zero,q.numel(),q.shape,channel,tail))
        return packed

    packed=pack_all()
    full_shape=(batch,context.kv_heads,upto,context.head_dim)

    def unpack_all():
        tensors=[]
        for data,scale,zero,count,shape,channel,tail in packed:
            codes=unpack_nibbles(data,count).reshape(shape)
            dequantized=(codes.float()*scale.float()+zero.float()).to(tail.dtype)
            tensors.append(torch.cat([from_groups(dequantized,channel,full_shape),tail],dim=2))
        return tensors

    token=torch.full((batch,1),13,device=device)
    step_position=torch.full((batch,1),context.length,device=device)

    def decode_fp16():
        with torch.no_grad():
            cache=cache_from_tensors(batched_keys,batched_values)
            model(input_ids=token,past_key_values=cache,position_ids=step_position,use_cache=True)

    def decode_unpacked():
        with torch.no_grad():
            tensors=unpack_all()
            new_keys=[]
            new_values=[]
            for layer in range(len(batched_keys)):
                new_keys.append(tensors[2*layer])
                new_values.append(tensors[2*layer+1])
            cache=cache_from_tensors(new_keys,new_values)
            model(input_ids=token,past_key_values=cache,position_ids=step_position,use_cache=True)

    measured=0
    for data,scale,zero,count,shape,channel,tail in packed:
        measured=measured+data.numel()+scale.numel()*2+zero.numel()*2+tail.numel()*2
    reset_peak()
    result={"layout":layout,"batch":batch,"context_length":context.length,
        "pack":summarize_times(time_call(pack_all,config["warmup"],config["repeats"])),
        "unpack_dequant":summarize_times(time_call(unpack_all,config["warmup"],config["repeats"])),
        "decode_step_fp16":summarize_times(time_call(decode_fp16,config["warmup"],config["repeats"])),
        "decode_step_with_unpack":summarize_times(time_call(decode_unpacked,config["warmup"],config["repeats"])),
        "packed_bytes_measured":int(measured),"packed_bytes_formula":int(batch*context.budget),"fp16_bytes":int(batch*4*context.layers*context.kv_heads*context.head_dim*context.length),
        "peak_allocated_mb":peak_mb(),"note":"unpacking dequantizes into an ordinary fp16 cache before SDPA, so allocated memory is fp16-sized"}
    return result


def step_summary(full_config,out_dir):
    config=full_config["kv"]
    rows=read_jsonl(out_dir/"c_eval_rows.jsonl")
    by_method={}
    for row in rows:
        if row["method"] not in by_method:
            by_method[row["method"]]=[]
        by_method[row["method"]].append(row)
    table={}
    for method in METHODS:
        group=by_method[method]
        correct=[]
        logprob=[]
        kl=[]
        evidence=[]
        evidence_fp16=[]
        kinds={}
        for row in group:
            correct.append(int(row["correct"]))
            logprob.append(row["gold_logprob"])
            kl.append(row["kl_mean_gold"])
            evidence.append(row["evidence_retained"])
            if row["evidence_retained_fp16"] is not None:
                evidence_fp16.append(row["evidence_retained_fp16"]) #only eviction and the hybrid select tokens
            kinds[row["answer_kind"]]=kinds.get(row["answer_kind"],0)+1
        evidence_fp16_mean=None #full cache and the int4 layouts keep every token
        if len(evidence_fp16)>0:
            evidence_fp16_mean=float(np.mean(evidence_fp16))
        reference_correct={}
        for row in by_method["full"]:
            reference_correct[row["id"]]=row["correct"]
        conditional=[]
        for row in group:
            if reference_correct[row["id"]]==True:
                conditional.append(int(row["correct"]))
        conditional_accuracy=None #stays None if the full cache got every case wrong
        if len(conditional)>0:
            conditional_accuracy=float(np.mean(conditional))
        table[method]={"accuracy":float(np.mean(correct)),"accuracy_given_reference_correct":conditional_accuracy,"reference_correct_cases":len(conditional),
            "fraction_of_full_fp16_bytes":mean_of(group,"fraction_of_full_fp16_bytes"),"mean_total_live_cache_bytes":mean_of(group,"total_live_cache_bytes"),"gold_logprob":float(np.mean(logprob)),"kl_mean_gold":float(np.mean(kl)),"evidence_retained":float(np.mean(evidence)),"evidence_retained_fp16":evidence_fp16_mean,"answer_kinds":kinds,"cases":len(group),
            "mean_packed_bytes":mean_of(group,"packed_bytes"),"mean_budget_bytes":mean_of(group,"budget_bytes")}
        cells={}
        for row in group:
            for key in [f"len{row['target_tokens']}",f"pos_{row['position']}",f"distractors_{row['distractors']}"]:
                if key not in cells:
                    cells[key]=[]
                cells[key].append(int(row["correct"]))
        table[method]["accuracy_by_factor"]={}
        for key in cells:
            table[method]["accuracy_by_factor"][key]=float(np.mean(cells[key]))
    #paired bootstrap over prompts: each compressed cache minus the full cache, and question-aware minus sink-recent
    pairs=[]
    for method in METHODS[1:]:
        pairs.append((method,"full"))
    pairs.append(("question_aware","sink_recent"))
    pairs.append(("hybrid","question_aware"))
    pairs.append(("int4_kivi","int4_tokenwise"))
    bootstrap={}
    for a,b in pairs:
        index_b={}
        for row in by_method[b]:
            index_b[row["id"]]=row
        diffs=[]
        for row in by_method[a]:
            diffs.append(int(row["correct"])-int(index_b[row["id"]]["correct"]))
        bootstrap[f"{a}_minus_{b}"]=bootstrap_mean_ci(diffs,config["bootstrap"],full_config["seed"])
    save_json(out_dir/"c_summary.json",{"table":table,"paired_bootstrap_accuracy":bootstrap})
    print("wrote",out_dir/"c_summary.json")


def mean_of(rows,key):
    values=[]
    for row in rows:
        values.append(row[key])
    return float(np.mean(values))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/task2.yaml")
    parser.add_argument("--step",required=True,choices=["c1","eval","timing","summary"])
    parser.add_argument("--limit",type=int,default=None,help="smoke test only: use a few cases")
    args=parser.parse_args()
    full_config=load_yaml(args.config)
    full_config["limit"]=args.limit
    set_seed(full_config["seed"])
    out_dir=repo_path(full_config["results_dir"])/"part_c"
    out_dir.mkdir(parents=True,exist_ok=True)
    save_json(out_dir/"config_used.json",full_config) #every reported hyperparameter
    if args.step=="c1":
        step_c1(full_config,out_dir)
    elif args.step=="eval":
        step_eval(full_config,out_dir)
    elif args.step=="timing":
        step_timing(full_config,out_dir)
    else:
        step_summary(full_config,out_dir)


if __name__=="__main__":
    main()
