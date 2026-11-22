"""Exact speculative sampling (Leviathan et al. 2023) and plain target-only / student-only sampling, batch 1.

Cache bookkeeping (both models): the cache always holds the KV of seq[:-1] or fewer tokens, so the next
forward feeds whatever tokens are not cached yet. One round:
  draft  : feed the uncached tail, sample x1 from q1, feed x1, sample x2 ... up to x_gamma   (gamma draft forwards)
  target : feed the uncached tail + x1..x_gamma in ONE forward -> p1..p_gamma and p_{gamma+1}
  verify : accept x_i with prob min(1, p_i(x_i)/q_i(x_i)); at the first rejection sample from norm(max(p_i-q_i,0))
           and stop; if all gamma are accepted, sample a bonus token from p_{gamma+1}
  rollback: crop both caches to old length + accepted count (rejected drafts and the new token are not cached)
"""
from __future__ import annotations

import time

import torch

from common.utils import sync


def probabilities(logits,temperature):
    return torch.softmax(logits.float()/temperature,dim=-1)


def verify_block(p_rows,q_rows,drafted,generator,buggy=False):
    """p_rows (g+1, V) target probabilities, q_rows (g, V) draft probabilities, drafted = g draft tokens.
    Returns (accepted tokens, the extra token, whether all were accepted).
    buggy=True is the negative control: on rejection it samples from p itself instead of the residual,
    which over-produces tokens that p and q both like, so the output law is wrong."""
    device=p_rows.device
    accepted=[]
    for i in range(len(drafted)):
        x=drafted[i]
        ratio=p_rows[i,x]/q_rows[i,x] #q(x)>0 because x was sampled from q
        u=torch.rand(1,generator=generator,device=device)
        if u<torch.clamp(ratio,max=1.0):
            accepted.append(x)
        else:
            if buggy==True:
                residual=p_rows[i].clone()
            else:
                residual=torch.clamp(p_rows[i]-q_rows[i],min=0.0)
                if float(residual.sum())<=0.0:
                    residual=p_rows[i].clone() #only when p==q exactly, then a rejection cannot really happen
            residual=residual/residual.sum()
            extra=int(torch.multinomial(residual,1,generator=generator))
            return accepted,extra,False
    extra=int(torch.multinomial(p_rows[len(drafted)],1,generator=generator)) #bonus token
    return accepted,extra,True


class CachedModel:
    """A model plus its kv cache and how many tokens of the sequence the cache holds."""

    def __init__(self,model):
        self.model=model
        self.cache=None
        self.cached=0
        self.forwards=0

    def feed(self,tokens,device):
        ids=torch.tensor([tokens],device=device)
        output=self.model(input_ids=ids,past_key_values=self.cache,use_cache=True)
        self.cache=output.past_key_values
        self.cached=self.cached+len(tokens)
        self.forwards=self.forwards+1
        return output.logits[0]

    def crop(self,length):
        if self.cache is not None and self.cached>length:
            self.cache.crop(length) #drop KV of rejected tokens
            self.cached=length


@torch.no_grad()
def speculative_generate(target,draft,prompt,gamma,max_new,eos_list,generator,temperature=1.0,ignore_eos=False,buggy=False,record=False):
    device=next(target.parameters()).device
    t=CachedModel(target)
    d=CachedModel(draft)
    seq=list(prompt)
    prompt_length=len(seq)
    sync()
    start=time.perf_counter()
    if len(seq)>1:
        t.feed(seq[:-1],device) #prefill all but the last prompt token
        d.feed(seq[:-1],device)
    sync()
    #per-round time split (manual step 6): draft proposals, target verification forward, accept/residual sampling, cache rollback
    stats={"rounds":0,"proposed":0,"examined":0,"accepted":0,"accepted_per_round":[],"proposed_per_round":[],
        "prefill_time":time.perf_counter()-start,"draft_time":0.0,"verify_time":0.0,"accept_time":0.0,"rollback_time":0.0}
    rows=[] #(position in seq, target probabilities used there), only when record=True
    finished=False
    while finished==False:
        generated=len(seq)-prompt_length
        g=min(gamma,max_new-generated-1) #never draft past the cap (variable proposal length)
        if g<0:
            g=0
        sync()
        t0=time.perf_counter()
        drafted=[]
        q_rows=[]
        for i in range(g):
            if i==0:
                logits=d.feed(seq[d.cached:],device)
            else:
                logits=d.feed([drafted[-1]],device)
            q=probabilities(logits[-1],temperature)
            x=int(torch.multinomial(q,1,generator=generator))
            drafted.append(x)
            q_rows.append(q)
        sync()
        t1=time.perf_counter()
        logits=t.feed(seq[t.cached:]+drafted,device)
        p_rows=probabilities(logits[-(g+1):],temperature) #rows for x1..x_g and the bonus
        sync()
        t2=time.perf_counter()
        if g>0:
            q_stack=torch.stack(q_rows)
        else:
            q_stack=None
        accepted,extra,all_accepted=verify_block(p_rows,q_stack,drafted,generator,buggy)
        sync()
        t3=time.perf_counter()
        stats["draft_time"]=stats["draft_time"]+(t1-t0)
        stats["verify_time"]=stats["verify_time"]+(t2-t1)
        stats["accept_time"]=stats["accept_time"]+(t3-t2)
        stats["rounds"]=stats["rounds"]+1
        stats["proposed"]=stats["proposed"]+g
        if all_accepted==True:
            stats["examined"]=stats["examined"]+g #every proposal was checked
        else:
            stats["examined"]=stats["examined"]+len(accepted)+1 #checking stops at the first rejection
        stats["accepted"]=stats["accepted"]+len(accepted)
        stats["accepted_per_round"].append(len(accepted))
        stats["proposed_per_round"].append(g)
        old_length=len(seq)
        new_tokens=accepted+[extra]
        if record==True:
            for i in range(len(new_tokens)):
                rows.append((old_length+i,p_rows[i].detach().cpu()))
        t.crop(old_length+len(accepted)) #keep prefix + accepted, the extra token is fed next round
        d.crop(old_length+len(accepted))
        sync()
        stats["rollback_time"]=stats["rollback_time"]+(time.perf_counter()-t3)
        for token in new_tokens:
            seq.append(token)
            if ignore_eos==False and token in eos_list:
                finished=True
                break
            if len(seq)-prompt_length>=max_new:
                finished=True
                break
    stats["target_forwards"]=t.forwards
    stats["draft_forwards"]=d.forwards
    stats["total_time"]=time.perf_counter()-start
    stats["host_overhead_time"]=stats["total_time"]-stats["prefill_time"]-stats["draft_time"]-stats["verify_time"]-stats["accept_time"]-stats["rollback_time"] #token bookkeeping, python loop
    response=seq[prompt_length:]
    if record==True:
        return response,stats,rows,t
    return response,stats


@torch.no_grad()
def plain_generate(model,prompt,max_new,eos_list,generator,temperature=1.0,ignore_eos=False):
    """Ancestral sampling from one model with its cache (target-only baseline or student-only)."""
    device=next(model.parameters()).device
    m=CachedModel(model)
    seq=list(prompt)
    prompt_length=len(seq)
    sync()
    start=time.perf_counter()
    logits=m.feed(seq,device)
    sync()
    prefill_time=time.perf_counter()-start
    while True:
        p=probabilities(logits[-1],temperature)
        token=int(torch.multinomial(p,1,generator=generator))
        seq.append(token)
        if ignore_eos==False and token in eos_list:
            break
        if len(seq)-prompt_length>=max_new:
            break
        logits=m.feed([token],device)
    sync()
    total=time.perf_counter()-start
    return seq[prompt_length:],{"target_forwards":m.forwards,"prefill_time":prefill_time,"decode_time":total-prefill_time,"total_time":total}


def make_generator(device,seed):
    generator=torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator
