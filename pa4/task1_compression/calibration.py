"""Calibration statistics for one transformer block at a time.
The same 128 x 512 C4 calibration sequences feed Wanda, AWQ and the sensitivity probes (manual)."""
from __future__ import annotations

import torch


#which linear input each eligible projection reads (q,k,v read the same normed hidden state, so do gate and up)
INPUT_OF={"q_proj":"attn_in","k_proj":"attn_in","v_proj":"attn_in","o_proj":"o_in","gate_proj":"mlp_in","up_proj":"mlp_in","down_proj":"down_in"}


class StopForward(Exception):
    """Raised by the down_proj hook so the forward pass ends right after the block we need."""
    pass


def sample_index(total_tokens,sample_tokens,seed):
    """Fixed random token rows (same rows for every method that uses the sample)."""
    generator=torch.Generator().manual_seed(seed)
    order=torch.randperm(total_tokens,generator=generator)
    chosen=order[:sample_tokens]
    chosen,_=torch.sort(chosen)
    return chosen


@torch.no_grad()
def block_stats(model,calibration,block_index,batch_size,sample_tokens,seed):
    """Runs the calibration set through the model as it is right now (earlier blocks may already be compressed,
    which is what sequential Wanda/AWQ do) and stops after block `block_index`.
    For each of the 4 inputs returns: sq_sum (sum x^2 per channel), abs_sum (sum |x|), count, sample (token rows)."""
    device=next(model.parameters()).device
    layer=model.model.layers[block_index]
    watched={"attn_in":layer.self_attn.q_proj,"o_in":layer.self_attn.o_proj,"mlp_in":layer.mlp.gate_proj,"down_in":layer.mlp.down_proj}
    sequence_length=len(calibration[0]["token_ids"])
    total_tokens=len(calibration)*sequence_length
    chosen=sample_index(total_tokens,sample_tokens,seed)
    stats={}
    for key in watched:
        stats[key]={"sq_sum":None,"abs_sum":None,"count":0,"sample":[]}
    state={"offset":0} #flat token offset of the current batch

    def make_hook(key):
        def hook(module,args):
            x=args[0].detach().float()
            flat=x.reshape(-1,x.shape[-1]) #(batch*tokens, channels)
            entry=stats[key]
            if entry["sq_sum"] is None:
                entry["sq_sum"]=torch.zeros(flat.shape[1],device=flat.device)
                entry["abs_sum"]=torch.zeros(flat.shape[1],device=flat.device)
            entry["sq_sum"]+=(flat*flat).sum(dim=0)
            entry["abs_sum"]+=flat.abs().sum(dim=0)
            entry["count"]+=flat.shape[0]
            low=state["offset"]
            high=low+flat.shape[0]
            inside=chosen[(chosen>=low)&(chosen<high)]-low #sampled rows that fall in this batch
            entry["sample"].append(flat[inside.to(flat.device)].to(torch.float16))
            if key=="down_in":
                raise StopForward() #nothing after this block is needed
        return hook

    handles=[]
    for key in watched:
        handles.append(watched[key].register_forward_pre_hook(make_hook(key)))
    try:
        for start in range(0,len(calibration),batch_size):
            batch=calibration[start:start+batch_size]
            ids=[]
            for record in batch:
                ids.append(record["token_ids"])
            ids=torch.tensor(ids,device=device)
            try:
                model(input_ids=ids,use_cache=False)
            except StopForward:
                pass #expected, we stopped after the block
            state["offset"]=state["offset"]+ids.numel()
    finally:
        for handle in handles:
            handle.remove() #never leave hooks on the model
    for key in stats:
        stats[key]["sample"]=torch.cat(stats[key]["sample"],dim=0)
        stats[key]["norm"]=torch.sqrt(stats[key]["sq_sum"]) #wanda ||X_j||_2 over all calibration tokens
        stats[key]["mean_abs"]=stats[key]["abs_sum"]/stats[key]["count"] #awq activation magnitude
    return stats
