"""Part A: dense causal attention, tiled online-softmax attention and PyTorch SDPA on the same random Q, K, V.

python -m task2_attention.exact_attention
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

from common.utils import load_yaml, save_json, repo_path, set_seed, time_call, summarize_times, reset_peak, peak_mb, peak_reserved_mb, get_device, package_versions


def dense_attention(q,k,v):
    """A1: explicit S = QK^T / sqrt(d), causal mask, stable softmax (subtract the row max), then P V.
    Works in the input dtype, so the fp16 run really does its math in fp16 (the softmax is done in fp32 and cast back)."""
    d=q.shape[-1]
    length=q.shape[-2]
    scores=(q@k.transpose(-2,-1))/math.sqrt(d) #(B,H,L,L), the full score matrix is materialized
    future=torch.ones(length,length,dtype=torch.bool,device=q.device).triu(1) #True above the diagonal
    scores=scores.masked_fill(future,float("-inf"))
    row_max=scores.amax(dim=-1,keepdim=True)
    weights=torch.exp((scores-row_max).float()) #stable: largest entry becomes exp(0)=1
    weights=weights/weights.sum(dim=-1,keepdim=True)
    return weights.to(v.dtype)@v


def tiled_attention(q,k,v,tile):
    """A2: exact blockwise forward without the L x L score matrix (FlashAttention-style online softmax).
    For each query tile we keep, per query row, the running max m, the running denominator l and the
    unnormalized output acc, all in fp32. A new key tile with scores s updates them:
        m_new = max(m, rowmax(s)),  l = e^{m-m_new} l + sum_j e^{s_j-m_new},  acc = e^{m-m_new} acc + e^{s-m_new} V_tile
    and the output is acc / l. Tiles fully in the future are skipped, the diagonal tile gets the causal mask.
    This is plain pytorch with autograd, so backward stores every tile's intermediates: it is NOT a FlashAttention kernel."""
    batch,heads,length,d=q.shape
    scale=1.0/math.sqrt(d)
    outputs=[]
    for q_start in range(0,length,tile):
        q_end=min(q_start+tile,length) #last tile can be partial
        q_tile=q[:,:,q_start:q_end].float()
        rows=q_end-q_start
        m=torch.full((batch,heads,rows,1),float("-inf"),device=q.device)
        l=torch.zeros((batch,heads,rows,1),device=q.device)
        acc=torch.zeros((batch,heads,rows,d),device=q.device)
        for k_start in range(0,q_end,tile): #keys after the last query of this tile are never needed
            k_end=min(k_start+tile,length)
            k_tile=k[:,:,k_start:k_end].float()
            v_tile=v[:,:,k_start:k_end].float()
            s=(q_tile@k_tile.transpose(-2,-1))*scale
            if k_end>q_start:
                query_index=torch.arange(q_start,q_end,device=q.device).unsqueeze(1)
                key_index=torch.arange(k_start,k_end,device=q.device).unsqueeze(0)
                s=s.masked_fill(key_index>query_index,float("-inf")) #causal mask only matters on/near the diagonal
            m_new=torch.maximum(m,s.amax(dim=-1,keepdim=True))
            correction=torch.exp(m-m_new) #rescales the old sums to the new max (exp(-inf)=0 on the first tile)
            p=torch.exp(s-m_new)
            l=correction*l+p.sum(dim=-1,keepdim=True)
            acc=correction*acc+p@v_tile
            m=m_new
        outputs.append((acc/l).to(q.dtype))
    return torch.cat(outputs,dim=2)


def sdpa_attention(q,k,v):
    return F.scaled_dot_product_attention(q,k,v,is_causal=True)


def make_qkv(config,length,dtype,device,seed,requires_grad):
    generator=torch.Generator(device="cpu").manual_seed(seed+length)
    shape=(config["batch"],config["heads"],length,config["head_dim"])
    tensors=[]
    for i in range(3):
        x=torch.randn(shape,generator=generator,dtype=torch.float32).to(device=device,dtype=dtype)
        x.requires_grad_(requires_grad)
        tensors.append(x)
    return tensors


def max_error(a,b):
    return float((a.float()-b.float()).abs().max())


def compare_to_reference(config,length,dtype,device,seed,tile):
    """Outputs and gradients of all three implementations against an fp32 dense reference on the same inputs."""
    q,k,v=make_qkv(config,length,dtype,device,seed,True)
    q32=q.detach().float().requires_grad_(True)
    k32=k.detach().float().requires_grad_(True)
    v32=v.detach().float().requires_grad_(True)
    reference=dense_attention(q32,k32,v32)
    upstream=torch.randn(reference.shape,generator=torch.Generator(device="cpu").manual_seed(seed+1),dtype=torch.float32).to(device)
    reference.backward(upstream)
    out={}
    for name in ["dense","tiled","sdpa"]:
        for x in [q,k,v]:
            x.grad=None
        if name=="dense":
            y=dense_attention(q,k,v)
        elif name=="tiled":
            y=tiled_attention(q,k,v,tile)
        else:
            y=sdpa_attention(q,k,v)
        y.backward(upstream.to(dtype))
        out[name]={"output_max_abs_error":max_error(y,reference),"dq_max_abs_error":max_error(q.grad,q32.grad),"dk_max_abs_error":max_error(k.grad,k32.grad),"dv_max_abs_error":max_error(v.grad,v32.grad),"finite":bool(torch.isfinite(y).all())}
    return out


def causality_check(config,device,seed,tile):
    """Changing future keys/values must not change earlier outputs."""
    length=300 #not a multiple of the tile, so partial tiles are covered too
    q,k,v=make_qkv(config,length,torch.float32,device,seed,False)
    cut=150
    k2=k.clone()
    v2=v.clone()
    k2[:,:,cut:]=torch.randn_like(k2[:,:,cut:])*10
    v2[:,:,cut:]=torch.randn_like(v2[:,:,cut:])*10
    out={}
    for name in ["dense","tiled","sdpa"]:
        if name=="dense":
            a=dense_attention(q,k,v)
            b=dense_attention(q,k2,v2)
        elif name=="tiled":
            a=tiled_attention(q,k,v,tile)
            b=tiled_attention(q,k2,v2,tile)
        else:
            a=sdpa_attention(q,k,v)
            b=sdpa_attention(q,k2,v2)
        out[name]=max_error(a[:,:,:cut],b[:,:,:cut]) #should be 0 (or rounding only)
    return out


def edge_cases(config,device,tile):
    """Extreme scores: one huge key logit per row (softmax overflow without the max trick) and the first query,
    which only sees itself. An all-masked row cannot occur with a causal mask (the diagonal is always visible)."""
    shape=(1,1,200,config["head_dim"])
    q=torch.ones(shape,device=device)
    k=torch.zeros(shape,device=device)
    v=torch.randn(shape,device=device)
    k[:,:,5]=1000.0 #score 1000*64/8, exp overflows in fp32 without the max subtraction
    out={}
    for dtype in [torch.float32,torch.float16]:
        qd=q.to(dtype)
        kd=k.to(dtype).clamp(max=60000)
        vd=v.to(dtype)
        reference=dense_attention(q,k,v)
        dense=dense_attention(qd,kd,vd)
        tiled=tiled_attention(qd,kd,vd,tile)
        out[str(dtype)]={"dense_error":max_error(dense,reference),"tiled_error":max_error(tiled,reference),"dense_finite":bool(torch.isfinite(dense).all()),"tiled_finite":bool(torch.isfinite(tiled).all()),"first_row_equals_v0":max_error(tiled[:,:,0],v[:,:,0])}
    return out


def sdpa_backends(q,k,v):
    """Which SDPA backends run on this gpu for these inputs (forcing one at a time)."""
    out={}
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
    except ImportError:
        return {"error":"torch.nn.attention.sdpa_kernel not available"}
    names={"flash":SDPBackend.FLASH_ATTENTION,"efficient":SDPBackend.EFFICIENT_ATTENTION,"math":SDPBackend.MATH}
    for name in names:
        try:
            with sdpa_kernel(names[name]):
                F.scaled_dot_product_attention(q,k,v,is_causal=True)
            out[name]="supported"
        except RuntimeError as error:
            out[name]="not supported: "+str(error).split("\n")[0][:200]
    out["default_selected"]=default_backend(q,k,v)
    return out


def default_backend(q,k,v):
    """The backend SDPA actually dispatches to with no forcing, read from the aten op names in the profiler."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        F.scaled_dot_product_attention(q,k,v,is_causal=True)
    names=[]
    for event in profile.events():
        names.append(event.name)
    joined=" ".join(names)
    if "flash_attention" in joined:
        return "flash"
    elif "efficient_attention" in joined:
        return "efficient"
    elif "cudnn_attention" in joined:
        return "cudnn"
    elif "attention_math" in joined:
        return "math"
    else:
        return "unknown: "+joined[:200]


def tile_sweep(config,device,seed):
    """A2: the tiled output and gradients against the fp32 dense reference for several tile sizes and both dtypes."""
    rows=[]
    for dtype_name in config["dtypes"]:
        dtype=getattr(torch,dtype_name)
        for tile in config["tile_sweep"]:
            errors=compare_to_reference(config,config["tile_sweep_length"],dtype,device,seed,tile)
            rows.append({"dtype":dtype_name,"tile":tile,"length":config["tile_sweep_length"],"tiled":errors["tiled"]})
            del_memory()
    return rows


def benchmark(config,length,dtype,device,seed,tile,name):
    """Forward and forward+backward time with cuda events, and peak allocated and reserved memory. OOM is recorded, not hidden."""
    result={"implementation":name,"length":length,"dtype":str(dtype)}
    if name=="tiled" and length>config["tiled_max_length"]:
        result["status"]="skipped (python tiled loop too slow past tiled_max_length)"
        return result
    try:
        q,k,v=make_qkv(config,length,dtype,device,seed,True)
        if name=="dense":
            function=dense_attention
        elif name=="tiled":
            def function(a,b,c):
                return tiled_attention(a,b,c,tile)
        else:
            function=sdpa_attention

        def forward():
            with torch.no_grad():
                function(q,k,v)

        def forward_backward():
            y=function(q,k,v)
            y.sum().backward()

        reset_peak()
        result["forward"]=summarize_times(time_call(forward,config["warmup"],config["repeats"]))
        result["forward_peak_mb"]=peak_mb()
        result["forward_peak_reserved_mb"]=peak_reserved_mb()
        reset_peak()
        result["forward_backward"]=summarize_times(time_call(forward_backward,config["warmup"],config["repeats"]))
        result["forward_backward_peak_mb"]=peak_mb()
        result["forward_backward_peak_reserved_mb"]=peak_reserved_mb()
        result["status"]="ok"
        if name=="sdpa":
            result["backends"]=sdpa_backends(q.detach(),k.detach(),v.detach())
    except torch.cuda.OutOfMemoryError:
        result["status"]="OOM"
    del_memory()
    return result


def del_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/task2.yaml")
    args=parser.parse_args()
    full_config=load_yaml(args.config)
    config=full_config["attention"]
    seed=full_config["seed"]
    set_seed(seed)
    device=get_device()
    tile=config["tile"]
    out_dir=repo_path(full_config["results_dir"])/"part_a"
    results={"versions":package_versions(),"config":config,"causality":causality_check(config,device,seed,tile),"edge_cases":edge_cases(config,device,tile),"tile_sweep":tile_sweep(config,device,seed),"accuracy":[],"timing":[]}
    for dtype_name in config["dtypes"]:
        dtype=getattr(torch,dtype_name)
        for length in config["lengths"]:
            if length<=config["reference_max_length"]:
                try:
                    entry={"length":length,"dtype":dtype_name,"errors":compare_to_reference(config,length,dtype,device,seed,tile)}
                except torch.cuda.OutOfMemoryError:
                    entry={"length":length,"dtype":dtype_name,"errors":"OOM"}
                del_memory()
                results["accuracy"].append(entry)
                print("accuracy",length,dtype_name,flush=True)
            for name in ["dense","tiled","sdpa"]:
                row=benchmark(config,length,dtype,device,seed,tile,name)
                results["timing"].append(row)
                print("timing",name,length,dtype_name,row["status"],flush=True)
            save_json(out_dir/"attention_results.json",results) #save as we go, a crash keeps earlier rows
    save_json(out_dir/"attention_results.json",results)


if __name__=="__main__":
    main()
