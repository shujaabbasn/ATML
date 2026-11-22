"""Q1 round-to-nearest, Q2 AWQ-style activation-aware scaling, Q3 mixed precision.
One quantizer for all three (manual: rounding, scale, zero-point, grouping must agree across Q1/Q2/Q3):
  group = 128 contiguous input columns of one output row
  s = (w_max - w_min) / (2^b - 1),  z = round(-w_min / s)
  q = clip(round(w / s) + z, 0, 2^b - 1),  w_hat = s * (q - z)
The reconstructed w_hat is written back as fp16 (fake quantization: matmuls stay dense fp16)."""
from __future__ import annotations

import math

import torch

from task1_compression.calibration import block_stats, INPUT_OF
from task1_compression.model_utils import attach_input_scale, quant_bytes_matrix


def fake_quantize(weight,bits,group_size):
    """Returns the dequantized fp32 matrix. Columns are padded to a multiple of group_size by repeating the
    last column (keeps that group's min/max), the padding is cut off again after reconstruction.
    Qwen2.5-1.5B widths (1536, 8960) are multiples of 128, so the padding branch is never used there."""
    w=weight.float()
    rows,columns=w.shape
    groups=math.ceil(columns/group_size)
    padded=groups*group_size
    if padded!=columns:
        extra=w[:,-1:].expand(rows,padded-columns)
        w=torch.cat([w,extra],dim=1)
    w=w.reshape(rows,groups,group_size)
    w_min=w.amin(dim=2,keepdim=True)
    w_max=w.amax(dim=2,keepdim=True)
    levels=2**bits-1
    scale=((w_max-w_min)/levels).clamp(min=1e-8) #a constant group would give scale 0
    zero=torch.round(-w_min/scale)
    q=torch.clamp(torch.round(w/scale)+zero,0,levels) #integer codes 0..2^b-1
    w_hat=scale*(q-zero)
    w_hat=w_hat.reshape(rows,padded)[:,:columns]
    return w_hat


@torch.no_grad()
def rtn_quantize(entries,bits,group_size):
    """Q1: plain round-to-nearest of every eligible matrix, no calibration."""
    records=[]
    for full,block_index,short,module in entries:
        weight=module.weight.data
        w_hat=fake_quantize(weight,bits,group_size)
        weight.copy_(w_hat.to(weight.dtype))
        records.append({"name":full,"block":block_index,"proj":short,"bits":bits,"bytes":quant_bytes_matrix(weight.shape[0],weight.shape[1],bits,group_size)})
    return records


def block_groups(layer):
    """The 4 scaling groups of a qwen2 block and where 1/s can be folded:
    attn_in: q,k,v after input_layernorm -> divide the norm weight by s
    o_in:    o_proj after v_proj -> only foldable if v_proj rows line up 1:1 with o_proj columns (not with GQA)
    mlp_in:  gate,up after post_attention_layernorm -> divide the norm weight
    down_in: down_proj after act(gate)*up -> divide up_proj rows (the product is linear in up)"""
    attention=layer.self_attn
    mlp=layer.mlp
    groups=[]
    groups.append({"key":"attn_in","linears":[("q_proj",attention.q_proj),("k_proj",attention.k_proj),("v_proj",attention.v_proj)],"fold":("norm",layer.input_layernorm)})
    if attention.v_proj.weight.shape[0]==attention.o_proj.weight.shape[1]:
        groups.append({"key":"o_in","linears":[("o_proj",attention.o_proj)],"fold":("rows",attention.v_proj)})
    else:
        groups.append({"key":"o_in","linears":[("o_proj",attention.o_proj)],"fold":("explicit",None)}) #qwen 1.5b: 2 kv heads vs 12 heads
    groups.append({"key":"mlp_in","linears":[("gate_proj",mlp.gate_proj),("up_proj",mlp.up_proj)],"fold":("norm",layer.post_attention_layernorm)})
    groups.append({"key":"down_in","linears":[("down_proj",mlp.down_proj)],"fold":("rows",mlp.up_proj)})
    return groups


def output_error(sample,weight,effective):
    """Squared error and squared reference norm of the linear outputs on the calibration token sample (no bias,
    the bias cancels in the difference)."""
    x=sample.float()
    reference=x@weight.float().t()
    approx=x@effective.t()
    diff=approx-reference
    return float((diff*diff).sum()),float((reference*reference).sum())


def scale_candidates(mean_abs,alpha):
    """AWQ search space: s = mean|x|^alpha, normalized so sqrt(max*min)=1 (as in the AutoAWQ code)."""
    s=mean_abs.clamp(min=1e-4).pow(alpha)
    s=s/torch.sqrt(s.max()*s.min())
    s=s.clamp(min=1e-4)
    return s


@torch.no_grad()
def search_group_scale(group,sample,mean_abs,bits_of,group_size,grid):
    """Grid search over alpha in {0, 1/grid, ..., 1}; loss = summed output error of the group's linears when
    W*s is quantized and the input is divided by s. alpha=0 gives s=1, i.e. plain RTN."""
    best=None
    log=[]
    for step in range(grid+1):
        alpha=step/grid
        s=scale_candidates(mean_abs,alpha)
        total=0.0
        reference_total=0.0
        for short,module in group["linears"]:
            weight=module.weight.data
            scaled=weight.float()*s.unsqueeze(0) #scale input columns
            effective=fake_quantize(scaled,bits_of(short,module),group_size)/s.unsqueeze(0)
            error,reference=output_error(sample,weight,effective)
            total=total+error
            reference_total=reference_total+reference
        relative=total/max(reference_total,1e-12)
        log.append({"alpha":alpha,"relative_error":relative})
        if best is None or relative<best[0]:
            best=(relative,alpha,s)
    return best,log


@torch.no_grad()
def apply_group_scale(group,s,block_name):
    """Multiply the linears' input columns by s and put 1/s into the previous op (or keep it as an fp16 input scale).
    In full precision this changes nothing; check_scaling_identity() measures that numerically."""
    for short,module in group["linears"]:
        weight=module.weight.data
        weight.copy_((weight.float()*s.unsqueeze(0)).to(weight.dtype))
    kind,previous=group["fold"]
    if kind=="norm":
        norm_weight=previous.weight.data
        norm_weight.copy_((norm_weight.float()/s).to(norm_weight.dtype))
        return "folded into "+block_name+" norm"
    elif kind=="rows":
        previous_weight=previous.weight.data
        previous_weight.copy_((previous_weight.float()/s.unsqueeze(1)).to(previous_weight.dtype)) #divide output rows
        if previous.bias is not None:
            previous.bias.data.copy_((previous.bias.data.float()/s).to(previous.bias.dtype))
        return "folded into previous linear rows"
    else:
        attach_input_scale(group["linears"][0][1],s) #fp16 1/s kept and counted in bytes
        return "explicit fp16 input scale"


@torch.no_grad()
def check_scaling_identity(group,s,sample):
    """Before quantization: compare (x/s)(W s)^T with x W^T on the calibration sample, using the fp16 weights
    the model would really hold after scaling. Returns the worst relative error over the group's linears."""
    worst=0.0
    for short,module in group["linears"]:
        weight=module.weight.data
        scaled=(weight.float()*s.unsqueeze(0)).to(weight.dtype).float() #round trip through fp16 like the real model
        x=sample.float()
        reference=x@weight.float().t()
        rescaled=(x/s.unsqueeze(0))@scaled.t()
        relative=float((rescaled-reference).norm()/reference.norm().clamp(min=1e-12))
        if relative>worst:
            worst=relative
    return worst


@torch.no_grad()
def awq_quantize(model,entries,calibration,bits_of,group_size,grid,calib_config,seed,quantize=True,probe_bits=None):
    """Q2 (and Q3 with a per-linear bits_of). For each block in order:
      1. collect calibration statistics with earlier blocks already processed (sequential like AutoAWQ)
      2. per scaling group grid-search alpha, record the error at alpha and at alpha=0 (RTN)
      3. check the full-precision rescaling identity, then apply the scale (fold or explicit)
      4. quantize the block's linears (skipped when quantize=False, used for the end-to-end identity check)
    probe_bits: list of widths; if given we also record each linear's relative output error at every width
    (the Q3 sensitivity measure). Reproduced from AWQ: activation-magnitude scale search with the normalized
    s=mean|x|^alpha form and folding into previous ops. Not reproduced: weight clipping search, and the loss is the
    per-linear output error instead of the whole attention/mlp module output."""
    block_log=[]
    sensitivity=[]
    records=[]
    for block_index in range(len(model.model.layers)):
        layer=model.model.layers[block_index]
        stats=block_stats(model,calibration,block_index,calib_config["batch"],calib_config["sample_tokens"],seed)
        for group in block_groups(layer):
            sample=stats[group["key"]]["sample"]
            mean_abs=stats[group["key"]]["mean_abs"]
            best,search_log=search_group_scale(group,sample,mean_abs,bits_of,group_size,grid)
            relative,alpha,s=best
            identity_error=check_scaling_identity(group,s,sample)
            if probe_bits is not None:
                for short,module in group["linears"]:
                    weight=module.weight.data
                    entry={"name":f"{block_index}.{short}","block":block_index,"proj":short,"rows":weight.shape[0],"columns":weight.shape[1],"errors":{}}
                    for bits in probe_bits:
                        effective=fake_quantize(weight.float()*s.unsqueeze(0),bits,group_size)/s.unsqueeze(0)
                        error,reference=output_error(sample,weight,effective)
                        entry["errors"][str(bits)]=error/max(reference,1e-12) #relative output error
                    sensitivity.append(entry)
            where=apply_group_scale(group,s,f"block {block_index}")
            block_log.append({"block":block_index,"group":group["key"],"alpha":alpha,"relative_error_awq":relative,"relative_error_rtn":search_log[0]["relative_error"],"identity_relative_error":identity_error,"fold":where,"search":search_log})
        if quantize==True:
            for full,entry_block,short,module in entries:
                if entry_block!=block_index:
                    continue
                weight=module.weight.data
                bits=bits_of(short,module)
                weight.copy_(fake_quantize(weight,bits,group_size).to(weight.dtype))
                records.append({"name":full,"block":block_index,"proj":short,"bits":bits,"bytes":quant_bytes_matrix(weight.shape[0],weight.shape[1],bits,group_size)})
    return records,block_log,sensitivity


def greedy_bit_allocation(sensitivity,widths,budget_bytes,group_size):
    """Q3 allocation rule: every linear starts at the smallest width; repeatedly take the single upgrade
    (to that linear's next width) with the largest drop in relative output error per extra byte that still
    fits in the budget. Stops when no upgrade fits. Unit = one projection matrix."""
    widths=sorted(widths)
    level={} #index into widths per linear
    used=0
    for entry in sensitivity:
        level[entry["name"]]=0
        used=used+quant_bytes_matrix(entry["rows"],entry["columns"],widths[0],group_size)
    if used>budget_bytes:
        raise RuntimeError("even the smallest width is over the budget")
    while True:
        best=None
        for entry in sensitivity:
            now=level[entry["name"]]
            if now+1>=len(widths):
                continue #already at the widest
            low_bits=widths[now]
            high_bits=widths[now+1]
            extra=quant_bytes_matrix(entry["rows"],entry["columns"],high_bits,group_size)-quant_bytes_matrix(entry["rows"],entry["columns"],low_bits,group_size)
            if used+extra>budget_bytes:
                continue
            gain=entry["errors"][str(low_bits)]-entry["errors"][str(high_bits)]
            ratio=gain/extra
            if best is None or ratio>best[0]:
                best=(ratio,entry["name"],extra)
        if best is None:
            break
        level[best[1]]=level[best[1]]+1
        used=used+best[2]
    bits={}
    for name in level:
        bits[name]=widths[level[name]]
    return bits,used
