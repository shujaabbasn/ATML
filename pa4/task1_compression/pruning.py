"""P1 magnitude, P2 Wanda, P3 OWL-style nonuniform Wanda, and the 2:4 structured diagnostic.
All masks are per output row: inside each row of W (one output neuron) we drop the lowest-scored inputs."""
from __future__ import annotations

import math

import numpy as np
import torch

from task1_compression.calibration import block_stats, INPUT_OF


def mask_lowest(score,counts):
    """True where a weight is pruned. counts[r] = how many weights row r loses.
    rank of each weight inside its row (0 = lowest score), prune every rank below that row's count."""
    order=torch.argsort(score,dim=1) #lowest first
    ranks=torch.empty_like(order)
    positions=torch.arange(score.shape[1],device=score.device).unsqueeze(0).expand_as(order)
    ranks.scatter_(1,order,positions) #ranks[r,order[r,i]]=i
    return ranks<counts.unsqueeze(1)


def uniform_counts(rows,columns,sparsity,device):
    """Same count in every row. For qwen 1.5b, s*columns is an integer for 0.25/0.5/0.75 so the sparsity is exact."""
    count=int(round(sparsity*columns))
    return torch.full((rows,),count,dtype=torch.long,device=device)


def apply_mask(module,mask):
    weight=module.weight.data
    weight[mask]=0 #zero-masked weights, still a dense fp16 matrix


def matrix_record(full,block_index,short,module):
    weight=module.weight.data
    zeros=int((weight==0).sum())
    return {"name":full,"block":block_index,"proj":short,"rows":weight.shape[0],"columns":weight.shape[1],"numel":weight.numel(),"zeros":zeros,"nonzero":weight.numel()-zeros,"sparsity":zeros/weight.numel()}


@torch.no_grad()
def magnitude_prune(entries,sparsity):
    """P1: score = |W_ij|, per-output-row comparison groups."""
    records=[]
    for full,block_index,short,module in entries:
        weight=module.weight.data
        counts=uniform_counts(weight.shape[0],weight.shape[1],sparsity,weight.device)
        mask=mask_lowest(weight.abs().float(),counts)
        apply_mask(module,mask)
        records.append(matrix_record(full,block_index,short,module))
    return records


@torch.no_grad()
def wanda_prune(model,entries,calibration,counts_for,calib_config,seed,score_log=None):
    """P2/P3: score = |W_ij| * ||X_j||_2 (Wanda). Blocks are done in order and the calibration statistics of block l
    are taken after blocks 0..l-1 were pruned (sequential, like the original Wanda code).
    counts_for(full_name, rows, columns, device) gives the per-row prune counts, so P2 and P3 share this exact code."""
    records=[]
    block_count=len(model.model.layers)
    for block_index in range(block_count):
        stats=block_stats(model,calibration,block_index,calib_config["batch"],1,seed) #only need norms here
        for full,entry_block,short,module in entries:
            if entry_block!=block_index:
                continue
            weight=module.weight.data
            norm=stats[INPUT_OF[short]]["norm"]
            score=weight.abs().float()*norm.unsqueeze(0) #broadcast ||X_j|| over rows
            counts=counts_for(full,weight.shape[0],weight.shape[1],weight.device)
            mask=mask_lowest(score,counts)
            if score_log is not None:
                magnitude_mask=mask_lowest(weight.abs().float(),counts)
                both=(mask&magnitude_mask).sum().item()
                either=(mask|magnitude_mask).sum().item()
                quantiles=torch.quantile(score.flatten()[::97].float(),torch.tensor([0.1,0.5,0.9,0.99],device=score.device)) #strided subsample, quantile has a size limit
                score_log.append({"name":full,"block":block_index,"proj":short,"mask_jaccard_vs_magnitude":both/max(either,1),"score_q10":float(quantiles[0]),"score_q50":float(quantiles[1]),"score_q90":float(quantiles[2]),"score_q99":float(quantiles[3]),"input_norm_max_over_median":float(norm.max()/norm.median())})
            apply_mask(module,mask)
            records.append(matrix_record(full,block_index,short,module))
    return records


@torch.no_grad()
def dense_input_norms(model,calibration,calib_config,seed):
    """One pass per block on the unpruned model, only the Wanda norms. Used by the OWL sensitivity measure."""
    norms={}
    for block_index in range(len(model.model.layers)):
        stats=block_stats(model,calibration,block_index,calib_config["batch"],1,seed)
        norms[block_index]={}
        for key in stats:
            norms[block_index][key]=stats[key]["norm"].clone()
    return norms


@torch.no_grad()
def owl_outlier_ratio(entries,norms,threshold):
    """OWL sensitivity per block: share of Wanda scores larger than threshold * mean score of their matrix.
    Blocks with more outlier scores are treated as more sensitive and get pruned less."""
    outliers={}
    totals={}
    for full,block_index,short,module in entries:
        weight=module.weight.data
        score=weight.abs().float()*norms[block_index][INPUT_OF[short]].unsqueeze(0)
        big=(score>threshold*score.mean()).sum().item()
        if block_index not in outliers:
            outliers[block_index]=0
            totals[block_index]=0
        outliers[block_index]=outliers[block_index]+big
        totals[block_index]=totals[block_index]+score.numel()
    ratio={}
    for block_index in outliers:
        ratio[block_index]=outliers[block_index]/totals[block_index]
    return ratio


def owl_allocation(ratio,block_sizes,target,lam,low,high):
    """Allocation rule (OWL paper): map outlier ratios linearly onto [target-lam, target+lam] sparsity,
    more outliers -> lower sparsity, then clip to the manual bounds [low, high] and shift back so the
    size-weighted mean equals the target. Unit = one transformer block (all 7 projections share s_l)."""
    blocks=sorted(ratio.keys())
    values=np.zeros(len(blocks),dtype=np.float64)
    sizes=np.zeros(len(blocks),dtype=np.float64)
    for i in range(len(blocks)):
        values[i]=ratio[blocks[i]] #outlier ratio of block i
        sizes[i]=block_sizes[blocks[i]] #eligible weights in block i
    spread=values.max()-values.min()
    if spread>0:
        normalized=(values-values.min())/spread #0..1, 1 = most outliers
    else:
        normalized=np.zeros_like(values)
    sparsity=target+lam-2*lam*normalized #most outliers -> target-lam
    for step in range(100):
        sparsity=np.clip(sparsity,low,high)
        gap=target-(sparsity*sizes).sum()/sizes.sum()
        if abs(gap)<1e-12:
            break
        sparsity=sparsity+gap #shift every block, then clip again
    out={}
    for i in range(len(blocks)):
        out[blocks[i]]=float(sparsity[i])
    return out


def exact_row_counts(entries,unit_sparsity,target):
    """Turns per-block sparsities into integer per-row counts whose total is exactly round(target * N).
    Each row first gets floor(s_l * columns); the leftover weights go one per row to the matrices with
    the largest fractional parts. Manual: realized global sparsity equal by integer weight counting."""
    total=0
    plan=[] #(full, rows, columns, wanted per row as float)
    for full,block_index,short,module in entries:
        rows,columns=module.weight.shape
        total=total+rows*columns
        plan.append((full,rows,columns,unit_sparsity[block_index]*columns))
    goal=int(round(target*total))
    counts={}
    used=0
    fractions=[]
    for full,rows,columns,wanted in plan:
        base=int(math.floor(wanted))
        counts[full]=[base]*rows
        used=used+base*rows
        fractions.append((wanted-base,full,rows))
    left=goal-used
    fractions.sort(reverse=True) #largest fractional part first
    for fraction,full,rows in fractions:
        if left<=0:
            break
        give=min(rows,left)
        for r in range(give):
            counts[full][r]=counts[full][r]+1
        left=left-give
    if left!=0:
        raise RuntimeError("could not reach the exact global prune count")
    return counts,goal,total


@torch.no_grad()
def nm_prune(model,entries,calibration,calib_config,seed,n,m):
    """2:4 diagnostic: in every row, inside each group of m contiguous input weights keep the n highest Wanda scores."""
    records=[]
    for block_index in range(len(model.model.layers)):
        stats=block_stats(model,calibration,block_index,calib_config["batch"],1,seed)
        for full,entry_block,short,module in entries:
            if entry_block!=block_index:
                continue
            weight=module.weight.data
            rows,columns=weight.shape
            score=weight.abs().float()*stats[INPUT_OF[short]]["norm"].unsqueeze(0)
            grouped=score.reshape(rows,columns//m,m) #(rows, groups, m)
            order=torch.argsort(grouped,dim=2) #lowest first inside each group
            drop=torch.zeros_like(grouped,dtype=torch.bool)
            drop.scatter_(2,order[:,:,:m-n],True) #the m-n lowest in each group
            apply_mask(module,drop.reshape(rows,columns))
            records.append(matrix_record(full,block_index,short,module))
    return records
