"""Part B: MQAR (query-position classification) with small 2-block transformers trained from scratch.

python -m task2_attention.mqar --arch dense --seed 6304   (every dense run in the config, one seed)
python -m task2_attention.mqar --arch all                 (every run in configs/task2.yaml mqar.runs x every seed)
python -m task2_attention.mqar --inference_sweep          (forward-only timing at T=256..8192, random weights and inputs)
python -m task2_attention.mqar --summary                  (tables over seeds from the saved results)

Each run trains ONE architecture at ONE K and ONE regime (the manual's B1/B2/B3 list), on its own stream.

Architectures (only the attention operator changes, everything else is shared):
  dense    causal softmax over all earlier positions
  local    causal sliding window, the preceding 64 positions (incl. itself)
  linear   causal ELU+1 linear attention (Katharopoulos et al.), running state S=sum phi(k)v^T, z=sum phi(k)
  dilated  B2 mixed local/dilated: heads 0,1 local W=64, heads 2,3 see i, i-4, ..., i-252 (64 positions, dilation 4)
  hybrid   B3 intervention: per head, a gated mix of a W=16 local softmax and an ELU+1 linear memory, both stored
           in fp16 so the per-layer state bytes do not exceed the fp32 ELU+1 linear comparator
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.utils import load_yaml, save_json, load_json, repo_path, data_path, read_jsonl, set_seed, get_device, sync, reset_peak, peak_mb, time_call, summarize_times


def generator_module():
    """The staff generator lives in the data repo's scripts folder."""
    folder=str(data_path("scripts"))
    if folder not in sys.path:
        sys.path.insert(0,folder)
    import mqar_generator
    return mqar_generator


#---------------- data ----------------

def records_to_arrays(records,length,max_k):
    tokens=np.zeros((len(records),length),dtype=np.int64)
    positions=np.full((len(records),max_k),-1,dtype=np.int64) #-1 = padding (K<16)
    answers=np.full((len(records),max_k),-1,dtype=np.int64)
    sources=np.full((len(records),max_k),-1,dtype=np.int64)
    ks=np.zeros(len(records),dtype=np.int64)
    regimes=[]
    for i in range(len(records)):
        record=records[i]
        tokens[i]=record["tokens"]
        k=record["K"]
        ks[i]=k
        positions[i,:k]=record["query_positions"]
        answers[i,:k]=record["answer_ids"]
        sources[i,:k]=record["source_positions"]
        regimes.append(record["regime"])
    return {"tokens":tokens,"positions":positions,"answers":answers,"sources":sources,"K":ks,"regime":np.array(regimes)}


def training_stream(config,k,regime,cache_path):
    """updates*batch training examples make_example('train', K, regime, index) for one (K, regime).
    Made once and cached; every architecture and seed with this (K, regime) reads the same stream (seeds only change the init)."""
    if cache_path.exists():
        return dict(np.load(cache_path,allow_pickle=False))
    generator=generator_module()
    count=config["updates"]*config["batch"]
    records=[]
    for i in range(count):
        records.append(generator.make_example("train",k,regime,i))
    arrays=records_to_arrays(records,config["sequence_length"],16)
    cache_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache_path,tokens=arrays["tokens"],positions=arrays["positions"],answers=arrays["answers"],sources=arrays["sources"],K=arrays["K"])
    return arrays


def labels_for(positions,answers,length):
    """-100 everywhere except the K query positions, which hold the value id (loss only on supervised positions)."""
    labels=torch.full((positions.shape[0],length),-100,dtype=torch.long)
    for row in range(positions.shape[0]):
        for j in range(positions.shape[1]):
            if positions[row,j]>=0:
                labels[row,positions[row,j]]=int(answers[row,j])
    return labels


#---------------- model ----------------

def allowed_mask(kind,length,heads,window,dilation):
    """(heads, L, L) bool, True where query i may attend key j. Used by the softmax operators and the reach analysis."""
    i=torch.arange(length).unsqueeze(1)
    j=torch.arange(length).unsqueeze(0)
    causal=j<=i
    local=causal&(i-j<window) #j in [i-window+1, i]
    masks=[]
    for h in range(heads):
        if kind=="dense" or kind=="linear":
            masks.append(causal)
        elif kind=="local" or kind=="hybrid":
            masks.append(local)
        elif kind=="dilated":
            if h<heads//2:
                masks.append(local)
            else:
                gap=i-j
                masks.append(causal&(gap%dilation==0)&(gap<=dilation*(window-1))) #i, i-4, ..., i-4*63
        else:
            raise ValueError(kind)
    return torch.stack(masks)


def elu_feature(x):
    return F.elu(x)+1.0 #positive feature map phi


def linear_attention(q,k,v,state_fp16=False):
    """Causal linear attention in its recurrent (prefix-sum) form: out_i = phi(q_i)S_i / (phi(q_i).z_i),
    S_i = sum_{j<=i} phi(k_j) v_j^T, z_i = sum_{j<=i} phi(k_j). The state is d x d + d per head, independent of i.
    With state_fp16 (hybrid) the stored state is the running mean S_i/i, z_i/i instead of the sum: the output is a
    ratio so dividing top and bottom by i changes nothing, but a running sum grows with i and passed fp16's 65504
    in the first hybrid runs (inf then NaN), while a mean stays at the size of one phi(k)v^T term."""
    phi_q=elu_feature(q)
    phi_k=elu_feature(k)
    state=torch.cumsum(phi_k.unsqueeze(-1)*v.unsqueeze(-2),dim=2) #(B,H,T,d,d)
    normalizer=torch.cumsum(phi_k,dim=2) #(B,H,T,d)
    if state_fp16==True:
        count=torch.arange(1,q.shape[2]+1,device=q.device,dtype=state.dtype).view(1,1,-1,1) #tokens summed so far at each position
        state=state/count.unsqueeze(-1) #running mean, same 2 bytes per number as the sum
        normalizer=normalizer/count #same division as the state so numerator/denominator is unchanged
        state=state.half().float() #hybrid only: the stored state is rounded to fp16 (b3 precision adjustment)
        normalizer=normalizer.half().float()
    numerator=(phi_q.unsqueeze(-2)@state).squeeze(-2) #(B,H,T,d)
    denominator=(phi_q*normalizer).sum(dim=-1,keepdim=True)
    return numerator/(denominator+1e-6)


def masked_softmax_attention(q,k,v,mask):
    """Correctness implementation of local/dilated attention: a dense score matrix with a mask.
    Its runtime is NOT the runtime of an efficient sparse kernel."""
    scores=(q@k.transpose(-2,-1))/math.sqrt(q.shape[-1])
    scores=scores.masked_fill(mask.logical_not(),float("-inf"))
    return torch.softmax(scores,dim=-1)@v


class Attention(nn.Module):
    def __init__(self,config,kind):
        super().__init__()
        self.kind=kind
        self.state_fp16=(kind=="hybrid" and config["hybrid_state_dtype"]=="float16")
        self.heads=config["heads"]
        self.head_dim=config["hidden"]//config["heads"] #128/4=32
        self.qkv=nn.Linear(config["hidden"],3*config["hidden"])
        self.out=nn.Linear(config["hidden"],config["hidden"])
        if kind=="hybrid":
            window=config["hybrid_window"]
        else:
            window=config["window"]
        self.register_buffer("mask",allowed_mask(kind,config["sequence_length"],config["heads"],window,config["dilation"]),persistent=False)
        self.gate=None #only hybrid gets one, added after init so shared weights start identical

    def add_gate(self,hidden):
        self.gate=nn.Linear(hidden,self.heads) #one mixing weight per head per position

    def forward(self,x):
        batch,length,hidden=x.shape
        q,k,v=self.qkv(x).split(hidden,dim=-1)
        q=q.view(batch,length,self.heads,self.head_dim).transpose(1,2) #(B,H,T,d)
        k=k.view(batch,length,self.heads,self.head_dim).transpose(1,2)
        v=v.view(batch,length,self.heads,self.head_dim).transpose(1,2)
        mask=self.mask[:,:length,:length]
        if self.kind=="linear":
            y=linear_attention(q,k,v)
        elif self.kind=="hybrid":
            if self.state_fp16==True:
                k=k.half().float() #the local buffer stores keys and values in fp16
                v=v.half().float()
            local=masked_softmax_attention(q,k,v,mask)
            memory=linear_attention(q,k,v,self.state_fp16)
            g=torch.sigmoid(self.gate(x)).transpose(1,2).unsqueeze(-1) #(B,H,T,1)
            y=g*local+(1-g)*memory
        else:
            y=masked_softmax_attention(q,k,v,mask)
        y=y.transpose(1,2).reshape(batch,length,hidden)
        return self.out(y)


class Block(nn.Module):
    def __init__(self,config,kind):
        super().__init__()
        self.norm1=nn.LayerNorm(config["hidden"])
        self.attention=Attention(config,kind)
        self.norm2=nn.LayerNorm(config["hidden"])
        self.mlp=nn.Sequential(nn.Linear(config["hidden"],config["mlp_ratio"]*config["hidden"]),nn.GELU(),nn.Linear(config["mlp_ratio"]*config["hidden"],config["hidden"]))

    def forward(self,x):
        x=x+self.attention(self.norm1(x)) #pre-norm
        x=x+self.mlp(self.norm2(x))
        return x


def sinusoidal(length,hidden):
    """Vaswani et al. positional table: sin on even channels, cos on odd channels."""
    position=torch.arange(length).float().unsqueeze(1)
    frequency=torch.exp(torch.arange(0,hidden,2).float()*(-math.log(10000.0)/hidden))
    table=torch.zeros(length,hidden)
    table[:,0::2]=torch.sin(position*frequency)
    table[:,1::2]=torch.cos(position*frequency)
    return table


class MQARModel(nn.Module):
    def __init__(self,config,kind):
        super().__init__()
        self.token=nn.Embedding(config["vocab"],config["hidden"])
        self.register_buffer("position",sinusoidal(config["sequence_length"],config["hidden"]),persistent=False) #fixed sinusoidal positions (manual)
        self.blocks=nn.ModuleList()
        for i in range(config["layers"]):
            self.blocks.append(Block(config,kind))
        self.norm=nn.LayerNorm(config["hidden"])
        self.head=nn.Linear(config["hidden"],config["vocab"]) #logits over the 256 ids at every position

    def forward(self,tokens):
        x=self.token(tokens)+self.position[:tokens.shape[1]].unsqueeze(0)
        for block in self.blocks:
            x=block(x)
        return self.head(self.norm(x))


def build_model(config,kind,seed):
    torch.manual_seed(seed)
    model=MQARModel(config,kind) #same seed -> same shared weights for every operator
    if kind=="hybrid":
        torch.manual_seed(seed+1000)
        for block in model.blocks:
            block.attention.add_gate(config["hidden"])
    return model


def count_parameters(model):
    total=0
    for parameter in model.parameters():
        if parameter.requires_grad:
            total=total+parameter.numel()
    return total


#---------------- analysis of what each operator can reach ----------------

def reach_matrix(config,kind):
    """reach[i,j]=True if position i can get information from position j through the two layers
    (union over heads of each layer's allowed positions, applied twice; the residual is the diagonal)."""
    if kind=="hybrid":
        mask=allowed_mask("dense",config["sequence_length"],config["heads"],config["window"],config["dilation"]) #linear memory sees everything
    else:
        mask=allowed_mask(kind,config["sequence_length"],config["heads"],config["window"],config["dilation"])
    one_layer=mask.any(dim=0).float()
    reach=one_layer
    for layer in range(config["layers"]-1):
        reach=(one_layer@reach)>0
        reach=reach.float()
    return reach>0


def state_floats(config,kind):
    """Persistent inference state per layer after a long prefix (what must be kept to answer later queries).
    Bytes use fp32 for every model except the hybrid, whose buffer and linear state are stored in hybrid_state_dtype."""
    heads=config["heads"]
    d=config["hidden"]//config["heads"]
    length=config["sequence_length"]
    if kind=="dense":
        tokens_per_head=[length]*heads #full kv cache at the end of the sequence
        recurrent=0
    elif kind=="local":
        tokens_per_head=[config["window"]]*heads
        recurrent=0
    elif kind=="dilated":
        span=config["dilation"]*(config["window"]-1)+1 #253 positions, later queries need other residues
        tokens_per_head=[config["window"]]*(heads//2)+[span]*(heads-heads//2)
        recurrent=0
    elif kind=="linear":
        tokens_per_head=[0]*heads
        recurrent=heads*(d*d+d)
    else:
        tokens_per_head=[config["hybrid_window"]]*heads
        recurrent=heads*(d*d+d)
    cached=0
    for count in tokens_per_head:
        cached=cached+count*2*d #one key and one value vector per cached token
    bytes_per_float=4
    if kind=="hybrid" and config["hybrid_state_dtype"]=="float16":
        bytes_per_float=2
    return {"cached_kv_floats":cached,"recurrent_state_floats":recurrent,"total_floats":cached+recurrent,"bytes_per_float":bytes_per_float,"total_bytes":bytes_per_float*(cached+recurrent)}


def check_b3_state(config):
    """Manual B3: the hybrid's total state bytes per layer (local buffer included) must match the linear comparator."""
    hybrid=state_floats(config,"hybrid")["total_bytes"]
    comparator=state_floats(config,config["b3_comparator"]["arch"])["total_bytes"]
    if hybrid>comparator or hybrid<0.95*comparator:
        raise RuntimeError(f"hybrid state {hybrid} bytes vs comparator {comparator} bytes per layer: not matched")
    return {"hybrid_state_bytes":hybrid,"comparator_state_bytes":comparator,"ratio":hybrid/comparator}


def attention_pairs(config,kind):
    """Theoretical (query, key) pairs scored per head per layer at T=sequence_length (softmax operators only)."""
    if kind=="linear":
        return 0 #no pairwise scores, a running state instead
    window=config["window"]
    if kind=="hybrid":
        window=config["hybrid_window"]
    mask=allowed_mask(kind,config["sequence_length"],config["heads"],window,config["dilation"])
    return float(mask.float().sum()/config["heads"])


def candidate_keys(config,kind):
    """Mean number of key positions a query can score directly (per head, averaged over positions)."""
    window=config["window"]
    if kind=="hybrid":
        window=config["hybrid_window"]
    mask=allowed_mask(kind,config["sequence_length"],config["heads"],window,config["dilation"])
    return float(mask.float().sum(dim=-1).mean())


#---------------- training and evaluation ----------------

def lr_lambda(config):
    def schedule(step):
        if step<config["warmup"]:
            return (step+1)/config["warmup"] #linear warmup
        else:
            progress=(step-config["warmup"])/max(1,config["updates"]-config["warmup"])
            return 1.0-(1.0-config["final_lr_fraction"])*progress #linear decay to 15% of peak
    return schedule


@torch.no_grad()
def evaluate(model,arrays,config,device,reach=None):
    """Accuracy over supervised query positions only; argmax over the full 256-id vocabulary."""
    model.eval()
    rows=[] #one entry per query: K, regime, distance, correct, reachable
    total_loss=0.0
    total_queries=0
    count=arrays["tokens"].shape[0]
    for start in range(0,count,config["eval_batch"]):
        end=min(start+config["eval_batch"],count)
        tokens=torch.tensor(arrays["tokens"][start:end],device=device)
        logits=model(tokens)
        positions=arrays["positions"][start:end]
        answers=arrays["answers"][start:end]
        labels=labels_for(positions,answers,tokens.shape[1]).to(device)
        total_loss=total_loss+float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),ignore_index=-100,reduction="sum"))
        predictions=logits.argmax(dim=-1).cpu().numpy()
        for row in range(end-start):
            example=start+row
            for j in range(positions.shape[1]):
                position=positions[row,j]
                if position<0:
                    continue
                source=arrays["sources"][example,j]
                correct=int(predictions[row,position]==answers[row,j])
                reachable=-1
                if reach is not None:
                    reachable=int(reach[position,source+1]) #the value token sits right after its key
                rows.append((int(arrays["K"][example]),str(arrays["regime"][example]),int(position-source),correct,reachable))
                total_queries=total_queries+1
    model.train()
    return rows,total_loss/max(total_queries,1)


DISTANCE_BINS=[(0,32),(32,64),(64,128),(128,256)] #manual's finer distance bins


def distance_bin(distance):
    for low,high in DISTANCE_BINS:
        if distance>=low and distance<high:
            return f"d{low}-{high}"
    return "d256+"


def accuracy_table(rows):
    """Accuracy per K x regime and per K x distance bin, pooled over supervised query positions."""
    table={}
    for k,regime,distance,correct,reachable in rows:
        for key in [f"K{k}_{regime}",f"K{k}_{distance_bin(distance)}"]:
            if key not in table:
                table[key]={"correct":0,"queries":0,"reachable":0}
            table[key]["correct"]=table[key]["correct"]+correct
            table[key]["queries"]=table[key]["queries"]+1
            if reachable==1:
                table[key]["reachable"]=table[key]["reachable"]+1
    for key in table:
        table[key]["accuracy"]=table[key]["correct"]/table[key]["queries"]
        table[key]["reachable_fraction"]=table[key]["reachable"]/table[key]["queries"]
    return table


def run_name(run,seed):
    return f"{run['arch']}_K{run['k']}_{run['regime']}_seed{seed}"


def set_masks(model,config,kind):
    """Swap every layer's attention mask (used for the test-time-only W=64 control on a trained dense model)."""
    for block in model.blocks:
        block.attention.mask=allowed_mask(kind,config["sequence_length"],config["heads"],config["window"],config["dilation"]).to(block.attention.mask.device)


def train_one(config,run,seed,out_dir,stream,dev,test,device):
    kind=run["arch"]
    set_seed(seed)
    model=build_model(config,kind,seed).to(device)
    parameters=count_parameters(model)
    baseline_parameters=count_parameters(build_model(config,"dense",seed))
    if abs(parameters/baseline_parameters-1.0)>0.05:
        raise RuntimeError(f"{kind} has {parameters} parameters, more than 5% away from the {baseline_parameters} baseline (manual B3 rule)")
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["lr"],weight_decay=config["weight_decay"])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda(config))
    reach=reach_matrix(config,kind).numpy()
    run_dir=out_dir/run_name(run,seed)
    run_dir.mkdir(parents=True,exist_ok=True)
    log=[]
    checkpoints=[]
    step_times=[]
    reset_peak()
    model.train()
    for step in range(config["updates"]):
        start=step*config["batch"]
        end=start+config["batch"]
        tokens=torch.tensor(stream["tokens"][start:end],device=device)
        labels=labels_for(stream["positions"][start:end],stream["answers"][start:end],tokens.shape[1]).to(device)
        sync()
        t0=time.perf_counter()
        logits=model(tokens)
        loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),ignore_index=-100) #mean over the query positions
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),config["clip"]))
        optimizer.step()
        scheduler.step()
        sync()
        step_times.append(time.perf_counter()-t0)
        if step%50==0:
            log.append({"step":step,"loss":float(loss),"grad_norm":grad_norm,"lr":scheduler.get_last_lr()[0]})
        if (step+1) in config["checkpoints"]:
            rows,dev_loss=evaluate(model,dev,config,device,reach)
            table=accuracy_table(rows)
            correct=0
            for row in rows:
                correct=correct+row[3]
            checkpoints.append({"update":step+1,"dev_loss":dev_loss,"dev_accuracy":correct/len(rows),"dev_table":table})
            torch.save(model.state_dict(),run_dir/f"model_update{step+1}.pt") #keep every checkpoint
            print(run_name(run,seed),"update",step+1,"dev acc",round(correct/len(rows),4),flush=True)
    peak=peak_mb()
    rows,test_loss=evaluate(model,test,config,device,reach) #held-out test only once, at the end
    np.savez_compressed(run_dir/"test_queries.npz",K=np.array(column(rows,0)),regime=np.array(column(rows,1)),distance=np.array(column(rows,2)),correct=np.array(column(rows,3)),reachable=np.array(column(rows,4)))
    tokens=torch.tensor(test["tokens"][:config["eval_batch"]],device=device)

    def inference():
        with torch.no_grad():
            model(tokens)

    inference_time=summarize_times(time_call(inference,3,10))
    mask_control=None
    control=config["mask_control"]
    if kind==control["arch"] and run["k"]==control["k"] and run["regime"]==control["regime"]:
        set_masks(model,config,control["against"]) #trained dense, W=64 window imposed at test time only
        control_rows,control_loss=evaluate(model,test,config,device,reach_matrix(config,control["against"]).numpy())
        set_masks(model,config,kind)
        mask_control={"test_time_mask":control["against"],"test_loss":control_loss,"test_table":accuracy_table(control_rows)}
    b3=None
    if kind=="hybrid":
        b3={"state_match":check_b3_state(config),"comparator":config["b3_comparator"],"rule":config["b3_rule"],"prediction":config["b3_prediction"],"window":config["hybrid_window"],"state_dtype":config["hybrid_state_dtype"]}
    result={"arch":kind,"train_k":run["k"],"train_regime":run["regime"],"seed":seed,"parameters":parameters,"dense_parameters":baseline_parameters,"parameter_ratio_vs_dense":parameters/baseline_parameters,
        "test_loss":test_loss,"test_table":accuracy_table(rows),"checkpoints":checkpoints,"train_log":log,
        "mean_step_ms":1000*float(np.mean(step_times[100:])),"train_peak_mb":peak,"inference_batch":config["eval_batch"],"inference":inference_time,
        "state":state_floats(config,kind),"candidate_keys_per_query":candidate_keys(config,kind),"attention_pairs_per_head_T256":attention_pairs(config,kind),
        "test_time_mask_control":mask_control,"b3":b3,"chance_label_space":1/64,
        "max_reach_positions":int(max_reach(reach)),"note":"local/dilated/hybrid local part use a masked dense implementation; runtimes are not sparse-kernel runtimes"}
    save_json(run_dir/"result.json",result)
    return result


def column(rows,index):
    values=[]
    for row in rows:
        values.append(row[index])
    return values


def max_reach(reach):
    """Largest i-j that is reachable (126 for two W=64 local layers)."""
    best=0
    length=reach.shape[0]
    for gap in range(length):
        i=np.arange(gap,length)
        if reach[i,i-gap].any():
            best=gap
    return best


def summary(config,out_dir):
    """Mean and std over seeds of the test accuracy per run (arch, training K, training regime) and test cell,
    plus the dense test-time-mask control and the chance baselines (1/64 label space, 1/K values in the example)."""
    table={}
    controls={}
    for folder in sorted(out_dir.iterdir()):
        if (folder/"result.json").exists()==False:
            continue
        result=load_json(folder/"result.json")
        label=f"{result['arch']}_K{result['train_k']}_{result['train_regime']}"
        if label not in table:
            table[label]={}
        for key in result["test_table"]:
            if key not in table[label]:
                table[label][key]=[]
            table[label][key].append(result["test_table"][key]["accuracy"])
        if result.get("test_time_mask_control") is not None:
            control_label=label+"_testmask_"+result["test_time_mask_control"]["test_time_mask"]
            if control_label not in controls:
                controls[control_label]={}
            for key in result["test_time_mask_control"]["test_table"]:
                if key not in controls[control_label]:
                    controls[control_label][key]=[]
                controls[control_label][key].append(result["test_time_mask_control"]["test_table"][key]["accuracy"])
    out={"runs":{},"test_time_mask_controls":{},"chance":{"label_space":1/64,"one_over_K":{"K4":0.25,"K8":0.125,"K16":0.0625}}}
    for source,target in [(table,out["runs"]),(controls,out["test_time_mask_controls"])]:
        for label in source:
            target[label]={}
            for key in source[label]:
                values=np.array(source[label][key])
                target[label][key]={"mean":float(values.mean()),"std":float(values.std()),"seeds":len(values)}
    save_json(out_dir/"summary.json",out)
    print("wrote",out_dir/"summary.json")


def inference_sweep(full_config,out_dir,device):
    """Forward-only timing of one attention operator at T=256..8192 with random weights and fixed random inputs,
    same B, H, d for every operator, so it measures kernels and not learnability. Local/dilated/hybrid build a dense
    (H,T,T) boolean mask and linear attention builds a (B,H,T,d,d) prefix tensor; both are recorded as such."""
    config=dict(full_config["mqar"])
    rows=[]
    kinds=["dense","local","dilated","linear","hybrid"]
    for length in config["inference_lengths"]:
        for kind in kinds:
            local=dict(config)
            local["sequence_length"]=length
            row={"arch":kind,"T":length,"batch":config["inference_batch"],"heads":config["heads"],"head_dim":config["hidden"]//config["heads"],
                "materializes":"dense boolean (H,T,T) mask and (B,H,T,T) scores"}
            if kind=="linear":
                row["materializes"]="(B,H,T,d,d) prefix-sum state tensor"
            if kind=="hybrid":
                row["materializes"]="both of the above"
            try:
                torch.manual_seed(full_config["seed"])
                operator=Attention(local,kind)
                if kind=="hybrid":
                    operator.add_gate(config["hidden"])
                operator=operator.to(device).eval()
                x=torch.randn(config["inference_batch"],length,config["hidden"],generator=torch.Generator().manual_seed(full_config["seed"])).to(device)

                def forward():
                    with torch.no_grad():
                        operator(x)

                reset_peak()
                row["forward"]=summarize_times(time_call(forward,3,10))
                row["peak_mb"]=peak_mb()
                row["status"]="ok"
                del operator
            except torch.cuda.OutOfMemoryError:
                row["status"]="OOM"
            except RuntimeError as error:
                if "out of memory" in str(error):
                    row["status"]="OOM"
                else:
                    raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            rows.append(row)
            print("sweep",kind,length,row["status"],flush=True)
    save_json(out_dir/"inference_sweep.json",{"rows":rows})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/task2.yaml")
    parser.add_argument("--arch",default=None)
    parser.add_argument("--seed",type=int,default=None)
    parser.add_argument("--summary",action="store_true")
    parser.add_argument("--inference_sweep",action="store_true")
    args=parser.parse_args()
    full_config=load_yaml(args.config)
    config=full_config["mqar"]
    out_dir=repo_path(full_config["results_dir"])/"part_b"
    if args.summary:
        summary(config,out_dir)
        return
    device=get_device()
    out_dir.mkdir(parents=True,exist_ok=True)
    save_json(out_dir/"config_used.json",full_config) #every reported hyperparameter, b3 prediction included, before any training
    if args.inference_sweep:
        inference_sweep(full_config,out_dir,device)
        return
    check_b3_state(config) #fail early if the hybrid is not state-matched
    #staff invariant checks on the frozen test split before any training (manual)
    generator=generator_module()
    test_records=read_jsonl(data_path("data/mqar/mqar_test.jsonl.gz"))
    dev_records=read_jsonl(data_path("data/mqar/mqar_development.jsonl.gz"))
    for record in test_records:
        generator.validate(record)
    for record in dev_records:
        generator.validate(record)
    test=records_to_arrays(test_records,config["sequence_length"],16)
    dev=records_to_arrays(dev_records,config["sequence_length"],16)
    runs=[]
    for run in config["runs"]:
        if args.arch is None or args.arch=="all" or args.arch==run["arch"]:
            runs.append(run)
    if args.seed is None:
        seeds=config["seeds"]
    else:
        seeds=[args.seed]
    for run in runs:
        stream=training_stream(config,run["k"],run["regime"],out_dir/f"train_stream_K{run['k']}_{run['regime']}.npz")
        for seed in seeds:
            if (out_dir/run_name(run,seed)/"result.json").exists():
                print("done already",run_name(run,seed))
                continue
            train_one(config,run,seed,out_dir,stream,dev,test,device)
    summary(config,out_dir)


if __name__=="__main__":
    main()
