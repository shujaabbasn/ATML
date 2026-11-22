from __future__ import annotations

import gzip
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml


#pa4 folder, every relative path in the configs is relative to this
REPO_ROOT=Path(__file__).resolve().parents[1]


def repo_path(path):
    """Turn a config path into an absolute path inside pa4/ (absolute paths are kept)."""
    p=Path(path)
    if p.is_absolute():
        return p
    else:
        return REPO_ROOT/p


def data_path(relative):
    """Path inside the cloned staff data repo (atml-pa4-datasets), PA4_DATA overrides the default folder."""
    root=os.environ.get("PA4_DATA","data_repo") #setup clones the staff repo into pa4/data_repo
    return repo_path(root)/relative


def load_yaml(path):
    return yaml.safe_load(repo_path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    """Reads .jsonl and .jsonl.gz the same way (staff files are both)."""
    p=Path(path)
    if p.is_absolute()==False:
        p=repo_path(path)
    rows=[] #one dict per line
    if str(p).endswith(".gz"):
        handle=gzip.open(p,"rt",encoding="utf-8")
    else:
        handle=p.open(encoding="utf-8")
    for line in handle:
        if line.strip()!="":
            rows.append(json.loads(line))
    handle.close()
    return rows


def write_jsonl(path,rows):
    p=repo_path(path)
    p.parent.mkdir(parents=True,exist_ok=True) #make the folder if needed
    if str(p).endswith(".gz"):
        handle=gzip.open(p,"wt",encoding="utf-8")
    else:
        handle=p.open("w",encoding="utf-8")
    for row in rows:
        handle.write(json.dumps(row,ensure_ascii=False)+"\n")
    handle.close()


def append_jsonl(path,record):
    p=repo_path(path)
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("a",encoding="utf-8") as f:
        f.write(json.dumps(record,ensure_ascii=False)+"\n")


def save_json(path,obj):
    p=repo_path(path)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding="utf-8")


def load_json(path):
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu") #only for the small local tests


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_call(function,warmup,repeats):
    """Runs function() warmup times untimed, then repeats times timed. Returns a list of milliseconds.
    On the gpu we use cuda events (manual: warmed-up, cuda-synchronized timing), on cpu perf_counter."""
    for i in range(warmup):
        function()
    sync()
    times=[] #one entry per timed repeat
    for i in range(repeats):
        if torch.cuda.is_available():
            start=torch.cuda.Event(enable_timing=True)
            end=torch.cuda.Event(enable_timing=True)
            start.record()
            function()
            end.record()
            torch.cuda.synchronize() #wait so elapsed_time is valid
            times.append(start.elapsed_time(end))
        else:
            t0=time.perf_counter()
            function()
            times.append((time.perf_counter()-t0)*1000.0)
    return times


def summarize_times(times):
    array=np.array(times,dtype=np.float64)
    return {"mean_ms":float(array.mean()),"std_ms":float(array.std()),"median_ms":float(np.median(array)),"repeats":len(times),"all_ms":array.tolist()}


def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated()/2**20
    else:
        return float("nan") #no allocator stats on cpu


def peak_reserved_mb():
    """Peak memory held by the caching allocator (allocated + cached free blocks)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_reserved()/2**20
    else:
        return float("nan")


def allocated_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated()/2**20
    else:
        return float("nan")


def bootstrap_mean_ci(values,repeats,seed):
    """Percentile 95% interval of the mean. For a paired comparison pass the per-example differences."""
    values=np.asarray(values,dtype=np.float64)
    rng=np.random.default_rng(seed)
    n=len(values)
    means=np.empty(repeats) #one bootstrap mean per resample
    for r in range(repeats):
        index=rng.integers(0,n,size=n) #resample examples with replacement
        means[r]=values[index].mean()
    return {"mean":float(values.mean()),"low":float(np.percentile(means,2.5)),"high":float(np.percentile(means,97.5)),"n":int(n),"repeats":int(repeats)}


def bootstrap_ppl_diff(nll_a,nll_b,window_lengths,repeats,seed):
    """Paired bootstrap of the perplexity difference exp(pooled nll_a)-exp(pooled nll_b). The unit is a whole fixed
    evaluation window (manual): resample windows, then pool the token NLL of the chosen windows before exponentiating."""
    nll_a=np.asarray(nll_a,dtype=np.float64)
    nll_b=np.asarray(nll_b,dtype=np.float64)
    lengths=np.asarray(window_lengths,dtype=np.int64)
    ends=np.cumsum(lengths)
    starts=ends-lengths
    sums_a=np.empty(len(lengths)) #nll sum per window
    sums_b=np.empty(len(lengths))
    for w in range(len(lengths)):
        sums_a[w]=nll_a[starts[w]:ends[w]].sum()
        sums_b[w]=nll_b[starts[w]:ends[w]].sum()
    rng=np.random.default_rng(seed)
    n=len(lengths)
    diffs=np.empty(repeats)
    for r in range(repeats):
        index=rng.integers(0,n,size=n) #same windows for both models = paired
        tokens=lengths[index].sum()
        diffs[r]=np.exp(sums_a[index].sum()/tokens)-np.exp(sums_b[index].sum()/tokens)
    point=np.exp(sums_a.sum()/lengths.sum())-np.exp(sums_b.sum()/lengths.sum())
    return {"diff":float(point),"low":float(np.percentile(diffs,2.5)),"high":float(np.percentile(diffs,97.5)),"windows":int(n),"repeats":int(repeats)}


def package_versions():
    import transformers
    out={"torch":torch.__version__,"transformers":transformers.__version__,"cuda":str(torch.version.cuda)}
    if torch.cuda.is_available():
        out["gpu"]=torch.cuda.get_device_name(0)
    return out
