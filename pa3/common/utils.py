from __future__ import annotations

import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml


#pa3 folder, every relative path in the configs is relative to this
REPO_ROOT=Path(__file__).resolve().parents[1]


def repo_path(path):
    """Turn a config path into an absolute path inside pa3/ (absolute paths are kept)."""
    p=Path(path)
    if p.is_absolute():
        return p
    else:
        return REPO_ROOT/p


def load_yaml(path):
    return yaml.safe_load(repo_path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    rows=[] #one dict per line
    with repo_path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip()!="":
                rows.append(json.loads(line))
    return rows


def write_jsonl(path,rows):
    p=repo_path(path)
    p.parent.mkdir(parents=True,exist_ok=True) #make the folder if needed
    with p.open("w",encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False)+"\n")


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


def rng_state():
    """Every rng we use, saved inside recovery checkpoints (manual: preserve rng state for resuming)."""
    state={"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"]=torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def wall_timer():
    start=time.perf_counter()
    return lambda: time.perf_counter()-start #seconds since the timer was made


def peak_vram_gib():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated()/2**30 #GiB
    else:
        return None #cpu run


def folder_size_bytes(path):
    """Total size of a checkpoint folder or file (manual: checkpoint sizes)."""
    p=repo_path(path)
    if p.is_file():
        return p.stat().st_size
    total=0
    for root,dirs,files in os.walk(p):
        for name in files:
            total=total+os.path.getsize(os.path.join(root,name))
    return total


def remove_folder(path):
    p=repo_path(path)
    if p.exists():
        shutil.rmtree(p)


def count_parameters(parameters):
    """Number of scalars, each tensor counted once even if it is tied (model.parameters() already dedups)."""
    total=0
    seen=set() #ids of tensors already counted
    for p in parameters:
        if id(p) not in seen:
            seen.add(id(p))
            total=total+p.numel()
    return total
