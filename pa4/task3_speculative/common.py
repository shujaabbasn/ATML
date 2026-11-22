"""Models, prompts and answer checking shared by every task 3 script."""
from __future__ import annotations

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.utils import read_jsonl, data_path, get_device


def load_causal(name):
    device=get_device()
    if device.type=="cuda":
        dtype=torch.float16
    else:
        dtype=torch.float32
    model=AutoModelForCausalLM.from_pretrained(name,dtype=dtype,attn_implementation="sdpa").to(device).eval()
    return model


def load_tokenizer(config):
    """Target and draft must tokenize identically (manual: one tokenizer, one chat template, one EOS rule)."""
    tokenizer=AutoTokenizer.from_pretrained(config["target"])
    draft_tokenizer=AutoTokenizer.from_pretrained(config["draft"])
    for text in ["Example 123","Hello there","A complicated example with 𝜋"]:
        if tokenizer.encode(text)!=draft_tokenizer.encode(text):
            raise RuntimeError("target and draft tokenizers differ")
    return tokenizer


def eos_ids(tokenizer):
    ids=[tokenizer.convert_tokens_to_ids("<|im_end|>"),tokenizer.convert_tokens_to_ids("<|endoftext|>")]
    return ids


def chat_ids(tokenizer,prompt):
    """Same chat template the staff prefix-bank script uses (one user turn, generation prompt added)."""
    return tokenizer.apply_chat_template([{"role":"user","content":prompt}],tokenize=True,add_generation_prompt=True)


def gsm8k_gold(answer_text):
    return answer_text.split("####")[-1].strip().replace(",","")


def train_prompts(config):
    """500 GSM8K train questions + 300 short-form train instructions. No test prompt ever enters training."""
    math=[]
    for record in read_jsonl(data_path(config["data"]["gsm8k_train"])):
        math.append({"id":f"GSM8K-TRAIN-{record['order']:04d}","prompt":record["example"]["question"],"domain":"math","answer":gsm8k_gold(record["example"]["answer"])})
    short=[]
    for record in read_jsonl(data_path(config["data"]["short_train"])):
        short.append({"id":record["id"],"prompt":record["input"],"domain":"short","answer":record["answer"]})
    return math,short


def test_prompts(config):
    """The 120 held-out prompts with the same ids the staff bank uses (GSM8K-0000.., SHORT-...)."""
    out=[]
    for record in read_jsonl(data_path(config["data"]["gsm8k_test"])):
        out.append({"id":f"GSM8K-{record['order']:04d}","prompt":record["example"]["question"],"domain":"math","subset":"math","answer":gsm8k_gold(record["example"]["answer"])})
    for record in read_jsonl(data_path(config["data"]["short_test"])):
        out.append({"id":record["id"],"prompt":record["input"],"domain":"short","subset":record["split"],"family":record["family"],"answer":record["answer"]})
    return out


def last_number(text):
    boxed=re.findall(r"\\boxed\{([^}]*)\}",text)
    if len(boxed)>0:
        candidates=re.findall(r"-?\d[\d,]*\.?\d*",boxed[-1])
    else:
        candidates=re.findall(r"-?\d[\d,]*\.?\d*",text)
    if len(candidates)==0:
        return None
    return candidates[-1].replace(",","").rstrip(".")


def is_correct(item,response):
    """GSM8K: final numeric answer equals the gold number. Short form: exact match after trimming case/punctuation."""
    if item["domain"]=="math":
        predicted=last_number(response)
        if predicted is None:
            return False
        try:
            return abs(float(predicted)-float(item["answer"]))<1e-6
        except ValueError:
            return False
    else:
        cleaned=response.strip().strip(".").strip().lower()
        return cleaned==str(item["answer"]).strip().lower()


def cap_for(item,caps):
    if item["domain"]=="math":
        return caps["cap_math"]
    else:
        return caps["cap_short"]
