from __future__ import annotations

import argparse

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.lora import inject_lora, lora_modules, merge_factors_into
from common.utils import load_json, load_yaml, repo_path, save_json
from datasets import load_dataset


#merges a trained adapter into a fresh backbone copy for evaluation (manual step 5)
#oplora uses its projected factors A~, B~, lora uses the raw ones
#then checks merged vs unmerged logits on a small batch


def adapter_factors(out_dir,method):
    if method=="oplora":
        return load_file(str(out_dir/"final"/"adapter_projected.safetensors")) #A~, B~
    else:
        return load_file(str(out_dir/"final"/"adapter_raw.safetensors")) #A, B


def check_logits(cfg,run_cfg,factors,merged_dir,tokenizer):
    """Unmerged = fp16 backbone + fp32 adapter (as in training, autocast), merged = fp16 merged model."""
    device="cuda"
    probe=load_json("data/task1/probe_ids.json")
    gsm=load_dataset(cfg["data"]["gsm8k"],"main",split="test")
    texts=[]
    for index in probe["gsm8k_test"][:4]: #small batch of 4 probe questions
        messages=[{"role":"user","content":"Q: "+gsm[index]["question"]+"\nA:"}]
        texts.append(tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False))
    tokenizer.padding_side="right"
    batch=tokenizer(texts,return_tensors="pt",padding=True).to(device)
    mask=batch["attention_mask"].bool()

    #unmerged: plain lora layers holding the (projected) factors, no bases needed since A~, B~ already satisfy the constraint
    unmerged=AutoModelForCausalLM.from_pretrained(cfg["base_model"],torch_dtype=torch.float16).to(device)
    inject_lora(unmerged,cfg["lora"]["targets"],int(run_cfg["rank"]),float(run_cfg["alpha"]),0.0,None)
    for name,module in lora_modules(unmerged).items():
        module.lora_A.data.copy_(factors[name+".lora_A"].to(device))
        module.lora_B.data.copy_(factors[name+".lora_B"].to(device))
    unmerged.eval()
    with torch.no_grad():
        with torch.autocast(device_type="cuda",dtype=torch.float16):
            logits_unmerged=unmerged(**batch).logits.float()
    del unmerged
    torch.cuda.empty_cache()

    merged=AutoModelForCausalLM.from_pretrained(str(merged_dir),torch_dtype=torch.float16).to(device)
    merged.eval()
    with torch.no_grad():
        logits_merged=merged(**batch).logits.float()
    del merged
    torch.cuda.empty_cache()

    difference=(logits_unmerged-logits_merged).abs()[mask] #only real tokens
    same_top1=(logits_unmerged.argmax(-1)==logits_merged.argmax(-1))[mask]
    return {
        "prompts":4,
        "tokens":int(mask.sum().item()),
        "max_abs_logit_diff":float(difference.max().item()),
        "mean_abs_logit_diff":float(difference.mean().item()),
        "top1_agreement":float(same_top1.float().mean().item()),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task1.yaml")
    ap.add_argument("--run",required=True,choices=["lora4","lora16","oplora16"])
    args=ap.parse_args()
    cfg=load_yaml(args.config)
    run_cfg=cfg["runs"][args.run]
    out_dir=repo_path(cfg["outputs_dir"])/args.run
    merged_dir=out_dir/"merged"
    tokenizer=AutoTokenizer.from_pretrained(cfg["base_model"])

    factors=adapter_factors(out_dir,run_cfg["method"])
    model=AutoModelForCausalLM.from_pretrained(cfg["base_model"],torch_dtype=torch.float32) #fresh backbone copy, merged in fp32
    scale=float(run_cfg["alpha"])/float(run_cfg["rank"]) #alpha/r
    merged_names=merge_factors_into(model,factors,scale)
    model=model.to(torch.float16) #evaluated in fp16
    model.save_pretrained(str(merged_dir),safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    del model

    check=check_logits(cfg,run_cfg,factors,merged_dir,tokenizer)
    check["run"]=args.run
    check["merged_modules"]=len(merged_names)
    if run_cfg["method"]=="oplora":
        check["factors"]="projected"
    else:
        check["factors"]="raw"
    save_json(repo_path(cfg["results_dir"])/"merge_check"/(args.run+".json"),check)
    print(check)


if __name__=="__main__":
    main()
