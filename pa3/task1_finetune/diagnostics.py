from __future__ import annotations

import argparse
import glob

import numpy as np
import torch
from lm_eval.tasks import TaskManager, get_task_dict
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.lora import find_target_linears, svd_bases
from common.utils import load_json, load_yaml, repo_path, save_json, write_jsonl


#manual step 6: weight-space and activation diagnostics on down_proj of layers 0, 13, 27
#  rho_k = ||U_k U_k^T dW||_F^2 / ||dW||_F^2     eta = ||dW||_F / ||W0||_F
#  R(z) = ||dW z||_2 / (||W0 z||_2 + 1e-8), z = starting model's down_proj input at the last prompt token
#everything in fp32. oplora uses its projected factors.

CONDITIONS=["full_ft_mid","full_ft","full_ft_l2sp","lora4","lora16","oplora16"]


def full_ft_tensor(folder,key):
    """Read one tensor from a saved full model without loading the rest."""
    for path in sorted(glob.glob(str(folder)+"/*.safetensors")):
        with safe_open(path,framework="pt") as f:
            if key in f.keys():
                return f.get_tensor(key).to(torch.float32)
    raise KeyError(key+" not found in "+str(folder))


def delta_weight(cfg,condition,layer,W0):
    """Delta W for one down_proj. Full ft: W_trained - W0. adapters: (alpha/r) B A (projected factors for oplora)."""
    out_dir=repo_path(cfg["outputs_dir"])
    module_name="model.layers."+str(layer)+".mlp.down_proj"
    if condition=="full_ft_mid":
        return full_ft_tensor(out_dir/"full_ft"/"midpoint",module_name+".weight")-W0
    elif condition in ("full_ft","full_ft_l2sp"):
        return full_ft_tensor(out_dir/condition/"final",module_name+".weight")-W0
    else:
        run_cfg=cfg["runs"][condition]
        if run_cfg["method"]=="oplora":
            factors=load_file(str(out_dir/condition/"final"/"adapter_projected.safetensors"))
        else:
            factors=load_file(str(out_dir/condition/"final"/"adapter_raw.safetensors"))
        A=factors[module_name+".lora_A"].to(torch.float32)
        B=factors[module_name+".lora_B"].to(torch.float32)
        scale=float(run_cfg["alpha"])/float(run_cfg["rank"])
        return scale*(B@A)


def probe_prompts(cfg,tokenizer):
    """The 64 probe prompts in the benchmark formats, built with lm-eval's own task definitions."""
    probe=load_json("data/task1/probe_ids.json")
    tasks=get_task_dict([cfg["eval"]["gsm8k_task"],"hellaswag"],TaskManager())
    gsm_task=tasks[cfg["eval"]["gsm8k_task"]]
    hellaswag_task=tasks["hellaswag"]
    gsm_docs=list(gsm_task.test_docs())
    hellaswag_docs=list(hellaswag_task.validation_docs())
    prompts=[]
    for index in probe["gsm8k_test"]:
        user_text=gsm_task.doc_to_text(gsm_docs[index]) #"Q: ...\nA:"
        messages=[{"role":"user","content":user_text}]
        text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False) #same chat format as the gsm8k eval
        prompts.append({"kind":"math","source":"gsm8k_test","index":index,"text":text})
    for index in probe["hellaswag_validation"]:
        text=hellaswag_task.doc_to_text(hellaswag_docs[index]) #stock completion context, no chat template, no ending
        prompts.append({"kind":"non_math","source":"hellaswag_validation","index":index,"text":text})
    return prompts


def cache_activations(cfg,model,tokenizer,prompts,layers):
    """z = input of down_proj at the last prompt token, for each probe prompt and layer. One prompt at a time, so no padding."""
    cache={}
    for layer in layers:
        cache[layer]=[]
    current={} #filled by the hooks during one forward
    hooks=[]
    for layer in layers:
        module=model.model.layers[layer].mlp.down_proj
        def hook(module,inputs,layer=layer): #layer=layer binds the loop value
            current[layer]=inputs[0][0,-1,:].detach().float().cpu() #batch 0, last position
        hooks.append(module.register_forward_pre_hook(hook))
    device=next(model.parameters()).device
    with torch.no_grad():
        for prompt in prompts:
            ids=tokenizer(prompt["text"],return_tensors="pt",add_special_tokens=False)["input_ids"].to(device)
            model(input_ids=ids)
            for layer in layers:
                cache[layer].append(current[layer])
    for h in hooks:
        h.remove()
    for layer in layers:
        cache[layer]=torch.stack(cache[layer]) #[64, 3072]
    return cache


def oplora_constraint_check(cfg,base_model):
    """Numerical check that A~ V_k = 0 and U_k^T B~ = 0 for every oplora module (separate from benchmark evidence)."""
    out_dir=repo_path(cfg["outputs_dir"])
    run_cfg=cfg["runs"]["oplora16"]
    factors=load_file(str(out_dir/"oplora16"/"final"/"adapter_projected.safetensors"))
    device=next(base_model.parameters()).device
    worst_B=0.0
    worst_A=0.0
    modules=0
    for full_name,parent,child_name,linear in find_target_linears(base_model,cfg["lora"]["targets"]):
        U_k,V_k=svd_bases(linear.weight.to(device),int(run_cfg["protect_rank"]))
        A=factors[full_name+".lora_A"].to(device)
        B=factors[full_name+".lora_B"].to(device)
        ratio_B=float((U_k.T@B).norm().item()/(B.norm().item()+1e-12)) #||U_k^T B~|| / ||B~||
        ratio_A=float((A@V_k).norm().item()/(A.norm().item()+1e-12)) #||A~ V_k|| / ||A~||
        if ratio_B>worst_B:
            worst_B=ratio_B
        if ratio_A>worst_A:
            worst_A=ratio_A
        modules=modules+1
    return {"modules":modules,"max_rel_UkT_Btilde":worst_B,"max_rel_Atilde_Vk":worst_A}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task1.yaml")
    args=ap.parse_args()
    cfg=load_yaml(args.config)
    layers=list(cfg["diagnostics"]["layers"]) #0, 13, 27
    k=int(cfg["diagnostics"]["protect_rank"]) #128, same bases as oplora
    results_dir=repo_path(cfg["results_dir"])/"diagnostics"
    device="cpu"
    if torch.cuda.is_available():
        device="cuda"

    tokenizer=AutoTokenizer.from_pretrained(cfg["base_model"])
    base=AutoModelForCausalLM.from_pretrained(cfg["base_model"],torch_dtype=torch.float32).to(device) #fp32 for the diagnostics
    base.eval()

    W0={}
    U={}
    for layer in layers:
        W0[layer]=base.model.layers[layer].mlp.down_proj.weight.detach().float().cpu() #[1024, 3072]
        U_k,V_k=svd_bases(W0[layer],k)
        U[layer]=U_k #[1024, 128]

    prompts=probe_prompts(cfg,tokenizer)
    cache=cache_activations(cfg,base,tokenizer,prompts,layers)

    weight_rows=[]
    activation_rows=[]
    for condition in CONDITIONS:
        for layer in layers:
            dW=delta_weight(cfg,condition,layer,W0[layer]) #[1024, 3072] fp32
            dW_norm_sq=float((dW*dW).sum().item())
            projected=U[layer]@(U[layer].T@dW) #U_k U_k^T dW
            rho=None #undefined for a zero update
            if dW_norm_sq>0:
                rho=float((projected*projected).sum().item())/dW_norm_sq
            eta=(dW_norm_sq**0.5)/float(W0[layer].norm().item())
            weight_rows.append({"condition":condition,"layer":layer,"rho_k":rho,"eta":eta,"dW_fro":dW_norm_sq**0.5,"W0_fro":float(W0[layer].norm().item())})
            z=cache[layer] #[64, 3072]
            update_out=(z@dW.T).norm(dim=1) #||dW z||_2 per prompt
            base_out=(z@W0[layer].T).norm(dim=1) #||W0 z||_2 per prompt
            ratio=update_out/(base_out+1e-8)
            for i in range(len(prompts)):
                activation_rows.append({"condition":condition,"layer":layer,"kind":prompts[i]["kind"],"source":prompts[i]["source"],"index":prompts[i]["index"],"R":float(ratio[i].item())})

    #summary of R(z) by condition, layer and prompt kind
    activation_summary=[]
    for condition in CONDITIONS:
        for layer in layers:
            for kind in ["math","non_math"]:
                values=[]
                for row in activation_rows:
                    if row["condition"]==condition and row["layer"]==layer and row["kind"]==kind:
                        values.append(row["R"])
                values=np.asarray(values)
                activation_summary.append({"condition":condition,"layer":layer,"kind":kind,"n":len(values),"mean_R":float(values.mean()),"median_R":float(np.median(values)),"std_R":float(values.std())})

    save_json(results_dir/"weight_space.json",weight_rows)
    write_jsonl(results_dir/"activation_ratios.jsonl",activation_rows)
    save_json(results_dir/"activation_summary.json",activation_summary)
    save_json(results_dir/"oplora_constraints.json",oplora_constraint_check(cfg,base))
    save_json(results_dir/"probe_prompts.json",prompts)
    for row in weight_rows:
        print(row)
    for row in activation_summary:
        print(row)


if __name__=="__main__":
    main()
