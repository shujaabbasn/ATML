from __future__ import annotations

import argparse
from pathlib import Path

from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import set_seed
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    trainable_parameters,
    value_parameter_groups,
)

import random

import numpy as np
import torch

from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, wall_timer
from common.metrics import masked_mean, sample_entropy, sampled_kl
from common.models import reference_mode, token_values
from task2_ppo.ppo import affected_token_fraction, compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards, value_mse_loss


def disable_dropout(model):
    """Set every dropout to 0 so pi_old and pi_theta are the same function before the first step."""
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):
            module.p=0.0 #lora dropout would otherwise make the ratio != 1 even with identical weights


def cast_input_to_float(module,args):
    """Forward pre-hook: feed the fp32 critic head an fp32 input (the backbone hands it fp16)."""
    return (args[0].float(),) #same tensor, just fp32


def critic_trainable_to_fp32(value_model):
    """Keep the critic's trainable parameters in fp32.

    The released critic loads in fp16 and its trainable head stays fp16. AdamW's eps=1e-8 rounds to 0 in fp16,
    so the very first critic step divided 0 by 0 and the critic became NaN (Kaggle run, update 1, epoch 2).
    """
    for name,parameter in value_model.named_parameters():
        if parameter.requires_grad==True:
            if parameter.dtype!=torch.float32:
                parameter.data=parameter.data.float() #same Parameter object, so the optimizer still holds it
    for name,module in value_model.named_modules():
        if isinstance(module,torch.nn.Linear):
            if "score" in name:
                if module.weight.requires_grad==True:
                    module.register_forward_pre_hook(cast_input_to_float) #trainable head copy now expects fp32 input


def ppo_prompt_order(prompt_rows,seed):
    """Fixed shuffled order of the training prompts, identical for every fork."""
    order=list(range(len(prompt_rows))) #0..n-1
    random.Random(seed).shuffle(order) #own rng, so nothing else can change the order
    return order


def response_values(value_model,sequences,attention_mask,prompt_width,steps):
    """V(s_t) for every response token, lined up with the log-prob positions."""
    values=token_values(value_model,sequences,attention_mask) #[batch, full length], one value per position
    values=values[:,prompt_width-1:-1] #position i predicts token i+1, same shift as response_token_logprobs
    return values[:,:steps].float() #[batch, response steps] in fp32 for the gae maths


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard"):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    policy=bundle["policy"] #midpoint lora, trainable
    value_model=bundle["value_model"] #midpoint critic + fresh zero lora + trainable head
    reward_model=bundle["reward_model"] #frozen course reward model (8-bit on gpu)
    reward_tokenizer=bundle["reward_tokenizer"] #base tokenizer, see models.py note
    tokenizer=bundle["tokenizer"] #policy tokenizer, left padding
    prompt_rows=bundle["prompt_rows"] #training prompt pool
    policy_optimizer=bundle["policy_optimizer"] #adamw, lr 3e-6
    value_optimizer=bundle["value_optimizer"] #adamw, lora 1e-4 / head 3e-4
    updates=int(cfg["updates"]) #20 standard, 8 for forks
    prompts_per_update=int(cfg["prompts_per_update"]) #config: 1
    ppo_epochs=int(cfg["ppo_epochs"]) #config: 2 passes over each rollout
    eps=float(cfg["clip_epsilon"]) #0.20 standard
    kl_beta=float(cfg["kl_beta"]) #0.10 standard
    gamma=float(cfg["gamma"]) #1.0
    lam=float(cfg["gae_lambda"]) #0.95
    value_coef=float(cfg["value_coef"]) #0.5
    missing_eos_penalty=float(cfg["missing_eos_penalty"]) #1.0 subtracted when the response never ends
    max_grad_norm=float(cfg["max_grad_norm"]) #1.0
    generation_cfg=cfg["generation"] #temperature 0.7, top_p 0.9, sampling
    device=next(policy.parameters()).device #cuda
    results_dir=repo_path(cfg["results_dir"])/run_name #results/task2_ppo/<run_name>/
    results_dir.mkdir(parents=True,exist_ok=True) #make it
    log_path=results_dir/"train_log.jsonl" #one line per update
    if log_path.exists():
        log_path.unlink() #fresh log for this run

    disable_dropout(policy) #ratio must be exactly 1 at theta=theta_old
    disable_dropout(value_model) #same for old vs new values
    critic_trainable_to_fp32(value_model) #fp16 head + adamw gave NaN after the first critic step, see the function
    order=ppo_prompt_order(prompt_rows,int(cfg["seed"])) #same prompt sequence for standard and every fork

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats() #peak vram for this run only
    elapsed=wall_timer() #wall clock from here
    total_generated_tokens=0 #manual: compare forks at equal generated-token budgets, so we record it

    for update in range(updates):
        #1) this update's prompts, from the fixed order
        batch_rows=[] #prompt rows for this update
        for k in range(prompts_per_update):
            index=order[(update*prompts_per_update+k)%len(order)] #wrap around if we ever run out
            batch_rows.append(prompt_rows[index]) #add
        prompts=[] #chat messages
        prompt_ids=[] #fixed ids for the log
        for row in batch_rows:
            prompts.append(prompt_messages(row)) #messages list
            prompt_ids.append(row.get("prompt_id")) #record which prompt was used

        #2) on-policy rollout with the current policy (this snapshot is pi_old)
        generated=batch_generate(
            policy,
            tokenizer,
            prompts,
            max_prompt_length=int(cfg["max_prompt_length"]), #256
            max_new_tokens=int(cfg["max_response_length"]), #512 continuation cap
            temperature=float(generation_cfg["temperature"]),
            top_p=float(generation_cfg["top_p"]),
            do_sample=bool(generation_cfg["do_sample"]),
        )
        sequences=generated["sequences"].clone() #clone: generate makes inference-mode tensors that cannot go through backward
        attention_mask=generated["attention_mask"].clone() #prompt mask + ones on the response
        response_ids=generated["response_ids"].clone() #a_t for every step
        mask=generated["response_mask"].clone().float() #valid response tokens (eos included)
        prompt_width=generated["prompt_width"] #where the response starts
        steps=response_ids.shape[1] #response length incl. padding

        #3) old/ref log-probs, old values and the learned reward, all without gradient
        with torch.no_grad():
            old_logp,old_logits=response_token_logprobs(policy,sequences,attention_mask,prompt_width,response_ids) #log pi_old(a_t|s_t)
            del old_logits #not needed
            with reference_mode(policy):
                ref_logp,ref_logits=response_token_logprobs(policy,sequences,attention_mask,prompt_width,response_ids) #log pi_ref, adapter off
            del ref_logits #not needed
            old_values=response_values(value_model,sequences,attention_mask,prompt_width,steps) #V_old(s_t)
            raw_reward=score_reward_pairs(reward_model,reward_tokenizer,prompts,generated["responses"],max_length=int(cfg["reward_max_length"])) #r_task
        old_logp=old_logp.float() #fp32 for the ratio
        ref_logp=ref_logp.float() #same
        raw_reward=raw_reward.to(device).float() #reward model may sit on another device
        task_reward=raw_reward.clone() #effective reward after the eos penalty
        for i in range(len(prompts)):
            if generated["terminated_with_eos"][i]==False:
                task_reward[i]=task_reward[i]-missing_eos_penalty #config: punish responses that never stop

        #4) kl-shaped token rewards, gae, returns (manual: r_t = r_task*1[t=T] - beta_kl*(log pi - log pi_ref))
        rewards=shaped_rewards(task_reward,old_logp,ref_logp,mask,kl_beta) #kl on every token, task reward on the last valid one
        advantages,returns=compute_gae(rewards,old_values,mask,gamma,lam) #A_t and the value targets
        advantages=normalize_advantages(advantages,mask) #whiten over valid tokens (starter helper), returns stay raw

        #5) ppo epochs on this rollout
        epoch_logs=[] #per-epoch diagnostics
        for epoch in range(ppo_epochs):
            #policy step
            new_logp,new_logits=response_token_logprobs(policy,sequences,attention_mask,prompt_width,response_ids) #with gradient this time
            del new_logits #drop the reference to the big logits tensor
            policy_loss,ratio,clip_fraction=ppo_policy_loss(new_logp,old_logp,advantages,mask,eps) #fixed clipped surrogate
            affected=affected_token_fraction(ratio,advantages,mask,eps) #tokens where clipping changes the objective
            approx_kl_old_new=masked_mean(old_logp-new_logp.detach().float(),mask) #how far this update has already moved from pi_old
            if torch.isfinite(policy_loss)==False:
                raise RuntimeError("policy loss is not finite at update "+str(update+1)+", stopping before the weights get corrupted") #fail loudly
            policy_optimizer.zero_grad() #clear
            policy_loss.backward() #gradient of -L_clip
            policy_grad_norm=torch.nn.utils.clip_grad_norm_(trainable_parameters(policy),max_grad_norm) #norm before clipping
            policy_optimizer.step() #update the lora

            #critic step
            new_values=response_values(value_model,sequences,attention_mask,prompt_width,steps) #V_theta(s_t) with gradient
            value_loss=value_mse_loss(new_values,returns.detach(),mask) #regress onto the gae returns
            if torch.isfinite(value_loss)==False:
                raise RuntimeError("value loss is not finite at update "+str(update+1)+", stopping before the weights get corrupted") #fail loudly
            value_optimizer.zero_grad() #clear
            (value_coef*value_loss).backward() #config value_coef 0.5
            value_params=[] #trainable critic params
            for group in value_optimizer.param_groups:
                for parameter in group["params"]:
                    value_params.append(parameter) #lora + head
            value_grad_norm=torch.nn.utils.clip_grad_norm_(value_params,max_grad_norm) #same clipping rule
            value_optimizer.step() #update the critic

            epoch_logs.append({
                "policy_loss":float(policy_loss.item()), #-clipped surrogate
                "value_loss":float(value_loss.item()), #mse to returns
                "clip_fraction":float(clip_fraction.item()), #manual: ratio outside [1-eps,1+eps]
                "affected_fraction":float(affected.item()), #ratio outside AND clipping changes the min
                "approx_kl_old_new":float(approx_kl_old_new.item()), #drift inside one update
                "ratio_mean":float(masked_mean(ratio,mask).item()), #should start at 1
                "policy_grad_norm":float(policy_grad_norm), #before clipping
                "value_grad_norm":float(value_grad_norm), #before clipping
            })

        response_lengths=mask.sum(-1) #valid tokens per response
        total_generated_tokens=total_generated_tokens+int(response_lengths.sum().item()) #running budget
        last=epoch_logs[-1] #last epoch is the one after the most movement
        policy_grad_total=0.0 #sum of grad norms over the epochs
        value_grad_total=0.0 #same for the critic
        for log in epoch_logs:
            policy_grad_total=policy_grad_total+log["policy_grad_norm"] #add
            value_grad_total=value_grad_total+log["value_grad_norm"] #add
        record={
            "run_name":run_name, #standard / clip_0.05 / kl_0.0 ...
            "update":update+1, #1-based for plots
            "prompt_ids":prompt_ids, #which prompt(s)
            "reward_raw":float(raw_reward.mean().item()), #learned reward of the rollout
            "reward_effective":float(task_reward.mean().item()), #after eos penalty
            "kl_from_reference":float(sampled_kl(old_logp,ref_logp,mask).item()), #manual sampled estimator
            "entropy":float(sample_entropy(old_logp,mask).item()), #mean token entropy (sampled)
            "response_length":float(response_lengths.float().mean().item()), #tokens
            "terminated_with_eos":float(np.mean(generated["terminated_with_eos"])), #fraction that ended
            "value_mean":float(masked_mean(old_values,mask).item()), #critic level
            "return_mean":float(masked_mean(returns,mask).item()), #target level
            "policy_loss":last["policy_loss"], #last epoch
            "value_loss":last["value_loss"],
            "clip_fraction":last["clip_fraction"],
            "affected_fraction":last["affected_fraction"],
            "approx_kl_old_new":last["approx_kl_old_new"],
            "ratio_mean":last["ratio_mean"],
            "policy_grad_norm":policy_grad_total/len(epoch_logs), #average over epochs
            "value_grad_norm":value_grad_total/len(epoch_logs), #average over epochs
            "epochs":epoch_logs, #full per-epoch detail
            "generated_tokens_so_far":total_generated_tokens, #budget check
            "seconds":elapsed(), #wall clock so far
        }
        append_jsonl(log_path,record) #one json line per update
        print("update",update+1,"reward",round(record["reward_raw"],3),"kl",round(record["kl_from_reference"],4),"len",record["response_length"]) #progress

    policy.save_pretrained(str(out)) #policy lora (task 4 reads outputs/task2_ppo/standard)
    tokenizer.save_pretrained(str(out)) #self-contained adapter folder
    value_model.save_pretrained(str(out)+"_value") #critic lora + head, separate folder

    peak_vram=None #none on cpu
    if torch.cuda.is_available():
        peak_vram=torch.cuda.max_memory_allocated()/2**30 #GiB, manual wants this for the standard run
    summary={
        "run_name":run_name, #condition
        "updates":updates, #update budget
        "prompts_per_update":prompts_per_update, #1
        "ppo_epochs":ppo_epochs, #2
        "clip_epsilon":eps, #clip setting
        "kl_beta":kl_beta, #kl setting
        "seed":int(cfg["seed"]), #6304
        "total_generated_tokens":total_generated_tokens, #token budget actually used
        "wall_clock_seconds":elapsed(), #manual: wall-clock time
        "peak_vram_gib":peak_vram, #manual: peak vram
        "output":str(out), #adapter folder
    }
    save_json(results_dir/"train_summary.json",summary) #small json for the report
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
