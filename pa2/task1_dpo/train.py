from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.logging_utils import set_seed
from common.models import load_policy, load_tokenizer, trainable_parameters
from task1_dpo.dpo import dpo_loss
from common.generation import response_sequence_logprobs
from common.logging_utils import append_jsonl, save_json, wall_timer
from common.models import reference_mode


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def drop_long_prompts(rows,tokenizer,max_length):
    """Remove pairs whose prompt alone does not fit, encode_prompt_response raises on them."""
    kept=[] #pairs that can be encoded
    dropped=[] #prompt_ids we removed, saved in the summary
    for row in rows:
        prompt=prompt_messages_from_preference(row) #same prompt the collate function builds
        prompt_ids=tokenizer.apply_chat_template(prompt,tokenize=True,add_generation_prompt=True) #same call as encode_prompt_response
        if len(prompt_ids)>=max_length:
            dropped.append(row.get("prompt_id")) #starter error says: increase max_length or filter this example
        else:
            kept.append(row) #fits
    return kept,dropped


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    rows,dropped=drop_long_prompts(rows,load_tokenizer(cfg["base_model"]),int(cfg["max_sequence_length"])) #filter before the 600 cut so every fork still gets 600
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
        "dropped_long_prompts":dropped, #added: prompt_ids filtered out above
    }


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)

    model=bundle["model"] #policy = base model + fresh zero-init lora, so at step 0 it equals the reference
    tokenizer=bundle["tokenizer"] #saved next to the adapter
    loader=bundle["loader"] #shuffled preference pairs, seed already set in prepare_dpo_run
    optimizer=bundle["optimizer"] #adamw over the lora parameters only
    beta=bundle["beta"] #0.10 standard, or the forked value
    accum_steps=int(cfg["grad_accum_steps"]) #config: 8 micro batches per optimizer step
    max_grad_norm=float(cfg["max_grad_norm"]) #config: 1.0
    epochs=int(cfg["epochs"]) #config: 1
    device=next(model.parameters()).device #cuda if available
    results_dir=repo_path(cfg["results_dir"])/run_name #results/task1_dpo/<run_name>/
    log_path=results_dir/"train_log.jsonl" #one line per optimizer step
    results_dir.mkdir(parents=True,exist_ok=True) #make sure it exists
    if log_path.exists():
        log_path.unlink() #fresh log, append_jsonl would otherwise keep lines from an older run

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats() #so peak vram is for this run only
    elapsed=wall_timer() #returns seconds since now when called

    optimizer.zero_grad() #start with clean grads
    optimizer_step=0 #how many real updates so far
    micro_in_window=0 #micro batches accumulated since the last update
    examples_seen=0 #pairs processed
    #running sums for one optimizer step, averaged when we log
    window={"loss":0.0,"accuracy":0.0,"logit_mean":0.0,"policy_margin_mean":0.0,"chosen_tokens":0.0,"rejected_tokens":0.0}

    model.train() #lora dropout on for the policy forward
    for epoch in range(epochs):
        for chosen_batch,rejected_batch in loader:
            for key in chosen_batch:
                chosen_batch[key]=chosen_batch[key].to(device) #input_ids, attention_mask, response_mask
            for key in rejected_batch:
                rejected_batch[key]=rejected_batch[key].to(device) #same for the rejected side

            #reference log-probs: same network with the adapter switched off, no gradient
            with torch.no_grad():
                with reference_mode(model):
                    ref_chosen_logp,_,_=response_sequence_logprobs(model,chosen_batch) #log pi_ref(y+|x), summed over response tokens
                    ref_rejected_logp,_,_=response_sequence_logprobs(model,rejected_batch) #log pi_ref(y-|x)

            #policy log-probs with gradient
            policy_chosen_logp,_,_=response_sequence_logprobs(model,chosen_batch) #log pi_theta(y+|x)
            policy_rejected_logp,_,_=response_sequence_logprobs(model,rejected_batch) #log pi_theta(y-|x)

            loss,stats=dpo_loss(policy_chosen_logp,policy_rejected_logp,ref_chosen_logp,ref_rejected_logp,beta) #fixed objective from dpo.py
            (loss/accum_steps).backward() #divide so 8 micro batches add up to one average gradient

            micro_in_window=micro_in_window+1 #one more micro batch in this window
            examples_seen=examples_seen+chosen_batch["input_ids"].shape[0] #pairs in this micro batch
            window["loss"]=window["loss"]+float(loss.item()) #running sums
            window["accuracy"]=window["accuracy"]+float(stats["preference_accuracy"].item())
            window["logit_mean"]=window["logit_mean"]+float(stats["logit_mean"].item())
            window["policy_margin_mean"]=window["policy_margin_mean"]+float(stats["policy_margin_mean"].item())
            window["chosen_tokens"]=window["chosen_tokens"]+float(chosen_batch["response_mask"].sum(-1).mean().item()) #response length, for length analysis
            window["rejected_tokens"]=window["rejected_tokens"]+float(rejected_batch["response_mask"].sum(-1).mean().item())

            if micro_in_window==accum_steps:
                grad_norm=torch.nn.utils.clip_grad_norm_(trainable_parameters(model),max_grad_norm) #returns the norm before clipping
                optimizer.step() #one real update
                optimizer.zero_grad() #clear for the next window
                optimizer_step=optimizer_step+1 #count it
                record={"run_name":run_name,"epoch":epoch,"step":optimizer_step,"examples_seen":examples_seen,"beta":beta,"grad_norm":float(grad_norm),"seconds":elapsed()}
                for key in window:
                    record["train_"+key]=window[key]/micro_in_window #average over the window
                    window[key]=0.0 #reset for the next window
                append_jsonl(log_path,record) #machine-readable log (manual: save logs for every result)
                micro_in_window=0 #new window

        #leftover micro batches at the end of the epoch still get an update
        if micro_in_window>0:
            grad_norm=torch.nn.utils.clip_grad_norm_(trainable_parameters(model),max_grad_norm) #same as above
            optimizer.step() #smaller window, so this last update is a bit smaller
            optimizer.zero_grad() #clear
            optimizer_step=optimizer_step+1 #count it
            record={"run_name":run_name,"epoch":epoch,"step":optimizer_step,"examples_seen":examples_seen,"beta":beta,"grad_norm":float(grad_norm),"seconds":elapsed(),"partial_window":micro_in_window}
            for key in window:
                record["train_"+key]=window[key]/micro_in_window #average over the partial window
                window[key]=0.0 #reset
            append_jsonl(log_path,record) #log it too
            micro_in_window=0 #reset

    model.save_pretrained(str(output)) #only the lora adapter, a few MB
    tokenizer.save_pretrained(str(output)) #so the adapter folder is self contained

    peak_vram=None #stays none on cpu
    if torch.cuda.is_available():
        peak_vram=torch.cuda.max_memory_allocated()/2**30 #GiB
    summary={
        "run_name":run_name, #standard, beta_0.03, length_balanced, ...
        "budget":"full_epoch", #manual: say clearly that the standard run and the short forks use different budgets
        "beta":beta, #dpo beta used
        "dataset":str(dataset_path or cfg["paths"]["dpo_standard_train"]), #which preference file
        "train_examples":len(bundle["rows"]), #after the long-prompt filter and max_examples
        "dropped_long_prompts":len(bundle["dropped_long_prompts"]), #pairs removed because the prompt alone is >=max_length
        "dropped_prompt_ids":bundle["dropped_long_prompts"], #which ones
        "max_examples":max_examples, #none for the full run, 600 for the beta forks
        "epochs":epochs, #config
        "optimizer_steps":optimizer_step, #real updates
        "batch_size":int(cfg["batch_size"]), #config
        "grad_accum_steps":accum_steps, #config
        "learning_rate":float(cfg["learning_rate"]), #config
        "seed":int(cfg["seed"]), #config
        "wall_clock_seconds":elapsed(), #total training time
        "peak_vram_gib":peak_vram, #peak allocated memory
        "output":str(output), #where the adapter went
    }
    if max_examples is not None:
        summary["budget"]="short_"+str(max_examples) #short fork
    save_json(results_dir/"train_summary.json",summary) #small json, committed with the repo
    return str(output)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
