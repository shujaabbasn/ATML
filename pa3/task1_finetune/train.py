from __future__ import annotations

import argparse
import glob

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from common.lora import adapter_state, inject_lora, lora_modules
from common.utils import (
    append_jsonl,
    count_parameters,
    folder_size_bytes,
    load_json,
    load_yaml,
    peak_vram_gib,
    read_jsonl,
    remove_folder,
    repo_path,
    rng_state,
    save_json,
    set_rng_state,
    set_seed,
    wall_timer,
    write_jsonl,
)
from task1_finetune.prepare_data import prompt_ids, response_ids


#one script for all five variants:
#  full_ft       all parameters, fp32 master weights, lambda=0
#  full_ft_l2sp  same + lambda/2 ||theta-theta0||^2
#  lora4/lora16  frozen fp16 backbone + fp32 adapters on all 7 projections
#  oplora16      lora16 with the orthogonal projection of B and A


def encode_example(tokenizer,row):
    """input ids and labels, -100 on the prompt so only the response and <|im_end|> are supervised."""
    prompt=prompt_ids(tokenizer,row["query"])
    response=response_ids(tokenizer,row["response"])
    input_ids=prompt+response
    labels=[-100]*len(prompt)+response
    return input_ids,labels


def pad_micro_batch(examples,pad_id,device):
    """Right-pad a micro batch. Padding gets attention 0 and label -100."""
    width=0
    for input_ids,labels in examples:
        if len(input_ids)>width:
            width=len(input_ids)
    all_ids=[]
    all_labels=[]
    all_mask=[]
    for input_ids,labels in examples:
        extra=width-len(input_ids)
        all_ids.append(input_ids+[pad_id]*extra)
        all_labels.append(labels+[-100]*extra)
        all_mask.append([1]*len(input_ids)+[0]*extra)
    return torch.tensor(all_ids,device=device),torch.tensor(all_labels,device=device),torch.tensor(all_mask,device=device)


def chunk_cross_entropy(lm_head,hidden,targets):
    """Summed cross-entropy of one chunk of response positions, logits in fp32."""
    logits=lm_head(hidden).float() #[chunk, vocab]
    return F.cross_entropy(logits,targets,reduction="sum")


def response_loss_sum(model,input_ids,attention_mask,labels,chunk_tokens):
    """Sum of next-token cross-entropy over the supervised positions of one micro batch.

    The lm_head is applied only to supervised positions and in chunks (re-computed in backward),
    because 2 x 2048 positions x 151,936 vocab logits in fp32 would not fit on a T4 next to full ft.
    """
    hidden=model.model(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state #[batch, seq, 1024]
    hidden=hidden[:,:-1,:] #position t predicts token t+1
    targets=labels[:,1:]
    valid=targets!=-100 #response tokens + <|im_end|>
    hidden=hidden[valid] #[n_valid, 1024]
    targets=targets[valid] #[n_valid]
    total=hidden.new_zeros((),dtype=torch.float32)
    for start in range(0,hidden.shape[0],chunk_tokens):
        end=start+chunk_tokens
        total=total+checkpoint(chunk_cross_entropy,model.lm_head,hidden[start:end],targets[start:end],use_reentrant=False)
    return total,int(valid.sum().item())


def param_groups(model,weight_decay):
    """Manual: weight decay 0.01, except biases and parameters with fewer than two dimensions."""
    decay=[]
    no_decay=[]
    for name,p in model.named_parameters():
        if p.requires_grad==False:
            continue
        if p.ndim<2 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    groups=[]
    if len(decay)>0:
        groups.append({"params":decay,"weight_decay":weight_decay})
    if len(no_decay)>0:
        groups.append({"params":no_decay,"weight_decay":0.0}) #lora has no 1-d trainable params, so this group can be empty
    return groups


def load_model(cfg,run_cfg,device):
    """Full ft: fp32 master weights. lora/oplora: fp16 frozen backbone + fp32 adapters."""
    method=run_cfg["method"]
    extra={"time_seconds":0.0} #method specific preprocessing (oplora svd)
    if method=="full":
        model=AutoModelForCausalLM.from_pretrained(cfg["base_model"],torch_dtype=torch.float32)
        model.to(device)
    else:
        model=AutoModelForCausalLM.from_pretrained(cfg["base_model"],torch_dtype=torch.float16)
        model.to(device)
        for p in model.parameters():
            p.requires_grad_(False) #freeze the whole backbone, biases included
        protect_rank=None
        if method=="oplora":
            protect_rank=int(run_cfg["protect_rank"])
        torch.manual_seed(int(cfg["seed"])) #same seed right before injection, so lora16 and oplora16 get identical A
        timer=wall_timer()
        names=inject_lora(model,cfg["lora"]["targets"],int(run_cfg["rank"]),float(run_cfg["alpha"]),float(cfg["lora"]["dropout"]),protect_rank)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        extra["time_seconds"]=timer() #for oplora this is the svd time of all 196 weights
        extra["adapted_modules"]=len(names)
        if method=="oplora":
            basis_bytes=0
            for module in lora_modules(model).values():
                basis_bytes=basis_bytes+module.U_k.numel()*4+module.V_k.numel()*4 #fp32
            extra["basis_storage_bytes"]=basis_bytes
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False}) #manual: gradient checkpointing enabled
    return model,extra


def l2sp_penalty_and_grad(model,theta0,lam):
    """Omega = 1/2 sum (theta - theta0)^2 over every unique parameter, and add lambda (theta - theta0) to the grads.

    theta0 lives on the cpu (pinned) and is copied over one tensor at a time, the copy is the transfer overhead.
    Called once per optimizer update, after unscaling, so the clipped gradient is the gradient of SFT + lambda*Omega.
    """
    omega=0.0
    for name,p in model.named_parameters(): #named_parameters skips the tied lm_head, so it is counted once
        reference=theta0[name].to(p.device,non_blocking=True)
        difference=p.detach()-reference
        omega=omega+0.5*float((difference*difference).sum().item())
        if lam>0 and p.grad is not None:
            p.grad.add_(difference,alpha=lam) #d/dtheta of lambda/2 ||theta-theta0||^2
    return omega


def parameter_displacement(model,base_state):
    """||theta - theta0||_2 over all unique parameters, in fp32."""
    total=0.0
    for name,p in model.named_parameters():
        difference=p.detach().float().cpu()-base_state[name].float()
        total=total+float((difference*difference).sum().item())
    return total**0.5


def save_model_weights(model,run_cfg,folder,tokenizer):
    """Full ft: whole model in fp32. adapters: raw factors, plus projected factors for oplora."""
    folder=repo_path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    if run_cfg["method"]=="full":
        model.save_pretrained(str(folder),safe_serialization=True)
        tokenizer.save_pretrained(str(folder))
    else:
        save_file(adapter_state(model,projected=False),str(folder/"adapter_raw.safetensors")) #trainable A, B
        if run_cfg["method"]=="oplora":
            timer=wall_timer()
            save_file(adapter_state(model,projected=True),str(folder/"adapter_projected.safetensors")) #A~, B~ used for inference
            save_json(folder/"export_time.json",{"projected_export_seconds":timer()}) #manual: oplora export overhead
    return folder_size_bytes(folder)


def trainable_state(model,run_cfg):
    if run_cfg["method"]=="full":
        return model.state_dict()
    else:
        state={}
        for name,p in model.named_parameters():
            if p.requires_grad:
                state[name]=p.detach().clone()
        return state


def save_recovery(model,run_cfg,optimizer,scheduler,scaler,update,examples_seen,seconds,bookkeeping,folder):
    """Everything needed to resume: weights, optimizer, scheduler, grad scaler, rng, data position."""
    folder=repo_path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    torch.save({
        "update":update, #optimizer updates done
        "examples_seen":examples_seen, #data position, examples are used in source order
        "model":trainable_state(model,run_cfg),
        "optimizer":optimizer.state_dict(),
        "scheduler":scheduler.state_dict(),
        "scaler":scaler.state_dict(),
        "rng":rng_state(),
        "seconds":seconds, #training time so far, so a resumed run still reports the full time
        "bookkeeping":bookkeeping, #checkpoint sizes, displacements, skipped updates, l2-sp time so far
    },str(folder/"recovery.pt"))
    return folder_size_bytes(folder)


def latest_recovery(out_dir):
    folders=sorted(glob.glob(str(repo_path(out_dir)/"recovery_u*")))
    best=None
    best_update=-1
    for folder in folders:
        update=int(folder.split("recovery_u")[-1])
        if update>best_update:
            best_update=update
            best=folder
    return best


def train(config_path,run_name,resume,keep_recovery):
    cfg=load_yaml(config_path)
    run_cfg=cfg["runs"][run_name]
    train_cfg=cfg["train"]
    set_seed(int(cfg["seed"]))
    device="cpu"
    if torch.cuda.is_available():
        device="cuda"
    out_dir=repo_path(cfg["outputs_dir"])/run_name #outputs/task1/<run>
    results_dir=repo_path(cfg["results_dir"])/"train"/run_name #results/task1/train/<run>
    log_path=results_dir/"train_log.jsonl"

    tokenizer=AutoTokenizer.from_pretrained(cfg["base_model"])
    rows=read_jsonl("data/task1/train.jsonl") #40,000 examples in source order, never shuffled
    model,method_extra=load_model(cfg,run_cfg,device)
    model.train() #only lora dropout reacts to this, qwen3 has attention dropout 0

    trainable=[] #parameters the optimizer updates
    for p in model.parameters():
        if p.requires_grad:
            trainable.append(p)
    optimizer=torch.optim.AdamW(param_groups(model,float(train_cfg["weight_decay"])),lr=float(run_cfg["lr"]),betas=tuple(train_cfg["adam_betas"]),eps=float(train_cfg["adam_eps"])) #fp32 states since trainable params are fp32
    total_updates=int(train_cfg["updates"]) #1,250
    scheduler=get_cosine_schedule_with_warmup(optimizer,int(train_cfg["warmup_updates"]),total_updates) #38 warmup, then cosine
    scaler=torch.amp.GradScaler("cuda",enabled=(device=="cuda")) #dynamic loss scaling for fp16

    lam=float(run_cfg.get("l2sp_lambda",0.0))
    theta0=None
    reference_bytes=0
    if lam>0:
        theta0={}
        for name,p in model.named_parameters():
            reference=p.detach().to("cpu",copy=True)
            if torch.cuda.is_available():
                reference=reference.pin_memory() #faster host to gpu copies
            theta0[name]=reference
            reference_bytes=reference_bytes+reference.numel()*reference.element_size()

    micro_batch=int(train_cfg["micro_batch"]) #2
    accum=int(train_cfg["grad_accum"]) #16
    effective=micro_batch*accum #32
    start_update=0
    seconds_before=0.0 #training time from earlier sessions
    bookkeeping={"penalty_seconds":0.0,"skipped_updates":[],"checkpoint_sizes":{},"displacement":{}}
    if resume==True:
        folder=latest_recovery(out_dir)
        if folder is not None:
            state=torch.load(folder+"/recovery.pt",map_location=device,weights_only=False)
            model.load_state_dict(state["model"],strict=(run_cfg["method"]=="full"))
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            set_rng_state(state["rng"])
            start_update=int(state["update"])
            seconds_before=float(state["seconds"])
            bookkeeping=state["bookkeeping"]
            print("resumed from",folder,"at update",start_update)
    if start_update==0 and log_path.exists():
        log_path.unlink() #fresh run, start a fresh log
    if start_update>0 and log_path.exists():
        kept=[]
        for record in read_jsonl(log_path):
            if record["update"]<=start_update:
                kept.append(record) #drop lines logged after the checkpoint, they are redone now
        write_jsonl(log_path,kept)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    session_timer=wall_timer()
    def elapsed():
        return seconds_before+session_timer() #total training seconds across sessions
    penalty_seconds=bookkeeping["penalty_seconds"] #time spent on the l2-sp copy + penalty (manual: transfer overhead)
    skipped_updates=bookkeeping["skipped_updates"] #updates the grad scaler skipped because of inf/nan
    checkpoint_sizes=bookkeeping["checkpoint_sizes"]
    displacement=bookkeeping["displacement"]
    base_state=None #starting weights on cpu, only loaded when a displacement is needed

    for update in range(start_update,total_updates):
        batch_rows=rows[update*effective:(update+1)*effective] #examples 32u .. 32u+31, the data position is the update index
        encoded=[]
        for row in batch_rows:
            encoded.append(encode_example(tokenizer,row))
        total_tokens=0 #valid response tokens in the effective batch, the loss normalizer
        for input_ids,labels in encoded:
            for label in labels[1:]:
                if label!=-100:
                    total_tokens=total_tokens+1

        sft_loss=0.0
        for start in range(0,effective,micro_batch):
            input_ids,labels,attention_mask=pad_micro_batch(encoded[start:start+micro_batch],tokenizer.pad_token_id,device)
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=(device=="cuda")):
                loss_sum,n_valid=response_loss_sum(model,input_ids,attention_mask,labels,int(train_cfg["loss_chunk_tokens"]))
            loss=loss_sum/total_tokens #manual: normalize over all valid response tokens in the effective batch
            scaler.scale(loss).backward()
            sft_loss=sft_loss+float(loss.item())

        scaler.unscale_(optimizer) #manual: unscale before clipping
        omega=None
        if theta0 is not None:
            penalty_timer=wall_timer()
            omega=l2sp_penalty_and_grad(model,theta0,lam)
            penalty_seconds=penalty_seconds+penalty_timer()
        grad_norm=torch.nn.utils.clip_grad_norm_(trainable,float(train_cfg["max_grad_norm"])) #norm before clipping
        lr_used=scheduler.get_last_lr()[0] #lr of this update
        scale_before=scaler.get_scale()
        scaler.step(optimizer) #skipped inside if any grad is inf/nan
        scaler.update()
        skipped=scaler.get_scale()<scale_before #the scaler only lowers its scale after a skipped step
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        done=update+1
        record={"update":done,"examples_seen":done*effective,"sft_loss":sft_loss,"lr":lr_used,"grad_norm":float(grad_norm),"loss_scale":scale_before,"skipped":bool(skipped),"tokens":total_tokens,"seconds":elapsed()}
        if omega is not None:
            record["omega"]=omega
            record["lambda_omega"]=lam*omega
            record["total_loss"]=sft_loss+lam*omega
        if skipped:
            skipped_updates.append(done)
        append_jsonl(log_path,record)
        if done%25==0:
            print(record)

        if done in train_cfg["recovery_updates"]:
            if done==int(train_cfg["midpoint_update"]):
                checkpoint_sizes["midpoint"]=save_model_weights(model,run_cfg,out_dir/"midpoint",tokenizer)
            if run_cfg["method"]=="full" and (done==int(train_cfg["midpoint_update"]) or done==total_updates):
                if base_state is None:
                    base_state=AutoModelForCausalLM.from_pretrained(cfg["base_model"],torch_dtype=torch.float32).state_dict()
                displacement["u"+str(done)]=parameter_displacement(model,base_state) #||theta - theta0||_2
            folder=out_dir/("recovery_u"+str(done).zfill(4))
            bookkeeping={"penalty_seconds":penalty_seconds,"skipped_updates":skipped_updates,"checkpoint_sizes":checkpoint_sizes,"displacement":displacement}
            checkpoint_sizes["recovery_u"+str(done)]=save_recovery(model,run_cfg,optimizer,scheduler,scaler,done,done*effective,elapsed(),bookkeeping,folder)
            for old in glob.glob(str(out_dir/"recovery_u*")):
                if old!=str(folder):
                    remove_folder(old) #keep only the newest recovery checkpoint, disk on kaggle is 20GB

    train_seconds=elapsed()
    checkpoint_sizes["final"]=save_model_weights(model,run_cfg,out_dir/"final",tokenizer)
    if keep_recovery==False:
        for old in glob.glob(str(out_dir/"recovery_u*")):
            remove_folder(old) #run finished, final/ holds the same weights

    summary={
        "run_name":run_name,
        "method":run_cfg["method"],
        "peak_lr":float(run_cfg["lr"]),
        "l2sp_lambda":lam,
        "rank":run_cfg.get("rank"),
        "alpha":run_cfg.get("alpha"),
        "protect_rank":run_cfg.get("protect_rank"),
        "updates":total_updates,
        "examples":total_updates*effective,
        "trainable_parameters":count_parameters(trainable),
        "total_parameters":count_parameters(model.parameters()),
        "train_seconds":train_seconds, #includes earlier sessions when resumed
        "peak_vram_note":"peak of the last session only" if start_update>0 else "whole run",
        "resumed_from_update":start_update,
        "peak_vram_gib":peak_vram_gib(),
        "checkpoint_bytes":checkpoint_sizes,
        "skipped_updates":skipped_updates,
        "method_preprocessing_seconds":method_extra["time_seconds"],
        "gpu":"cpu",
    }
    if torch.cuda.is_available():
        summary["gpu"]=torch.cuda.get_device_name(0)
    if "basis_storage_bytes" in method_extra:
        summary["oplora_basis_storage_bytes"]=method_extra["basis_storage_bytes"]
        summary["oplora_svd_seconds"]=method_extra["time_seconds"]
        summary["oplora_projected_export_seconds"]=load_json(out_dir/"final"/"export_time.json")["projected_export_seconds"]
    if theta0 is not None:
        summary["l2sp_reference_storage_bytes"]=reference_bytes
        summary["l2sp_penalty_and_transfer_seconds"]=penalty_seconds
    if len(displacement)>0:
        summary["displacement_l2"]=displacement
    save_json(results_dir/"train_summary.json",summary)
    print(summary)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task1.yaml")
    ap.add_argument("--run",required=True,choices=["full_ft","full_ft_l2sp","lora4","lora16","oplora16"])
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--keep-recovery",action="store_true")
    args=ap.parse_args()
    train(args.config,args.run,args.resume,args.keep_recovery)


if __name__=="__main__":
    main()
