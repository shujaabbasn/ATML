"""Steps 3-4: off-policy and on-policy forward-KL distillation of the 0.5B student (LoRA) towards the 1.5B teacher.

python -m task3_speculative.distill --step teacher_bank                      one teacher response per training prompt
python -m task3_speculative.distill --step train --method off_policy --seed 6304
python -m task3_speculative.distill --step train --method on_policy --seed 6304

Objective for both: L = 1/|H| sum_h KL(p_T(.|h) || q_psi(.|h)) over the full vocabulary at every supervised
response prefix h. Off-policy: h comes from the fixed teacher responses. On-policy: h comes from fresh rollouts of
the current student. The sampled token is never used as a hard label.
Budget: 30,000 math + 10,000 short teacher-scored tokens per run; one optimizer update per 256 scored tokens
(the last update has 40,000 mod 256 = 64). A response that would pass the quota is cut and the cut is logged.
"""
from __future__ import annotations

import argparse
import random
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, PeftModel

from common.utils import load_yaml, save_json, load_json, read_jsonl, write_jsonl, append_jsonl, repo_path, set_seed, get_device, sync, peak_mb, reset_peak, package_versions
from task3_speculative.common import load_causal, load_tokenizer, eos_ids, chat_ids, train_prompts
from task3_speculative.sampler import plain_generate, make_generator


def cap_of(item,config):
    if item["domain"]=="math":
        return config["cap_math"]
    else:
        return config["cap_short"]


@torch.no_grad()
def teacher_bank(config):
    """Fixed off-policy data: one teacher sample per training prompt (temperature 1, top-p 1), timed so the cost can
    be charged to every off-policy run."""
    device=get_device()
    distill=config["distill"]
    tokenizer=load_tokenizer(config)
    teacher=load_causal(config["target"])
    eos_list=eos_ids(tokenizer)
    math,short=train_prompts(config)
    rows=[]
    sync()
    started=time.perf_counter()
    items=math+short
    for index in range(len(items)):
        item=items[index]
        prompt_ids=chat_ids(tokenizer,item["prompt"])
        rollout_seed=config["seed"]+31*index
        generator=make_generator(device,rollout_seed)
        response,stats=plain_generate(teacher,prompt_ids,cap_of(item,distill),eos_list,generator,distill["temperature"])
        rows.append({"id":item["id"],"domain":item["domain"],"rollout_seed":rollout_seed,"prompt_ids":prompt_ids,"response_ids":response,"ended_with_eos":response[-1] in eos_list})
        if (index+1)%100==0:
            print("teacher bank",index+1,"/",len(items),flush=True)
    sync()
    seconds=time.perf_counter()-started
    out_dir=repo_path(config["results_dir"])/"distill"
    write_jsonl(out_dir/"teacher_train_responses.jsonl.gz",rows)
    tokens=0
    for row in rows:
        tokens=tokens+len(row["response_ids"])
    save_json(out_dir/"teacher_bank_cost.json",{"seconds":seconds,"responses":len(rows),"generated_tokens":tokens,"versions":package_versions()})


def kl_positions(teacher,student,sequence,prompt_length,first,last,device,times):
    """KL(teacher || student) at response positions first..last-1 (0 = first response token). Logit at index i
    predicts token i+1, so response token r is predicted at index prompt_length-1+r.
    times["teacher"] collects the teacher scoring time separately from the student forward/backward."""
    ids=torch.tensor([sequence[:prompt_length+last]],device=device)
    sync()
    t0=time.perf_counter()
    with torch.no_grad():
        teacher_logits=teacher(input_ids=ids).logits[0,prompt_length-1+first:prompt_length-1+last].float()
    sync()
    times["teacher"]=times["teacher"]+(time.perf_counter()-t0)
    with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=(device.type=="cuda")):
        student_logits=student(input_ids=ids).logits[0,prompt_length-1+first:prompt_length-1+last]
    teacher_log=F.log_softmax(teacher_logits,dim=-1)
    student_log=F.log_softmax(student_logits.float(),dim=-1)
    return (teacher_log.exp()*(teacher_log-student_log)).sum(dim=-1) #full common vocabulary


class Trainer:
    """Turns a stream of (sequence, prompt length, scored positions) into updates of exactly 256 scored tokens."""

    def __init__(self,config,teacher,student,device,log_path):
        distill=config["distill"]
        self.teacher=teacher
        self.student=student
        self.device=device
        self.per_update=distill["tokens_per_update"]
        self.total=distill["quota_math"]+distill["quota_short"]
        self.full_updates=self.total//self.per_update #156
        self.last_size=self.total-self.full_updates*self.per_update #64
        self.trainable=[]
        for parameter in student.parameters():
            if parameter.requires_grad:
                self.trainable.append(parameter)
        self.optimizer=torch.optim.AdamW(self.trainable,lr=distill["lr"],weight_decay=distill["weight_decay"])
        self.scaler=torch.amp.GradScaler("cuda",enabled=(device.type=="cuda")) #fp16 autocast, fp32 adapters
        self.clip=distill["clip"]
        self.update=0
        self.in_update=0
        self.kl_sum=0.0
        self.log_path=log_path
        self.skipped=0
        self.times={"teacher":0.0} #teacher scoring seconds
        self.time_teacher_and_student=0.0
        self.scored_so_far=0
        self.teacher_tokens=0 #tokens pushed through the teacher (prompt + scored part), teacher utilization
        self.started=time.perf_counter()

    def size_now(self):
        if self.update<self.full_updates:
            return self.per_update
        else:
            return self.last_size

    def add(self,sequence,prompt_length,count):
        """Score response positions 0..count-1 of this sequence, splitting them across update boundaries.
        After an update the remaining positions are re-forwarded with the new weights."""
        done=0
        while done<count:
            size=self.size_now()
            if size==0:
                return
            room=size-self.in_update
            take=min(room,count-done)
            sync()
            t0=time.perf_counter()
            kl=kl_positions(self.teacher,self.student,sequence,prompt_length,done,done+take,self.device,self.times)
            self.teacher_tokens=self.teacher_tokens+prompt_length+done+take
            loss=kl.sum()/size #normalized by the supervised tokens of this update
            self.scaler.scale(loss).backward()
            sync()
            self.time_teacher_and_student=self.time_teacher_and_student+(time.perf_counter()-t0)
            self.kl_sum=self.kl_sum+float(kl.detach().sum())
            self.in_update=self.in_update+take
            self.scored_so_far=self.scored_so_far+take
            done=done+take
            if self.in_update==size:
                self.step(size)

    def step(self,size):
        self.scaler.unscale_(self.optimizer)
        grad_norm=float(torch.nn.utils.clip_grad_norm_(self.trainable,self.clip))
        scale_before=self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        skipped=self.scaler.get_scale()<scale_before #grad scaler skips the step on inf/nan
        if skipped==True:
            self.skipped=self.skipped+1
        self.optimizer.zero_grad(set_to_none=True)
        append_jsonl(self.log_path,{"update":self.update,"tokens":size,"scored_tokens_so_far":self.scored_so_far,"mean_kl":self.kl_sum/size,"lr":self.optimizer.param_groups[0]["lr"],
            "grad_norm":grad_norm,"skipped_overflow":bool(skipped),"seconds_since_start":time.perf_counter()-self.started}) #loss vs teacher-scored-token curve
        if self.update%20==0:
            print("update",self.update,"kl",round(self.kl_sum/size,4),flush=True)
        self.update=self.update+1
        self.in_update=0
        self.kl_sum=0.0


def train(config,method,seed):
    device=get_device()
    distill=config["distill"]
    name=f"{method}_s{seed}"
    out_dir=repo_path(config["results_dir"])/"drafts"/name
    out_dir.mkdir(parents=True,exist_ok=True)
    log_path=out_dir/"updates.jsonl"
    if log_path.exists():
        log_path.unlink() #fresh log for a fresh run
    set_seed(seed)
    tokenizer=load_tokenizer(config)
    eos_list=eos_ids(tokenizer)
    teacher=load_causal(config["target"])
    base=load_causal(config["draft"])
    torch.manual_seed(seed) #same lora init for off- and on-policy at the same seed
    lora=LoraConfig(r=distill["lora_r"],lora_alpha=distill["lora_alpha"],lora_dropout=distill["lora_dropout"],target_modules=distill["lora_targets"],task_type="CAUSAL_LM")
    student=get_peft_model(base,lora)
    for parameter in student.parameters():
        if parameter.requires_grad:
            parameter.data=parameter.data.float() #fp32 adapters (and fp32 adam states), backbone stays fp16
    trainer=Trainer(config,teacher,student,device,log_path)
    math,short=train_prompts(config)
    pools={"math":math,"short":short}
    quotas={"math":distill["quota_math"],"short":distill["quota_short"]}
    used={"math":0,"short":0}
    order={}
    pointer={"math":0,"short":0}
    rng=random.Random(seed)
    for domain in pools:
        indices=list(range(len(pools[domain])))
        rng.shuffle(indices) #fixed prompt order per seed, cycled when a pool runs out
        order[domain]=indices
    bank={}
    bank_seconds=0.0
    if method=="off_policy":
        for row in read_jsonl(repo_path(config["results_dir"])/"distill"/"teacher_train_responses.jsonl.gz"):
            bank[row["id"]]=row
        bank_seconds=load_json(repo_path(config["results_dir"])/"distill"/"teacher_bank_cost.json")["seconds"]
    truncations=[]
    rollout_seconds=0.0
    rollout_tokens=0
    responses_used=0
    rollout_counter=0
    reset_peak()
    started=time.perf_counter()
    while used["math"]<quotas["math"] or used["short"]<quotas["short"]:
        #interleave: take the domain that is furthest behind its own quota
        if used["math"]>=quotas["math"]:
            domain="short"
        elif used["short"]>=quotas["short"]:
            domain="math"
        elif used["math"]/quotas["math"]<=used["short"]/quotas["short"]:
            domain="math"
        else:
            domain="short"
        item=pools[domain][order[domain][pointer[domain]%len(order[domain])]]
        pointer[domain]=pointer[domain]+1
        prompt_ids=chat_ids(tokenizer,item["prompt"])
        if method=="off_policy":
            response=bank[item["id"]]["response_ids"]
            response_seed=bank[item["id"]]["rollout_seed"]
        else:
            student.eval()
            sync()
            t0=time.perf_counter()
            response_seed=seed*1000+rollout_counter
            generator=make_generator(device,response_seed)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=(device.type=="cuda")):
                response,unused=plain_generate(student,prompt_ids,cap_of(item,distill),eos_list,generator,distill["temperature"])
            sync()
            rollout_seconds=rollout_seconds+(time.perf_counter()-t0)
            rollout_tokens=rollout_tokens+len(response)
            rollout_counter=rollout_counter+1
            student.train()
        count=len(response)
        left=quotas[domain]-used[domain]
        truncated_by_quota=False
        if count>left:
            truncations.append({"id":item["id"],"domain":domain,"response_tokens":count,"scored":left})
            count=left #score only the permitted prefix
            truncated_by_quota=True
        if len(response)>0 and response[-1] in eos_list:
            termination="eos"
        else:
            termination="cap"
        #manual: every presentation keeps source id, rollout seed, response ids, termination reason and teacher-scored tokens
        append_jsonl(out_dir/"presentations.jsonl",{"presentation":responses_used,"id":item["id"],"domain":domain,"rollout_seed":response_seed,"producer":method,"update_at_start":trainer.update,
            "response_ids":response,"response_tokens":len(response),"termination":termination,"teacher_scored_tokens":count,"truncated_by_quota":truncated_by_quota})
        trainer.add(prompt_ids+response,len(prompt_ids),count)
        used[domain]=used[domain]+count
        responses_used=responses_used+1
    sync()
    total_seconds=time.perf_counter()-started
    student.save_pretrained(str(out_dir/"adapter"))
    trainable_count=0
    student_count=0
    for parameter in student.parameters():
        student_count=student_count+parameter.numel()
        if parameter.requires_grad:
            trainable_count=trainable_count+parameter.numel()
    teacher_count=0
    for parameter in teacher.parameters():
        teacher_count=teacher_count+parameter.numel()
    cost={"method":method,"seed":seed,"updates":trainer.update,"adapter_parameters":trainable_count,"student_parameters_with_adapter":student_count,"teacher_parameters":teacher_count,
        "teacher_forward_tokens":trainer.teacher_tokens,"scored_tokens_total":used["math"]+used["short"],"config":distill,"skipped_updates":trainer.skipped,"scored_tokens":used,"responses_used":responses_used,"truncations":truncations,
        "teacher_bank_generation_seconds_charged":bank_seconds,"student_rollout_seconds":rollout_seconds,"student_rollout_tokens":rollout_tokens,
        "teacher_scoring_seconds":trainer.times["teacher"],"student_forward_backward_update_seconds":trainer.time_teacher_and_student-trainer.times["teacher"],"training_loop_seconds":total_seconds,
        "end_to_end_seconds":total_seconds+bank_seconds,"peak_mb":peak_mb(),"versions":package_versions()}
    save_json(out_dir/"cost.json",cost)
    del teacher
    del student
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    merge(config,name,tokenizer)


@torch.no_grad()
def merge(config,name,tokenizer):
    """Merge the adapter into a fresh fp16 0.5B backbone, check merged vs unmerged logits, save the merged draft."""
    device=get_device()
    out_dir=repo_path(config["results_dir"])/"drafts"/name
    base=load_causal(config["draft"])
    unmerged=PeftModel.from_pretrained(base,str(out_dir/"adapter")).to(device).eval()
    math,short=train_prompts(config)
    checks=[]
    references=[]
    for item in math[:2]+short[:2]:
        ids=torch.tensor([chat_ids(tokenizer,item["prompt"])],device=device)
        references.append((ids,unmerged(input_ids=ids).logits.float()))
    sync()
    t0=time.perf_counter()
    merged=unmerged.merge_and_unload()
    sync()
    merge_seconds=time.perf_counter()-t0
    for ids,reference in references:
        logits=merged(input_ids=ids).logits.float()
        checks.append({"max_abs_logit_diff":float((logits-reference).abs().max()),"argmax_agree":float((logits.argmax(-1)==reference.argmax(-1)).float().mean())})
    merged.save_pretrained(str(out_dir/"merged"))
    tokenizer.save_pretrained(str(out_dir/"merged"))
    save_json(out_dir/"merge_check.json",{"checks":checks,"merge_seconds":merge_seconds})
    print("merged",name,checks)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/task3.yaml")
    parser.add_argument("--step",required=True,choices=["teacher_bank","train"])
    parser.add_argument("--method",choices=["off_policy","on_policy"])
    parser.add_argument("--seed",type=int)
    args=parser.parse_args()
    config=load_yaml(args.config)
    if args.step=="teacher_bank":
        teacher_bank(config)
    else:
        train(config,args.method,args.seed)


if __name__=="__main__":
    main()
