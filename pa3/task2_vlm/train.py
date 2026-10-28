from __future__ import annotations

import argparse
import glob
import math

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.lora import inject_lora, lora_modules
from common.utils import (
    append_jsonl,
    count_parameters,
    load_json,
    load_yaml,
    peak_vram_gib,
    read_jsonl,
    repo_path,
    rng_state,
    save_json,
    set_rng_state,
    set_seed,
    wall_timer,
    write_jsonl,
)
from task2_vlm.model import Connector, PromptBuilder, collate_right, embed_inputs, per_example_nll
from task2_vlm.prepare_data import counting_question, presence_question


#three-stage curriculum, one continuing model:
#  stage 1 caption alignment       connector only, 2 passes over 18,000 captions (2,250 updates)
#  stage 2 visual instruction      connector + new lora, draws: presence 50%, counting 25%, caption replay 25% (1,500 updates)
#  stage 3 natural vqa             same connector + lora, draws: vqa 70%, presence/counting/caption replay 10% each (750 updates)
#each draw gives two examples (a complete pair, or two captions). micro batch 8, accumulation 2 -> 16 examples per update


class Data:
    """Everything the sampler and the dev evaluation read: splits, cached features and the grounding pools."""

    def __init__(self,cfg):
        splits=load_json(cfg["data"]["splits"])
        cache=torch.load(str(repo_path(cfg["data"]["features"])),weights_only=False)
        self.index=cache["index"] #image key -> row
        self.features=cache["features"] #[N, 196, 768] fp16 on cpu
        self.splits=splits
        prompt=cfg["data"]["caption_prompt"]
        self.captions=[] #18,000 caption examples
        for row in splits["caption_train"]:
            self.captions.append({"kind":"caption","question":prompt,"target":row["caption"],"image":row["image"]})
        self.caption_dev=[]
        for row in splits["caption_dev"]:
            self.caption_dev.append({"kind":"caption","question":prompt,"target":row["caption"],"image":row["image"]})
        self.vqa_train=splits["vqa_train"]
        #presence pool: category -> images with the object, and images without it
        pools=splits["grounding_pools"]
        self.pool_images=pools["images"]
        self.presence={}
        self.counting={}
        cmin=int(cfg["data"]["count_min"])
        cmax=int(cfg["data"]["count_max"])
        for name in pools["categories"]:
            positives=[]
            negatives=[]
            by_count={}
            for image in self.pool_images:
                count=pools["counts"].get(image,{}).get(name,0)
                if count==0:
                    negatives.append(image)
                else:
                    positives.append(image) #crowd boxes (-1) still mean the object is there
                if count>=cmin and count<=cmax:
                    if count not in by_count:
                        by_count[count]=[]
                    by_count[count].append(image)
            if len(positives)>0 and len(negatives)>0:
                self.presence[name]=(positives,negatives)
            if len(by_count)>=2:
                self.counting[name]=by_count
        self.presence_names=sorted(self.presence.keys())
        self.counting_names=sorted(self.counting.keys())

    def feature_batch(self,images,device):
        rows=[]
        for image in images:
            rows.append(self.index[image])
        return self.features[rows].to(device,non_blocking=True) #[n, 196, 768] fp16


def pair_examples(pair,kind):
    examples=[]
    for example in pair["examples"]:
        examples.append({"kind":kind,"question":pair["question"],"target":example["target"],"image":example["image"]})
    return examples


def draw(data,rng,component):
    """One draw = two examples. Categories, counts, images and pairs are picked uniformly at random from the pools."""
    if component=="caption":
        first=data.captions[int(rng.integers(len(data.captions)))]
        second=data.captions[int(rng.integers(len(data.captions)))]
        return [first,second]
    elif component=="presence":
        name=data.presence_names[int(rng.integers(len(data.presence_names)))]
        positives,negatives=data.presence[name]
        question=presence_question(name)
        positive=positives[int(rng.integers(len(positives)))]
        negative=negatives[int(rng.integers(len(negatives)))]
        return [{"kind":"presence","question":question,"target":"yes","image":positive},{"kind":"presence","question":question,"target":"no","image":negative}]
    elif component=="counting":
        name=data.counting_names[int(rng.integers(len(data.counting_names)))]
        by_count=data.counting[name]
        counts=rng.choice(sorted(by_count.keys()),size=2,replace=False) #two different counts in 1..4
        examples=[]
        for count in counts:
            images=by_count[int(count)]
            examples.append({"kind":"counting","question":counting_question(name),"target":str(int(count)),"image":images[int(rng.integers(len(images)))]})
        return examples
    else:
        pair=data.vqa_train[int(rng.integers(len(data.vqa_train)))]
        return pair_examples(pair,"vqa")


def stage_lr_lambda(updates,warmup_fraction,final_fraction):
    """Per-stage 5% linear warmup, then cosine decay from the peak to 10% of it."""
    warmup=int(math.ceil(warmup_fraction*updates))
    def factor(step):
        if step<warmup:
            return (step+1)/warmup
        else:
            progress=(step-warmup)/max(1,updates-warmup)
            return final_fraction+(1.0-final_fraction)*0.5*(1.0+math.cos(math.pi*progress))
    return factor


def encode(builder,example,include_end,with_image=True):
    kind=example["kind"]
    return builder.prompt(example["question"],with_image),builder.target(example["target"],kind,include_end)


def micro_batch_loss(qwen,connector,builder,data,examples,device,effective_batch):
    """Example-weighted loss: mean over each example's supervised tokens, summed and divided by the effective batch (16)."""
    encoded=[]
    images=[]
    for example in examples:
        encoded.append(encode(builder,example,include_end=True)) #caption/answer tokens + <|im_end|>
        images.append(example["image"])
    ids,labels,mask,positions=collate_right(encoded,builder.tokenizer.pad_token_id,device)
    features=data.feature_batch(images,device)
    embeds=embed_inputs(qwen,connector,ids,features,builder.image_token_id)
    nll=per_example_nll(qwen,embeds,mask,positions,labels) #[8]
    return nll.sum()/effective_batch,nll.detach()


def reference_nll(qwen,connector,builder,data,items,image_mode,device,include_end,batch_size=8):
    """Teacher-forced mean token nll of each item's reference. image_mode: the item field holding the image
    ('image', 'partner_image', 'mismatch_image') or 'none'."""
    values=[]
    for start in range(0,len(items),batch_size):
        chunk=items[start:start+batch_size]
        encoded=[]
        images=[]
        for item in chunk:
            if image_mode=="none":
                encoded.append(encode(builder,item,include_end,with_image=False)) #visual tokens removed, masks/positions rebuilt
            else:
                encoded.append(encode(builder,item,include_end,with_image=True))
                images.append(item[image_mode])
        ids,labels,mask,positions=collate_right(encoded,builder.tokenizer.pad_token_id,device)
        features=None
        if image_mode!="none":
            features=data.feature_batch(images,device)
        with torch.no_grad():
            with torch.autocast(device_type="cuda",dtype=torch.float16):
                embeds=embed_inputs(qwen,connector,ids,features,builder.image_token_id)
                nll=per_example_nll(qwen,embeds,mask,positions,labels)
        values=values+nll.float().cpu().tolist()
    return values


def suite_items(pairs):
    """Flatten pairs into question items that know their partner image."""
    items=[]
    for pair in pairs:
        for side in range(2):
            example=pair["examples"][side]
            partner=pair["examples"][1-side]
            items.append({"kind":pair["suite"],"qtype":pair["qtype"],"pair_id":pair["pair_id"],"side":side,"question":pair["question"],
                "target":example["target"],"canonical":example["canonical"],"references":example["references"],
                "image":example["image"],"partner_image":partner["image"],"partner_canonical":partner["canonical"],
                "mismatch_image":example.get("mismatch_image")}) #mismatch image only exists for the test suites
    return items


def dev_evaluation(qwen,connector,builder,data,device,global_update,stage_name):
    """Caption reference loss, and reference-answer nll under correct / complementary / no image (termination excluded)."""
    qwen.eval()
    connector.eval()
    record={"update":global_update,"stage":stage_name}
    caption_loss=reference_nll(qwen,connector,builder,data,data.caption_dev,"image",device,include_end=True)
    record["caption_ref_loss"]=float(np.mean(caption_loss))
    for suite in ["grounding_dev","vqa_dev"]:
        items=suite_items(data.splits[suite])
        correct=reference_nll(qwen,connector,builder,data,items,"image",device,include_end=False)
        partner=reference_nll(qwen,connector,builder,data,items,"partner_image",device,include_end=False)
        no_image=reference_nll(qwen,connector,builder,data,items,"none",device,include_end=False)
        delta=np.asarray(partner)-np.asarray(correct) #manual: Delta_NLL = NLL(y|q,I_partner) - NLL(y|q,I_correct)
        record[suite]={"nll_correct":float(np.mean(correct)),"nll_complementary":float(np.mean(partner)),"nll_no_image":float(np.mean(no_image)),"delta_nll":float(delta.mean())}
        by_type={}
        for i in range(len(items)):
            qtype=items[i]["qtype"]
            if qtype not in by_type:
                by_type[qtype]={"correct":[],"partner":[],"none":[]}
            by_type[qtype]["correct"].append(correct[i])
            by_type[qtype]["partner"].append(partner[i])
            by_type[qtype]["none"].append(no_image[i])
        for qtype in by_type:
            values=by_type[qtype]
            record[suite][qtype]={"nll_correct":float(np.mean(values["correct"])),"nll_complementary":float(np.mean(values["partner"])),
                "nll_no_image":float(np.mean(values["none"])),"delta_nll":float(np.mean(np.asarray(values["partner"])-np.asarray(values["correct"])))}
    qwen.train()
    connector.train()
    return record


def calibrate_connector(qwen,connector,builder,data,cfg,device):
    """Rule (training data only): scale = mean ||text embedding|| of caption tokens / mean ||connector output|| at init,
    both over the first 512 training caption images."""
    rows=data.captions[0:int(cfg["model"]["calibration_images"])*int(cfg["data"]["captions_per_train_image"]):int(cfg["data"]["captions_per_train_image"])] #first caption of the first 512 images
    text_norms=[]
    visual_norms=[]
    with torch.no_grad():
        for start in range(0,len(rows),64):
            chunk=rows[start:start+64]
            for row in chunk:
                ids=torch.tensor(builder.target(row["target"],"caption",include_end=False),device=device)
                text_norms.append(qwen.get_input_embeddings()(ids).float().norm(dim=-1)) #per token norms
            images=[]
            for row in chunk:
                images.append(row["image"])
            features=data.feature_batch(images,device)
            visual_norms.append(connector(features).float().norm(dim=-1).flatten()) #per visual token norms
    text_mean=float(torch.cat(text_norms).mean().item())
    visual_mean=float(torch.cat(visual_norms).mean().item())
    scale=text_mean/visual_mean
    connector.output_scale.fill_(scale)
    return {"rule":"output_scale = mean text-embedding norm / mean initial connector-output norm, first 512 caption training images","images":len(rows),
        "text_embedding_norm":text_mean,"connector_output_norm_before":visual_mean,"scale":scale}


def interface_checks(qwen,connector,builder,data,cfg,device):
    """Manual step 1: feature shapes, frozen parameter sets, connector gradients, split isolation."""
    checks={}
    checks["feature_shape"]=list(data.features.shape[1:]) #[196, 768]
    frozen_with_grad=0
    for p in qwen.parameters():
        if p.requires_grad:
            frozen_with_grad=frozen_with_grad+1
    checks["qwen_trainable_tensors"]=frozen_with_grad #0 before lora
    examples=data.captions[:2]
    with torch.autocast(device_type="cuda",dtype=torch.float16):
        loss,_=micro_batch_loss(qwen,connector,builder,data,examples,device,2)
    loss.backward()
    grad_norm=0.0
    for p in connector.parameters():
        if p.grad is not None:
            grad_norm=grad_norm+float(p.grad.float().norm().item())**2
    checks["connector_grad_norm"]=grad_norm**0.5 #must be > 0: gradients reach the connector through frozen qwen
    qwen_grads=0
    for p in qwen.parameters():
        if p.grad is not None:
            qwen_grads=qwen_grads+1
    checks["qwen_params_with_grad"]=qwen_grads #must be 0
    connector.zero_grad(set_to_none=True)
    ids=builder.prompt("test question",with_image=True)
    checks["visual_tokens_in_prompt"]=ids.count(builder.image_token_id)
    checks["prompt_with_image"]=builder.tokenizer.decode(ids)
    checks["prompt_without_image"]=builder.tokenizer.decode(builder.prompt("test question",with_image=False))
    checks["split_isolation"]=load_json(repo_path(cfg["results_dir"])/"data_summary.json")["isolation"] #written by prepare_data
    return checks


def build_optimizer(connector,qwen,stage_cfg,train_cfg):
    groups=[{"params":list(connector.parameters()),"lr":float(stage_cfg["connector_lr"])}]
    if stage_cfg["lora_lr"] is not None:
        lora_params=[]
        for module in lora_modules(qwen).values():
            lora_params.append(module.lora_A)
            lora_params.append(module.lora_B)
        groups.append({"params":lora_params,"lr":float(stage_cfg["lora_lr"])})
    optimizer=torch.optim.AdamW(groups,weight_decay=float(train_cfg["weight_decay"]))
    factor=stage_lr_lambda(int(stage_cfg["updates"]),float(train_cfg["warmup_fraction"]),float(train_cfg["final_lr_fraction"]))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,[factor]*len(groups))
    return optimizer,scheduler


def adapter_weights(connector,qwen):
    connector_state={}
    for key,value in connector.state_dict().items():
        connector_state[key]=value.detach().cpu()
    state={"connector":connector_state}
    lora={}
    for name,module in lora_modules(qwen).items():
        lora[name+".lora_A"]=module.lora_A.detach().cpu()
        lora[name+".lora_B"]=module.lora_B.detach().cpu()
    state["lora"]=lora
    return state


def train(config_path,visual_tokens,resume):
    cfg=load_yaml(config_path)
    train_cfg=cfg["train"]
    seed=int(cfg["seed"])
    set_seed(seed)
    device="cuda"
    run_name="tokens"+str(visual_tokens) #tokens196 main run, tokens49 optional
    out_dir=repo_path(cfg["outputs_dir"])/run_name
    results_dir=repo_path(cfg["results_dir"])/run_name
    out_dir.mkdir(parents=True,exist_ok=True)

    data=Data(cfg)
    tokenizer=AutoTokenizer.from_pretrained(cfg["language_model"])
    builder=PromptBuilder(tokenizer,visual_tokens,cfg["caps"])
    qwen=AutoModelForCausalLM.from_pretrained(cfg["language_model"],torch_dtype=torch.float16).to(device) #frozen fp16 backbone
    for p in qwen.parameters():
        p.requires_grad_(False)
    qwen.config.use_cache=False
    qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False}) #non-reentrant: grads reach the connector without input grads
    torch.manual_seed(seed)
    connector=Connector(int(cfg["model"]["vision_dim"]),int(cfg["model"]["hidden_dim"]),visual_tokens==49).to(device) #fp32 trainable
    lora_cfg=train_cfg["lora"]
    stages=train_cfg["stages"]
    micro=int(train_cfg["micro_batch"]) #8
    accum=int(train_cfg["grad_accum"]) #2
    effective=micro*accum #16
    stage_ends=[]
    total=0
    for stage_cfg in stages:
        total=total+int(stage_cfg["updates"])
        stage_ends.append(total) #2250, 3750, 4500

    #state that a resume restores
    stage_index=0
    step_in_stage=0
    sampler_state=None
    resume_state=None
    seconds_before=0.0
    if resume==True:
        files=sorted(glob.glob(str(out_dir/"checkpoint_u*.pt")))
        if len(files)>0:
            resume_state=torch.load(files[-1],map_location="cpu",weights_only=False)
            stage_index=int(resume_state["stage_index"])
            step_in_stage=int(resume_state["step_in_stage"])
            sampler_state=resume_state["sampler"]
            seconds_before=float(resume_state["seconds"])
            connector.load_state_dict(resume_state["connector"])
            print("resuming from",files[-1])

    log_path=results_dir/"train_log.jsonl"
    dev_path=results_dir/"dev_eval.jsonl"
    if resume_state is not None:
        for path in [log_path,dev_path]:
            if path.exists():
                kept=[]
                for record in read_jsonl(path):
                    if record["update"]<=resume_state["global_update"]:
                        kept.append(record) #lines written after the checkpoint are redone now
                write_jsonl(path,kept)
    if resume_state is None:
        for path in [log_path,dev_path]:
            if path.exists():
                path.unlink()
        save_json(results_dir/"calibration.json",calibrate_connector(qwen,connector,builder,data,cfg,device))
        save_json(results_dir/"interface_checks.json",interface_checks(qwen,connector,builder,data,cfg,device))
        save_json(results_dir/"config_used.json",cfg)
        append_jsonl(dev_path,dev_evaluation(qwen,connector,builder,data,device,0,"before_training")) #manual: evaluate before training

    stage_stats=[]
    if resume_state is not None and "stage_stats" in resume_state:
        stage_stats=resume_state["stage_stats"]
    session_timer=wall_timer()
    for s in range(stage_index,len(stages)):
        stage_cfg=stages[s]
        if s>=1 and len(lora_modules(qwen))==0:
            torch.manual_seed(seed+s) #seeded adapter init
            inject_lora(qwen,lora_cfg["targets"],int(lora_cfg["rank"]),float(lora_cfg["alpha"]),float(lora_cfg["dropout"])) #new lora from stage 2 on, backbone stays frozen
        if resume_state is not None and s==stage_index and len(resume_state["lora"])>0:
            for name,module in lora_modules(qwen).items():
                module.lora_A.data.copy_(resume_state["lora"][name+".lora_A"].to(device))
                module.lora_B.data.copy_(resume_state["lora"][name+".lora_B"].to(device))
        optimizer,scheduler=build_optimizer(connector,qwen,stage_cfg,train_cfg)
        scaler=torch.amp.GradScaler("cuda") #dynamic loss scaling, fp32 trainables
        rng=np.random.default_rng(seed+1000*(s+1)) #seeded draw sampler for this stage
        start_step=0
        if resume_state is not None and s==stage_index and step_in_stage>0: #a checkpoint at a stage end resumes into a fresh stage
            optimizer.load_state_dict(resume_state["optimizer"])
            scheduler.load_state_dict(resume_state["scheduler"])
            scaler.load_state_dict(resume_state["scaler"])
            rng.bit_generator.state=sampler_state
            set_rng_state(resume_state["rng"])
            start_step=step_in_stage
        trainable=list(connector.parameters())
        for module in lora_modules(qwen).values():
            trainable.append(module.lora_A)
            trainable.append(module.lora_B)
        stage_trainable=count_parameters(trainable)
        torch.cuda.reset_peak_memory_stats()
        stage_timer=wall_timer()
        updates=int(stage_cfg["updates"])
        offset=stage_ends[s]-updates #global updates before this stage
        if s==0:
            first_pass=np.random.default_rng(seed).permutation(len(data.captions)) #two passes, each a seeded shuffle
            second_pass=np.random.default_rng(seed+1).permutation(len(data.captions))
            caption_order=np.concatenate([first_pass,second_pass])
        qwen.train()
        connector.train()
        for step in range(start_step,updates):
            #16 examples for this update
            examples=[]
            mixture={}
            if s==0:
                for position in caption_order[step*effective:(step+1)*effective]:
                    examples.append(data.captions[int(position)])
                mixture["caption"]=effective
            else:
                names=sorted(stage_cfg["mixture"].keys())
                probabilities=[]
                for name in names:
                    probabilities.append(float(stage_cfg["mixture"][name]))
                for d in range(effective//2): #8 draws, two examples each
                    component=names[int(rng.choice(len(names),p=probabilities))]
                    mixture[component]=mixture.get(component,0)+2
                    examples=examples+draw(data,rng,component)
            losses=[]
            for m in range(accum):
                with torch.autocast(device_type="cuda",dtype=torch.float16):
                    loss,nll=micro_batch_loss(qwen,connector,builder,data,examples[m*micro:(m+1)*micro],device,effective)
                scaler.scale(loss).backward()
                losses=losses+nll.float().cpu().tolist()
            scaler.unscale_(optimizer) #manual: unscale before clipping
            grad_norm=torch.nn.utils.clip_grad_norm_(trainable,float(train_cfg["max_grad_norm"]))
            lrs=scheduler.get_last_lr()
            scale_before=scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped=scaler.get_scale()<scale_before
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_update=offset+step+1
            lora_lr=None #stage 1 has no lora
            if len(lrs)>1:
                lora_lr=lrs[1]
            record={"update":global_update,"stage":stage_cfg["name"],"loss":float(np.mean(losses)),"connector_lr":lrs[0],
                "lora_lr":lora_lr,"grad_norm":float(grad_norm),"skipped":bool(skipped),"mixture":mixture,
                "seconds":seconds_before+session_timer()}
            append_jsonl(log_path,record)
            if global_update%50==0:
                print(record)

            is_stage_end=(step+1==updates)
            if global_update%int(train_cfg["dev_eval_every"])==0 or is_stage_end:
                append_jsonl(dev_path,dev_evaluation(qwen,connector,builder,data,device,global_update,stage_cfg["name"]))
            if is_stage_end:
                torch.save(adapter_weights(connector,qwen),str(out_dir/("stage"+str(s+1)+"_weights.pt"))) #connector/adapter weights after each stage
                stage_stats.append({"stage":stage_cfg["name"],"updates":updates,"trainable_parameters":stage_trainable,
                    "seconds":stage_timer(),"peak_vram_gib":peak_vram_gib(),"examples":updates*effective})
            if global_update%int(train_cfg["checkpoint_every"])==0 or is_stage_end:
                next_stage=s
                next_step=step+1
                if is_stage_end:
                    next_stage=s+1
                    next_step=0
                state=adapter_weights(connector,qwen)
                state.update({"stage_index":next_stage,"step_in_stage":next_step,"global_update":global_update,
                    "optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),
                    "sampler":rng.bit_generator.state,"rng":rng_state(),"seconds":seconds_before+session_timer(),"stage_stats":stage_stats})
                torch.save(state,str(out_dir/("checkpoint_u"+str(global_update).zfill(4)+".pt")))
        resume_state=None #later stages start fresh

    summary={"run":run_name,"visual_tokens":visual_tokens,"stages":stage_stats,"total_updates":stage_ends[-1],
        "example_presentations":stage_ends[-1]*effective,"train_seconds":seconds_before+session_timer(),"gpu":torch.cuda.get_device_name(0)}
    save_json(results_dir/"train_summary.json",summary)
    print(summary)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task2.yaml")
    ap.add_argument("--visual-tokens",type=int,default=196,choices=[196,49])
    ap.add_argument("--resume",action="store_true")
    args=ap.parse_args()
    train(args.config,args.visual_tokens,args.resume)


if __name__=="__main__":
    main()
