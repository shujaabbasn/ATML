from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.lora import inject_lora, lora_modules
from common.utils import load_yaml, repo_path, save_json, set_seed, write_jsonl
from task2_vlm.model import Connector, PromptBuilder, greedy_generate
from task2_vlm.train import Data, reference_nll, suite_items
from task2_vlm.vqa_eval import normalize_answer, vqa_soft_accuracy


#final evaluation of the completed stage 3 model on both test suites (manual: final evaluation)
#greedy, thinking disabled, 16-new-token cap, batch 8, same question and references under four image conditions

CONDITIONS={"correct":"image","complementary_swap":"partner_image","mismatched":"mismatch_image","no_image":"none"} #condition -> item field


def load_final_model(cfg,visual_tokens,device):
    tokenizer=AutoTokenizer.from_pretrained(cfg["language_model"])
    qwen=AutoModelForCausalLM.from_pretrained(cfg["language_model"],torch_dtype=torch.float16).to(device)
    for p in qwen.parameters():
        p.requires_grad_(False)
    lora_cfg=cfg["train"]["lora"]
    inject_lora(qwen,lora_cfg["targets"],int(lora_cfg["rank"]),float(lora_cfg["alpha"]),0.0) #dropout off for evaluation
    weights=torch.load(str(repo_path(cfg["outputs_dir"])/("tokens"+str(visual_tokens))/"stage3_weights.pt"),map_location="cpu",weights_only=False)
    for name,module in lora_modules(qwen).items():
        module.lora_A.data.copy_(weights["lora"][name+".lora_A"].to(device))
        module.lora_B.data.copy_(weights["lora"][name+".lora_B"].to(device))
    connector=Connector(int(cfg["model"]["vision_dim"]),int(cfg["model"]["hidden_dim"]),visual_tokens==49).to(device)
    connector.load_state_dict(weights["connector"])
    qwen.eval()
    connector.eval()
    return tokenizer,qwen,connector


def generate_answers(qwen,connector,builder,data,items,field,device,batch_size,max_new_tokens):
    stop_ids=[builder.end_id,builder.tokenizer.eos_token_id]
    predictions=[]
    seconds=0.0
    for start in range(0,len(items),batch_size):
        chunk=items[start:start+batch_size]
        prompts=[]
        images=[]
        for item in chunk:
            prompts.append(builder.prompt(item["question"],with_image=(field!="none")))
            if field!="none":
                images.append(item[field])
        features=None
        if field!="none":
            features=data.feature_batch(images,device)
        torch.cuda.synchronize()
        begin=time.perf_counter()
        with torch.autocast(device_type="cuda",dtype=torch.float16):
            tokens,finished=greedy_generate(qwen,connector,prompts,features,builder.image_token_id,builder.tokenizer.pad_token_id,stop_ids,max_new_tokens)
        torch.cuda.synchronize()
        seconds=seconds+time.perf_counter()-begin
        for i in range(len(chunk)):
            predictions.append({"text":builder.tokenizer.decode(tokens[i],skip_special_tokens=True).strip(),"tokens":len(tokens[i]),"stopped":bool(finished[i])})
    return predictions,seconds/len(items)


def score(item,prediction):
    """Grounding: normalized exact match against the canonical answer. Natural vqa: soft consensus accuracy over 10 answers."""
    if item["kind"]=="grounding":
        return float(normalize_answer(prediction)==item["canonical"])
    else:
        return vqa_soft_accuracy(prediction,item["references"])


def pair_table(rows):
    """pair_id -> [side 0 row, side 1 row] for one condition."""
    pairs={}
    for row in rows:
        if row["pair_id"] not in pairs:
            pairs[row["pair_id"]]=[None,None]
        pairs[row["pair_id"]][row["side"]]=row
    return pairs


def condition_metrics(rows):
    pairs=pair_table(rows)
    accuracy=[]
    nll=[]
    empty=0
    for row in rows:
        accuracy.append(row["score"])
        nll.append(row["nll"])
        if row["normalized"]=="":
            empty=empty+1 #empty outputs stay in the denominators
    both_correct=0
    identical=0
    for pair_id in pairs:
        first,second=pairs[pair_id]
        if first["normalized"]==first["canonical"] and second["normalized"]==second["canonical"]:
            both_correct=both_correct+1 #pair accuracy: both match their original canonical answers
        if first["normalized"]==second["normalized"]:
            identical=identical+1 #same answer for two images with different answers
    metrics={"n_questions":len(rows),"n_pairs":len(pairs),"accuracy":float(np.mean(accuracy)),"pair_accuracy":both_correct/len(pairs),
        "identical_answer_rate":identical/len(pairs),"empty_answer_rate":empty/len(rows),"mean_reference_nll":float(np.mean(nll))}
    by_type={}
    for row in rows:
        if row["qtype"] not in by_type:
            by_type[row["qtype"]]=[]
        by_type[row["qtype"]].append(row["score"])
    metrics["accuracy_by_type"]={}
    for qtype in by_type:
        metrics["accuracy_by_type"][qtype]=float(np.mean(by_type[qtype]))
    follows=0
    for row in rows:
        if row["normalized"]==row["partner_canonical"]:
            follows=follows+1
    metrics["matches_partner_answer_rate"]=follows/len(rows) #under the swap: did the answer follow the partner image
    return metrics


def bootstrap_difference(rows_a,rows_b,key,iterations,rng):
    """95% interval of mean(rows_a[key]) - mean(rows_b[key]), resampling complete pairs with the same pair ids in both."""
    pairs_a=pair_table(rows_a)
    pairs_b=pair_table(rows_b)
    pair_ids=sorted(pairs_a.keys())
    per_pair=[] #pair-level difference = mean over the two questions
    for pair_id in pair_ids:
        a=(pairs_a[pair_id][0][key]+pairs_a[pair_id][1][key])/2
        b=(pairs_b[pair_id][0][key]+pairs_b[pair_id][1][key])/2
        per_pair.append(a-b)
    per_pair=np.asarray(per_pair)
    means=[]
    for i in range(iterations):
        sample=rng.integers(len(per_pair),size=len(per_pair)) #same sampled pair ids for both conditions
        means.append(per_pair[sample].mean())
    return {"estimate":float(per_pair.mean()),"ci95_low":float(np.percentile(means,2.5)),"ci95_high":float(np.percentile(means,97.5)),"iterations":iterations}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task2.yaml")
    ap.add_argument("--visual-tokens",type=int,default=196,choices=[196,49])
    args=ap.parse_args()
    cfg=load_yaml(args.config)
    set_seed(int(cfg["seed"]))
    device="cuda"
    eval_cfg=cfg["eval"]
    data=Data(cfg)
    tokenizer,qwen,connector=load_final_model(cfg,args.visual_tokens,device)
    builder=PromptBuilder(tokenizer,args.visual_tokens,cfg["caps"])
    results_dir=repo_path(cfg["results_dir"])/("tokens"+str(args.visual_tokens))/"final_eval"
    rng=np.random.default_rng(int(cfg["seed"]))

    all_metrics={}
    for suite in ["grounding_test","vqa_test"]:
        items=suite_items(data.splits[suite])
        rows_by_condition={}
        all_metrics[suite]={}
        for condition,field in CONDITIONS.items():
            predictions,seconds_per_answer=generate_answers(qwen,connector,builder,data,items,field,device,int(eval_cfg["batch_size"]),int(eval_cfg["max_new_tokens"]))
            nll=reference_nll(qwen,connector,builder,data,items,field,device,include_end=False)
            rows=[]
            for i in range(len(items)):
                item=items[i]
                rows.append({"suite":suite,"condition":condition,"pair_id":item["pair_id"],"side":item["side"],"qtype":item["qtype"],
                    "question":item["question"],"image":item.get(field),"own_image":item["image"],
                    "reference":item["target"],"canonical":item["canonical"],"references":item["references"],"partner_canonical":item["partner_canonical"],
                    "prediction":predictions[i]["text"],"normalized":normalize_answer(predictions[i]["text"]),"new_tokens":predictions[i]["tokens"],
                    "stopped":predictions[i]["stopped"],"score":score(item,predictions[i]["text"]),"nll":nll[i],"kind":item["kind"]})
            rows_by_condition[condition]=rows
            metrics=condition_metrics(rows)
            metrics["generation_seconds_per_answer"]=seconds_per_answer
            all_metrics[suite][condition]=metrics
            write_jsonl(results_dir/(suite+"_"+condition+".jsonl"),rows)
            print(suite,condition,metrics)
        #correct-image advantage over the complementary swap, in accuracy and in nll
        all_metrics[suite]["advantage_accuracy_correct_minus_swap"]=bootstrap_difference(rows_by_condition["correct"],rows_by_condition["complementary_swap"],"score",int(eval_cfg["bootstrap"]),rng)
        all_metrics[suite]["advantage_nll_swap_minus_correct"]=bootstrap_difference(rows_by_condition["complementary_swap"],rows_by_condition["correct"],"nll",int(eval_cfg["bootstrap"]),rng)
        for condition in CONDITIONS:
            if condition!="correct":
                all_metrics[suite]["accuracy_correct_minus_"+condition]=bootstrap_difference(rows_by_condition["correct"],rows_by_condition[condition],"score",int(eval_cfg["bootstrap"]),rng)
    save_json(results_dir/"metrics.json",all_metrics)


if __name__=="__main__":
    main()
