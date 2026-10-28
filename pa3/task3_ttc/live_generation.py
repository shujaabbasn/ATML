from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.utils import append_jsonl, load_json, load_yaml, read_jsonl, repo_path, save_json
from task3_ttc.answers import extract_last_boxed, math_correct, math_equivalent, math_parse
from task3_ttc.offline_eval import load_prepared


#step 4: on the fixed 50 math evaluation questions, same 3,072-token total output allowance per question
#  single      one response, cap 3,072
#  parallel    three independent responses, cap 1,024 each, self-consistency picks the answer
#  refinement  first parallel response (cap 1,024) + one critique-and-revision response (cap 2,048), revised answer returned
#fp16, no quantization, batch size 1, native chat template. weights fixed, gold answers only used for grading afterwards


def generate_once(model,tokenizer,messages,cap,seed,cfg):
    live=cfg["live"]
    text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs=tokenizer(text,return_tensors="pt",add_special_tokens=False).to(model.device)
    torch.manual_seed(seed) #independently seeded call
    torch.cuda.synchronize()
    start=time.perf_counter()
    with torch.no_grad():
        output=model.generate(**inputs,max_new_tokens=cap,do_sample=True,temperature=float(live["temperature"]),top_p=float(live["top_p"]),
            top_k=int(live["top_k"]),repetition_penalty=float(live["repetition_penalty"]),pad_token_id=tokenizer.pad_token_id)
    torch.cuda.synchronize()
    seconds=time.perf_counter()-start
    new_tokens=output[0,inputs["input_ids"].shape[1]:].tolist()
    stop_ids=[tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids("<|im_end|>"),tokenizer.convert_tokens_to_ids("<|endoftext|>")]
    stopped=False
    retained=[]
    for token in new_tokens:
        if token in stop_ids:
            stopped=True
            break
        retained.append(token)
    response=tokenizer.decode(retained,skip_special_tokens=True)
    answer=extract_last_boxed(response)
    return {"seed":seed,"input_tokens":int(inputs["input_ids"].shape[1]),"output_tokens":len(new_tokens),"retained_tokens":len(retained),
        "truncated":bool(stopped==False and len(new_tokens)>=cap),"seconds":seconds,"response":response,"answer":answer,"cap":cap}


def majority_answer(answers):
    """Self-consistency over the three parallel answers with the same equivalence rule; ties -> earliest. None = NO_ANSWER."""
    groups=[] #(representative index, members)
    for i in range(len(answers)):
        placed=False
        for group in groups:
            first=answers[group[0]]
            if (first is None and answers[i] is None) or (first is not None and answers[i] is not None and math_equivalent(first,answers[i],math_parse(first),math_parse(answers[i]))):
                group.append(i)
                placed=True
                break
        if placed==False:
            groups.append([i])
    best=0
    for g in range(1,len(groups)):
        if len(groups[g])>len(groups[best]):
            best=g
    return answers[groups[best][0]]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task3.yaml")
    args=ap.parse_args()
    cfg=load_yaml(args.config)
    live=cfg["live"]
    seed=int(cfg["seed"])
    results_dir=repo_path(cfg["results_dir"])/"live"
    out_path=results_dir/"generations.jsonl"
    prepared=load_prepared(cfg,"math")
    questions={}
    for q in prepared["questions"]:
        questions[q["sample_id"]]=q
    live_ids=load_json(repo_path(cfg["results_dir"])/"splits"/"math_splits.json")["live_50"]

    done=set() #resume: skip questions already written
    if out_path.exists():
        for row in read_jsonl(out_path):
            done.add(row["sample_id"])

    tokenizer=AutoTokenizer.from_pretrained(live["model"],revision=live["revision"])
    model=AutoModelForCausalLM.from_pretrained(live["model"],revision=live["revision"],torch_dtype=torch.float16).to("cuda") #fp16, no quantization
    model.eval()
    torch.cuda.reset_peak_memory_stats()

    for index in range(len(live_ids)):
        sample_id=live_ids[index]
        if sample_id in done:
            continue
        q=questions[sample_id]
        user=q["question_text"]+"\n\n"+live["instruction"] #problem, two newlines, instruction (supplied README)
        base_messages=[{"role":"system","content":live["system"]},{"role":"user","content":user}]
        call_seed=seed*1000+index*10 #seeds 6304000 + 10*question index + call number, saved with every call

        single=generate_once(model,tokenizer,base_messages,int(live["single_cap"]),call_seed+0,cfg)
        parallel=[]
        for k in range(int(live["parallel_samples"])):
            parallel.append(generate_once(model,tokenizer,base_messages,int(live["parallel_cap"]),call_seed+1+k,cfg))
        initial=parallel[0] #shared initial response, reused for refinement
        review_messages=base_messages+[{"role":"assistant","content":initial["response"]},{"role":"user","content":live["review"]}]
        revision=generate_once(model,tokenizer,review_messages,int(live["revision_cap"]),call_seed+5,cfg)

        parallel_answers=[]
        parallel_output=0
        parallel_input=0
        parallel_seconds=0.0
        for p in parallel:
            parallel_answers.append(p["answer"])
            parallel_output=parallel_output+p["output_tokens"]
            parallel_input=parallel_input+p["input_tokens"]
            parallel_seconds=parallel_seconds+p["seconds"]
        parallel_answer=majority_answer(parallel_answers)
        gold=q["gold"]
        row={
            "sample_id":sample_id,"gold":gold,"prompt":user,
            "single":single,"parallel":parallel,"revision":revision,
            "answers":{"single":single["answer"],"parallel":parallel_answer,"initial":initial["answer"],"refinement":revision["answer"]},
            "correct":{"single":math_correct(gold,single["answer"]),"parallel":math_correct(gold,parallel_answer),
                "initial":math_correct(gold,initial["answer"]),"refinement":math_correct(gold,revision["answer"])},
            "output_tokens":{"single":single["output_tokens"],"parallel":parallel_output,
                "refinement":initial["output_tokens"]+revision["output_tokens"]}, #shared initial response charged to both
            "input_tokens":{"single":single["input_tokens"],"parallel":parallel_input,
                "refinement":initial["input_tokens"]+revision["input_tokens"]}, #the revision prompt is longer, counted in full
            "seconds":{"single":single["seconds"],"parallel":parallel_seconds,"refinement":initial["seconds"]+revision["seconds"]},
        }
        append_jsonl(out_path,row)
        print(index,sample_id,row["correct"],row["output_tokens"])

    #summary over the 50 questions
    rows=read_jsonl(out_path)
    summary={"questions":len(rows),"peak_vram_gib_this_session":torch.cuda.max_memory_allocated()/2**30,"gpu":torch.cuda.get_device_name(0),
        "model":live["model"],"revision":live["revision"]}
    for strategy in ["single","parallel","refinement"]:
        correct=[]
        truncated=[] #1 if any call of the strategy hit its cap
        missing=[] #1 if there is no extractable final answer
        output_tokens=[]
        input_tokens=[]
        seconds=[]
        for r in rows:
            if strategy=="single":
                calls=[r["single"]]
            elif strategy=="parallel":
                calls=r["parallel"]
            else:
                calls=[r["parallel"][0],r["revision"]]
            hit_cap=0.0
            for call in calls:
                if call["truncated"]:
                    hit_cap=1.0
            truncated.append(hit_cap)
            correct.append(float(r["correct"][strategy]))
            missing.append(float(r["answers"][strategy] is None))
            output_tokens.append(r["output_tokens"][strategy])
            input_tokens.append(r["input_tokens"][strategy])
            seconds.append(r["seconds"][strategy])
        summary[strategy]={"accuracy":float(np.mean(correct)),"mean_output_tokens":float(np.mean(output_tokens)),"mean_input_tokens":float(np.mean(input_tokens)),
            "mean_seconds":float(np.mean(seconds)),"truncation_rate":float(np.mean(truncated)),"no_final_answer_rate":float(np.mean(missing)),"extraction_rate":1.0-float(np.mean(missing))}
    transitions={"wrong_to_correct":0,"correct_to_wrong":0,"stayed_correct":0,"stayed_wrong":0}
    for r in rows:
        before=r["correct"]["initial"]
        after=r["correct"]["refinement"]
        if before==False and after==True:
            transitions["wrong_to_correct"]=transitions["wrong_to_correct"]+1
        elif before==True and after==False:
            transitions["correct_to_wrong"]=transitions["correct_to_wrong"]+1
        elif before==True:
            transitions["stayed_correct"]=transitions["stayed_correct"]+1
        else:
            transitions["stayed_wrong"]=transitions["stayed_wrong"]+1
    summary["refinement_transitions_vs_shared_initial"]=transitions
    initial=[]
    for r in rows:
        initial.append(float(r["correct"]["initial"]))
    summary["initial_accuracy"]=float(np.mean(initial))
    save_json(results_dir/"summary.json",summary)
    print(summary)


if __name__=="__main__":
    main()
