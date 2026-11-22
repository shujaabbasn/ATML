"""Steps 2, 6, 7: held-out speculative decoding, standalone answer quality, the fixed-64 throughput diagnostic and the cost model.

python -m task3_speculative.spec_eval --mode speed --drafts original off_policy_s6304 ...       primary EOS-aware benchmark, 64-token cap
python -m task3_speculative.spec_eval --mode quality --drafts original off_policy_s6304 ...     answer quality, caps 512 math / 32 short
python -m task3_speculative.spec_eval --mode throughput --drafts original off_policy_s6304 ...  fixed 64 tokens, EOS ignored, 24 math prompts + cost model
python -m task3_speculative.spec_eval --mode unmerged                                          one adapter timed unmerged (wrapper overhead only)
python -m task3_speculative.spec_eval --mode summary

Every condition uses the same 120 prompts, the same chat template, the same EOS stopping and the same per-prompt
seeds (seed = 6304 + 1000*prompt_index + s, s in 0..seeds_per_prompt-1), so comparisons are paired.
In the speed benchmark every speculative run has an adjacent target-only run on the same prompt and seed, and the
order of the two alternates (counterbalanced) from one (prompt, seed) to the next.
Cost model (manual): S_hat = (1 + sum_{j=1..gamma} alpha^j) / (gamma c + v_gamma), c = draft step / target step,
v_gamma = target verification forward / target step. Break-even alpha = smallest alpha with 1 + sum alpha^j > gamma c + v_gamma.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from common.utils import load_yaml, save_json, load_json, read_jsonl, append_jsonl, repo_path, get_device, sync, time_call, summarize_times, bootstrap_mean_ci, reset_peak, peak_mb, package_versions
from task3_speculative.common import load_causal, load_tokenizer, eos_ids, chat_ids, test_prompts, is_correct, cap_for, last_number
from task3_speculative.sampler import speculative_generate, plain_generate, make_generator, CachedModel


def draft_path(config,name):
    if name=="original":
        return config["draft"]
    return str(repo_path(config["results_dir"])/"drafts"/name/"merged")


def prompt_seed(config,index,s):
    return config["seed"]+1000*index+s


def spec_fields(stats):
    """Everything a speculative row keeps from the sampler stats."""
    row={}
    for key in ["rounds","proposed","examined","accepted","target_forwards","draft_forwards","accepted_per_round","proposed_per_round",
        "prefill_time","draft_time","verify_time","accept_time","rollback_time","host_overhead_time","total_time"]:
        row[key]=stats[key]
    return row


def timed_run(function):
    """Run one generation with synchronized wall time and peak allocated memory."""
    reset_peak()
    sync()
    t0=time.perf_counter()
    response,stats=function()
    sync()
    return response,stats,time.perf_counter()-t0,peak_mb()


#---------------- primary EOS-aware speed benchmark ----------------

@torch.no_grad()
def speed(config,drafts):
    """Manual step 2 and 6: 64-token cap with EOS, gamma 2 and 4, two paired seeds, adjacent target-only run."""
    device=get_device()
    evaluation=config["evaluation"]
    tokenizer=load_tokenizer(config)
    eos_list=eos_ids(tokenizer)
    target=load_causal(config["target"])
    prompts=test_prompts(config)
    out_dir=repo_path(config["results_dir"])/"speed"
    save_json(repo_path(config["results_dir"])/"config_used.json",config) #every reported hyperparameter
    cap=evaluation["speed_cap"]
    for name in drafts:
        draft=load_causal(draft_path(config,name))
        for gamma in evaluation["gammas"]:
            condition=f"spec_{name}_g{gamma}"
            path=out_dir/f"{condition}.jsonl"
            if path.exists():
                print("done already",condition)
                continue
            part=out_dir/f"{condition}.jsonl.part" #rows go here first, renamed only when every prompt is done
            if part.exists():
                part.unlink() #left by a session that was cut off, redo the whole condition
            for index in range(len(prompts)):
                item=prompts[index]
                prompt_ids=chat_ids(tokenizer,item["prompt"])
                for s in range(evaluation["seeds_per_prompt"]):
                    seed=prompt_seed(config,index,s)

                    def run_spec():
                        return speculative_generate(target,draft,prompt_ids,gamma,cap,eos_list,make_generator(device,seed))

                    def run_target():
                        return plain_generate(target,prompt_ids,cap,eos_list,make_generator(device,seed))

                    if (index+s)%2==0: #counterbalanced order
                        response,stats,seconds,peak=timed_run(run_spec)
                        target_response,target_stats,target_seconds,target_peak=timed_run(run_target)
                        order="spec_first"
                    else:
                        target_response,target_stats,target_seconds,target_peak=timed_run(run_target)
                        response,stats,seconds,peak=timed_run(run_spec)
                        order="target_first"
                    row={"condition":condition,"draft":name,"gamma":gamma,"id":item["id"],"domain":item["domain"],"subset":item["subset"],"seed_index":s,"order":order,
                        "response":tokenizer.decode(response,skip_special_tokens=True),"tokens":len(response),"ended_eos":len(response)>0 and response[-1] in eos_list,
                        "seconds":seconds,"tokens_per_second":len(response)/seconds,"peak_mb":peak,
                        "target_tokens":len(target_response),"target_ended_eos":len(target_response)>0 and target_response[-1] in eos_list,
                        "target_seconds":target_seconds,"target_tokens_per_second":len(target_response)/target_seconds,"target_peak_mb":target_peak,
                        "target_prefill_time":target_stats["prefill_time"],"target_decode_time":target_stats["decode_time"]}
                    row.update(spec_fields(stats))
                    append_jsonl(part,row)
                if index%20==0:
                    print(condition,index,"/",len(prompts),flush=True)
            part.rename(path) #complete, so a rerun can skip it
        del draft
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


#---------------- answer quality (standalone models) ----------------

def answer_flags(item,text,tokens,ended_eos,cap):
    """Correct, extractable (math: a final number found; short: a non-empty answer) and truncated (hit the cap, no EOS)."""
    if item["domain"]=="math":
        extractable=last_number(text) is not None
    else:
        extractable=len(text.strip())>0
    truncated=(tokens>=cap and ended_eos==False)
    return {"correct":is_correct(item,text),"extractable":extractable,"truncated":truncated}


@torch.no_grad()
def quality(config,drafts):
    """Manual step 7: target-only and each standalone student at the answer caps (512 math, 32 short), two seeds.
    Speculative outputs follow the target law, so they are not used to rank student accuracy."""
    device=get_device()
    evaluation=config["evaluation"]
    tokenizer=load_tokenizer(config)
    eos_list=eos_ids(tokenizer)
    prompts=test_prompts(config)
    out_dir=repo_path(config["results_dir"])/"quality"
    models=[("target_only",config["target"])]
    for name in drafts:
        models.append((f"student_only_{name}",draft_path(config,name)))
    for condition,path_name in models:
        path=out_dir/f"{condition}.jsonl"
        if path.exists():
            print("done already",condition)
            continue
        part=out_dir/f"{condition}.jsonl.part" #rows go here first, renamed only when every prompt is done
        if part.exists():
            part.unlink() #left by a session that was cut off, redo the whole condition
        model=load_causal(path_name)
        for index in range(len(prompts)):
            item=prompts[index]
            prompt_ids=chat_ids(tokenizer,item["prompt"])
            cap=cap_for(item,evaluation)
            for s in range(evaluation["seeds_per_prompt"]):
                generator=make_generator(device,prompt_seed(config,index,s))
                response,stats=plain_generate(model,prompt_ids,cap,eos_list,generator)
                text=tokenizer.decode(response,skip_special_tokens=True)
                ended_eos=len(response)>0 and response[-1] in eos_list
                row={"condition":condition,"id":item["id"],"domain":item["domain"],"subset":item["subset"],"seed_index":s,"cap":cap,"response":text,"tokens":len(response),"ended_eos":ended_eos}
                row.update(answer_flags(item,text,len(response),ended_eos,cap))
                append_jsonl(part,row)
        part.rename(path) #complete, so a rerun can skip it
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("quality done",condition,flush=True)


#---------------- cost components and the fixed-64 diagnostic ----------------

@torch.no_grad()
def forward_latency(model,length,new_tokens,device,warmup,repeats):
    """Time one cached forward of `new_tokens` tokens on top of a `length`-token prefix (cache cropped back each repeat)."""
    m=CachedModel(model)
    m.feed([13]*length,device)
    base=m.cached

    def one_forward():
        m.feed([13]*new_tokens,device)
        m.crop(base)

    return summarize_times(time_call(one_forward,warmup,repeats))


def alpha_from_rounds(rows,gamma):
    """Per-token acceptance alpha from full rounds (those that really proposed gamma tokens): solve
    E[accepted] = sum_{i=1..gamma} alpha^i by bisection (accepted count before the extra token)."""
    accepted=[]
    for row in rows:
        for a,g in zip(row["accepted_per_round"],row["proposed_per_round"]):
            if g==gamma:
                accepted.append(a)
    if len(accepted)==0:
        return None
    mean=float(np.mean(accepted))
    low=0.0
    high=1.0
    for step in range(60):
        middle=(low+high)/2
        if geometric(middle,gamma)-1<mean:
            low=middle
        else:
            high=middle
    return (low+high)/2


def geometric(alpha,gamma):
    """1 + sum_{j=1..gamma} alpha^j, expected tokens per round."""
    value=1.0
    for j in range(1,gamma+1):
        value=value+alpha**j
    return value


def predicted_speedup(alpha,gamma,c,v):
    return geometric(alpha,gamma)/(gamma*c+v)


def break_even_alpha(gamma,c,v):
    """Smallest alpha in [0,1] with 1 + sum alpha^j > gamma c + v_gamma, None if even alpha=1 does not reach it."""
    cost=gamma*c+v
    if geometric(1.0,gamma)<=cost:
        return None
    if geometric(0.0,gamma)>cost:
        return 0.0
    low=0.0
    high=1.0
    for step in range(60):
        middle=(low+high)/2
        if geometric(middle,gamma)>cost:
            high=middle
        else:
            low=middle
    return high


@torch.no_grad()
def throughput(config,drafts):
    """Fixed 64 generated tokens with EOS ignored (timing only) on the first 24 GSM8K test prompts, condition order
    rotated per prompt; plus isolated draft/target step costs at several prefix lengths, v_gamma, S_hat and break-even alpha."""
    device=get_device()
    evaluation=config["evaluation"]
    tokenizer=load_tokenizer(config)
    eos_list=eos_ids(tokenizer)
    target=load_causal(config["target"])
    prompts=test_prompts(config)
    chosen=[]
    for item in prompts:
        if item["domain"]=="math" and len(chosen)<evaluation["throughput_prompts"]:
            chosen.append(item)
    loaded={}
    for name in drafts:
        loaded[name]=load_causal(draft_path(config,name))
    lengths=evaluation["latency_prompt_lengths"]
    results={"prompts":[],"latency":{"target":{},"drafts":{},"verify":{}},"conditions":{},"versions":package_versions()}
    for item in chosen:
        results["prompts"].append(item["id"])
    #isolated single-step costs at several prefix lengths, and the gamma-token verification forward (gamma drafts + 1 carried token)
    for length in lengths:
        results["latency"]["target"][str(length)]=forward_latency(target,length,1,device,evaluation["warmup"],evaluation["repeats"])
        for gamma in evaluation["gammas"]:
            results["latency"]["verify"][f"{length}_g{gamma}"]=forward_latency(target,length,gamma+1,device,evaluation["warmup"],evaluation["repeats"])
        for name in drafts:
            if name not in results["latency"]["drafts"]:
                results["latency"]["drafts"][name]={}
            results["latency"]["drafts"][name][str(length)]=forward_latency(loaded[name],length,1,device,evaluation["warmup"],evaluation["repeats"])
    conditions=[("target_only",None,0)]
    for name in drafts:
        for gamma in evaluation["gammas"]:
            conditions.append((f"spec_{name}_g{gamma}",name,gamma))
    per_condition={}
    for condition,name,gamma in conditions:
        per_condition[condition]={"seconds":[],"tokens":[]}
    #global warmup
    warm_ids=chat_ids(tokenizer,chosen[0]["prompt"])
    for repeat in range(evaluation["warmup"]):
        plain_generate(target,warm_ids,evaluation["throughput_tokens"],eos_list,make_generator(device,config["seed"]),ignore_eos=True)
    for index in range(len(chosen)):
        prompt_ids=chat_ids(tokenizer,chosen[index]["prompt"])
        for offset in range(len(conditions)):
            condition,name,gamma=conditions[(index+offset)%len(conditions)] #rotate the order per prompt
            generator=make_generator(device,config["seed"]+index)
            sync()
            t0=time.perf_counter()
            if gamma==0:
                response,stats=plain_generate(target,prompt_ids,evaluation["throughput_tokens"],eos_list,generator,ignore_eos=True)
            else:
                response,stats=speculative_generate(target,loaded[name],prompt_ids,gamma,evaluation["throughput_tokens"],eos_list,generator,ignore_eos=True)
            sync()
            per_condition[condition]["seconds"].append(time.perf_counter()-t0)
            per_condition[condition]["tokens"].append(len(response))
    base=per_condition["target_only"]
    speed_dir=repo_path(config["results_dir"])/"speed"
    for condition,name,gamma in conditions:
        times=np.array(per_condition[condition]["seconds"])
        tokens=np.array(per_condition[condition]["tokens"])
        ratios=(tokens/times)/(np.array(base["tokens"])/np.array(base["seconds"])) #per-prompt throughput ratio vs target-only
        entry={"tokens_per_second":float(tokens.sum()/times.sum()),"per_prompt_seconds":times.tolist(),"measured_speedup_per_prompt":bootstrap_mean_ci(ratios,evaluation["bootstrap"],config["seed"])}
        if gamma>0:
            c_values=[]
            v_values=[]
            for length in lengths:
                c_values.append(results["latency"]["drafts"][name][str(length)]["mean_ms"]/results["latency"]["target"][str(length)]["mean_ms"])
                v_values.append(results["latency"]["verify"][f"{length}_g{gamma}"]["mean_ms"]/results["latency"]["target"][str(length)]["mean_ms"])
            c=float(np.mean(c_values))
            v=float(np.mean(v_values))
            entry["c_by_length"]=c_values
            entry["v_by_length"]=v_values
            entry["c"]=c
            entry["v_gamma"]=v
            entry["break_even_alpha"]=break_even_alpha(gamma,c,v)
            path=speed_dir/f"{condition}.jsonl"
            if path.exists():
                alpha=alpha_from_rounds(read_jsonl(path),gamma)
                entry["alpha_speed_benchmark"]=alpha
                if alpha is not None:
                    entry["predicted_speedup"]=predicted_speedup(alpha,gamma,c,v)
        results["conditions"][condition]=entry
        print(condition,entry["tokens_per_second"],flush=True)
    save_json(repo_path(config["results_dir"])/"throughput.json",results)


@torch.no_grad()
def unmerged(config):
    """Manual step 6: one adapter executed through the PEFT wrapper (not merged), timed next to its merged copy on the
    fixed-64 diagnostic at the larger gamma. Only quantifies wrapper overhead; not the primary speed comparison."""
    from peft import PeftModel
    device=get_device()
    evaluation=config["evaluation"]
    tokenizer=load_tokenizer(config)
    eos_list=eos_ids(tokenizer)
    target=load_causal(config["target"])
    name=evaluation["unmerged_draft"]
    adapter=repo_path(config["results_dir"])/"drafts"/name/"adapter"
    wrapped=PeftModel.from_pretrained(load_causal(config["draft"]),str(adapter)).to(device).eval()
    merged=load_causal(draft_path(config,name))
    gamma=max(evaluation["gammas"])
    prompts=test_prompts(config)
    chosen=[]
    for item in prompts:
        if item["domain"]=="math" and len(chosen)<evaluation["throughput_prompts"]:
            chosen.append(item)
    result={"draft":name,"gamma":gamma,"step_latency":{},"spec_seconds":{}}
    for label,model in [("unmerged",wrapped),("merged",merged)]:
        result["step_latency"][label]={}
        for length in evaluation["latency_prompt_lengths"]:
            result["step_latency"][label][str(length)]=forward_latency(model,length,1,device,evaluation["warmup"],evaluation["repeats"])
        seconds=[]
        for index in range(len(chosen)):
            prompt_ids=chat_ids(tokenizer,chosen[index]["prompt"])
            generator=make_generator(device,config["seed"]+index)
            sync()
            t0=time.perf_counter()
            speculative_generate(target,model,prompt_ids,gamma,evaluation["throughput_tokens"],eos_list,generator,ignore_eos=True)
            sync()
            seconds.append(time.perf_counter()-t0)
        result["spec_seconds"][label]={"mean":float(np.mean(seconds)),"per_prompt":seconds}
    save_json(repo_path(config["results_dir"])/"unmerged_timing.json",result)
    print("unmerged",result["spec_seconds"]["unmerged"]["mean"],"merged",result["spec_seconds"]["merged"]["mean"])


#---------------- summary ----------------

def length_bin(tokens):
    if tokens<=16:
        return "1-16"
    elif tokens<=32:
        return "17-32"
    else:
        return "33-64"


def prompt_means(rows,key_function):
    """Average a per-run value over the paired seeds of each prompt -> one value per prompt (bootstrap unit)."""
    by_prompt={}
    for row in rows:
        if row["id"] not in by_prompt:
            by_prompt[row["id"]]=[]
        by_prompt[row["id"]].append(key_function(row))
    values={}
    for prompt in by_prompt:
        values[prompt]=float(np.mean(by_prompt[prompt]))
    return values


def throughput_ratio(row):
    return row["tokens_per_second"]/row["target_tokens_per_second"]


def acceptance(row):
    return row["accepted"]/max(row["proposed"],1)


def speed_table(rows,evaluation,seed):
    accepted=0
    proposed=0
    examined=0
    rounds=0
    tokens=0
    target_forwards=0
    draft_forwards=0
    eos=0
    parts={"prefill_time":0.0,"draft_time":0.0,"verify_time":0.0,"accept_time":0.0,"rollback_time":0.0,"host_overhead_time":0.0}
    peaks=[]
    for row in rows:
        accepted=accepted+row["accepted"]
        proposed=proposed+row["proposed"]
        examined=examined+row["examined"]
        rounds=rounds+row["rounds"]
        tokens=tokens+row["tokens"]
        target_forwards=target_forwards+row["target_forwards"]
        draft_forwards=draft_forwards+row["draft_forwards"]
        eos=eos+int(row["ended_eos"])
        peaks.append(row["peak_mb"])
        for key in parts:
            parts[key]=parts[key]+row[key]
    per_round_ms={}
    for key in parts:
        per_round_ms[key]=1000*parts[key]/max(rounds,1)
    ratios=list(prompt_means(rows,throughput_ratio).values())
    rates=list(prompt_means(rows,acceptance).values())
    return {"runs":len(rows),"accepted_over_examined":accepted/max(examined,1),"accepted_over_proposed":accepted/max(proposed,1),
        "accepted_prefix_per_verification":accepted/max(rounds,1),"target_passes_per_output_token":target_forwards/max(tokens,1),
        "draft_forwards_per_output_token":draft_forwards/max(tokens,1),"tokens_per_second":float(np.mean(column(rows,"tokens_per_second"))),
        "adjacent_target_tokens_per_second":float(np.mean(column(rows,"target_tokens_per_second"))),
        "throughput_ratio_vs_target":bootstrap_mean_ci(ratios,evaluation["bootstrap"],seed),"acceptance_per_prompt":bootstrap_mean_ci(rates,evaluation["bootstrap"],seed),
        "eos_rate":eos/len(rows),"cap_truncation_rate":1-eos/len(rows),"peak_mb":float(np.max(peaks)),"target_peak_mb":float(np.max(column(rows,"target_peak_mb"))),
        "time_per_round_ms":per_round_ms,"mean_seconds":float(np.mean(column(rows,"seconds"))),"mean_target_seconds":float(np.mean(column(rows,"target_seconds")))}


def column(rows,key):
    values=[]
    for row in rows:
        values.append(row[key])
    return values


def paired_difference(rows_a,rows_b,function,evaluation,seed):
    """Prompt-level paired bootstrap of function(a) - function(b)."""
    a=prompt_means(rows_a,function)
    b=prompt_means(rows_b,function)
    diffs=[]
    for prompt in a:
        if prompt in b:
            diffs.append(a[prompt]-b[prompt])
    return bootstrap_mean_ci(diffs,evaluation["bootstrap"],seed)


def summary(config):
    evaluation=config["evaluation"]
    root=repo_path(config["results_dir"])
    out={"speed":{},"speed_differences":{},"quality":{},"deployment":{}}
    speed_rows={}
    for path in sorted((root/"speed").glob("*.jsonl")):
        speed_rows[path.stem]=read_jsonl(path)
    for condition in speed_rows:
        rows=speed_rows[condition]
        groups={"all":rows}
        for row in rows:
            for scope in [row["subset"],"len_"+length_bin(row["tokens"])]:
                if scope not in groups:
                    groups[scope]=[]
                groups[scope].append(row)
        out["speed"][condition]={}
        for scope in groups:
            out["speed"][condition][scope]=speed_table(groups[scope],evaluation,config["seed"])
    #each trained draft against the original draft, same gamma, paired by prompt
    for condition in speed_rows:
        if condition.startswith("spec_original_"):
            continue
        gamma_part=condition.split("_")[-1]
        reference=f"spec_original_{gamma_part}"
        if reference in speed_rows:
            out["speed_differences"][f"{condition}_minus_original"]={"throughput_ratio":paired_difference(speed_rows[condition],speed_rows[reference],throughput_ratio,evaluation,config["seed"]),
                "acceptance":paired_difference(speed_rows[condition],speed_rows[reference],acceptance,evaluation,config["seed"])}
    for gamma in evaluation["gammas"]:
        for seed in config["distill"]["seeds"]:
            on=f"spec_on_policy_s{seed}_g{gamma}"
            off=f"spec_off_policy_s{seed}_g{gamma}"
            if on in speed_rows and off in speed_rows:
                out["speed_differences"][f"on_minus_off_s{seed}_g{gamma}"]={"throughput_ratio":paired_difference(speed_rows[on],speed_rows[off],throughput_ratio,evaluation,config["seed"]),
                    "acceptance":paired_difference(speed_rows[on],speed_rows[off],acceptance,evaluation,config["seed"])}
    #standalone answer quality
    for path in sorted((root/"quality").glob("*.jsonl")):
        rows=read_jsonl(path)
        groups={"all":rows}
        for row in rows:
            if row["subset"] not in groups:
                groups[row["subset"]]=[]
            groups[row["subset"]].append(row)
        out["quality"][path.stem]={}
        for scope in groups:
            group=groups[scope]
            out["quality"][path.stem][scope]={"runs":len(group),"accuracy":float(np.mean(column(group,"correct"))),"extractable_rate":float(np.mean(column(group,"extractable"))),
                "truncated_rate":float(np.mean(column(group,"truncated"))),"eos_rate":float(np.mean(column(group,"ended_eos")))}
    #deployment break-even: upfront distillation + merge cost over the measured per-request saving
    for condition in speed_rows:
        draft=speed_rows[condition][0]["draft"]
        if draft=="original":
            continue
        cost_path=root/"drafts"/draft/"cost.json"
        merge_path=root/"drafts"/draft/"merge_check.json"
        if cost_path.exists()==False:
            continue
        upfront=load_json(cost_path)["end_to_end_seconds"]
        if merge_path.exists():
            upfront=upfront+load_json(merge_path)["merge_seconds"]
        savings=[]
        for row in speed_rows[condition]:
            savings.append(row["target_seconds"]-row["seconds"]) #positive = speculative request was faster
        saving=float(np.mean(savings))
        requests=None #no break-even when speculation is not faster
        if saving>0:
            requests=upfront/saving
        out["deployment"][condition]={"upfront_seconds":upfront,"mean_saving_per_request_seconds":saving,"requests_to_break_even":requests}
    save_json(root/"heldout_summary.json",out)
    print("wrote heldout_summary.json")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/task3.yaml")
    parser.add_argument("--mode",required=True,choices=["speed","quality","throughput","unmerged","summary"])
    parser.add_argument("--drafts",nargs="*",default=["original"])
    args=parser.parse_args()
    config=load_yaml(args.config)
    if args.mode=="speed":
        speed(config,args.drafts)
    elif args.mode=="quality":
        quality(config,args.drafts)
    elif args.mode=="throughput":
        throughput(config,args.drafts)
    elif args.mode=="unmerged":
        unmerged(config)
    else:
        summary(config)


if __name__=="__main__":
    main()
