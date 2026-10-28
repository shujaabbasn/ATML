from __future__ import annotations

import argparse
import pickle

import numpy as np

from common.utils import load_yaml, repo_path, save_json, write_jsonl
from task3_ttc.selection import (
    descending_average_ranks,
    group_candidates,
    pooled_kl,
    method_cost,
    rollout_scores,
    select_all,
    self_consistency,
    visconf,
)


#steps 1-3 on the supplied rollouts:
#  for N in {1,2,4,8,16,32} and each of the ten orders: candidates = first N rollouts of the order
#  accuracy, oracle coverage, distinct answer groups and generated-token cost per method,
#  averaged over orders within a question, then over questions
#  step 3: within-question signal analysis at N = 32

SIGNAL_COMPARISONS=[ #(a, b): accuracy of a minus b, bootstrapped
    ("self_certainty_rwc","highest_self_certainty"),
    ("vision_rwc","highest_vision"),
    ("visconf_rwc","highest_visconf"),
    ("highest_visconf","highest_self_certainty"),
    ("visconf_rwc","self_certainty_rwc"),
    ("highest_visconf_tokenwise","highest_visconf"),
    ("visconf_tokenwise_rwc","visconf_rwc"),
]


def load_prepared(cfg,dataset):
    with open(repo_path(cfg["data"]["prepared_dir"])/(dataset+".pkl"),"rb") as f:
        return pickle.load(f)


def evaluate_question(q,orders,budgets,dataset):
    """Per-question results averaged over the ten orders."""
    scores=rollout_scores(q)
    out={}
    for n in budgets:
        per_method={}
        coverage=[]
        distinct=[]
        for order in orders:
            candidates=order[:n]
            chosen=select_all(q,candidates,scores,dataset)
            coverage.append(float(q["correct"][candidates].any())) #oracle: any correct candidate (analysis upper bound)
            distinct.append(len(group_candidates(candidates,q["eq"])))
            for method,rollout in chosen.items():
                if method not in per_method:
                    per_method[method]={"correct":[],"cost":[],"selected_rollouts":[],"selected_answers":[]}
                per_method[method]["correct"].append(float(q["correct"][rollout]))
                per_method[method]["cost"].append(method_cost(q,candidates,method))
                per_method[method]["selected_rollouts"].append(int(rollout))
                per_method[method]["selected_answers"].append(q["keys"][rollout])
        out[n]={"coverage":float(np.mean(coverage)),"distinct_groups":float(np.mean(distinct)),"methods":per_method}
    return out


def question_accuracies(per_question,ids,n,method):
    """Order-averaged accuracy of one method at budget n, one value per question."""
    values=[]
    for s in ids:
        values.append(float(np.mean(per_question[s][n]["methods"][method]["correct"])))
    return values


def paired_bootstrap(a,b,iterations,rng):
    """95% interval of mean(a - b) over questions (values already averaged over orders), 1,000 paired resamples."""
    difference=np.asarray(a)-np.asarray(b)
    means=[]
    for i in range(iterations):
        sample=rng.integers(len(difference),size=len(difference))
        means.append(difference[sample].mean())
    return {"estimate":float(difference.mean()),"ci95_low":float(np.percentile(means,2.5)),"ci95_high":float(np.percentile(means,97.5))}


def quartile_gap(values,correct):
    """mean correctness of the top 8 minus bottom 8 rollouts by this score (ties broken by rollout id)."""
    order=np.argsort(-np.asarray(values),kind="stable")
    quarter=len(values)//4
    top=correct[order[:quarter]].mean()
    bottom=correct[order[-quarter:]].mean()
    return float(top-bottom)


def spearman(x,y):
    rx=descending_average_ranks(x)
    ry=descending_average_ranks(y)
    if np.std(rx)==0 or np.std(ry)==0:
        return None #undefined when one side is constant
    return float(np.corrcoef(rx,ry)[0,1])


def signal_analysis(questions,split_ids,dataset):
    """Step 3 at N = 32: length / score / correctness associations within each question."""
    wanted=set(split_ids)
    gaps={}
    mixed_gaps={}
    length_corr={}
    vision_corr_by_question=[]
    covariance_check=0.0
    for q in questions:
        if q["sample_id"] not in wanted:
            continue
        scores=rollout_scores(q)
        candidates=list(range(32))
        values={"mean_logprob":scores["L"],"self_certainty":scores["K"],"shortest":-scores["length"]}
        if dataset=="mathvista":
            Z,Z_token=visconf(q,candidates)
            values["vision"]=scores["V"]
            values["visconf"]=Z
            values["visconf_tokenwise"]=Z_token
            #confidence-vision correlation inside each rollout, c from the 32-candidate calibration
            pooled=pooled_kl(q,candidates)
            q05=np.percentile(pooled,5)
            q95=np.percentile(pooled,95)
            correlations=[]
            for r in candidates:
                c=np.clip((q["kl"][r].astype(np.float64)-q05)/max(q95-q05,1e-8),0.0,1.0)
                v=q["vision"][r].astype(np.float64)
                covariance=float((c*v).mean()-c.mean()*v.mean()) #Cov_t(c, v)
                covariance_check=max(covariance_check,abs((Z_token[r]-Z[r])-covariance)) #identity Z_token - Z = Cov
                if np.std(c)>0 and np.std(v)>0:
                    correlations.append(float(np.corrcoef(c,v)[0,1]))
            if len(correlations)>0:
                vision_corr_by_question.append(float(np.mean(correlations)))
        correct=q["correct"].astype(float)
        is_mixed=correct.min()!=correct.max() #some right, some wrong
        for name in values:
            if name not in gaps:
                gaps[name]=[]
                mixed_gaps[name]=[]
                length_corr[name]=[]
            gap=quartile_gap(values[name],correct)
            gaps[name].append(gap)
            if is_mixed:
                mixed_gaps[name].append(gap)
            rho=spearman(scores["length"],values[name])
            if rho is not None:
                length_corr[name].append(rho)
    summary={}
    for name in gaps:
        summary[name]={"top_minus_bottom_quartile_correct":float(np.mean(gaps[name])),"same_on_mixed_questions":None,
            "mixed_questions":len(mixed_gaps[name]),"spearman_length_vs_score":None}
        if len(mixed_gaps[name])>0:
            summary[name]["same_on_mixed_questions"]=float(np.mean(mixed_gaps[name]))
        if len(length_corr[name])>0:
            summary[name]["spearman_length_vs_score"]=float(np.mean(length_corr[name]))
    if dataset=="mathvista":
        values=np.asarray(vision_corr_by_question)
        summary["confidence_vision_correlation"]={"questions":len(values),"mean":float(values.mean()),"median":float(np.median(values)),
            "q25":float(np.percentile(values,25)),"q75":float(np.percentile(values,75)),"max_abs_identity_error":covariance_check}
    return summary


def illustrative_trajectories(questions,eval_ids,count=3):
    """Rule (fixed before looking): the first 3 held-out mathvista questions, in permutation order, where highest-VisConf
    and self-consistency pick different answers at N = 32. Token series of both picked rollouts are saved."""
    by_id={}
    for q in questions:
        by_id[q["sample_id"]]=q
    picked=[]
    for sample_id in eval_ids:
        q=by_id[sample_id]
        candidates=list(range(32))
        Z,Z_token=visconf(q,candidates)
        best=int(np.argmax(Z))
        majority=self_consistency(candidates,q["eq"])
        if q["eq"][best,majority]==False:
            pooled=pooled_kl(q,candidates)
            q05=np.percentile(pooled,5)
            q95=np.percentile(pooled,95)
            series={}
            for label,r in [("visconf_pick",best),("self_consistency_pick",majority)]:
                c=np.clip((q["kl"][r].astype(np.float64)-q05)/max(q95-q05,1e-8),0.0,1.0)
                series[label]={"rollout":r,"answer":q["keys"][r],"correct":bool(q["correct"][r]),"Z":float(Z[r]),"c":c.round(4).tolist(),"v":q["vision"][r].round(4).tolist()}
            picked.append({"sample_id":sample_id,"gold":q["gold"],"series":series})
            if len(picked)==count:
                break
    return picked


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task3.yaml")
    ap.add_argument("--dataset",required=True,choices=["math","mathvista"])
    args=ap.parse_args()
    cfg=load_yaml(args.config)
    budgets=list(cfg["offline"]["budgets"])
    prepared=load_prepared(cfg,args.dataset)
    questions=prepared["questions"]
    splits=prepared["splits"]
    orders=prepared["orders"]
    results_dir=repo_path(cfg["results_dir"])/"offline"/args.dataset
    split_of={}
    for sample_id in splits["dev"]:
        split_of[sample_id]="dev"
    for sample_id in splits["eval"]:
        split_of[sample_id]="eval"

    per_question={}
    rows=[]
    for i in range(len(questions)):
        q=questions[i]
        result=evaluate_question(q,orders[q["sample_id"]],budgets,args.dataset)
        per_question[q["sample_id"]]=result
        for n in budgets:
            for method,values in result[n]["methods"].items():
                rows.append({"sample_id":q["sample_id"],"split":split_of[q["sample_id"]],"N":n,"method":method,"accuracy":float(np.mean(values["correct"])),
                    "mean_cost":float(np.mean(values["cost"])),"coverage":result[n]["coverage"],"distinct_groups":result[n]["distinct_groups"],
                    "selected_rollouts":values["selected_rollouts"],"selected_answers":values["selected_answers"]})
        if i%100==0:
            print(args.dataset,i,"/",len(questions))
    write_jsonl(results_dir/"per_question.jsonl",rows) #candidate ids = the saved orders cut at N

    #aggregate: mean over questions of the order-averaged values, per split
    methods=sorted(per_question[questions[0]["sample_id"]][budgets[0]]["methods"].keys())
    summary={}
    for split in ["dev","eval"]:
        ids=splits[split]
        summary[split]={}
        for n in budgets:
            coverage=[]
            distinct=[]
            for s in ids:
                coverage.append(per_question[s][n]["coverage"])
                distinct.append(per_question[s][n]["distinct_groups"])
            entry={"coverage":float(np.mean(coverage)),"distinct_groups":float(np.mean(distinct)),"methods":{}}
            for method in methods:
                accuracy=[] #per question, already averaged over the ten orders
                cost=[]
                for s in ids:
                    accuracy.append(float(np.mean(per_question[s][n]["methods"][method]["correct"])))
                    cost.append(float(np.mean(per_question[s][n]["methods"][method]["cost"])))
                acc=float(np.mean(accuracy))
                entry["methods"][method]={"accuracy":acc,"mean_token_cost":float(np.mean(cost)),
                    "no_correct_candidate":1.0-entry["coverage"], #sampling limit
                    "correct_available_but_missed":entry["coverage"]-acc} #selection limit
            summary[split][n]=entry

    #paired question-level bootstrap on the held-out questions
    rng=np.random.default_rng(int(cfg["seed"]))
    eval_ids=splits["eval"]
    bootstrap={}
    for n in budgets:
        bootstrap[n]={}
        base=question_accuracies(per_question,eval_ids,n,"self_consistency")
        for method in methods:
            if method!="self_consistency":
                values=question_accuracies(per_question,eval_ids,n,method)
                bootstrap[n][method+"_minus_self_consistency"]=paired_bootstrap(values,base,int(cfg["offline"]["bootstrap"]),rng)
        for a,b in SIGNAL_COMPARISONS:
            if a in methods and b in methods:
                values_a=question_accuracies(per_question,eval_ids,n,a)
                values_b=question_accuracies(per_question,eval_ids,n,b)
                bootstrap[n][a+"_minus_"+b]=paired_bootstrap(values_a,values_b,int(cfg["offline"]["bootstrap"]),rng)

    save_json(results_dir/"summary.json",summary)
    save_json(results_dir/"bootstrap_eval.json",bootstrap)
    save_json(results_dir/"signals_n32_eval.json",signal_analysis(questions,eval_ids,args.dataset))
    if args.dataset=="mathvista":
        save_json(results_dir/"illustrative_trajectories.json",illustrative_trajectories(questions,eval_ids))
    for n in budgets:
        print("N =",n,"coverage",round(summary["eval"][n]["coverage"],4))
        for method in methods:
            print("   ",method,round(summary["eval"][n]["methods"][method]["accuracy"],4))


if __name__=="__main__":
    main()
