from __future__ import annotations

import argparse
import importlib.metadata
import pickle
import urllib.request

import gdown
import numpy as np
import pandas as pd

from common.utils import load_json, load_yaml, repo_path, save_json
from task3_ttc.answers import NO_ANSWER, math_correct, math_equivalent, math_parse, mathvista_gold_key, mathvista_key


#step 1 preparation (offline, supplied rollouts only):
#  join generations + probability (+ attention) on (dataset, sample_id, rollout_id), check every token list lines up
#  answer equivalence between rollouts of a question (no gold used) and correctness labels (gold used, analysis only)
#  development / held-out split and the ten rollout orders per question, all with seed 6304


def download(cfg):
    root=repo_path(cfg["data"]["root"])
    if not (root/cfg["data"]["math_folder"]/"generations.parquet").exists():
        root.mkdir(parents=True,exist_ok=True)
        gdown.download_folder(cfg["data"]["drive_folder"],output=str(root),quiet=False) #the supplied ATML-PA3-Data folder
    metadata=repo_path(cfg["data"]["mathvista_metadata"])
    if not metadata.exists():
        metadata.parent.mkdir(parents=True,exist_ok=True)
        urllib.request.urlretrieve(cfg["data"]["mathvista_metadata_url"],metadata) #released question metadata


def load_joined(folder,with_attention):
    keys=["dataset","sample_id","rollout_id"]
    generations=pd.read_parquet(folder/"generations.parquet")
    probabilities=pd.read_parquet(folder/"probability_metrics.parquet")
    joined=generations.merge(probabilities,on=keys,how="inner",validate="one_to_one")
    if with_attention==True:
        attention=pd.read_parquet(folder/"attention_metrics.parquet")
        attention=attention.rename(columns={"n_generated_text_tokens":"attn_preceding_counts"}) #per-token list, not the scalar length
        joined=joined.merge(attention,on=keys,how="inner",validate="one_to_one")
    assert len(joined)==len(generations)==32000
    return joined.sort_values(["sample_id","rollout_id"]).reset_index(drop=True)


def check_alignment(df,with_attention):
    """Every per-token list must have n_generated_text_tokens entries (README: lists align with generated_token_ids)."""
    problems=0
    for row in df.itertuples(index=False):
        n=int(row.n_generated_text_tokens)
        lengths=[len(row.generated_token_ids),len(row.logp),len(row.kl_u_p)]
        if with_attention==True:
            lengths=lengths+[len(row.img_attn_total),len(row.prompt_text_attn_total),len(row.generated_text_attn_total),len(row.attn_preceding_counts)]
            if list(row.attn_preceding_counts)!=list(range(n)):
                problems=problems+1 #README: preceding-token counts are [0, ..., N-1]
        for length in lengths:
            if length!=n:
                problems=problems+1
    return problems


def equivalence_matrix(answers,dataset,meta):
    """32 x 32 symmetric matrix, True when two rollouts give equivalent answers. Uses only the answers, never the gold."""
    size=len(answers)
    matrix=np.zeros((size,size),dtype=bool)
    if dataset=="math":
        distinct=set() #different non-missing answer strings
        for a in answers:
            if a is not None:
                distinct.add(a)
        distinct=sorted(distinct)
        parsed={}
        for a in distinct:
            parsed[a]=math_parse(a)
        same={} #(a, b) -> equal, computed once per distinct pair
        for i in range(len(distinct)):
            for j in range(i,len(distinct)):
                equal=math_equivalent(distinct[i],distinct[j],parsed[distinct[i]],parsed[distinct[j]])
                same[(distinct[i],distinct[j])]=equal
                same[(distinct[j],distinct[i])]=equal
        for i in range(size):
            for j in range(size):
                if answers[i] is None or answers[j] is None:
                    matrix[i,j]=(answers[i] is None and answers[j] is None) #all missing extractions share NO_ANSWER
                else:
                    matrix[i,j]=same[(answers[i],answers[j])]
    else:
        keys=[] #normalized key, equal keys = same group
        for a in answers:
            keys.append(mathvista_key(a,meta))
        for i in range(size):
            for j in range(size):
                matrix[i,j]=keys[i]==keys[j]
    return matrix


def float_lists(column):
    """Per-rollout token lists as float32 arrays."""
    arrays=[]
    for x in column:
        arrays.append(np.asarray(x,dtype=np.float32))
    return arrays


def prepare_dataset(cfg,dataset,folder_name,with_attention,metadata):
    folder=repo_path(cfg["data"]["root"])/folder_name
    df=load_joined(folder,with_attention)
    problems=check_alignment(df,with_attention)
    questions=[]
    for sample_id,group in df.groupby("sample_id",sort=True):
        group=group.sort_values("rollout_id")
        assert list(group["rollout_id"])==list(range(32))
        meta=None #math has no extra metadata
        if metadata is not None:
            meta=metadata.get(sample_id)
        answers=[]
        for value in group["extracted_answer"]:
            if pd.isna(value):
                answers.append(None)
            else:
                answers.append(str(value))
        gold=str(group["gold_answer"].iloc[0])
        correct=[]
        keys=[]
        for a in answers:
            if dataset=="math":
                correct.append(math_correct(gold,a))
                keys.append(NO_ANSWER if a is None else a)
            else:
                key=mathvista_key(a,meta)
                keys.append(key)
                correct.append(key!=NO_ANSWER and key==mathvista_gold_key(gold,meta))
        q={
            "sample_id":sample_id,
            "question_text":group["question_text"].iloc[0],
            "gold":gold,
            "answers":answers, #raw extracted answers, None = missing
            "keys":keys, #display keys (mathvista: normalized key)
            "eq":equivalence_matrix(answers,dataset,meta),
            "correct":np.asarray(correct,dtype=bool), #analysis only, never used by selection or stopping
            "tokens":group["n_generated_text_tokens"].to_numpy().astype(np.int64), #retained token count = cost proxy
            "capped":group["hit_max_new_tokens"].to_numpy().astype(bool),
            "logp":float_lists(group["logp"]),
            "kl":float_lists(group["kl_u_p"]),
            "texts":list(group["generated_text"]),
        }
        if with_attention==True:
            vision=[]
            for img,prompt,generated in zip(group["img_attn_total"],group["prompt_text_attn_total"],group["generated_text_attn_total"]):
                img=np.asarray(img,dtype=np.float64)
                denominator=img+np.asarray(prompt,dtype=np.float64)+np.asarray(generated,dtype=np.float64)
                v=np.where(denominator>0,img/np.where(denominator>0,denominator,1.0),0.0) #v_it = A_img / (A_img + A_prompt + A_generated)
                vision.append(v.astype(np.float32))
            q["vision"]=vision
            q["question_type"]=meta["question_type"]
            q["answer_type"]=meta["answer_type"]
        questions.append(q)
    return questions,problems


def make_splits(cfg,sample_ids):
    """Sort ids lexicographically, permute with default_rng(6304): first 200 dev, remaining 800 held-out evaluation."""
    ordered=sorted(sample_ids)
    permutation=np.random.default_rng(int(cfg["seed"])).permutation(len(ordered))
    permuted=[ordered[i] for i in permutation]
    dev=permuted[:int(cfg["splits"]["dev_questions"])]
    evaluation=permuted[int(cfg["splits"]["dev_questions"]):]
    return {"permuted":permuted,"dev":dev,"eval":evaluation}


def make_orders(cfg,sample_ids):
    """Ten seeded random permutations of the 32 rollout ids per question, reused by every method and stopping policy."""
    rng=np.random.default_rng(int(cfg["seed"]))
    orders={}
    for sample_id in sorted(sample_ids):
        orders[sample_id]=[]
        for o in range(int(cfg["offline"]["orders"])):
            orders[sample_id].append(rng.permutation(32).tolist())
    return orders


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task3.yaml")
    args=ap.parse_args()
    cfg=load_yaml(args.config)
    download(cfg)
    metadata=load_json(cfg["data"]["mathvista_metadata"])
    prepared=repo_path(cfg["data"]["prepared_dir"])
    prepared.mkdir(parents=True,exist_ok=True)
    report={"math_verify_version":importlib.metadata.version("math-verify")} #manual: report the scorer version
    for dataset,folder_name,with_attention,meta in [("math",cfg["data"]["math_folder"],False,None),("mathvista",cfg["data"]["mathvista_folder"],True,metadata)]:
        questions,problems=prepare_dataset(cfg,dataset,folder_name,with_attention,meta)
        sample_ids=[q["sample_id"] for q in questions]
        splits=make_splits(cfg,sample_ids)
        orders=make_orders(cfg,sample_ids)
        with open(prepared/(dataset+".pkl"),"wb") as f:
            pickle.dump({"questions":questions,"splits":splits,"orders":orders},f)
        split_ids={"dev":splits["dev"],"eval":splits["eval"]}
        if dataset=="math":
            split_ids["live_50"]=splits["eval"][:int(cfg["splits"]["live_questions"])] #first 50 math evaluation questions in the permutation
        save_json(repo_path(cfg["results_dir"])/"splits"/(dataset+"_splits.json"),split_ids)
        save_json(repo_path(cfg["results_dir"])/"splits"/(dataset+"_orders.json"),orders)
        missing=0
        correct=0
        for q in questions:
            for a in q["answers"]:
                if a is None:
                    missing=missing+1
            correct=correct+int(q["correct"].sum())
        report[dataset]={"questions":len(questions),"rollouts":32*len(questions),"alignment_problems":problems,"missing_extractions":missing,
            "correct_rollouts":correct,"retained_tokens":int(sum(int(q["tokens"].sum()) for q in questions)),"capped":int(sum(int(q["capped"].sum()) for q in questions))}
        print(dataset,report[dataset])
    save_json(repo_path(cfg["results_dir"])/"prepare_report.json",report)


if __name__=="__main__":
    main()
