"""Manual, "Before You Submit": check that calibration or training data did not overlap with held-out evaluations.

python overlap_check.py      (after setup_data.py --public; the mqar part needs the cached training streams)

Task 1: C4 calibration vs development rows, and no 64-token chunk of calibration/development text inside the
        WikiText-2 or C4 validation windows.
Task 2: no MQAR training sequence equals a frozen development or test sequence.
Task 3: no GSM8K or short-form training prompt equals a held-out prompt, and the teacher training bank only
        contains training ids.
"""
from __future__ import annotations

import hashlib

import numpy as np

from common.utils import read_jsonl, data_path, repo_path, save_json


def chunk_hashes(token_ids,size):
    out=set()
    for start in range(0,len(token_ids)-size+1,size):
        piece=token_ids[start:start+size]
        out.add(hashlib.sha256(str(piece).encode()).hexdigest())
    return out


def sliding_hashes(token_ids,size):
    """Every window position, so a chunk is found even when it is not aligned with the other file."""
    out=set()
    for start in range(0,len(token_ids)-size+1):
        piece=token_ids[start:start+size]
        out.add(hashlib.sha256(str(piece).encode()).hexdigest())
    return out


def task1():
    calibration=read_jsonl(data_path("data/public_locked/c4_calibration_locked.jsonl.gz"))
    development=read_jsonl(data_path("data/public_locked/c4_development_locked.jsonl.gz"))
    calibration_rows=set()
    for record in calibration:
        calibration_rows.add(record["source_row"])
    development_rows=set()
    for record in development:
        development_rows.add(record["source_row"])
    train_chunks=set()
    for record in calibration+development:
        train_chunks=train_chunks|chunk_hashes(record["token_ids"],64)
    found={}
    for name,path in [("wikitext2_test","data/public_locked/wikitext2_test_windows_locked.jsonl.gz"),("c4_validation","data/public_locked/c4_validation_windows_locked.jsonl.gz")]:
        evaluation=set()
        for record in read_jsonl(data_path(path)):
            evaluation=evaluation|sliding_hashes(record["token_ids"],64)
        found[name]=len(train_chunks&evaluation)
    return {"calibration_sequences":len(calibration),"development_sequences":len(development),"calibration_development_shared_rows":len(calibration_rows&development_rows),
        "calibration_split":"c4 train","evaluation_split":"c4 validation / wikitext-2 test","shared_64_token_chunks_with_evaluation":found,
        "pass":len(calibration_rows&development_rows)==0 and found["wikitext2_test"]==0 and found["c4_validation"]==0}


def task2():
    held_out=set()
    for path in ["data/mqar/mqar_test.jsonl.gz","data/mqar/mqar_development.jsonl.gz"]:
        for record in read_jsonl(data_path(path)):
            held_out.add(hashlib.sha256(str(record["tokens"]).encode()).hexdigest())
    stream_paths=sorted(repo_path("results/task2/part_b").glob("train_stream_K*.npz")) #one stream per (K, regime)
    if len(stream_paths)==0:
        return {"status":"train streams not generated yet, run task2_attention.mqar first"}
    shared=0
    total=0
    streams=[]
    for stream_path in stream_paths:
        tokens=np.load(stream_path)["tokens"]
        streams.append(stream_path.name)
        for row in range(tokens.shape[0]):
            key=hashlib.sha256(str(tokens[row].tolist()).encode()).hexdigest()
            if key in held_out:
                shared=shared+1
            total=total+1
    return {"streams":streams,"train_sequences":total,"held_out_sequences":len(held_out),"identical_sequences":shared,"pass":shared==0}


def task3():
    train_questions=set()
    test_questions=set()
    train_rows=set()
    test_rows=set()
    for record in read_jsonl(data_path("data/public_locked/gsm8k_train_locked.jsonl.gz")):
        train_questions.add(record["example"]["question"].strip())
        train_rows.add(record["source_row_index"])
    for record in read_jsonl(data_path("data/public_locked/gsm8k_test_locked.jsonl.gz")):
        test_questions.add(record["example"]["question"].strip())
        test_rows.add(record["source_row_index"])
    short_train=set()
    short_test=set()
    for record in read_jsonl(data_path("data/short_form/short_train.jsonl")):
        short_train.add(record["input"].strip())
    for record in read_jsonl(data_path("data/short_form/short_test.jsonl")):
        short_test.add(record["input"].strip())
    out={"gsm8k_train":len(train_questions),"gsm8k_test":len(test_questions),"gsm8k_shared_questions":len(train_questions&test_questions),
        "gsm8k_splits":"train rows from the train split, test rows from the test split",
        "short_train":len(short_train),"short_test":len(short_test),"short_shared_inputs":len(short_train&short_test)}
    bank_path=repo_path("results/task3/distill/teacher_train_responses.jsonl.gz")
    if bank_path.exists():
        test_like=0
        for row in read_jsonl(bank_path):
            if row["id"].startswith("GSM8K-TRAIN")==False and row["id"].startswith("SHORT-train")==False:
                test_like=test_like+1
        out["teacher_bank_non_training_ids"]=test_like
    else:
        out["teacher_bank_non_training_ids"]="teacher bank not generated yet"
    out["pass"]=out["gsm8k_shared_questions"]==0 and out["short_shared_inputs"]==0 and out["teacher_bank_non_training_ids"] in [0,"teacher bank not generated yet"]
    return out


def main():
    result={"task1":task1(),"task2":task2(),"task3":task3()}
    save_json("results/overlap_check.json",result)
    for task in result:
        print(task,result[task])


if __name__=="__main__":
    main()
