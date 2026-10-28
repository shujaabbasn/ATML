from __future__ import annotations

import argparse
import unicodedata

import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer

from common.utils import load_yaml, repo_path, save_json, set_seed, wall_timer, write_jsonl


#lm-evaluation-harness evaluation of one of the seven conditions (manual: evaluation settings)
#  gsm8k_cot  zero-shot, non-thinking chat template, greedy, 1,024-new-token cap, stock stop rules + filters
#  hellaswag / boolq / arc_easy  zero-shot candidate likelihood, stock completion prompts, no chat template


def condition_path(cfg,name):
    """Which weights each evaluation condition loads."""
    out_dir=repo_path(cfg["outputs_dir"])
    if name=="baseline":
        return cfg["base_model"] #unchanged starting model
    elif name=="full_ft_mid":
        return str(out_dir/"full_ft"/"midpoint") #update 625
    elif name in ("full_ft","full_ft_l2sp"):
        return str(out_dir/name/"final") #fp32 weights, loaded as fp16 below
    else:
        return str(out_dir/name/"merged") #adapter merged into a fresh backbone copy (merge.py)


class TruncationRecordingHFLM(HFLM):
    """The stock lm-eval HF model. It only also records, per prompt, whether generation stopped on its own
    (eos or a stop string) or ran into the token cap. Re-tokenizing the returned text cannot tell this reliably,
    because repeated newlines and similar text merge into fewer tokens."""

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.generation_records={} #prompt text -> what happened during generation

    def _model_generate(self,context,max_length,stop,**generation_kwargs):
        mask=generation_kwargs.get("attention_mask") #left padding mask of the batch
        output=super()._model_generate(context,max_length,stop,**generation_kwargs)
        eos_ids=[self.tokenizer.eos_token_id,self.tokenizer.convert_tokens_to_ids("<|im_end|>")] #ids that end generation
        configured=self.model.generation_config.eos_token_id #qwen3: <|im_end|> and <|endoftext|>
        if isinstance(configured,int):
            eos_ids.append(configured)
        elif configured is not None:
            for token in configured:
                eos_ids.append(token)
        cap=max_length-context.shape[1] #max_gen_toks
        for row in range(context.shape[0]):
            prompt_ids=context[row]
            if mask is not None:
                prompt_ids=prompt_ids[mask[row].bool()] #drop the left padding
            new_ids=output[row,context.shape[1]:].tolist()
            generated=0
            hit_eos=False
            for token in new_ids:
                if token in eos_ids:
                    hit_eos=True
                    break
                generated=generated+1
            text=self.tokenizer.decode(new_ids[:generated])
            hit_stop_string=False
            for stop_string in stop:
                if stop_string in text:
                    hit_stop_string=True #lm-eval cuts the answer here
            self.generation_records[self.tokenizer.decode(prompt_ids)]={"generated_tokens":generated,"hit_eos":hit_eos,
                "hit_stop_string":hit_stop_string,"truncated":(hit_eos==False and hit_stop_string==False and generated>=cap)}
        return output


def run_gsm8k(cfg,model_path,tokenizer):
    eval_cfg=cfg["eval"]
    max_gen_toks=int(eval_cfg["max_gen_toks"]) #1,024
    model=TruncationRecordingHFLM(pretrained=model_path,dtype="float16",enable_thinking=False,batch_size=eval_cfg["gsm8k_batch_size"]) #fp16, non-thinking template
    output=lm_eval.simple_evaluate(
        model=model,
        tasks=[eval_cfg["gsm8k_task"]],
        num_fewshot=0,
        apply_chat_template=True,
        fewshot_as_multiturn=False,
        gen_kwargs="max_gen_toks="+str(max_gen_toks), #only the cap changes, do_sample=false and the until list stay stock
        log_samples=True,
        random_seed=int(cfg["seed"]),
        numpy_random_seed=int(cfg["seed"]),
        torch_random_seed=int(cfg["seed"]),
        fewshot_random_seed=int(cfg["seed"]),
    )
    task_name=eval_cfg["gsm8k_task"]
    scores=output["results"][task_name]
    #lm-eval logs one entry per (doc, filter), we join strict-match and flexible-extract per doc
    per_doc={}
    missing_records=0
    for sample in output["samples"][task_name]:
        doc_id=sample["doc_id"]
        if doc_id not in per_doc:
            response=sample["resps"][0][0] #generated text, already cut at the stop strings
            prompt=sample["arguments"][0][0] #the exact chat-templated prompt lm-eval generated from
            record=model.generation_records.get(prompt)
            if record is None:
                record=model.generation_records.get(unicodedata.normalize("NFC",prompt)) #qwen's tokenizer normalizes to NFC
            if record is None:
                missing_records=missing_records+1
                record={"generated_tokens":None,"hit_eos":None,"hit_stop_string":None,"truncated":None}
            per_doc[doc_id]={
                "doc_id":doc_id,
                "question":sample["doc"]["question"],
                "target":sample["target"],
                "response":response,
                "response_tokens":len(tokenizer(response,add_special_tokens=False)["input_ids"]), #re-tokenized, approximate
                "generated_tokens":record["generated_tokens"], #tokens before eos (includes text after a stop string)
                "hit_eos":record["hit_eos"],
                "hit_stop_string":record["hit_stop_string"],
                "truncated":record["truncated"], #ran into the 1,024-token cap
            }
        extracted=sample["filtered_resps"][0]
        per_doc[doc_id][sample["filter"]+"_answer"]=extracted
        per_doc[doc_id][sample["filter"]+"_correct"]=float(sample["exact_match"])
        per_doc[doc_id][sample["filter"]+"_extractable"]=bool(extracted!="[invalid]") #the regex filter returns [invalid] when nothing matched
    rows=[]
    for doc_id in sorted(per_doc):
        rows.append(per_doc[doc_id])
    truncated=0
    unextractable=0
    for row in rows:
        if row["truncated"]==True:
            truncated=truncated+1
        if row["flexible-extract_extractable"]==False:
            unextractable=unextractable+1
    summary={
        "flexible_exact_match":float(scores["exact_match,flexible-extract"]), #primary metric
        "flexible_exact_match_stderr":float(scores["exact_match_stderr,flexible-extract"]),
        "strict_exact_match":float(scores["exact_match,strict-match"]),
        "n":len(rows),
        "truncation_rate":truncated/len(rows),
        "flexible_unextractable_rate":unextractable/len(rows),
        "docs_without_generation_record":missing_records, #should be 0
    }
    return summary,rows


def run_likelihood(cfg,model_path):
    eval_cfg=cfg["eval"]
    output=lm_eval.simple_evaluate(
        model="hf",
        model_args="pretrained="+model_path+",dtype=float16",
        tasks=list(eval_cfg["other_tasks"]),
        num_fewshot=0,
        batch_size=eval_cfg["other_batch_size"],
        apply_chat_template=False, #stock completion prompts
        log_samples=True,
        random_seed=int(cfg["seed"]),
        numpy_random_seed=int(cfg["seed"]),
        torch_random_seed=int(cfg["seed"]),
        fewshot_random_seed=int(cfg["seed"]),
    )
    summary={}
    rows=[]
    for task_name in eval_cfg["other_tasks"]:
        scores=output["results"][task_name]
        entry={"acc":float(scores["acc,none"]),"acc_stderr":float(scores["acc_stderr,none"])}
        if "acc_norm,none" in scores:
            entry["acc_norm"]=float(scores["acc_norm,none"]) #boolq has only acc
            entry["acc_norm_stderr"]=float(scores["acc_norm_stderr,none"])
        entry["n"]=len(output["samples"][task_name])
        summary[task_name]=entry
        for sample in output["samples"][task_name]:
            row={"task":task_name,"doc_id":sample["doc_id"],"target":sample["target"],"acc":float(sample["acc"])}
            if "acc_norm" in sample:
                row["acc_norm"]=float(sample["acc_norm"])
            loglikelihoods=[]
            for response in sample["resps"]:
                loglikelihoods.append(float(response[0][0])) #log p(candidate | context)
            row["candidate_loglikelihoods"]=loglikelihoods
            rows.append(row)
    return summary,rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task1.yaml")
    ap.add_argument("--name",required=True,choices=["baseline","full_ft_mid","full_ft","full_ft_l2sp","lora4","lora16","oplora16"])
    args=ap.parse_args()
    cfg=load_yaml(args.config)
    set_seed(int(cfg["seed"]))
    model_path=condition_path(cfg,args.name)
    tokenizer=AutoTokenizer.from_pretrained(cfg["base_model"])
    results_dir=repo_path(cfg["results_dir"])/"eval"/args.name

    timer=wall_timer()
    gsm_summary,gsm_rows=run_gsm8k(cfg,model_path,tokenizer)
    gsm_seconds=timer()
    write_jsonl(results_dir/"gsm8k_samples.jsonl",gsm_rows)
    scores={"condition":args.name,"model_path":model_path,"gsm8k":gsm_summary,"gsm8k_seconds":gsm_seconds,"lm_eval_version":lm_eval.__version__}
    save_json(results_dir/"scores.json",scores)
    print(scores)

    timer=wall_timer()
    other_summary,other_rows=run_likelihood(cfg,model_path)
    scores["likelihood_seconds"]=timer()
    scores.update(other_summary)
    write_jsonl(results_dir/"likelihood_samples.jsonl",other_rows)
    save_json(results_dir/"scores.json",scores)
    print(scores)


if __name__=="__main__":
    main()
