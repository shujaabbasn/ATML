from __future__ import annotations

import argparse

from common.data import load_yaml, read_jsonl
from common.models import load_policy, load_reward_model, load_tokenizer

from common.data import prompt_messages, repo_path, write_jsonl
from common.evaluation import generation_metrics, pick_examples
from common.logging_utils import save_json, set_seed


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def evaluate_policy(bundle,adapter,name,batch_size=8,limit=None):
    """Same held-out protocol as task 2 (same prompts, reward model, decoding), results to results/task3_grpo/<name>/."""
    cfg=bundle["cfg"] #merged config
    reward_model,reward_tokenizer=bundle["reward"] #frozen course reward model
    rows=bundle["rows"] #held-out prompt pool, fixed order
    if limit is not None:
        rows=rows[:limit] #fixed prefix, recorded in the summary
    prompts=[] #prompt_id + messages
    for row in rows:
        prompts.append({"prompt_id":row["prompt_id"],"messages":prompt_messages(row)}) #same prompts for every condition
    set_seed(int(cfg["seed"])) #same sampling stream for every condition
    summary,records=generation_metrics(
        bundle["policy"],bundle["tokenizer"],reward_model,reward_tokenizer,prompts,cfg,
        max_new_tokens=int(cfg["cache_generation_cap"]), #768: the cache's cap and task 2's frozen-evaluation cap, so the two tasks compare
        max_prompt_length=int(cfg["max_prompt_length"]), #256, same as training
        batch_size=batch_size,
        reward_max_length=1280, #same as task 2
        missing_eos_penalty=0.0, #grpo has no eos penalty (it masks truncated completions instead)
    )
    summary["name"]=name #condition
    summary["adapter"]=str(adapter) #checkpoint
    summary["limit"]=limit #none means all eval prompts
    out_dir=repo_path(cfg["results_dir"])/name #results/task3_grpo/<name>/
    save_json(out_dir/"eval_summary.json",summary) #numbers for the report
    write_jsonl(out_dir/"eval_generations.jsonl",records) #every response
    save_json(out_dir/"examples.json",pick_examples(records)) #candidates for qualitative evidence
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--batch-size",type=int,default=8) #generation batch, does not change the protocol
    ap.add_argument("--limit",type=int) #first N eval prompts only; if used, use the SAME N for every condition
    args = ap.parse_args()
    bundle = load_evaluation_bundle(args.config, args.adapter)
    summary=evaluate_policy(bundle,args.adapter,args.name,args.batch_size,args.limit) #runs and saves
    print(summary) #quick look


if __name__ == "__main__":
    main()
