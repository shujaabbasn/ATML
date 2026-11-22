"""Step 1 correctness tests for the speculative sampler.

python -m task3_speculative.sampler_tests            toy markov tests + real-model cache tests
python -m task3_speculative.sampler_tests --toy_only
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import torch
from scipy import stats as scipy_stats

from common.utils import load_yaml, save_json, repo_path, get_device
from task3_speculative.sampler import verify_block, speculative_generate, plain_generate, make_generator, CachedModel
from task3_speculative.common import load_causal, load_tokenizer, eos_ids, chat_ids, test_prompts


#a 3-token markov chain: initial distribution + transition matrix, for target p and draft q (ours, fixed numbers)
P0=torch.tensor([0.5,0.3,0.2])
P=torch.tensor([[0.6,0.3,0.1],[0.2,0.5,0.3],[0.3,0.3,0.4]])
Q0=torch.tensor([0.2,0.4,0.4])
Q=torch.tensor([[0.3,0.3,0.4],[0.5,0.2,0.3],[0.1,0.6,0.3]])


def row_for(previous,start,transition):
    if previous is None:
        return start
    return transition[previous]


def toy_generate(length,gamma,generator,buggy,p_start,p_step,q_start,q_step):
    """Same verify_block as the real sampler, with probability rows read from the toy tables."""
    seq=[]
    rounds=[]
    while len(seq)<length:
        g=min(gamma,length-len(seq)-1)
        previous=None
        if len(seq)>0:
            previous=seq[-1]
        drafted=[]
        q_rows=[]
        for i in range(g):
            q=row_for(previous,q_start,q_step)
            x=int(torch.multinomial(q,1,generator=generator))
            drafted.append(x)
            q_rows.append(q)
            previous=x
        p_rows=[]
        previous=None
        if len(seq)>0:
            previous=seq[-1]
        for i in range(g+1):
            p_rows.append(row_for(previous,p_start,p_step))
            if i<g:
                previous=drafted[i]
        if g>0:
            q_stack=torch.stack(q_rows)
        else:
            q_stack=None
        accepted,extra,all_accepted=verify_block(torch.stack(p_rows),q_stack,drafted,generator,buggy)
        rounds.append((g,len(accepted)))
        seq.extend(accepted)
        seq.append(extra)
    return tuple(seq[:length]),rounds


def exact_probability(sequence,start,step):
    probability=float(start[sequence[0]])
    for i in range(1,len(sequence)):
        probability=probability*float(step[sequence[i-1],sequence[i]])
    return probability


def chi_square(counts,trials,length,min_expected):
    """Pearson chi-square of observed sequence counts vs the exact target law; cells with expected < min_expected
    are pooled into one cell so every tested cell has expected count >= min_expected."""
    observed=[]
    expected=[]
    pooled_observed=0
    pooled_expected=0.0
    for sequence in itertools.product(range(3),repeat=length):
        e=trials*exact_probability(sequence,P0,P)
        o=counts.get(sequence,0)
        if e>=min_expected:
            observed.append(o)
            expected.append(e)
        else:
            pooled_observed=pooled_observed+o
            pooled_expected=pooled_expected+e
    if pooled_expected>0:
        observed.append(pooled_observed)
        expected.append(pooled_expected)
    expected=np.array(expected,dtype=np.float64)
    expected=expected*trials/expected.sum() #the float32 tables sum to 1 only up to rounding
    result=scipy_stats.chisquare(np.array(observed,dtype=np.float64),expected)
    return {"statistic":float(result.statistic),"p_value":float(result.pvalue),"cells":len(observed),"min_expected":float(min(expected))}


def toy_tests(config):
    tests=config["sampler_tests"]
    out={}
    for buggy in [False,True]:
        generator=torch.Generator().manual_seed(config["seed"])
        counts={}
        for trial in range(tests["trials"]):
            sequence,rounds=toy_generate(tests["length"],tests["gamma"],generator,buggy,P0,P,Q0,Q)
            counts[sequence]=counts.get(sequence,0)+1
        result=chi_square(counts,tests["trials"],tests["length"],tests["min_expected"])
        if buggy==False:
            result["passes"]=result["p_value"]>=tests["alpha"] #correct sampler: no rejection at alpha=0.01
            out["exact_sampler"]=result
        else:
            result["rejected"]=result["p_value"]<tests["alpha"] #negative control must be rejected
            out["negative_control_sample_from_p"]=result
        print("toy buggy=",buggy,result,flush=True)
    #bonus-token test: when q equals p every draft is accepted and each round emits gamma+1 tokens
    generator=torch.Generator().manual_seed(config["seed"])
    full_rounds=0
    total_rounds=0
    for trial in range(2000):
        sequence,rounds=toy_generate(7,3,generator,False,P0,P,P0,P)
        for g,accepted in rounds:
            total_rounds=total_rounds+1
            if accepted==g:
                full_rounds=full_rounds+1
    out["bonus_token_q_equals_p"]={"rounds":total_rounds,"rounds_all_accepted":full_rounds,"passes":full_rounds==total_rounds}
    return out


@torch.no_grad()
def real_tests(config):
    """Prefill equivalence, cache rollback against a full recompute, cap handling and draft==target acceptance."""
    device=get_device()
    tests=config["sampler_tests"]
    tokenizer=load_tokenizer(config)
    target=load_causal(config["target"])
    draft=load_causal(config["draft"])
    eos_list=eos_ids(tokenizer)
    prompts=test_prompts(config)
    chosen=[prompts[0],prompts[1],prompts[-1]][:tests["real_prompts"]]
    rows=[]
    for index in range(len(chosen)):
        prompt=chat_ids(tokenizer,chosen[index]["prompt"])
        #1. one-shot prefill vs prefill(prompt[:-1]) + one cached step
        whole=target(input_ids=torch.tensor([prompt],device=device)).logits[0,-1].float()
        m=CachedModel(target)
        m.feed(prompt[:-1],device)
        stepped=m.feed(prompt[-1:],device)[-1].float()
        prefill_diff=float((whole-stepped).abs().max())
        #2. every probability row used during speculative sampling vs a fresh full forward of the final sequence
        generator=make_generator(device,config["seed"]+index)
        response,stats,recorded,t_state=speculative_generate(target,draft,prompt,4,tests["real_tokens"],eos_list,generator,record=True)
        full_sequence=prompt+response
        full_logits=target(input_ids=torch.tensor([full_sequence],device=device)).logits[0].float()
        full_probs=torch.softmax(full_logits,dim=-1)
        worst=0.0
        for position,row in recorded:
            if position<len(full_sequence):
                diff=float((full_probs[position-1].cpu()-row).abs().max()) #logit at position-1 predicts token at position
                if diff>worst:
                    worst=diff
        #3. the final target cache: feeding the uncached tail must give the full-forward last row
        tail_logits=t_state.feed(full_sequence[t_state.cached:],device)[-1].float()
        cache_diff=float((tail_logits-full_logits[-1]).abs().max())
        #4. cap: gamma 4 with a 5-token cap never returns more than 5 tokens
        short,short_stats=speculative_generate(target,draft,prompt,4,5,eos_list,make_generator(device,1),ignore_eos=True)
        #5. draft == target: every proposal is accepted
        same,same_stats=speculative_generate(target,target,prompt,4,24,eos_list,make_generator(device,2),ignore_eos=True)
        #7. eos: rerun with the same seed and one generated token declared as eos -> output stops right at its first occurrence
        stop_token=response[len(response)//2]
        first=response.index(stop_token)
        stopped,stopped_stats=speculative_generate(target,draft,prompt,4,tests["real_tokens"],[stop_token],make_generator(device,config["seed"]+index))
        eos_ok=(stopped==response[:first+1])
        #8. one-token proposals (gamma 1) and the cap still hold
        single,single_stats=speculative_generate(target,draft,prompt,1,12,eos_list,make_generator(device,4),ignore_eos=True)
        single_ok=(len(single)==12 and max(single_stats["proposed_per_round"])<=1)
        #6. same-condition target-only determinism: same seed, same tokens
        a,unused=plain_generate(target,prompt,16,eos_list,make_generator(device,3))
        b,unused=plain_generate(target,prompt,16,eos_list,make_generator(device,3))
        rows.append({"prompt_id":chosen[index]["id"],"prefill_max_abs_logit_diff":prefill_diff,"rollback_max_abs_prob_diff":worst,"final_cache_max_abs_logit_diff":cache_diff,
            "spec_tokens":len(response),"spec_rounds":stats["rounds"],"cap5_length":len(short),"cap5_ok":len(short)==5,
            "draft_equals_target_acceptance":same_stats["accepted"]/max(same_stats["proposed"],1),"target_only_seed_repeatable":a==b,
            "eos_stop_ok":eos_ok,"gamma1_ok":single_ok,"variable_proposal_lengths_seen":sorted(set(short_stats["proposed_per_round"]+stats["proposed_per_round"])),
            "padding":"not applicable: batch 1, no padded positions"})
        print(rows[-1],flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/task3.yaml")
    parser.add_argument("--toy_only",action="store_true")
    args=parser.parse_args()
    config=load_yaml(args.config)
    out={"config":config["sampler_tests"],"toy_tables":{"P0":P0.tolist(),"P":P.tolist(),"Q0":Q0.tolist(),"Q":Q.tolist()},"toy":toy_tests(config)}
    if args.toy_only==False:
        out["real"]=real_tests(config)
    save_json(repo_path(config["results_dir"])/"sampler_tests.json",out)


if __name__=="__main__":
    main()
