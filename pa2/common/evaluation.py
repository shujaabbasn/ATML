from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from common.data import render_prompt
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.metrics import safe_corr, sample_entropy, sampled_kl, word_limit_compliance
from common.models import reference_mode


#shared by task 1 and task 2 evaluation so both use the exact same protocol
#(manual: keep the prompt set, decoding config, max length and seed fixed across conditions)


def pad_and_stack(tensors):
    """Right-pad a list of [batch, steps] tensors with zeros to one width and stack them."""
    width=0 #widest batch decides the final width
    for tensor in tensors:
        if tensor.shape[1]>width:
            width=tensor.shape[1] #new widest
    padded=[] #same tensors, zero padded on the right
    for tensor in tensors:
        extra=width-tensor.shape[1] #how many zero columns this batch needs
        padded.append(F.pad(tensor,(0,extra))) #pad only the last dim, on the right
    return torch.cat(padded,dim=0) #[all responses, width]


def length_stats(lengths):
    """Mean plus dispersion of response lengths (manual: mean and std or iqr)."""
    values=np.asarray(lengths,dtype=float) #response token counts
    if len(values)==0:
        return {} #nothing generated
    return {
        "mean":float(values.mean()), #average generated tokens
        "std":float(values.std()), #spread
        "median":float(np.median(values)), #middle response
        "q25":float(np.percentile(values,25)), #lower quartile
        "q75":float(np.percentile(values,75)), #upper quartile
        "iqr":float(np.percentile(values,75)-np.percentile(values,25)), #interquartile range
        "min":float(values.min()), #shortest
        "max":float(values.max()), #longest
    }


def generation_metrics(policy,tokenizer,reward_model,reward_tokenizer,prompt_rows,cfg,max_new_tokens,max_prompt_length,batch_size,reward_max_length,missing_eos_penalty=0.0,logprob_batch_size=2):
    """Generate one response per prompt and measure reward, kl, entropy, length and word limits.

    prompt_rows is a list of {"prompt_id":..,"messages":[..]} in a fixed order.
    """
    generation_cfg=cfg["generation"] #temperature/top_p/do_sample from base.yaml, same for every condition
    records=[] #one record per generated response
    policy_logp_batches=[] #per batch [batch, steps] policy log-probs of the sampled tokens
    ref_logp_batches=[] #same tokens under the frozen reference
    mask_batches=[] #valid response tokens
    for start in range(0,len(prompt_rows),batch_size):
        chunk=prompt_rows[start:start+batch_size] #fixed order, so every condition sees the same batches
        prompts=[] #chat messages for batch_generate
        for row in chunk:
            prompts.append(row["messages"]) #already a list of role/content dicts
        generated=batch_generate(
            policy,
            tokenizer,
            prompts,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
            temperature=float(generation_cfg["temperature"]),
            top_p=float(generation_cfg["top_p"]),
            do_sample=bool(generation_cfg["do_sample"]),
        )
        sequences=generated["sequences"].clone() #clone so these are normal tensors, not inference-mode ones
        attention_mask=generated["attention_mask"].clone() #prompt padding mask + ones over the response
        response_ids=generated["response_ids"].clone() #sampled tokens after the prompt
        response_mask=generated["response_mask"].clone().float() #1 up to and including eos, 0 after
        #score log-probs a few rows at a time: 8 rows x 768 tokens x 151k vocab in fp32 ran out of memory on the t4 (task 2)
        #same padded rows, just fewer at once, so the numbers are the same as scoring all 8 together
        policy_logp_parts=[] #log pi_theta for each slice of rows
        ref_logp_parts=[] #log pi_ref for each slice of rows
        with torch.no_grad():
            for row_start in range(0,sequences.shape[0],logprob_batch_size):
                row_end=row_start+logprob_batch_size #end of this slice
                part_logp,part_logits=response_token_logprobs(policy,sequences[row_start:row_end],attention_mask[row_start:row_end],generated["prompt_width"],response_ids[row_start:row_end]) #log pi_theta(a_t|s_t)
                del part_logits #only need the gathered log-probs, logits are huge (vocab 151k)
                policy_logp_parts.append(part_logp) #keep
                with reference_mode(policy):
                    part_ref_logp,part_ref_logits=response_token_logprobs(policy,sequences[row_start:row_end],attention_mask[row_start:row_end],generated["prompt_width"],response_ids[row_start:row_end]) #adapter off = frozen reference
                del part_ref_logits #same reason
                ref_logp_parts.append(part_ref_logp) #keep
        policy_logp=torch.cat(policy_logp_parts,dim=0) #back to [batch, steps]
        ref_logp=torch.cat(ref_logp_parts,dim=0) #same
        raw_rewards=score_reward_pairs(reward_model,reward_tokenizer,prompts,generated["responses"],max_length=reward_max_length).cpu() #course reward model, one scalar per response
        policy_logp=policy_logp.float().cpu() #move to cpu so gpu memory stays flat across batches
        ref_logp=ref_logp.float().cpu() #same
        response_mask=response_mask.cpu() #same
        policy_logp_batches.append(policy_logp) #kept for the corpus-level kl at the end
        ref_logp_batches.append(ref_logp) #same
        mask_batches.append(response_mask) #same
        for i in range(len(chunk)):
            row=chunk[i] #the prompt this response belongs to
            prompt_text=row["messages"][-1]["content"] #last user turn, word limits live here
            rendered_length=len(tokenizer(render_prompt(tokenizer,row["messages"]))["input_ids"]) #prompt tokens before any truncation
            raw_reward=float(raw_rewards[i].item()) #learned reward score
            effective_reward=raw_reward #reward after the missing-eos penalty (ppo config uses 1.0)
            if generated["terminated_with_eos"][i]==False:
                effective_reward=raw_reward-float(missing_eos_penalty) #no eos means the response got cut
            records.append({
                "prompt_id":row["prompt_id"], #fixed id so conditions can be paired up
                "prompt":prompt_text, #for qualitative examples
                "response":generated["responses"][i], #decoded text without special tokens
                "response_tokens":int(generated["response_lengths"][i]), #manual: generated tokens excluding padding
                "terminated_with_eos":bool(generated["terminated_with_eos"][i]), #finished on its own
                "truncated":bool(generated["truncated"][i]), #hit the generation cap
                "prompt_was_truncated":bool(rendered_length>max_prompt_length), #batch_generate cut the prompt
                "reward":raw_reward, #raw reward-model score
                "effective_reward":effective_reward, #after eos penalty
                "kl":float(sampled_kl(policy_logp[i],ref_logp[i],response_mask[i]).item()), #per-response kl, only for examples
                "word_limit_compliance":word_limit_compliance(prompt_text,generated["responses"][i]), #none if the prompt has no limit
            })

    if len(records)==0:
        return {},records #nothing to summarise

    #corpus-level kl and entropy with the course helpers, one masked mean over every response token
    all_policy_logp=pad_and_stack(policy_logp_batches) #[n responses, longest response]
    all_ref_logp=pad_and_stack(ref_logp_batches) #same shape
    all_mask=pad_and_stack(mask_batches) #zeros on padded columns so they never count
    kl=float(sampled_kl(all_policy_logp,all_ref_logp,all_mask).item()) #manual: released sampled-response estimator
    entropy=float(sample_entropy(all_policy_logp,all_mask).item()) #manual: mean token-level entropy (sampled estimate)

    rewards=[] #raw scores
    effective_rewards=[] #after eos penalty
    lengths=[] #response token counts
    truncated_count=0 #hit the cap
    eos_count=0 #ended with eos
    prompt_truncated_count=0 #prompt longer than max_prompt_length
    compliance_values=[] #only prompts that state a word limit
    for record in records:
        rewards.append(record["reward"]) #collect
        effective_rewards.append(record["effective_reward"]) #collect
        lengths.append(record["response_tokens"]) #collect
        if record["truncated"]==True:
            truncated_count=truncated_count+1 #count cut responses
        if record["terminated_with_eos"]==True:
            eos_count=eos_count+1 #count finished responses
        if record["prompt_was_truncated"]==True:
            prompt_truncated_count=prompt_truncated_count+1 #should be 0, reported so it is visible
        if record["word_limit_compliance"] is not None:
            compliance_values.append(record["word_limit_compliance"]) #1.0 or 0.0
    word_limit_rate=None #stays none when no prompt has a limit
    if len(compliance_values)>0:
        word_limit_rate=float(np.mean(compliance_values)) #fraction within the limit

    summary={
        "n_prompts":len(records), #how many responses were generated
        "reward_mean":float(np.mean(rewards)), #manual: reward-model score, only for within-task comparison
        "reward_std":float(np.std(rewards)), #spread of reward
        "effective_reward_mean":float(np.mean(effective_rewards)), #same after eos penalty
        "kl_from_reference":kl, #sampled kl, policy vs frozen reference
        "entropy":entropy, #diversity/confidence diagnostic, not quality
        "response_length":length_stats(lengths), #mean, std, median, iqr
        "truncation_rate":truncated_count/len(records), #fraction that hit the cap
        "eos_rate":eos_count/len(records), #fraction that ended on their own
        "prompts_truncated":prompt_truncated_count, #sanity check
        "word_limit_compliance":word_limit_rate, #none unless the prompts have word limits
        "word_limit_prompts_scored":len(compliance_values), #how many prompts that rate is over
        "reward_length_corr":safe_corr(rewards,lengths), #length bias check for the reward model
        "max_new_tokens":max_new_tokens, #generation cap used
        "max_prompt_length":max_prompt_length, #prompt cap used
        "decoding":dict(generation_cfg), #exact decoding settings
    }
    return summary,records


def pick_examples(records,how_many=5):
    """Small sets of responses worth reading for the qualitative section."""
    by_reward=sorted(records,key=lambda record:record["reward"]) #lowest reward first
    by_length=sorted(records,key=lambda record:record["response_tokens"]) #shortest first
    truncated=[] #responses that hit the cap
    for record in records:
        if record["truncated"]==True:
            truncated.append(record) #collect
    return {
        "highest_reward":by_reward[-how_many:], #check if high reward is actually good
        "lowest_reward":by_reward[:how_many], #check if low reward is actually bad
        "longest":by_length[-how_many:], #length bias check
        "shortest":by_length[:how_many], #brevity check
        "truncated":truncated[:how_many], #cut-off responses
    }
