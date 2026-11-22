"""Task 1 driver. Every condition starts from the restored fp16 weights and goes through the same evaluation.

python -m task1_compression.run --condition baseline
python -m task1_compression.run --condition p1      (also p2, p3, q1, q2, q3, nm24, all)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from common.utils import load_yaml, save_json, write_jsonl, repo_path, set_seed, package_versions, data_path
from task1_compression.model_utils import load_model, eligible_linears, backup_state, restore_state, count_eligible, fp16_bytes_outside, quant_bytes_matrix, sparse_bytes_matrix, nm_bytes_matrix, input_scale_bytes
from task1_compression.evaluate import perplexity, hellaswag, hellaswag_items, resources, load_eval_data, HELLASWAG_TEMPLATE
from task1_compression.pruning import magnitude_prune, wanda_prune, uniform_counts, dense_input_norms, owl_outlier_ratio, owl_allocation, exact_row_counts, nm_prune
from task1_compression.quantization import rtn_quantize, awq_quantize, greedy_bit_allocation


def dev_windows(development):
    """The 32 development sequences as perplexity windows (all 511 next tokens scored). Only used for selection."""
    windows=[]
    for record in development:
        length=len(record["token_ids"])
        windows.append({"token_ids":record["token_ids"],"score_target_offsets":[1,length],"scored_tokens":length-1})
    return windows


@torch.no_grad()
def hidden_states(model,sequences):
    """Output of every block for a few dev sequences, kept on the cpu."""
    device=next(model.parameters()).device
    ids=[]
    for record in sequences:
        ids.append(record["token_ids"])
    ids=torch.tensor(ids,device=device)
    output=model(input_ids=ids,output_hidden_states=True,use_cache=False)
    states=[]
    for h in output.hidden_states[1:]: #index 0 is the embedding output
        states.append(h.detach().to("cpu",torch.float16))
    return states


def layer_errors(reference,current):
    """Relative error ||h_l - h_l_dense|| / ||h_l_dense|| per block on dev data (uniform vs adaptive diagnostic)."""
    out=[]
    for i in range(len(reference)):
        a=reference[i].float()
        b=current[i].float()
        out.append(float((b-a).norm()/a.norm()))
    return out


class Runner:
    def __init__(self,config,skip_timing):
        self.config=config
        self.skip_timing=skip_timing
        set_seed(config["seed"])
        self.model,self.tokenizer=load_model(config["model"])
        self.pad_id=self.tokenizer.eos_token_id #qwen has no pad token, padded positions are masked anyway
        self.entries=eligible_linears(self.model,config["eligible"])
        self.backup=backup_state(self.model)
        self.data=load_eval_data(config)
        self.calibration=self.data["calibration"]
        self.dev=dev_windows(self.data["development"])
        self.items=hellaswag_items(self.data["hellaswag"],config["eval"]["hellaswag_examples"])
        pool=[]
        for record in self.data["c4_val"]:
            pool.extend(record["token_ids"])
            if len(pool)>=4096:
                break
        self.token_pool=pool #real text tokens for the latency prompts
        self.dev_reference=hidden_states(self.model,self.data["development"][:config["diagnostic"]["dev_sequences"]])
        self.calibration_reference=hidden_states(self.model,self.calibration[:config["diagnostic"]["calibration_sequences"]]) #manual P1: layer-output error on calibration activations
        self.eligible_count=count_eligible(self.entries)
        self.outside_bytes=fp16_bytes_outside(self.model,self.entries)
        self.out_dir=repo_path(config["results_dir"])
        save_json(self.out_dir/"setup.json",{"model":config["model"],"eligible_matrices":len(self.entries),"eligible_weights":self.eligible_count,"fp16_bytes_outside_eligible":self.outside_bytes,"hellaswag_template":HELLASWAG_TEMPLATE,"config":config,"versions":package_versions(),"calibration_sequences":len(self.calibration),"calibration_source_rows":self.source_rows(self.calibration)})

    def restore(self):
        restore_state(self.model,self.backup)

    def source_rows(self,records):
        rows=[]
        for record in records:
            rows.append(record["source_row"]) #c4 row id, for the report and the overlap check
        return rows

    def dev_ppl(self):
        summary,nll=perplexity(self.model,self.dev,self.config["eval"]["ppl_batch"],self.pad_id)
        return summary["ppl"]

    def storage(self,kind,records):
        """Packed storage of the eligible weights in the method's own format + fp16 everything else."""
        if kind=="dense":
            eligible_bytes=self.eligible_count*2
            nominal="fp16"
        elif kind=="sparse":
            eligible_bytes=0
            for record in records:
                eligible_bytes=eligible_bytes+sparse_bytes_matrix(record["rows"],record["columns"],record["nonzero"])
            nominal="1 mask bit + 16 bits per nonzero"
        elif kind=="nm24":
            eligible_bytes=0
            for record in records:
                eligible_bytes=eligible_bytes+nm_bytes_matrix(record["rows"],record["columns"],2,4)
            nominal="2:4, fp16 values + 2-bit indices"
        else:
            eligible_bytes=0
            for record in records:
                eligible_bytes=eligible_bytes+record["bytes"]
            eligible_bytes=eligible_bytes+input_scale_bytes(self.model) #awq o_proj scales
            nominal="int, group 128, fp16 scale + int zero-point"
        total=eligible_bytes+self.outside_bytes
        baseline_total=self.eligible_count*2+self.outside_bytes
        return {"format":nominal,"eligible_packed_bytes":int(eligible_bytes),"bits_per_eligible_weight":8*eligible_bytes/self.eligible_count,"fp16_outside_bytes":int(self.outside_bytes),"total_packed_bytes":int(total),"compression_ratio_total":baseline_total/total,"compression_ratio_eligible":(self.eligible_count*2)/eligible_bytes,"in_memory_representation":"dense fp16 tensors (zero-masked or fake-quantized), no sparse/low-bit kernel executed","in_memory_eligible_bytes":int(self.eligible_count*2)}

    def evaluate(self,name,kind,records,extra):
        """Full evaluation of the current weights, written to results/task1/<name>/."""
        started=time.time()
        folder=self.out_dir/name
        config=self.config
        wiki,wiki_nll=perplexity(self.model,self.data["wikitext"],config["eval"]["ppl_batch"],self.pad_id)
        c4,c4_nll=perplexity(self.model,self.data["c4_val"],config["eval"]["ppl_batch"],self.pad_id)
        hs,per_example=hellaswag(self.model,self.tokenizer,self.items,config["eval"]["hellaswag_batch"],self.pad_id)
        diagnostic=per_example[:config["eval"]["hellaswag_diagnostic"]]
        diag_acc=0
        diag_norm=0
        for row in diagnostic:
            diag_acc=diag_acc+row["acc"]
            diag_norm=diag_norm+row["acc_norm"]
        zeros=0
        for full,block_index,short,module in self.entries:
            zeros=zeros+int((module.weight.data==0).sum())
        metrics={"condition":name,"kind":kind,"wikitext2_ppl":wiki,"c4_val_ppl":c4,"hellaswag":hs,
            "hellaswag_diagnostic512":{"acc":diag_acc/len(diagnostic),"acc_norm":diag_norm/len(diagnostic),"examples":len(diagnostic)},
            "dev_ppl":self.dev_ppl(),"achieved_sparsity_eligible":zeros/self.eligible_count,
            "storage":self.storage(kind,records),"layer_relative_error_dev":layer_errors(self.dev_reference,hidden_states(self.model,self.data["development"][:config["diagnostic"]["dev_sequences"]])),
            "layer_relative_error_calibration":layer_errors(self.calibration_reference,hidden_states(self.model,self.calibration[:config["diagnostic"]["calibration_sequences"]]))}
        if self.skip_timing==False:
            metrics["resources"]=resources(self.model,self.token_pool,config["timing"]["prefill_lengths"],config["timing"]["decode_tokens"],config["timing"]["warmup"],config["timing"]["repeats"])
        metrics.update(extra)
        metrics["eval_seconds"]=time.time()-started
        save_json(folder/"metrics.json",metrics)
        save_json(folder/"matrices.json",records)
        wiki_lengths=[]
        for record in self.data["wikitext"]:
            wiki_lengths.append(record["scored_tokens"])
        c4_lengths=[]
        for record in self.data["c4_val"]:
            c4_lengths.append(record["scored_tokens"])
        np.savez_compressed(folder/"token_nll.npz",wikitext=wiki_nll,c4_val=c4_nll,wikitext_lengths=np.array(wiki_lengths),c4_val_lengths=np.array(c4_lengths)) #window lengths: the bootstrap resamples whole windows
        write_jsonl(folder/"hellaswag_scores.jsonl",per_example)
        print(name,"wiki",round(wiki["ppl"],3),"c4",round(c4["ppl"],3),"hs",round(hs["acc"],4),round(hs["acc_norm"],4),flush=True)
        return metrics

    #---------------- conditions ----------------

    def baseline(self):
        self.restore()
        records=[]
        for full,block_index,short,module in self.entries:
            records.append({"name":full,"block":block_index,"proj":short,"rows":module.weight.shape[0],"columns":module.weight.shape[1]})
        self.evaluate("baseline_fp16","dense",records,{"eligible_matrices":len(self.entries)})

    def p1(self):
        for sparsity in self.config["pruning"]["sparsities"]:
            self.restore()
            started=time.time()
            records=magnitude_prune(self.entries,sparsity)
            seconds=time.time()-started
            self.evaluate(f"p1_magnitude_s{int(sparsity*100)}","sparse",records,{"target_sparsity":sparsity,"compression_seconds":seconds})

    def uniform_counts_for(self,sparsity):
        def counts_for(full,rows,columns,device):
            return uniform_counts(rows,columns,sparsity,device)
        return counts_for

    def p2(self):
        for sparsity in self.config["pruning"]["sparsities"]:
            self.restore()
            score_log=[]
            started=time.time()
            records=wanda_prune(self.model,self.entries,self.calibration,self.uniform_counts_for(sparsity),self.config["calibration"],self.config["seed"],score_log)
            seconds=time.time()-started
            save_json(self.out_dir/f"p2_wanda_s{int(sparsity*100)}"/"score_distributions.json",score_log)
            self.evaluate(f"p2_wanda_s{int(sparsity*100)}","sparse",records,{"target_sparsity":sparsity,"compression_seconds":seconds})

    def p3(self):
        """OWL-style nonuniform Wanda. lambda is picked on the development set only (never on test sets)."""
        config=self.config["pruning"]
        self.restore()
        norms=dense_input_norms(self.model,self.calibration,self.config["calibration"],self.config["seed"])
        ratio=owl_outlier_ratio(self.entries,norms,config["owl_threshold"])
        block_sizes={}
        for full,block_index,short,module in self.entries:
            if block_index not in block_sizes:
                block_sizes[block_index]=0
            block_sizes[block_index]=block_sizes[block_index]+module.weight.numel()
        for target in config["adaptive_targets"]:
            selection=[]
            candidates=[0.0]+list(config["owl_lambdas"]) #lambda=0 is uniform wanda, kept as the reference row
            best=None
            for lam in candidates:
                allocation=owl_allocation(ratio,block_sizes,target,lam,config["min_sparsity"],config["max_sparsity"])
                counts,goal,total=exact_row_counts(self.entries,allocation,target)
                self.restore()
                wanda_prune(self.model,self.entries,self.calibration,self.counts_from(counts),self.config["calibration"],self.config["seed"])
                ppl=self.dev_ppl()
                selection.append({"lambda":lam,"dev_ppl":ppl,"allocation":allocation})
                print("p3 target",target,"lambda",lam,"dev ppl",round(ppl,3),flush=True)
                if lam>0 and (best is None or ppl<best[0]):
                    best=(ppl,lam,allocation,counts)
            folder=self.out_dir/f"p3_owl_s{int(target*100)}"
            save_json(folder/"selection_dev.json",{"outlier_ratio":ratio,"threshold":config["owl_threshold"],"candidates":selection,"chosen_lambda":best[1],"rule":"lowest development-set perplexity among lambda>0"})
            self.restore()
            started=time.time()
            records=wanda_prune(self.model,self.entries,self.calibration,self.counts_from(best[3]),self.config["calibration"],self.config["seed"])
            seconds=time.time()-started
            self.evaluate(f"p3_owl_s{int(target*100)}","sparse",records,{"target_sparsity":target,"lambda":best[1],"allocation":best[2],"outlier_ratio":ratio,"compression_seconds":seconds})

    def counts_from(self,counts):
        def counts_for(full,rows,columns,device):
            return torch.tensor(counts[full],dtype=torch.long,device=device)
        return counts_for

    def q1(self):
        for bits in self.config["quant"]["bits"]:
            self.restore()
            started=time.time()
            records=rtn_quantize(self.entries,bits,self.config["quant"]["group_size"])
            seconds=time.time()-started
            self.evaluate(f"q1_rtn_int{bits}","quant",records,{"bits":bits,"compression_seconds":seconds})

    def constant_bits(self,bits):
        def bits_of(short,module):
            return bits
        return bits_of

    def q2(self):
        quant=self.config["quant"]
        #end-to-end check that the scaling alone (no quantization) keeps the model's outputs
        self.restore()
        before=hidden_states(self.model,self.data["development"][:2])
        before_ppl=self.dev_ppl()
        awq_quantize(self.model,self.entries,self.calibration,self.constant_bits(4),quant["group_size"],quant["awq_grid"],self.config["calibration"],self.config["seed"],quantize=False)
        after=hidden_states(self.model,self.data["development"][:2])
        after_ppl=self.dev_ppl()
        save_json(self.out_dir/"q2_scaling_identity.json",{"layer_relative_change":layer_errors(before,after),"dev_ppl_before":before_ppl,"dev_ppl_after_scaling_only":after_ppl})
        for bits in quant["bits"]:
            self.restore()
            started=time.time()
            records,block_log,unused=awq_quantize(self.model,self.entries,self.calibration,self.constant_bits(bits),quant["group_size"],quant["awq_grid"],self.config["calibration"],self.config["seed"])
            seconds=time.time()-started
            save_json(self.out_dir/f"q2_awq_int{bits}"/"awq_search.json",block_log)
            self.evaluate(f"q2_awq_int{bits}","quant",records,{"bits":bits,"compression_seconds":seconds})

    def q3(self):
        quant=self.config["quant"]
        self.restore()
        #sensitivity probe: awq scales searched at 4 bits, error of every linear at every allowed width
        unused,probe_log,sensitivity=awq_quantize(self.model,self.entries,self.calibration,self.constant_bits(4),quant["group_size"],quant["awq_grid"],self.config["calibration"],self.config["seed"],quantize=False,probe_bits=quant["mixed_bits"])
        save_json(self.out_dir/"q3_sensitivity.json",sensitivity)
        name_of={}
        for full,block_index,short,module in self.entries:
            name_of[id(module)]=f"{block_index}.{short}"
        for target in quant["mixed_targets"]:
            budget=0
            for full,block_index,short,module in self.entries:
                budget=budget+quant_bytes_matrix(module.weight.shape[0],module.weight.shape[1],target,quant["group_size"]) #same as q2 uniform int<target> (o_proj scales are equal in both)
            bits,used=greedy_bit_allocation(sensitivity,quant["mixed_bits"],budget,quant["group_size"])

            def bits_of(short,module):
                return bits[name_of[id(module)]]

            self.restore()
            started=time.time()
            records,block_log,unused=awq_quantize(self.model,self.entries,self.calibration,bits_of,quant["group_size"],quant["awq_grid"],self.config["calibration"],self.config["seed"])
            seconds=time.time()-started
            folder=self.out_dir/f"q3_mixed_match_int{target}"
            save_json(folder/"allocation.json",{"bits":bits,"budget_bytes":budget,"allocated_bytes":used,"gap_bytes":budget-used,"rule":"greedy: largest relative-output-error drop per extra byte","widths":quant["mixed_bits"]})
            save_json(folder/"awq_search.json",block_log)
            self.evaluate(f"q3_mixed_match_int{target}","quant",records,{"budget_from":f"q2 int{target}","budget_bytes":budget,"allocated_bytes":used,"compression_seconds":seconds})

    def nm24(self):
        self.restore()
        started=time.time()
        records=nm_prune(self.model,self.entries,self.calibration,self.config["calibration"],self.config["seed"],2,4)
        seconds=time.time()-started
        self.evaluate("nm24_wanda_s50","nm24",records,{"target_sparsity":0.5,"compression_seconds":seconds,"note":"quality/representation study, no 2:4 sparse tensor-core kernel was executed (t4 has none)"})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/task1.yaml")
    parser.add_argument("--condition",required=True,choices=["baseline","p1","p2","p3","q1","q2","q3","nm24","all"])
    parser.add_argument("--skip_timing",action="store_true")
    args=parser.parse_args()
    config=load_yaml(args.config)
    runner=Runner(config,args.skip_timing)
    if args.condition=="all":
        order=["baseline","p1","p2","p3","q1","q2","q3","nm24"]
    else:
        order=[args.condition]
    for condition in order:
        getattr(runner,condition)()
        runner.restore()


if __name__=="__main__":
    main()
