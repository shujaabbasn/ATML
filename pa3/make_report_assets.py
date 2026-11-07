"""Build every table and figure used in the PA3 report from the saved result files.

Run from this pa3/ folder:  python make_report_assets.py
Writes PDF figures to figures/ and LaTeX tables and facts.json to outputs/tables/.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg") #no display needed
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent #this pa3 folder
RESULTS=ROOT/"results" #saved experiment outputs in the repo
TABLES=ROOT/"outputs"/"tables" #latex snippets, outputs/ is not in the repo
FIGURES=ROOT/"figures" #pdf figures
TABLES.mkdir(parents=True,exist_ok=True) #make sure they exist
FIGURES.mkdir(parents=True,exist_ok=True)

plt.rcParams.update({"font.size":8,"axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False}) #small plots for a two column width

CONDITIONS=["baseline","full_ft_mid","full_ft","full_ft_l2sp","lora4","lora16","oplora16"] #seven evaluation conditions
RUNS=["full_ft","full_ft_l2sp","lora4","lora16","oplora16"] #five training runs
NAMES={"baseline":"Baseline","full_ft_mid":"Full FT (mid)","full_ft":"Full FT","full_ft_l2sp":"Full FT + L2-SP","lora4":"LoRA-4","lora16":"LoRA-16","oplora16":"OPLoRA-16"}
COLORS={"full_ft":"#1f4e9c","full_ft_l2sp":"#d9731a","lora4":"#2d8a4e","lora16":"#b0312f","oplora16":"#7a4fa3"}


def read_json(path):
    return json.load(open(path)) #parsed json


def read_jsonl(path):
    rows=[] #one dict per line
    for line in open(path):
        if line.strip():
            rows.append(json.loads(line)) #parse
    return rows


def pct(value,digits=1):
    return f"{float(value)*100:.{digits}f}" #fraction to percent


def signed(value,digits=1):
    text=f"{float(value)*100:+.{digits}f}" #fraction to signed percentage points
    if text=="-0.0" or text=="+0.0":
        return "0.0" #no signed zero in the table
    return text.replace("-","$-$") #proper minus sign in latex


def write_table(name,lines):
    (TABLES/name).write_text("\n".join(lines)+"\n") #save snippet
    print("wrote",TABLES/name)


def smooth(values,window):
    kernel=np.ones(window)/window #plain moving average
    return np.convolve(values,kernel,mode="valid")


# ---------------- Task 1 ----------------
def scores(condition):
    return read_json(RESULTS/"task1"/"eval"/condition/"scores.json") #all benchmarks for one condition


def t1_last_number_score(condition):
    """Share right on the stock filter or when the last number in the response equals the target (my check, not a harness metric)."""
    right=0
    total=0
    dollars=0 #responses where the stock filter took $$ as the answer
    for row in read_jsonl(RESULTS/"task1"/"eval"/condition/"gsm8k_samples.jsonl"):
        total=total+1
        if row["flexible-extract_answer"]=="$$":
            dollars=dollars+1
        found=re.findall(r"-?\d[\d,]*(?:\.\d+)?",row["response"]) #every number in the response
        last_matches=False
        if len(found)>0:
            last_matches=float(found[-1].replace(",",""))==float(row["target"].replace(",",""))
        if row["flexible-extract_correct"]==1.0 or last_matches==True:
            right=right+1
    return right/total,dollars


def t1_scores_table():
    base=scores("baseline") #changes are measured from here
    lines=[] #table body
    for condition in CONDITIONS:
        s=scores(condition)
        cells=[NAMES[condition]]
        for value,start in [(s["gsm8k"]["flexible_exact_match"],base["gsm8k"]["flexible_exact_match"]),(s["hellaswag"]["acc_norm"],base["hellaswag"]["acc_norm"]),(s["boolq"]["acc"],base["boolq"]["acc"]),(s["arc_easy"]["acc_norm"],base["arc_easy"]["acc_norm"])]:
            if condition=="baseline":
                cells.append(pct(value)+" (--)") #no change for the baseline itself
            else:
                cells.append(pct(value)+" ("+signed(value-start)+")") #score and change in points
        cells.insert(2,pct(t1_last_number_score(condition)[0])) #gsm8k rescored on the last number
        cells.append(pct(s["hellaswag"]["acc"])) #raw hellaswag
        cells.append(pct(s["arc_easy"]["acc"])) #raw arc
        cells.append(pct(s["gsm8k"]["truncation_rate"])) #gsm8k truncation
        lines.append(" & ".join(cells)+" \\\\")
    write_table("t1_scores.tex",lines)


def t1_resources_table():
    lines=[] #table body
    for run in RUNS:
        s=read_json(RESULTS/"task1"/"train"/run/"train_summary.json")
        log=read_jsonl(RESULTS/"task1"/"train"/run/"train_log.jsonl")
        last=[]
        for row in log[-50:]:
            last.append(row["sft_loss"]) #final training loss level
        displacement="--"
        if "displacement_l2" in s:
            displacement=f"{s['displacement_l2']['u1250']:.2f}" #full model displacement at final
        lines.append(" & ".join([NAMES[run],f"{s['trainable_parameters']/1e6:.1f}",f"{s['train_seconds']/3600:.2f}",f"{s['peak_vram_gib']:.1f}",f"{s['checkpoint_bytes']['final']/1e6:.0f}",f"{float(np.mean(last)):.3f}",displacement])+" \\\\")
    write_table("t1_resources.tex",lines)


def t1_diagnostics_table():
    weights={} #(condition, layer) -> row
    for row in read_json(RESULTS/"task1"/"diagnostics"/"weight_space.json"):
        weights[(row["condition"],row["layer"])]=row
    activations={} #(condition, layer, kind) -> row
    for row in read_json(RESULTS/"task1"/"diagnostics"/"activation_summary.json"):
        activations[(row["condition"],row["layer"],row["kind"])]=row
    lines=[] #table body
    for condition in ["full_ft_mid","full_ft","full_ft_l2sp","lora4","lora16","oplora16"]:
        cells=[NAMES[condition]]
        for layer in [0,13,27]:
            rho=weights[(condition,layer)]["rho_k"]
            if rho<0.001:
                cells.append(f"{rho:.0e}".replace("e-0","e-")) #essentially zero, keep the exponent
            else:
                cells.append(f"{rho:.3f}")
        for layer in [0,13,27]:
            cells.append(f"{weights[(condition,layer)]['eta']*100:.2f}") #eta in percent of the starting norm
        for layer in [0,13,27]:
            cells.append(f"{activations[(condition,layer,'math')]['mean_R']:.3f}") #math prompts
            cells.append(f"{activations[(condition,layer,'non_math')]['mean_R']:.3f}") #non math prompts
        lines.append(" & ".join(cells)+" \\\\")
    write_table("t1_diagnostics.tex",lines)


def t1_qualitative_table():
    rows=list(csv.DictReader(open(RESULTS/"task1"/"qualitative"/"annotation_sheet.csv")))
    lines=[] #table body
    facts={}
    for condition in CONDITIONS:
        comply=0 #full format compliance on json and template prompts
        good=0 #coherent and relevant
        repeat=0 #some or severe repetition
        severe=0
        math=0 #math framing
        bad=0 #inappropriate content
        cut=0 #truncated
        for row in rows:
            if row["condition"]!=condition:
                continue
            if row["format_compliance"]=="yes":
                comply=comply+1
            if row["coherence_relevance"]=="good":
                good=good+1
            if row["repetition"]!="none":
                repeat=repeat+1
            if row["repetition"]=="severe":
                severe=severe+1
            if row["unnecessary_math_framing"]=="yes":
                math=math+1
            if row["inappropriate_content"]=="yes":
                bad=bad+1
            if row["truncated"]=="True":
                cut=cut+1
        facts[condition]={"comply":comply,"good":good,"repeat":repeat,"severe":severe,"math":math,"inappropriate":bad,"truncated":cut}
        lines.append(" & ".join([NAMES[condition],str(comply)+"/4",str(good)+"/6",str(repeat)+"/6",str(severe)+"/6",str(math)+"/6",str(bad)+"/6",str(cut)+"/6"])+" \\\\")
    write_table("t1_qualitative.tex",lines)
    return facts


def t1_loss_figure():
    fig,ax=plt.subplots(figsize=(5.5,1.6))
    for run in RUNS:
        log=read_jsonl(RESULTS/"task1"/"train"/run/"train_log.jsonl")
        values=[]
        for row in log:
            values.append(row["sft_loss"]) #the sft loss, not the l2-sp total
        style="-"
        if run=="full_ft" or run=="lora16":
            style="--" #these two lie under another curve, dashes keep both visible
        ax.plot(np.arange(25,len(values)+1),smooth(values,25),color=COLORS[run],linewidth=1,linestyle=style,label=NAMES[run])
    ax.set_xlabel("Optimizer update")
    ax.set_ylabel("SFT loss")
    ax.set_ylim(0.15,0.45)
    ax.legend(ncol=5,loc="upper right",fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGURES/"t1_loss.pdf")
    plt.close(fig)


def t1_l2sp_facts():
    log=read_jsonl(RESULTS/"task1"/"train"/"full_ft_l2sp"/"train_log.jsonl")
    summary=read_json(RESULTS/"task1"/"train"/"full_ft_l2sp"/"train_summary.json")
    return {"omega_final":log[-1]["omega"],"lambda_omega_final":log[-1]["lambda_omega"],"sft_final":log[-1]["sft_loss"],"reference_gb":summary["l2sp_reference_storage_bytes"]/1e9,"penalty_seconds":summary["l2sp_penalty_and_transfer_seconds"]}


def t1_boolq_yes_rates():
    """How often each condition prefers yes on boolq, candidate order in lm-eval is [no, yes]."""
    rates={}
    for condition in CONDITIONS:
        yes=0
        gold=0
        total=0
        for row in read_jsonl(RESULTS/"task1"/"eval"/condition/"likelihood_samples.jsonl"):
            if row["task"]!="boolq":
                continue
            if row["candidate_loglikelihoods"][1]>row["candidate_loglikelihoods"][0]:
                yes=yes+1
            if row["target"]==1:
                gold=gold+1
            total=total+1
        rates[condition]=yes/total
        rates["gold"]=gold/total
    return rates


# ---------------- Task 2 ----------------
TOKEN_FOLDERS=["tokens196","tokens49"]


def t2_final_table():
    lines=[] #table body
    for folder,suites in [("tokens196",["grounding_test","vqa_test"])]:
        metrics=read_json(RESULTS/"task2"/folder/"final_eval"/"metrics.json")
        for suite in suites:
            for condition,label in [("correct","Correct"),("complementary_swap","Swap"),("mismatched","Mismatched"),("no_image","No image")]:
                m=metrics[suite][condition]
                cells=[]
                if suite=="grounding_test":
                    cells.append("Grounding")
                else:
                    cells.append("VQA")
                cells.append(label)
                cells.append(pct(m["accuracy"]))
                by_type=[]
                for key in m["accuracy_by_type"]:
                    by_type.append(pct(m["accuracy_by_type"][key]))
                cells.append(" / ".join(by_type))
                cells.append(pct(m["pair_accuracy"]))
                cells.append(pct(m["identical_answer_rate"]))
                cells.append(pct(m["matches_partner_answer_rate"]))
                cells.append(f"{m['mean_reference_nll']:.2f}")
                lines.append(" & ".join(cells)+" \\\\")
            lines.append("\\midrule")
    lines.pop() #no rule after the last suite
    write_table("t2_final.tex",lines)


def t2_compare_table():
    lines=[] #table body
    for folder,label in [("tokens196","196 tokens"),("tokens49","49 tokens")]:
        metrics=read_json(RESULTS/"task2"/folder/"final_eval"/"metrics.json")
        summary=read_json(RESULTS/"task2"/folder/"train_summary.json")
        peak=0.0
        for stage in summary["stages"]:
            if stage["peak_vram_gib"]>peak:
                peak=stage["peak_vram_gib"] #highest stage
        cells=[label]
        for suite in ["grounding_test","vqa_test"]:
            m=metrics[suite]
            advantage=m["advantage_accuracy_correct_minus_swap"]
            nll=m["advantage_nll_swap_minus_correct"]
            cells.append(pct(m["correct"]["accuracy"]))
            cells.append(pct(m["correct"]["pair_accuracy"]))
            cells.append(f"{advantage['estimate']*100:.1f} [{advantage['ci95_low']*100:.1f}, {advantage['ci95_high']*100:.1f}]")
            cells.append(f"{nll['estimate']:.2f} [{nll['ci95_low']:.2f}, {nll['ci95_high']:.2f}]")
        cells.append(f"{summary['train_seconds']/3600:.2f}")
        cells.append(f"{peak:.2f}")
        lines.append(" & ".join(cells)+" \\\\")
    write_table("t2_compare.tex",lines)


def t2_by_type_table():
    """Likelihood and pair results per question type, accuracy for both runs, for the appendix."""
    main=read_json(RESULTS/"task2"/"tokens196"/"final_eval"/"metrics.json")
    pooled=read_json(RESULTS/"task2"/"tokens49"/"final_eval"/"metrics.json")
    lines=[] #table body
    for suite,kinds in [("grounding_test",[("presence","Presence"),("counting","Counting")]),("vqa_test",[("yes/no","Yes-no"),("number","Number"),("other","Other")])]:
        rows={}
        for condition in ["correct","complementary_swap","mismatched"]:
            rows[condition]=read_jsonl(RESULTS/"task2"/"tokens196"/"final_eval"/(suite+"_"+condition+".jsonl"))
        for kind,label in kinds:
            pairs={}
            for row in rows["correct"]:
                if row["qtype"]==kind:
                    pairs.setdefault(row["pair_id"],[]).append(row)
            identical=0
            both=0
            for sides in pairs.values():
                if sides[0]["normalized"]==sides[1]["normalized"]:
                    identical=identical+1
                if sides[0]["normalized"]==sides[0]["canonical"] and sides[1]["normalized"]==sides[1]["canonical"]:
                    both=both+1
            nll=[]
            for condition in ["correct","complementary_swap","mismatched"]:
                values=[]
                for row in rows[condition]:
                    if row["qtype"]==kind:
                        values.append(row["nll"])
                nll.append(f"{float(np.mean(values)):.2f}")
            says_no="--" #only yes-no style questions can answer no
            if kind=="presence" or kind=="yes/no":
                count=0
                total=0
                for row in rows["mismatched"]:
                    if row["qtype"]==kind:
                        total=total+1
                        if row["normalized"]=="no":
                            count=count+1
                says_no=str(count)+"/"+str(total)
            lines.append(" & ".join([label,str(len(pairs)),str(identical),str(both),pct(main[suite]["correct"]["accuracy_by_type"][kind]),pct(pooled[suite]["correct"]["accuracy_by_type"][kind])]+nll+[says_no])+" \\\\")
    write_table("t2_by_type.tex",lines)


def t2_curves_figure():
    fig,axes=plt.subplots(1,4,figsize=(5.5,1.35))
    styles={"tokens196":("#1f4e9c","196"),"tokens49":("#d9731a","49")}
    for folder in TOKEN_FOLDERS:
        color,label=styles[folder]
        log=read_jsonl(RESULTS/"task2"/folder/"train_log.jsonl")
        values=[]
        for row in log:
            values.append(row["loss"])
        axes[0].plot(np.arange(50,len(values)+1),smooth(values,50),color=color,linewidth=0.8,label=label)
        dev=read_jsonl(RESULTS/"task2"/folder/"dev_eval.jsonl")
        updates=[]
        caption=[]
        grounding=[]
        vqa=[]
        delta_g=[]
        delta_v=[]
        for row in dev:
            updates.append(row["update"])
            caption.append(row["caption_ref_loss"])
            grounding.append(row["grounding_dev"]["nll_correct"])
            vqa.append(row["vqa_dev"]["nll_correct"])
            delta_g.append(row["grounding_dev"]["delta_nll"])
            delta_v.append(row["vqa_dev"]["delta_nll"])
        axes[1].plot(updates,caption,color=color,linewidth=0.9,marker=".",markersize=2.5)
        axes[2].plot(updates,grounding,color=color,linewidth=0.9,marker=".",markersize=2.5)
        axes[2].plot(updates,vqa,color=color,linewidth=0.9,marker=".",markersize=2.5,linestyle="--")
        axes[3].plot(updates,delta_g,color=color,linewidth=0.9,marker=".",markersize=2.5)
        axes[3].plot(updates,delta_v,color=color,linewidth=0.9,marker=".",markersize=2.5,linestyle="--")
    titles=["Training loss (nats)","Dev caption loss (nats)","Dev answer NLL (nats)","Dev $\\Delta_{\\mathrm{NLL}}$ (nats)"]
    for ax,title in zip(axes,titles):
        ax.set_title(title,fontsize=6.8,pad=2)
        ax.axvline(2250,color="#999999",linewidth=0.5) #stage 1 ends
        ax.axvline(3750,color="#999999",linewidth=0.5) #stage 2 ends
        ax.set_xticks([0,2250,3750])
        ax.tick_params(labelsize=6,length=2,pad=1)
        ax.set_xlabel("Update",fontsize=6.5,labelpad=1)
    axes[1].legend(axes[0].get_lines()[:2],["196","49"],title="Visual tokens",fontsize=6,title_fontsize=6,loc="upper right",borderpad=0)
    axes[1].set_ylim(2.2,3.0)
    axes[2].set_ylim(0,6)
    axes[3].axhline(0,color="#999999",linewidth=0.5)
    fig.tight_layout(pad=0.2,w_pad=0.5)
    fig.savefig(FIGURES/"t2_curves.pdf")
    plt.close(fig)


def t2_facts():
    facts={"feature_cache":read_json(RESULTS/"task2"/"feature_cache.json"),"data":read_json(RESULTS/"task2"/"data_summary.json")}
    for folder in TOKEN_FOLDERS:
        facts[folder]={"train":read_json(RESULTS/"task2"/folder/"train_summary.json"),"calibration":read_json(RESULTS/"task2"/folder/"calibration.json"),"dev_final":read_jsonl(RESULTS/"task2"/folder/"dev_eval.jsonl")[-1]}
        metrics=read_json(RESULTS/"task2"/folder/"final_eval"/"metrics.json")
        facts[folder]["generation_seconds_per_answer"]=metrics["grounding_test"]["correct"]["generation_seconds_per_answer"]
    return facts


# ---------------- Task 3 ----------------
BUDGETS=["1","2","4","8","16","32"]


def t3_methods_table():
    rows=[("first_answer","First answer"),("self_consistency","Self-consistency"),("shortest_rollout","Shortest rollout"),("highest_logprob","Highest log-prob"),("highest_self_certainty","Highest self-certainty"),("self_certainty_rwc","Self-certainty + RWC"),("highest_vision","Highest vision"),("vision_rwc","Vision + RWC"),("highest_visconf","Highest VisConf"),("visconf_rwc","VisConf + RWC"),("highest_visconf_tokenwise","Tokenwise VisConf"),("visconf_tokenwise_rwc","Tokenwise VisConf + RWC")]
    lines=[] #table body
    for method,label in rows:
        cells=[label]
        for dataset in ["math","mathvista"]:
            summary=read_json(RESULTS/"task3"/"offline"/dataset/"summary.json")["eval"]
            boot=read_json(RESULTS/"task3"/"offline"/dataset/"bootstrap_eval.json")["32"]
            if method not in summary["32"]["methods"]:
                cells.extend(["--","--","--"]) #vision methods only exist for mathvista
                continue
            cells.append(pct(summary["8"]["methods"][method]["accuracy"]))
            cells.append(pct(summary["32"]["methods"][method]["accuracy"]))
            if method=="self_consistency":
                cells.append("--")
            else:
                diff=boot[method+"_minus_self_consistency"]
                cells.append(f"{diff['estimate']*100:+.1f} [{diff['ci95_low']*100:+.1f}, {diff['ci95_high']*100:+.1f}]".replace("-","$-$"))
        lines.append(" & ".join(cells)+" \\\\")
    write_table("t3_methods.tex",lines)


def t3_adaptive_table():
    lines=[] #table body
    for dataset,name in [("math","MATH"),("mathvista","MathVista")]:
        result=read_json(RESULTS/"task3"/"adaptive"/dataset/"eval_results.json")
        fixed=result["fixed_n_self_consistency"]
        for family,label in [("agreement","Agreement"),("quality","Quality-aware")]:
            r=result[family]
            setting="$\\tau$="+str(r["tau"])
            if r["gamma"] is not None:
                setting=setting+", $\\gamma$="+str(r["gamma"])
            lines.append(" & ".join([name,label,setting,pct(r["accuracy"]),f"{r['mean_token_cost']:.0f}",f"{r['mean_rollouts']:.1f}","/".join([pct(r["stop_distribution"]["4"],0),pct(r["stop_distribution"]["8"],0),pct(r["stop_distribution"]["16"],0),pct(r["stop_distribution"]["32"],0)]),pct(r["incorrect_early_stops"]/r["early_stops"]),str(r["incorrect_early_stops_fixed32_correct"])])+" \\\\")
        for budget in ["8","16","32"]:
            r=fixed[budget]
            lines.append(" & ".join([name,"Fixed $N$="+budget,"--",pct(r["accuracy"]),f"{r['mean_token_cost']:.0f}",budget+".0","--","--","--"])+" \\\\")
        lines.append("\\midrule")
    lines.pop() #no rule at the end
    write_table("t3_adaptive.tex",lines)


def t3_live_table():
    live=read_json(RESULTS/"task3"/"live"/"summary.json")
    lines=[] #table body
    for key,label in [("single","Single 3072"),("parallel","Parallel 3$\\times$1024"),("refinement","Refinement 1024+2048")]:
        r=live[key]
        lines.append(" & ".join([label,pct(r["accuracy"],0),f"{r['mean_input_tokens']:.0f}",f"{r['mean_output_tokens']:.0f}",f"{r['mean_seconds']:.1f}",pct(r["truncation_rate"],0),pct(r["extraction_rate"],0)])+" \\\\")
    write_table("t3_live.tex",lines)


def t3_figure():
    fig,axes=plt.subplots(1,4,figsize=(6.2,1.5))
    for index,(dataset,name,quality,quality_label) in enumerate([("math","MATH","self_certainty_rwc","Self-cert. + RWC"),("mathvista","MathVista","visconf_rwc","VisConf + RWC")]):
        summary=read_json(RESULTS/"task3"/"offline"/dataset/"summary.json")["eval"]
        coverage=[]
        first=[]
        consensus=[]
        rwc=[]
        logprob=[]
        for budget in BUDGETS:
            coverage.append(summary[budget]["coverage"]*100)
            first.append(summary[budget]["methods"]["first_answer"]["accuracy"]*100)
            consensus.append(summary[budget]["methods"]["self_consistency"]["accuracy"]*100)
            rwc.append(summary[budget]["methods"][quality]["accuracy"]*100)
            logprob.append(summary[budget]["methods"]["highest_logprob"]["accuracy"]*100)
        x=[1,2,4,8,16,32]
        ax=axes[index]
        ax.plot(x,coverage,color="#555555",linewidth=0.9,linestyle="--",marker=".",markersize=3,label="Oracle coverage")
        ax.plot(x,rwc,color="#b0312f",linewidth=1,marker=".",markersize=3,label="Quality + RWC")
        ax.plot(x,consensus,color="#1f4e9c",linewidth=1,marker=".",markersize=3,label="Self-consistency")
        ax.plot(x,logprob,color="#2d8a4e",linewidth=0.9,marker=".",markersize=3,label="Highest log-prob")
        ax.plot(x,first,color="#999999",linewidth=0.8,label="First answer")
        ax.set_xscale("log",base=2)
        ax.set_xticks(x)
        ax.set_xticklabels(["1","2","4","8","16","32"])
        ax.minorticks_off()
        ax.set_title(name+": acc. (%) vs. $N$",fontsize=7,pad=2)
        ax.set_xlabel("Candidates $N$",fontsize=6.5,labelpad=1)
        ax.tick_params(labelsize=6,length=2,pad=1)
        ax.set_ylim(45,95)
        result=read_json(RESULTS/"task3"/"adaptive"/dataset/"eval_results.json")
        ax=axes[index+2]
        cost=[]
        accuracy=[]
        for budget in BUDGETS:
            cost.append(result["fixed_n_self_consistency"][budget]["mean_token_cost"]/1000)
            accuracy.append(result["fixed_n_self_consistency"][budget]["accuracy"]*100)
        ax.plot(cost,accuracy,color="#1f4e9c",linewidth=1,marker=".",markersize=3,label="Fixed $N$")
        ax.scatter([result["agreement"]["mean_token_cost"]/1000],[result["agreement"]["accuracy"]*100],color="#d9731a",s=14,marker="^",label="Agreement",zorder=3)
        ax.scatter([result["quality"]["mean_token_cost"]/1000],[result["quality"]["accuracy"]*100],color="#b0312f",s=14,marker="s",label="Quality-aware",zorder=3)
        ax.set_title(name+": acc. (%) vs. cost",fontsize=7,pad=2)
        ax.set_xlabel("Mean tokens (thousands)",fontsize=6.5,labelpad=1)
        ax.tick_params(labelsize=6,length=2,pad=1)
    axes[0].legend(fontsize=5.2,loc="upper left",borderpad=0,handlelength=1.4,labelspacing=0.2)
    axes[2].legend(fontsize=5.5,loc="lower right",borderpad=0,handlelength=1.2,labelspacing=0.2)
    fig.tight_layout(pad=0.2,w_pad=0.9)
    fig.savefig(FIGURES/"t3_curves.pdf")
    plt.close(fig)


def t3_facts():
    facts={"prepare":read_json(RESULTS/"task3"/"prepare_report.json"),"live":read_json(RESULTS/"task3"/"live"/"summary.json")}
    for dataset in ["math","mathvista"]:
        summary=read_json(RESULTS/"task3"/"offline"/dataset/"summary.json")["eval"]
        facts[dataset]={"signals":read_json(RESULTS/"task3"/"offline"/dataset/"signals_n32_eval.json"),"n1_coverage":summary["1"]["coverage"],"n32_coverage":summary["32"]["coverage"],"distinct_groups":summary["32"]["distinct_groups"],"n32_self_consistency":summary["32"]["methods"]["self_consistency"],"dev_selection":read_json(RESULTS/"task3"/"adaptive"/dataset/"dev_selection.json")["chosen"],"dev_fixed32":read_json(RESULTS/"task3"/"adaptive"/dataset/"dev_selection.json")["fixed32_self_consistency_dev_accuracy"]}
    return facts


# ---------------- Appendix tables, so every number in the text sits in a table ----------------
def minus(text):
    return text.replace("-","$-$") #proper minus sign in latex


def interval(block,scale=100,digits=1):
    """estimate [low, high] from a saved or computed bootstrap block."""
    estimate=f"{block['estimate']*scale:.{digits}f}"
    low=f"{block['ci95_low']*scale:.{digits}f}"
    high=f"{block['ci95_high']*scale:.{digits}f}"
    return minus(estimate+" ["+low+", "+high+"]")


def paired_interval(values,iterations,rng):
    """same resampling as paired_bootstrap in task3_ttc/offline_eval.py, values are per-question or per-pair."""
    values=np.asarray(values,dtype=float)
    means=[]
    for i in range(iterations):
        sample=rng.integers(len(values),size=len(values))
        means.append(values[sample].mean())
    return {"estimate":float(values.mean()),"ci95_low":float(np.percentile(means,2.5)),"ci95_high":float(np.percentile(means,97.5))}


def t1_gsm8k_checks_table():
    """What the responses look like: extraction of $$, the answer marker, length and the boolq yes rate."""
    rates=t1_boolq_yes_rates()
    lines=[] #table body
    for condition in CONDITIONS:
        rows=read_jsonl(RESULTS/"task1"/"eval"/condition/"gsm8k_samples.jsonl")
        dollars=0
        marker=0
        tokens=[]
        for row in rows:
            if row["flexible-extract_answer"]=="$$":
                dollars=dollars+1
            if "The answer is:" in row["response"]:
                marker=marker+1
            tokens.append(row["response_tokens"])
        lines.append(" & ".join([NAMES[condition],str(dollars),f"{marker:,}",f"{float(np.mean(tokens)):.0f}",pct(rates[condition])])+" \\\\")
    lines.append("\\midrule")
    lines.append(" & ".join(["Gold answers","--","--","--",pct(rates["gold"])])+" \\\\")
    write_table("t1_gsm8k_checks.tex",lines)
    return rates["gold"]


def t1_overhead_table():
    """Extra storage and time per method plus the checks that the method did what it should."""
    l2sp=read_json(RESULTS/"task1"/"train"/"full_ft_l2sp"/"train_summary.json")
    oplora=read_json(RESULTS/"task1"/"train"/"oplora16"/"train_summary.json")
    log=read_jsonl(RESULTS/"task1"/"train"/"full_ft_l2sp"/"train_log.jsonl")
    constraints=read_json(RESULTS/"task1"/"diagnostics"/"oplora_constraints.json")
    worst=max(constraints["max_rel_UkT_Btilde"],constraints["max_rel_Atilde_Vk"])
    merge={}
    for run in ["lora4","lora16","oplora16"]:
        merge[run]=read_json(RESULTS/"task1"/"merge_check"/(run+".json"))
    lines=[]
    lines.append(" & ".join(["Full FT + L2-SP",f"{l2sp['l2sp_reference_storage_bytes']/1e6:,.0f} (reference)",f"{l2sp['l2sp_penalty_and_transfer_seconds']:.0f} (penalty, transfer)","--","--",f"$\\Omega$={log[-1]['omega']:.1f}, $\\lambda\\Omega$={log[-1]['lambda_omega']:.3f}"])+" \\\\")
    for run in ["lora4","lora16"]:
        lines.append(" & ".join([NAMES[run],"--","--",pct(merge[run]["top1_agreement"]),f"{merge[run]['max_abs_logit_diff']:.2f}","--"])+" \\\\")
    lines.append(" & ".join(["OPLoRA-16",f"{oplora['oplora_basis_storage_bytes']/1e6:.0f} (bases)",f"{oplora['oplora_svd_seconds']:.0f} (SVD), {oplora['oplora_projected_export_seconds']:.2f} (export)",pct(merge["oplora16"]["top1_agreement"]),f"{merge['oplora16']['max_abs_logit_diff']:.2f}",f"violation {worst:.1e}, {constraints['modules']} modules".replace("e-0","e-")])+" \\\\")
    write_table("t1_overhead.tex",lines)


def t2_dev_table():
    lines=[] #table body
    for folder,label,updates in [("tokens196","196 tokens",[0,2250,2500,3750,4500]),("tokens49","49 tokens",[2250,3750,4500])]:
        rows={}
        for row in read_jsonl(RESULTS/"task2"/folder/"dev_eval.jsonl"):
            rows[row["update"]]=row
        first=label
        for update in updates:
            r=rows[update]
            delta_g=f"{r['grounding_dev']['delta_nll']:.2f}"
            if delta_g=="-0.00":
                delta_g="0.00" #no signed zero
            lines.append(" & ".join([first,f"{update:,}",f"{r['caption_ref_loss']:.2f}",f"{r['grounding_dev']['nll_correct']:.2f}",delta_g,f"{r['vqa_dev']['nll_correct']:.2f}",f"{r['vqa_dev']['delta_nll']:.2f}"])+" \\\\")
            first=""
        lines.append("\\midrule")
    lines.pop() #no rule at the end
    write_table("t2_dev.tex",lines)


def t2_swap_table():
    """How answers react to the swap, and whether the nll agrees, split by pairs answered differently or identically."""
    metrics=read_json(RESULTS/"task2"/"tokens196"/"final_eval"/"metrics.json")
    lines=[] #table body
    for suite,label in [("grounding_test","Grounding"),("vqa_test","VQA")]:
        correct=read_jsonl(RESULTS/"task2"/"tokens196"/"final_eval"/(suite+"_correct.jsonl"))
        swap=read_jsonl(RESULTS/"task2"/"tokens196"/"final_eval"/(suite+"_complementary_swap.jsonl"))
        swapped={}
        for row in swap:
            swapped[(row["pair_id"],row["side"])]=row
        changed=0
        followed=0
        exact=0
        for row in correct:
            other=swapped[(row["pair_id"],row["side"])]
            if row["normalized"]==row["canonical"]:
                exact=exact+1
            if other["normalized"]!=row["normalized"]:
                changed=changed+1
                if other["normalized"]==other["partner_canonical"]:
                    followed=followed+1
        pairs={}
        for row in correct:
            pairs.setdefault(row["pair_id"],[]).append(row)
        same=[]
        different=[]
        for pair_id,sides in pairs.items():
            advantage=(swapped[(pair_id,sides[0]["side"])]["nll"]-sides[0]["nll"]+swapped[(pair_id,sides[1]["side"])]["nll"]-sides[1]["nll"])/2 #swap minus correct, mean of the pair
            if sides[0]["normalized"]==sides[1]["normalized"]:
                same.append(advantage)
            else:
                different.append(advantage)
        positive=0
        for value in same:
            if value>0:
                positive=positive+1
        block=paired_interval(same,4000,np.random.default_rng(6304)) #4,000 pair resamples as in task 2
        lines.append(" & ".join([label,str(changed)+"/"+str(len(correct)),str(followed),pct(exact/len(correct)),interval(metrics[suite]["accuracy_correct_minus_mismatched"]),f"{float(np.mean(different)):.2f} ({len(different)})",f"{block['estimate']:.2f} [{block['ci95_low']:.2f}, {block['ci95_high']:.2f}] ({len(same)})",str(positive)])+" \\\\")
    write_table("t2_swap.tex",lines)


def t2_cost_table():
    cache=read_json(RESULTS/"task2"/"feature_cache.json")
    runs={}
    metrics={}
    for folder in TOKEN_FOLDERS:
        runs[folder]=read_json(RESULTS/"task2"/folder/"train_summary.json")
        metrics[folder]=read_json(RESULTS/"task2"/folder/"final_eval"/"metrics.json")
    lines=[] #table body
    lines.append(" & ".join(["Feature cache","--","--",f"{cache['seconds']:.0f}",f"{cache['peak_vram_gib']:.2f}","--","--"])+" \\\\")
    for index,label in enumerate(["Stage 1 (caption)","Stage 2 (visual instruction)","Stage 3 (natural VQA)"]):
        a=runs["tokens196"]["stages"][index]
        b=runs["tokens49"]["stages"][index]
        lines.append(" & ".join([label,f"{a['updates']:,}",f"{a['trainable_parameters']/1e6:.2f}",f"{a['seconds']:,.0f}",f"{a['peak_vram_gib']:.2f}",f"{b['seconds']:,.0f}",f"{b['peak_vram_gib']:.2f}"])+" \\\\")
    lines.append(" & ".join(["All stages (hours)",f"{runs['tokens196']['total_updates']:,}","--",f"{runs['tokens196']['train_seconds']/3600:.2f}","--",f"{runs['tokens49']['train_seconds']/3600:.2f}","--"])+" \\\\")
    speed=[]
    for folder in TOKEN_FOLDERS:
        speed.append(f"{metrics[folder]['grounding_test']['correct']['generation_seconds_per_answer']*1000:.0f} / {metrics[folder]['vqa_test']['correct']['generation_seconds_per_answer']*1000:.0f}")
    lines.append(" & ".join(["ms per answer (grounding / VQA)","--","--",speed[0],"--",speed[1],"--"])+" \\\\")
    write_table("t2_cost.tex",lines)


def t3_pool_table():
    report=read_json(RESULTS/"task3"/"prepare_report.json")
    lines=[] #table body
    for dataset,name in [("math","MATH"),("mathvista","MathVista")]:
        splits=read_json(RESULTS/"task3"/"splits"/(dataset+"_splits.json"))
        r=report[dataset]
        lines.append(" & ".join([name,f"{r['questions']:,}",str(len(splits["dev"])),str(len(splits["eval"])),f"{r['rollouts']:,}",f"{r['missing_extractions']:,}",f"{r['capped']:,}",f"{r['correct_rollouts']:,}"])+" \\\\")
    write_table("t3_pool.tex",lines)


def t3_decomposition_table():
    lines=[] #table body
    for dataset,name in [("math","MATH"),("mathvista","MathVista")]:
        summary=read_json(RESULTS/"task3"/"offline"/dataset/"summary.json")["eval"]
        rows=[("coverage",[]),("SC",[]),("no correct",[]),("missed",[]),("groups",[])]
        for budget in BUDGETS:
            s=summary[budget]
            rows[0][1].append(pct(s["coverage"]))
            rows[1][1].append(pct(s["methods"]["self_consistency"]["accuracy"]))
            rows[2][1].append(pct(s["methods"]["self_consistency"]["no_correct_candidate"]))
            rows[3][1].append(pct(s["methods"]["self_consistency"]["correct_available_but_missed"]))
            rows[4][1].append(f"{s['distinct_groups']:.1f}")
        for label,cells in rows:
            lines.append(" & ".join([name+" "+label]+cells)+" \\\\")
        lines.append("\\midrule")
    lines.pop() #no rule at the end
    write_table("t3_decomposition.tex",lines)


def t3_isolation_table():
    math=read_json(RESULTS/"task3"/"offline"/"math"/"bootstrap_eval.json")["32"]
    vista=read_json(RESULTS/"task3"/"offline"/"mathvista"/"bootstrap_eval.json")["32"]
    lines=[] #table body
    lines.append("MATH & Self-certainty + RWC minus highest self-certainty & "+interval(math["self_certainty_rwc_minus_highest_self_certainty"])+" \\\\")
    for key,label in [("self_certainty_rwc_minus_highest_self_certainty","Self-certainty + RWC minus highest self-certainty"),("vision_rwc_minus_highest_vision","Vision + RWC minus highest vision"),("visconf_rwc_minus_highest_visconf","VisConf + RWC minus highest VisConf"),("highest_visconf_minus_highest_self_certainty","Highest VisConf minus highest self-certainty"),("visconf_rwc_minus_self_certainty_rwc","VisConf + RWC minus self-certainty + RWC"),("highest_visconf_tokenwise_minus_highest_visconf","Highest tokenwise minus highest VisConf"),("visconf_tokenwise_rwc_minus_visconf_rwc","Tokenwise + RWC minus VisConf + RWC")]:
        lines.append("MathVista & "+label+" & "+interval(vista[key])+" \\\\")
    write_table("t3_isolation.tex",lines)


def t3_signals_table():
    lines=[] #table body
    for dataset,name,keys in [("math","MATH",[("mean_logprob","Log-probability"),("self_certainty","Self-certainty"),("shortest","Shortest rollout")]),("mathvista","MathVista",[("mean_logprob","Log-probability"),("self_certainty","Self-certainty"),("shortest","Shortest rollout"),("vision","Vision"),("visconf","VisConf"),("visconf_tokenwise","Tokenwise VisConf")])]:
        signals=read_json(RESULTS/"task3"/"offline"/dataset/"signals_n32_eval.json")
        first=name
        for key,label in keys:
            s=signals[key]
            spearman="--"
            if key!="shortest":
                spearman=minus(f"{s['spearman_length_vs_score']:+.2f}")
            lines.append(" & ".join([first,label,pct(s["top_minus_bottom_quartile_correct"]),pct(s["same_on_mixed_questions"]),str(s["mixed_questions"]),spearman])+" \\\\")
            first=""
        if dataset=="mathvista":
            c=signals["confidence_vision_correlation"]
            lines.append(" & ".join(["","Confidence vs.\\ vision (mean correlation)","--","--","--",minus(f"{c['mean']:+.2f}")])+" \\\\")
        lines.append("\\midrule")
    lines.pop() #no rule at the end
    write_table("t3_signals.tex",lines)


def t3_trajectory_table():
    lines=[] #table body
    for row in read_json(RESULTS/"task3"/"offline"/"mathvista"/"illustrative_trajectories.json"):
        for key,label in [("visconf_pick","Highest VisConf"),("self_consistency_pick","Self-consistency")]:
            s=row["series"][key]
            half=len(s["v"])//2
            right="wrong"
            if s["correct"]==True:
                right="right"
            lines.append(" & ".join([row["sample_id"],row["gold"],label,s["answer"],right,str(len(s["v"])),f"{float(np.mean(s['v'][:half])):.3f}",f"{float(np.mean(s['v'][half:])):.3f}",minus(f"{float(np.corrcoef(s['c'],s['v'])[0,1]):.2f}")])+" \\\\")
    write_table("t3_trajectories.tex",lines)


def t3_stops_table():
    """Fixed-32 comparison and what stopped runs looked like, from eval_stops.jsonl."""
    lines=[] #table body
    for dataset,name in [("math","MATH"),("mathvista","MathVista")]:
        result=read_json(RESULTS/"task3"/"adaptive"/dataset/"eval_results.json")
        stops=read_jsonl(RESULTS/"task3"/"adaptive"/dataset/"eval_stops.jsonl")
        rng=np.random.default_rng(6304) #one generator per dataset, as in offline_eval.py
        for family,label in [("agreement","Agreement"),("quality","Quality-aware")]:
            by_question={}
            for row in stops:
                if row["family"]==family:
                    by_question.setdefault(row["sample_id"],[]).append(row["correct"]-row["fixed32_correct"])
            values=[]
            for sample in by_question.values():
                values.append(float(np.mean(sample))) #average over the 10 orders first
            block=paired_interval(values,1000,rng)
            share=result[family]["mean_token_cost"]/result["fixed_n_self_consistency"]["32"]["mean_token_cost"]
            four=[row for row in stops if row["family"]==family and row["stop_n"]==4]
            full=[row for row in stops if row["family"]==family and row["stop_n"]==32]
            good=[row["pool_correct_fraction"] for row in stops if row["family"]==family and row["stop_n"]<32 and row["correct"]==1]
            wrong=[row["pool_correct_fraction"] for row in stops if row["family"]==family and row["stop_n"]<32 and row["correct"]==0]
            back=[row["pool_correct_fraction"] for row in stops if row["family"]==family and row["stop_n"]<32 and row["correct"]==0 and row["fixed32_correct"]==1]
            lines.append(" & ".join([name,label,interval(block,digits=2),pct(share,0),f"{float(np.mean([r['support'] for r in four])):.2f}",pct(np.mean([r["correct"] for r in four])),f"{float(np.mean([r['support'] for r in full])):.2f}",pct(np.mean([r["correct"] for r in full])),pct(np.mean(good),0),pct(np.mean(wrong),0),pct(np.mean(back),0),f"{len(wrong):,}",str(len(back))])+" \\\\")
    write_table("t3_stops.tex",lines)


def t3_grid_table():
    math=read_json(RESULTS/"task3"/"adaptive"/"math"/"dev_selection.json")
    vista=read_json(RESULTS/"task3"/"adaptive"/"mathvista"/"dev_selection.json")
    lines=[] #table body
    lines.append(" & ".join(["Fixed $N$=32","--","--",pct(math["fixed32_self_consistency_dev_accuracy"],2),f"{math['grid']['agreement'][0]['mean_token_cost']:,.0f}",pct(vista["fixed32_self_consistency_dev_accuracy"],2),f"{vista['grid']['agreement'][0]['mean_token_cost']:,.0f}"])+" \\\\")
    for index in range(1,len(math["grid"]["agreement"])):
        a=math["grid"]["agreement"][index]
        b=vista["grid"]["agreement"][index]
        lines.append(" & ".join(["Agreement",str(a["tau"]),"--",pct(a["accuracy"],2),f"{a['mean_token_cost']:,.0f}",pct(b["accuracy"],2),f"{b['mean_token_cost']:,.0f}"])+" \\\\")
    for index in range(1,len(math["grid"]["quality"])):
        a=math["grid"]["quality"][index]
        b=vista["grid"]["quality"][index]
        if a["tau"]==0.5:
            lines.append(" & ".join(["Quality-aware",str(a["tau"]),str(a["gamma"]),pct(a["accuracy"],2),f"{a['mean_token_cost']:,.0f}",pct(b["accuracy"],2),f"{b['mean_token_cost']:,.0f}"])+" \\\\")
    write_table("t3_grid.tex",lines)


def t3_live_detail_table():
    """Correct counts out of 50 and what refinement changed, read from generations.jsonl."""
    rows=read_jsonl(RESULTS/"task3"/"live"/"generations.jsonl")
    counts={"single":0,"parallel":0,"initial":0,"refinement":0}
    changes={"Wrong to correct":[],"Correct to wrong":[],"Stayed correct":[],"Stayed wrong":[]}
    cut=[]
    for row in rows:
        for key in counts:
            if row["correct"][key]:
                counts[key]=counts[key]+1
        before=bool(row["correct"]["initial"])
        after=bool(row["correct"]["refinement"])
        if before==False and after==True:
            changes["Wrong to correct"].append(row["sample_id"])
            if row["parallel"][0]["truncated"]==True and row["answers"]["initial"] is None:
                cut.append(row["sample_id"]) #the shared initial response ran out with no answer
        elif before==True and after==False:
            changes["Correct to wrong"].append(row["sample_id"])
        elif before==True:
            changes["Stayed correct"].append(row["sample_id"])
        else:
            changes["Stayed wrong"].append(row["sample_id"])
    lines=[]
    lines.append(" & ".join(["Correct of "+str(len(rows)),"Single "+str(counts["single"]),"Refinement "+str(counts["refinement"]),"Shared initial "+str(counts["initial"]),"Parallel "+str(counts["parallel"])])+" \\\\")
    lines.append(" & ".join(["Refinement change","Wrong to correct "+str(len(changes["Wrong to correct"])),"Correct to wrong "+str(len(changes["Correct to wrong"])),"Stayed correct "+str(len(changes["Stayed correct"])),"Stayed wrong "+str(len(changes["Stayed wrong"]))])+" \\\\")
    write_table("t3_live_detail.tex",lines)
    return {"wrong_to_correct":changes["Wrong to correct"],"correct_to_wrong":changes["Correct to wrong"],"cut_off_with_no_answer":cut}


def t3_further_stops_table():
    examples=read_json(RESULTS/"task3"/"examples.json")["premature_stopping"]
    lines=[] #table body
    for row in examples:
        if row["sample_id"] in ["test:534","test:3409","test:3504"]:
            label="Agreement"
            if row["family"]=="quality":
                label="Quality-aware"
            if row["family"]=="quality" and row["sample_id"]=="test:534":
                continue #the agreement order of this question is already listed
            lines.append(" & ".join([label,row["sample_id"],str(row["order"]),str(row["stop_n"]),row["selected_answer"],row["fixed32_answer"],pct(row["pool_correct_fraction"])])+" \\\\")
    write_table("t3_further_stops.tex",lines)


def main():
    t1_scores_table()
    t1_resources_table()
    t1_diagnostics_table()
    qualitative=t1_qualitative_table()
    t1_loss_figure()
    t2_final_table()
    t2_compare_table()
    t2_by_type_table()
    t2_curves_figure()
    t3_methods_table()
    t3_adaptive_table()
    t3_live_table()
    t3_figure()
    gold_yes=t1_gsm8k_checks_table()
    t1_overhead_table()
    t2_dev_table()
    t2_swap_table()
    t2_cost_table()
    t3_pool_table()
    t3_decomposition_table()
    t3_isolation_table()
    t3_signals_table()
    t3_trajectory_table()
    t3_stops_table()
    t3_grid_table()
    live_changes=t3_live_detail_table()
    t3_further_stops_table()
    print("gold yes",gold_yes,"live changes",live_changes)
    facts={"t1_qualitative":qualitative,"t1_l2sp":t1_l2sp_facts(),"t1_boolq_yes":t1_boolq_yes_rates(),"t2":t2_facts(),"t3":t3_facts()}
    (TABLES/"facts.json").write_text(json.dumps(facts,indent=2))
    print("wrote",TABLES/"facts.json")


if __name__=="__main__":
    main()
