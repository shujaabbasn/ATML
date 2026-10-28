from __future__ import annotations

import argparse

import numpy as np

from common.utils import load_yaml, repo_path, save_json, write_jsonl
from task3_ttc.offline_eval import load_prepared
from task3_ttc.selection import group_candidates, method_cost, rollout_scores, rwc, rwc_margin, self_consistency, visconf


#step 5: offline simulation of stopping between completed rollouts, revealed in the saved random orders
#check after n in {4, 8, 16}, otherwise stop at 32. both controllers return the current self-consistency answer.
#  agreement      unique, non-NO_ANSWER leading answer with support >= tau * n
#  quality-aware  agreement + self-consistency winner == rwc winner + rwc margin >= gamma
#                 (rwc on self-certainty for math, VisConf for mathvista, recomputed on the revealed rollouts only)


def checkpoint_state(q,candidates,scores,dataset):
    """Everything a controller can see after revealing these candidates. No gold, no unrevealed rollouts."""
    eq=q["eq"]
    groups=group_candidates(candidates,eq)
    sizes=[]
    for group in groups:
        sizes.append(len(group))
    leader=0
    for g in range(1,len(groups)):
        if sizes[g]>sizes[leader]:
            leader=g
    unique=True
    for g in range(len(groups)):
        if g!=leader and sizes[g]==sizes[leader]:
            unique=False #tie for the lead
    leader_missing=q["answers"][groups[leader][0]] is None #the NO_ANSWER group leads
    sc_rollout=self_consistency(candidates,eq)
    if dataset=="math":
        values=[] #self-certainty
        for r in candidates:
            values.append(scores["K"][r])
    else:
        Z,Z_token=visconf(q,candidates)
        values=list(Z) #VisConf, calibrated on the revealed rollouts
    rwc_rollout,totals=rwc(candidates,eq,values)
    return {
        "n":len(candidates),
        "support":sizes[leader]/len(candidates),
        "unique_leader":unique,
        "leader_is_no_answer":leader_missing,
        "sc_rollout":sc_rollout,
        "rwc_agrees":bool(eq[sc_rollout,rwc_rollout]),
        "rwc_margin":rwc_margin(totals),
        "cost":method_cost(q,candidates,"self_consistency"), #all revealed rollouts are consumed
        "sc_correct":float(q["correct"][sc_rollout]),
    }


def stop_agreement(state,tau):
    return state["unique_leader"] and state["leader_is_no_answer"]==False and state["support"]>=tau


def stop_quality(state,tau,gamma):
    return stop_agreement(state,tau) and state["rwc_agrees"] and state["rwc_margin"]>=gamma


def run_policy(states,family,tau,gamma):
    """states: checkpoint states for n = 4, 8, 16, 32 of one order. Returns the state we stop at."""
    for state in states[:-1]:
        if family=="agreement" and stop_agreement(state,tau):
            return state
        if family=="quality" and stop_quality(state,tau,gamma):
            return state
    return states[-1] #otherwise stop at 32


def policy_table(question_states,ids,family,tau,gamma):
    """Accuracy / cost / rollouts averaged over orders within a question, then over questions."""
    accuracy=[]
    cost=[]
    rollouts=[]
    stops=[]
    for sample_id in ids:
        acc=[]
        cst=[]
        rol=[]
        for states in question_states[sample_id]:
            if family=="fixed32":
                state=states[-1]
            else:
                state=run_policy(states,family,tau,gamma)
            acc.append(state["sc_correct"])
            cst.append(state["cost"])
            rol.append(state["n"])
            stops.append(state["n"])
        accuracy.append(np.mean(acc))
        cost.append(np.mean(cst))
        rollouts.append(np.mean(rol))
    distribution={}
    for n in [4,8,16,32]:
        distribution[str(n)]=stops.count(n)/len(stops)
    return {"accuracy":float(np.mean(accuracy)),"mean_token_cost":float(np.mean(cost)),"mean_rollouts":float(np.mean(rollouts)),"stop_distribution":distribution,"per_question_accuracy":accuracy}


def setting_sort_key(c):
    """Lowest cost first, then higher accuracy, then smaller tau and gamma (the fallback has none, so it sorts last on ties)."""
    tau=2.0
    gamma=2.0
    if c["tau"] is not None:
        tau=c["tau"]
    if c["gamma"] is not None:
        gamma=c["gamma"]
    return (c["mean_token_cost"],-c["accuracy"],tau,gamma)


def select_setting(question_states,dev_ids,family,taus,gammas,reference,tolerance):
    """Lowest mean dev cost with dev accuracy >= fixed-32 self-consistency - 1pp. Ties: higher accuracy, smaller tau, smaller gamma.
    Always-32 is in every family as the fallback."""
    candidates=[{"tau":None,"gamma":None,"fallback_32":True,**policy_table(question_states,dev_ids,"fixed32",None,None)}]
    for tau in taus:
        if family=="agreement":
            candidates.append({"tau":tau,"gamma":None,"fallback_32":False,**policy_table(question_states,dev_ids,"agreement",tau,None)})
        else:
            for gamma in gammas:
                candidates.append({"tau":tau,"gamma":gamma,"fallback_32":False,**policy_table(question_states,dev_ids,"quality",tau,gamma)})
    allowed=[] #settings within one percentage point of fixed-32 self-consistency
    for c in candidates:
        if c["accuracy"]>=reference-tolerance/100.0:
            allowed.append(c)
    allowed.sort(key=setting_sort_key)
    for c in candidates:
        del c["per_question_accuracy"]
    return allowed[0],candidates


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task3.yaml")
    ap.add_argument("--dataset",required=True,choices=["math","mathvista"])
    args=ap.parse_args()
    cfg=load_yaml(args.config)
    adaptive_cfg=cfg["adaptive"]
    prepared=load_prepared(cfg,args.dataset)
    questions={}
    for q in prepared["questions"]:
        questions[q["sample_id"]]=q
    splits=prepared["splits"]
    orders=prepared["orders"]
    checkpoints=list(adaptive_cfg["check_points"])+[int(adaptive_cfg["max_candidates"])] #4, 8, 16, 32
    results_dir=repo_path(cfg["results_dir"])/"adaptive"/args.dataset

    question_states={}
    for sample_id,q in questions.items():
        scores=rollout_scores(q)
        question_states[sample_id]=[]
        for order in orders[sample_id]:
            states=[]
            for n in checkpoints:
                states.append(checkpoint_state(q,order[:n],scores,args.dataset))
            question_states[sample_id].append(states)

    #development: choose one setting per controller family, then freeze it
    reference=policy_table(question_states,splits["dev"],"fixed32",None,None)["accuracy"]
    chosen={}
    grids={}
    for family in ["agreement","quality"]:
        best,grid=select_setting(question_states,splits["dev"],family,adaptive_cfg["taus"],adaptive_cfg["gammas"],reference,float(adaptive_cfg["tolerance_pp"]))
        chosen[family]=best
        grids[family]=grid
    save_json(results_dir/"dev_selection.json",{"fixed32_self_consistency_dev_accuracy":reference,"chosen":chosen,"grid":grids})

    #held-out evaluation with the frozen settings
    eval_ids=splits["eval"]
    held_out={}
    for family in ["agreement","quality"]:
        setting=chosen[family]
        if setting["fallback_32"]:
            table=policy_table(question_states,eval_ids,"fixed32",None,None)
        else:
            table=policy_table(question_states,eval_ids,family,setting["tau"],setting["gamma"])
        del table["per_question_accuracy"]
        held_out[family]={"tau":setting["tau"],"gamma":setting["gamma"],"fallback_32":setting["fallback_32"],**table}
    #fixed-N self-consistency on the same orders, for comparison at similar costs
    fixed={}
    for n in cfg["offline"]["budgets"]:
        accuracy=[]
        cost=[]
        for sample_id in eval_ids:
            q=questions[sample_id]
            acc=[]
            cst=[]
            for order in orders[sample_id]:
                candidates=order[:n]
                acc.append(float(q["correct"][self_consistency(candidates,q["eq"])]))
                cst.append(method_cost(q,candidates,"self_consistency"))
            accuracy.append(np.mean(acc))
            cost.append(np.mean(cst))
        fixed[str(n)]={"accuracy":float(np.mean(accuracy)),"mean_token_cost":float(np.mean(cost))}
    held_out["fixed_n_self_consistency"]=fixed

    #incorrect early stops, and how many fixed-32 would have recovered
    stop_rows=[]
    for family in ["agreement","quality"]:
        setting=chosen[family]
        early_wrong=0
        recoverable=0
        early=0
        for sample_id in eval_ids:
            q=questions[sample_id]
            pool_fraction=float(q["correct"].mean()) #retrospective only
            for o in range(len(orders[sample_id])):
                states=question_states[sample_id][o]
                if setting["fallback_32"]:
                    state=states[-1]
                else:
                    state=run_policy(states,family,setting["tau"],setting["gamma"])
                stop_rows.append({"family":family,"sample_id":sample_id,"order":o,"stop_n":state["n"],"support":state["support"],"rwc_margin":state["rwc_margin"],
                    "correct":state["sc_correct"],"fixed32_correct":states[-1]["sc_correct"],"pool_correct_fraction":pool_fraction,
                    "selected_answer":q["keys"][state["sc_rollout"]],"fixed32_answer":q["keys"][states[-1]["sc_rollout"]],"gold":q["gold"]})
                if state["n"]<32:
                    early=early+1
                    if state["sc_correct"]==0.0:
                        early_wrong=early_wrong+1
                        if states[-1]["sc_correct"]==1.0:
                            recoverable=recoverable+1
        held_out[family]["early_stops"]=early
        held_out[family]["incorrect_early_stops"]=early_wrong
        held_out[family]["incorrect_early_stops_fixed32_correct"]=recoverable
    save_json(results_dir/"eval_results.json",held_out)
    write_jsonl(results_dir/"eval_stops.jsonl",stop_rows)
    print(held_out)


if __name__=="__main__":
    main()
