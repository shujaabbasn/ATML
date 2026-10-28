from __future__ import annotations

import numpy as np


#quality scores, visconf calibration and the selection rules, all computed on the revealed candidates only
#ties: earliest supporting rollout in the current order wins. rwc uses average ranks for tied scores.


def rollout_scores(q):
    """Per-rollout summaries that do not depend on the candidate set: L_i, K_i, V_i and length."""
    L=[] #mean log-probability of the sampled tokens
    K=[] #self-certainty, mean KL(U || p)
    for r in range(len(q["logp"])):
        L.append(float(q["logp"][r].mean()))
        K.append(float(q["kl"][r].mean()))
    scores={"L":np.asarray(L),"K":np.asarray(K),"length":q["tokens"].astype(float)}
    if "vision" in q:
        V=[] #visual engagement
        for r in range(len(q["vision"])):
            V.append(float(q["vision"][r].mean()))
        scores["V"]=np.asarray(V)
    return scores


def pooled_kl(q,candidates):
    """All token KL values of the candidate set G in one array."""
    parts=[]
    for r in candidates:
        parts.append(q["kl"][r])
    return np.concatenate(parts).astype(np.float64)


def visconf(q,candidates):
    """VisConf on candidate set G: pool token KL of G, calibrate with its 5th/95th percentiles (linear interpolation),
    C_i = mean_t c_it, Z_i = V_i C_i, and the tokenwise fusion Z_token_i = mean_t c_it v_it."""
    pooled=pooled_kl(q,candidates)
    q05=np.percentile(pooled,5) #numpy default is linear interpolation
    q95=np.percentile(pooled,95)
    spread=max(q95-q05,1e-8)
    Z=[]
    Z_token=[]
    for r in candidates:
        c=np.clip((q["kl"][r].astype(np.float64)-q05)/spread,0.0,1.0) #c_it
        v=q["vision"][r].astype(np.float64) #v_it
        Z.append(float(v.mean())*float(c.mean())) #Z_i = V_i C_i
        Z_token.append(float((c*v).mean())) #Z_token_i
    return np.asarray(Z),np.asarray(Z_token)


def group_candidates(candidates,eq):
    """Answer groups among the revealed candidates, in order of first appearance. Uses the answer-equivalence matrix only."""
    groups=[]
    for r in candidates:
        placed=False
        for group in groups:
            if eq[group[0],r]:
                group.append(r)
                placed=True
                break
        if placed==False:
            groups.append([r])
    return groups


def weighted_vote(candidates,eq,weights):
    """Sum candidate weights per answer group. Returns (selected rollout, sorted group totals).
    Ties go to the group whose earliest supporter came first, and the earliest supporter represents the group."""
    groups=group_candidates(candidates,eq)
    weight_of={}
    for i in range(len(candidates)):
        weight_of[candidates[i]]=weights[i]
    totals=[]
    for group in groups:
        total=0.0
        for r in group:
            total=total+weight_of[r]
        totals.append(total)
    best=0
    for g in range(1,len(groups)):
        if totals[g]>totals[best]:
            best=g #strictly greater, so an earlier group keeps a tie
    return groups[best][0],sorted(totals,reverse=True),groups,best


def self_consistency(candidates,eq):
    return weighted_vote(candidates,eq,[1.0]*len(candidates))[0] #one vote per rollout, NO_ANSWER votes too


def highest_score(candidates,values):
    """values aligned with candidates. Ties -> earliest in the order."""
    best=0
    for i in range(1,len(candidates)):
        if values[i]>values[best]:
            best=i
    return candidates[best]


def descending_average_ranks(values):
    """Rank 1 = highest value. Tied values share the average of their ranks."""
    values=np.asarray(values,dtype=float)
    order=np.argsort(-values,kind="stable")
    ranks=np.zeros(len(values))
    i=0
    while i<len(values):
        j=i
        while j+1<len(values) and values[order[j+1]]==values[order[i]]:
            j=j+1
        average=(i+1+j+1)/2.0 #ranks i+1 .. j+1 share their mean
        for k in range(i,j+1):
            ranks[order[k]]=average
        i=j+1
    return ranks


def rwc_weights(values):
    """w_i = (N - r_i) / (N - 1) for N > 1."""
    n=len(values)
    ranks=descending_average_ranks(values)
    weights=[]
    for r in ranks:
        weights.append((n-r)/(n-1))
    return weights


def rwc(candidates,eq,values):
    if len(candidates)==1:
        return candidates[0],[1.0] #at N = 1 every rule returns the only candidate
    selected,totals,groups,best=weighted_vote(candidates,eq,rwc_weights(values))
    return selected,totals


def rwc_margin(totals):
    """(largest - second largest group vote) / total vote weight, runner-up = 0 when only one group exists."""
    total=sum(totals)
    if total<=0:
        return 0.0
    runner_up=0.0 #only one answer group
    if len(totals)>1:
        runner_up=totals[1]
    return (totals[0]-runner_up)/total


def select_all(q,candidates,scores,dataset):
    """Every selection rule at this candidate set -> {method: selected rollout}."""
    eq=q["eq"]
    L=[]
    K=[]
    negative_length=[] #shortest rollout = highest value
    for r in candidates:
        L.append(scores["L"][r])
        K.append(scores["K"][r])
        negative_length.append(-scores["length"][r])
    chosen={
        "first_answer":candidates[0],
        "self_consistency":self_consistency(candidates,eq),
        "highest_logprob":highest_score(candidates,L),
        "highest_self_certainty":highest_score(candidates,K),
        "self_certainty_rwc":rwc(candidates,eq,K)[0],
        "shortest_rollout":highest_score(candidates,negative_length), #length-only control
    }
    if dataset=="mathvista":
        V=[]
        for r in candidates:
            V.append(scores["V"][r])
        Z,Z_token=visconf(q,candidates) #recalibrated on this prefix only
        chosen["highest_vision"]=highest_score(candidates,V)
        chosen["vision_rwc"]=rwc(candidates,eq,V)[0]
        chosen["highest_visconf"]=highest_score(candidates,list(Z))
        chosen["visconf_rwc"]=rwc(candidates,eq,list(Z))[0]
        chosen["highest_visconf_tokenwise"]=highest_score(candidates,list(Z_token))
        chosen["visconf_tokenwise_rwc"]=rwc(candidates,eq,list(Z_token))[0]
    return chosen


def method_cost(q,candidates,method):
    """Generated-token cost: retained tokens of every candidate the method consumed. FirstAns only uses the first."""
    if method=="first_answer":
        return int(q["tokens"][candidates[0]])
    else:
        total=0
        for r in candidates:
            total=total+int(q["tokens"][r])
        return total
