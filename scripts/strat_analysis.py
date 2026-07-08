#!/usr/bin/env python3
"""难度分层分析(任务1,零训练): 按 base avg@12 逐题 pass-rate 分层(AIME24+25 pool),
各层内 v0 vs OPSD 逐题 pass-rate 的 3-seed mean±std。预注册预测: Δ 随难度上升而增大。
产出 docs/figs/strat_aime.png + 打印表。复现: python scripts/strat_analysis.py"""
import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statistics import mean, stdev
def probs(fp):
    d=json.load(open(fp)); return {r['problem_id']: r['num_correct']/r['val_n'] for r in d['results']}
def mpath(m,s,c,ds):
    if s==42: return f"results/v0_eval/v0ckpt{c}_{ds}.json" if m=="v0" else f"results/repro_eval/ckpt{c}_{ds}.json"
    return f"results/multiseed_eval/{('v0s' if m=='v0' else 'opsds')}{s}ckpt{c}_{ds}.json"
seeds=[42,1,2]; DS=["aime24","aime25"]
base={}
for ds in DS:
    for pid,r in probs(f"results/repro_eval/base_{ds}.json").items(): base[(ds,pid)]=r
tiers=["hard\n(base=0%)","mid\n(0-50%]","easy\n(>50%)"]
def tkey(x): return 0 if x==0 else (1 if x<=0.5 else 2)
members=[[],[],[]]
for k,v in base.items(): members[tkey(v)].append(k)
def tier_stats(m,c,ti):
    keys=members[ti]; sm=[]
    for s in seeds:
        rate={}
        for ds in DS: rate.update({(ds,pid):r for pid,r in probs(mpath(m,s,c,ds)).items()})
        sm.append(mean([rate[k] for k in keys if k in rate])*100)
    return mean(sm), stdev(sm)
fig,axes=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
for ax,c in zip(axes,[100,150]):
    x=range(3); w=0.36
    o=[tier_stats("opsd",c,i) for i in range(3)]; v=[tier_stats("v0",c,i) for i in range(3)]
    ax.bar([i-w/2 for i in x],[a[0] for a in o],w,yerr=[a[1] for a in o],capsize=4,label="OPSD",color="#4C78A8")
    ax.bar([i+w/2 for i in x],[a[0] for a in v],w,yerr=[a[1] for a in v],capsize=4,label="v0",color="#F58518")
    for i in x:
        d=v[i][0]-o[i][0]; ax.text(i,max(o[i][0],v[i][0])+4,f"Δ{d:+.1f}",ha="center",fontsize=9)
    ax.set_xticks(list(x)); ax.set_xticklabels([f"{t}\nn={len(members[i])}" for i,t in enumerate(tiers)])
    ax.set_title(f"ckpt{c}"); ax.set_ylim(0,105); ax.grid(axis="y",alpha=0.3)
axes[0].set_ylabel("per-problem pass-rate (avg@12, %)"); axes[0].legend(loc="upper left")
fig.suptitle("AIME24+25 difficulty-stratified: v0 vs OPSD (3-seed mean±std)  —  prediction (Δ↑ with difficulty) NOT supported")
fig.tight_layout(); fig.savefig("docs/figs/strat_aime.png",dpi=120)
print("saved docs/figs/strat_aime.png")
for c in [100,150]:
    print(f"ckpt{c}:", {["hard","mid","easy"][i]: f"OPSD {tier_stats('opsd',c,i)[0]:.1f} v0 {tier_stats('v0',c,i)[0]:.1f} Δ{tier_stats('v0',c,i)[0]-tier_stats('opsd',c,i)[0]:+.1f}" for i in range(3)})
