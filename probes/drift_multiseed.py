#!/usr/bin/env python3
"""漂移分解 3-seed 复核聚合（预注册见 docs/drift_multiseed_recon.md）。

"拽回起点"成分 = overall drift share_k = 100·Σ_tok drift_k / (Σ drift_k + Σ teach)，
逐 seed 在其 drift_shard 数据上聚合（drift_k=JSD(S_k,S0), teach=JSD(T_S,S0)），
与 seed42 主曲线合并为 mean±std，并判读预注册预言 (a)/(b)。CPU。

seed 数据文件：seed42=drift_shard*.jsonl；s1=drift_s1_shard*.jsonl；s2=drift_s2_shard*.jsonl。
用法: python probes/drift_multiseed.py
"""
import glob, json, os
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"data")
STEPS=[25,50,75,100,125,150]
SEEDS={"s42":"drift_shard*.jsonl","s1":"drift_s1_shard*.jsonl","s2":"drift_s2_shard*.jsonl"}

def seed_curve(pattern):
    files=sorted(glob.glob(os.path.join(DATA,pattern)))
    if not files: return None,0
    d_sum={k:0.0 for k in STEPS}; t_sum={k:0.0 for k in STEPS}; n=0
    for fp in files:
        for line in open(fp):
            r=json.loads(line); n+=1
            teach=np.asarray(r["teach"],float).sum()
            for k in STEPS:
                dk=np.asarray(r["per_k"][str(k)]["drift"],float).sum()
                d_sum[k]+=dk; t_sum[k]+=teach
    curve=[100*d_sum[k]/(d_sum[k]+t_sum[k]) if (d_sum[k]+t_sum[k])>0 else float("nan") for k in STEPS]
    return curve,n

def main():
    curves={}; ns={}
    for name,pat in SEEDS.items():
        c,n=seed_curve(pat)
        if c is not None: curves[name]=c; ns[name]=n
    print("=== per-seed overall drift share (%) by step", STEPS, "===")
    for name in ["s42","s1","s2"]:
        if name in curves:
            print(f"  {name} (n={ns[name]}): {[round(x,1) for x in curves[name]]}")
        else:
            print(f"  {name}: MISSING (未跑/数据缺)")

    have=[n for n in ["s42","s1","s2"] if n in curves]
    if have:
        M=np.array([curves[n] for n in have])
        mean=M.mean(0); std=M.std(0,ddof=0)
        print("\n=== merged mean±std (seeds:", have, ") ===")
        for i,k in enumerate(STEPS):
            print(f"  step{k}: {mean[i]:.1f} ± {std[i]:.1f}")

    # ---- 预注册判读 ----
    print("\n=== 预注册判读 ===")
    # (a) 各曲线首尾差 >10pp
    a_ok={}
    for n in have:
        d=curves[n][-1]-curves[n][0]; a_ok[n]=bool(d>10)
        print(f"  (a) {n}: 首步{curves[n][0]:.1f} -> 尾步{curves[n][-1]:.1f}  Δ={d:+.1f}pp  {'PASS' if d>10 else 'FAIL'}")
    # (b) 150 步各 seed >50%
    b_ok={}
    for n in have:
        v=curves[n][-1]; b_ok[n]=bool(v>50)
        print(f"  (b) {n}: step150 = {v:.1f}%  {'>50 PASS' if v>50 else '≤50 FAIL'}")

    new_have=[n for n in ["s1","s2"] if n in curves]   # 预言只对两条"新"曲线要求 (a)
    a_all = bool(all(a_ok[n] for n in new_have)) if new_have else None
    b_all = bool(all(b_ok[n] for n in have)) if have else None
    print("\n=== 结论 ===")
    print(f"  (a) 两条新曲线均首尾>10pp: {a_all}")
    print(f"  (b) 三 seed 150步均>50%: {b_all}")
    if a_all and b_all: verdict="(a)&(b) 成立：'拽回起点持续上升且150步过半'完整成立"
    elif a_all and (b_all is False): verdict="(a)成立(b)不成立：主张降级为'份额持续上升',不再引用'过半'"
    elif a_all is False: verdict="(a)不成立：中心论点需重审"
    else: verdict="数据不全，无法判读"
    print("  ->", verdict)

    out=os.path.join(HERE,"analysis","drift_multiseed.json")
    json.dump({"steps":STEPS,"curves":curves,"n":ns,
               "mean":(mean.tolist() if have else None),"std":(std.tolist() if have else None),
               "a_ok":a_ok,"b_ok":b_ok,"a_all":a_all,"b_all":b_all,"verdict":verdict},
              open(out,"w"),indent=2,ensure_ascii=False)
    print("[saved]",out)

if __name__=="__main__": main()
