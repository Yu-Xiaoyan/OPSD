#!/usr/bin/env python3
"""分诊 rationale (a): correct 桶漂移份额 vs wrong 桶（3-seed）。预注册见 docs/triage_rationale_recon.md。

drift share_k(bucket) = 100·Σ_{rollout∈bucket} Σdrift_k / Σ(Σdrift_k+Σteach)，逐 seed 逐桶聚合。
预言 (a): 150 步 correct 桶 ≥ wrong 桶 + 10pp（3-seed 均值）。CPU。
"""
import glob, json, os
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"data")
STEPS=[25,50,75,100,125,150]
SEEDS={"s42":"drift_shard*.jsonl","s1":"drift_s1_shard*.jsonl","s2":"drift_s2_shard*.jsonl"}
BUCKETS=["correct","wrong"]

def seed_bucket_curves(pattern):
    files=sorted(glob.glob(os.path.join(DATA,pattern)))
    agg={b:{k:[0.0,0.0] for k in STEPS} for b in BUCKETS}; n={b:0 for b in BUCKETS}
    for fp in files:
        for line in open(fp):
            r=json.loads(line); b=r.get("bucket")
            if b not in BUCKETS: continue
            n[b]+=1; teach=np.asarray(r["teach"],float).sum()
            for k in STEPS:
                dk=np.asarray(r["per_k"][str(k)]["drift"],float).sum()
                agg[b][k][0]+=dk; agg[b][k][1]+=teach
    curves={b:[100*agg[b][k][0]/(agg[b][k][0]+agg[b][k][1]) if (agg[b][k][0]+agg[b][k][1])>0 else float("nan")
               for k in STEPS] for b in BUCKETS}
    return curves,n

def main():
    per_seed={}; ns={}
    for name,pat in SEEDS.items():
        c,n=seed_bucket_curves(pat)
        if any(v>0 for v in n.values()): per_seed[name]=c; ns[name]=n
    seeds=[s for s in ["s42","s1","s2"] if s in per_seed]
    print("=== per-seed per-bucket drift share (%) @ steps",STEPS,"===")
    for s in seeds:
        for b in BUCKETS:
            print(f"  {s} {b} (n={ns[s][b]}): {[round(x,1) for x in per_seed[s][b]]}")
    # 3-seed 均值曲线（每桶）
    mean={b:np.mean([per_seed[s][b] for s in seeds],axis=0) for b in BUCKETS}
    std ={b:np.std ([per_seed[s][b] for s in seeds],axis=0) for b in BUCKETS}
    print("\n=== 3-seed 均值 (correct − wrong) by step ===")
    for i,k in enumerate(STEPS):
        print(f"  step{k}: correct={mean['correct'][i]:.1f} wrong={mean['wrong'][i]:.1f} Δ(c−w)={mean['correct'][i]-mean['wrong'][i]:+.1f}pp")
    d150=float(mean["correct"][-1]-mean["wrong"][-1])
    a_ok=bool(d150>=10)
    # per-seed 150 Δ（稳健性参照）
    per_seed_d150={s:float(per_seed[s]["correct"][-1]-per_seed[s]["wrong"][-1]) for s in seeds}
    minN=min(min(ns[s]["correct"] for s in seeds), min(ns[s]["wrong"] for s in seeds))
    prov = minN<20
    print("\n=== (a) 判读 ===")
    print(f"  150 步 3-seed 均值 Δ(correct−wrong) = {d150:+.1f}pp  (判据 ≥10)  -> {'PASS' if a_ok else 'FAIL'}"
          + ("  [provisional: correct/wrong n<20]" if prov else ""))
    print(f"  per-seed 150 Δ: "+", ".join(f'{s}={per_seed_d150[s]:+.1f}' for s in seeds))
    out={"steps":STEPS,"per_seed":per_seed,"n":ns,
         "mean":{b:mean[b].tolist() for b in BUCKETS},"std":{b:std[b].tolist() for b in BUCKETS},
         "delta_c_minus_w_150":d150,"per_seed_delta150":per_seed_d150,
         "a_pass":a_ok,"provisional":bool(prov),
         "verdict":f"(a) {'PASS' if a_ok else 'FAIL'}: 150步correct−wrong={d150:+.1f}pp (≥10)"}
    json.dump(out,open(os.path.join(HERE,"analysis","triage_a_drift.json"),"w"),indent=2,ensure_ascii=False)
    print("[saved] probes/analysis/triage_a_drift.json")

if __name__=="__main__": main()
