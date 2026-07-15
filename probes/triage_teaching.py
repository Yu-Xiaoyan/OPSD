#!/usr/bin/env python3
"""裁决二 (i): 按桶测教学含量 = teacher-student 分歧绝对量 JSD(T_S,S_k)。纯重聚合，CPU。
预注册见 docs/triage_rationale_recon.md（裁决二）。

per_k[k]["total"] = JSD(T_S, S_k) 逐 token；按桶取每 rollout 均值，报 wrong/correct 比值。
主判 step=50（rollout 由 ckpt50 生成，on-policy）；附全步轨迹。3-seed。
判据: (i) ≥1.5 correct降权获据 / <1.2 取消降权 / 1.2~1.5 人判。
"""
import glob, json, os
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"data")
STEPS=[25,50,75,100,125,150]
SEEDS={"s42":"drift_shard*.jsonl","s1":"drift_s1_shard*.jsonl","s2":"drift_s2_shard*.jsonl"}
BUCKETS=["correct","wrong"]

def seed_bucket_teach(pattern):
    """返回 {bucket:{step:[per-rollout mean JSD(T_S,S_k)]}}, n{bucket}."""
    files=sorted(glob.glob(os.path.join(DATA,pattern)))
    per={b:{k:[] for k in STEPS} for b in BUCKETS}; n={b:0 for b in BUCKETS}
    for fp in files:
        for line in open(fp):
            r=json.loads(line); b=r.get("bucket")
            if b not in BUCKETS: continue
            n[b]+=1
            for k in STEPS:
                tot=np.asarray(r["per_k"][str(k)]["total"],float)
                per[b][k].append(float(tot.mean()))    # 逐 token 均值 = 每 rollout 教学含量
    return per,n

def main():
    # 合并三 seed（按 rollout 池化：各 seed 各 rollout 的 per-rollout 教学含量并入桶）
    pool={b:{k:[] for k in STEPS} for b in BUCKETS}; ntot={b:0 for b in BUCKETS}
    per_seed_ratio={}   # {seed:{step:wrong/correct}}
    for name,pat in SEEDS.items():
        per,n=seed_bucket_teach(pat)
        if not any(n.values()): continue
        per_seed_ratio[name]={}
        for b in BUCKETS:
            ntot[b]+=n[b]
            for k in STEPS: pool[b][k]+=per[b][k]
        for k in STEPS:
            mc=np.mean(per["correct"][k]); mw=np.mean(per["wrong"][k])
            per_seed_ratio[name][k]=float(mw/mc) if mc>0 else float("nan")
    # 池化后的桶均值与比值
    meanc={k:float(np.mean(pool["correct"][k])) for k in STEPS}
    meanw={k:float(np.mean(pool["wrong"][k])) for k in STEPS}
    ratio={k:(meanw[k]/meanc[k] if meanc[k]>0 else float("nan")) for k in STEPS}
    print("=== (i) 教学含量 JSD(T_S,S_k) 逐 rollout 均值, 池化3-seed ===")
    print(f"  n: correct={ntot['correct']} wrong={ntot['wrong']}")
    for k in STEPS:
        print(f"  step{k}: correct={meanc[k]:.4f} wrong={meanw[k]:.4f} wrong/correct={ratio[k]:.2f}x")
    r50=ratio[50]
    if r50>=1.5: verdict="(i) ≥1.5 -> correct支降权重新获据(教学含量低,与漂移无关)"
    elif r50<1.2: verdict="(i) <1.2 -> 取消correct支降权改等权, v2简化为'错支手术+全局漂移扣除'"
    else: verdict="(i) 1.2~1.5 -> 轻降权, 人判"
    print("\n=== (i) 判读 (主判 step50) ===")
    print(f"  wrong/correct @step50 = {r50:.2f}x -> {verdict}")
    print(f"  per-seed @step50 ratio: "+", ".join(f'{s}={per_seed_ratio[s][50]:.2f}' for s in per_seed_ratio))
    print("\n=== (ii) 辅助: δ 正向质量(P_T加权) 桶间比值 = triage(b) 换口径复测 ===")
    print("  wrong/correct = 1.55x (probes/analysis/triage_b_delta.json; 冲突位 P_T 加权 δ+ 均值)")
    out={"steps":STEPS,"n":ntot,"mean_correct":meanc,"mean_wrong":meanw,"ratio_wc":ratio,
         "ratio_at_50":r50,"per_seed_ratio_50":{s:per_seed_ratio[s][50] for s in per_seed_ratio},
         "ii_delta_PT_ratio_from_b":1.55,"verdict":verdict}
    json.dump(out,open(os.path.join(HERE,"analysis","triage_teaching.json"),"w"),indent=2,ensure_ascii=False)
    print("[saved] probes/analysis/triage_teaching.json")

if __name__=="__main__": main()
