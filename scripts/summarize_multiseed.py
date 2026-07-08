#!/usr/bin/env python3
"""T2: 3-seed {42,1,2} mean±std for v0 vs OPSD, AIME24/25(avg@12) + MATH500(avg@4)
@ ckpt100/150. seed42+base: results/v0_eval|repro_eval; seed1/2: results/multiseed_eval.
Integrity-checks val_n and protocol lock (temp/top_p==1.0). Stdlib only."""
import json
from statistics import mean, stdev
def p(m,s,c,ds):
    if s==42: return f"results/v0_eval/v0ckpt{c}_{ds}.json" if m=="v0" else f"results/repro_eval/ckpt{c}_{ds}.json"
    return f"results/multiseed_eval/{('v0s' if m=='v0' else 'opsds')}{s}ckpt{c}_{ds}.json"
def load(fp):
    try: return json.load(open(fp))
    except Exception: return None
seeds=[42,1,2]; bms=[("aime24",12),("aime25",12),("math500",4)]; warns=[]
def cell(m,c,ds,n):
    out=[]
    for s in seeds:
        d=load(p(m,s,c,ds))
        if d is None: warns.append(f"MISSING {m} s{s} c{c} {ds}"); continue
        if d.get("val_n")!=n: warns.append(f"VALN {m} s{s} c{c} {ds}={d.get('val_n')}")
        if abs(d.get("temperature",1)-1)>1e-9 or abs(d.get("top_p",1)-1)>1e-9: warns.append(f"PROTO {m} s{s} c{c} {ds}")
        out.append(d["average_at_n_pct"])
    return out
def basev(ds,n):
    d=load(f"results/repro_eval/base_{ds}.json"); return d["average_at_n_pct"] if d else None
print("T2: 3-seed {42,1,2} mean±std (sample sd)")
for ds,n in bms:
    b=basev(ds,n); print(f"\n{ds.upper()} (avg@{n}) base={b:.1f}")
    print(f"  {'ckpt':<5}{'OPSD':>13}{'v0':>13}{'Δ':>8}{'v0-base':>9}")
    for c in [100,150]:
        ov,vv=cell("opsd",c,ds,n),cell("v0",c,ds,n)
        if len(ov)==3==len(vv):
            om,vm=mean(ov),mean(vv)
            print(f"  {c:<5}{om:>7.1f}±{stdev(ov):>4.1f}{vm:>7.1f}±{stdev(vv):>4.1f}{vm-om:>+8.1f}{vm-b:>+9.1f}")
        else: print(f"  {c:<5} incomplete OPSD={ov} v0={vv}")
print("\nintegrity warns:", warns or "none")
