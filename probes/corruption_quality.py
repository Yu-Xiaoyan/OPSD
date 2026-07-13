#!/usr/bin/env python3
"""裁决探针：分布级腐蚀质量 = mean token JSD(T_S, T_S̃)，在给定 checkpoint 的 rollout 上。
teacher 固定 = base（A/B 通用），差异只来自各自 generation dump 的 rollout。
用法: python probes/corruption_quality.py --gen_dir <run>/generations --steps 100,150 --tag A --limit 60
输出: probes/analysis/corruption_quality_<tag>.json  {step: {mean_jsd, n_rollout, n_token}}
"""
import argparse, json, glob, os, re, sys
import numpy as np, torch
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from scoring import build_teacher_prompt_text, forward_rollout_logits
from divergence import token_jsd
from corrupt_solution import corrupt_solution
from corrupt_answers import corrupt_answer
import leakage_detector as L

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.expanduser("~/models/Qwen3-1.7B"))
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--steps", default="100,150")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=60)  # 每 step 最多几条 rollout
    ap.add_argument("--out", default=None)
    a=ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("HF_DATASETS_OFFLINE","1")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    tok=AutoTokenizer.from_pretrained(a.base)
    model=AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2").to("cuda").eval()  # base = fixed teacher
    ds=load_dataset("siyanzhao/Openthoughts_math_30k_opsd", split="train")
    sol={L.problem_key(p): (s, str(ans)) for p,s,ans in zip(ds["problem"],ds["solution"],ds["Answer"]) if p}
    res={}
    for step in [int(x) for x in a.steps.split(",")]:
        fps=glob.glob(f"{a.gen_dir}/generations_step_{step}.json")
        if not fps: res[step]={"error":"no dump"}; continue
        gens=json.load(open(fps[0]))["generations"][:a.limit]
        jsd_sum=0.0; ntok=0; nr=0
        for g in gens:
            prob=L.extract_problem_from_prompt(g["prompt"])
            if not prob: continue
            info=sol.get(L.problem_key(prob))
            if not info: continue
            s_full, gt=info
            ids=tok(g["completion"], add_special_tokens=False).input_ids[:1024]
            if len(ids)<8: continue
            corr=corrupt_answer(gt, n=1, seed=nr)
            if not corr: continue
            csol,nrep=corrupt_solution(s_full, gt, corr[0])
            if nrep==0: continue   # 腐蚀未生效则跳过（否则 T_S̃=T_S）
            with torch.no_grad():
                TS,_,_=forward_rollout_logits(model, tok, build_teacher_prompt_text(tok,prob,s_full), ids)
                TSt,_,_=forward_rollout_logits(model, tok, build_teacher_prompt_text(tok,prob,csol), ids)
            T=min(TS.shape[0],TSt.shape[0])
            cj=token_jsd(TS[:T],TSt[:T])
            jsd_sum+=float(cj.sum()); ntok+=T; nr+=1
            del TS,TSt,cj
        res[step]={"mean_jsd_per_token": jsd_sum/max(1,ntok), "n_rollout":nr, "n_token":ntok}
        print(f"[{a.tag}] step{step}: mean_jsd={jsd_sum/max(1,ntok):.5f} nr={nr} ntok={ntok}", flush=True)
    out=a.out or f"probes/analysis/corruption_quality_{a.tag}.json"
    json.dump({"tag":a.tag,"gen_dir":a.gen_dir,"steps":res}, open(out,"w"), indent=2)
    print("[saved]",out)

if __name__=="__main__": main()
