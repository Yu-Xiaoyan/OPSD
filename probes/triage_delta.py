#!/usr/bin/env python3
"""分诊 rationale (b): wrong 桶腐蚀敏感(δ 正向质量, P_T 加权) vs correct 桶。
预注册见 docs/triage_rationale_recon.md。

每 rollout 标量 = Σ_{t∈冲突} Σ_v P_T·max(δ,0) / T （冲突位置 p2/p1≥0.3，与 R-b 一致）。
teacher=base 固定；δ=log π_T − log π_T̃（特权答案腐蚀）。按桶取均值，判 (b): wrong ≥1.5×correct。
用法: python probes/triage_delta.py --limit_per_bucket 100
"""
import argparse, json, os, sys
import numpy as np, torch
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0,HERE); sys.path.insert(0,ROOT)
from scoring import build_teacher_prompt_text, forward_rollout_logits
from corrupt_solution import corrupt_solution
CONFLICT_THR=0.3

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.expanduser("~/models/Qwen3-1.7B"))
    ap.add_argument("--rollouts", default=os.path.join(HERE,"data","rollouts_ckpt50_max4096.jsonl"))
    ap.add_argument("--limit_per_bucket", type=int, default=1000)
    ap.add_argument("--tok_cap", type=int, default=768)
    ap.add_argument("--out", default=os.path.join(HERE,"analysis","triage_b_delta.json"))
    a=ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("HF_DATASETS_OFFLINE","1")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    tok=AutoTokenizer.from_pretrained(a.base)
    model=AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2").to("cuda").eval()
    tr=load_dataset("siyanzhao/Openthoughts_math_30k_opsd", split="train")
    recs=[json.loads(l) for l in open(a.rollouts)]
    scores={"correct":[], "wrong":[]}
    cnt={"correct":0,"wrong":0}
    for r in recs:
        b=r.get("bucket")
        if b not in scores: continue
        if cnt[b]>=a.limit_per_bucket: continue
        ids=r["completion_token_ids"][:a.tok_cap]
        if len(ids)<8: continue
        corrs=r.get("corrupted_answers") or []
        if not corrs: continue
        sol=tr[r["problem_id"]]["solution"]; gt=r["gt_answer"]; problem=r["problem"]
        csol,nrep=corrupt_solution(sol, gt, corrs[0])
        if nrep==0: continue
        with torch.no_grad():
            pT,_,_ =forward_rollout_logits(model, tok, build_teacher_prompt_text(tok,problem,sol), ids)
            pTt,_,_=forward_rollout_logits(model, tok, build_teacher_prompt_text(tok,problem,csol), ids)
        T=min(pT.shape[0],pTt.shape[0]); pT,pTt=pT[:T].cuda(),pTt[:T].cuda()
        logPT=torch.log_softmax(pT,-1); logPTt=torch.log_softmax(pTt,-1); prob=logPT.exp()
        top2=prob.topk(2,dim=-1).values; ratio=top2[:,1]/top2[:,0].clamp_min(1e-12)
        conf=(ratio>=CONFLICT_THR)
        delta=logPT-logPTt
        wd=(prob*delta.clamp_min(0)).sum(-1)                 # [T] Σ_v P_T·max(δ,0)
        score=float(wd[conf].sum())/T if T>0 else 0.0        # /T 口径
        scores[b].append(score); cnt[b]+=1
        del pT,pTt,logPT,logPTt,prob,delta,wd
        if (cnt["correct"]+cnt["wrong"])%20==0:
            torch.cuda.empty_cache(); print(f"  ...correct={cnt['correct']} wrong={cnt['wrong']}", flush=True)
    mc=float(np.mean(scores["correct"])) if scores["correct"] else float("nan")
    mw=float(np.mean(scores["wrong"])) if scores["wrong"] else float("nan")
    ratio_wc=mw/mc if mc>0 else float("nan")
    b_ok=bool(ratio_wc>=1.5)
    prov = min(len(scores["correct"]),len(scores["wrong"]))<20
    out={"n":{k:len(v) for k,v in scores.items()},
         "mean_score":{"correct":mc,"wrong":mw},
         "wrong_over_correct":ratio_wc,"b_pass":b_ok,"provisional":bool(prov),
         "conflict_thr":CONFLICT_THR,"tok_cap":a.tok_cap,
         "verdict":f"(b) {'PASS' if b_ok else 'FAIL'}: wrong/correct={ratio_wc:.2f} (≥1.5)"}
    os.makedirs(os.path.dirname(a.out),exist_ok=True)
    json.dump(out,open(a.out,"w"),indent=2,ensure_ascii=False)
    print(json.dumps(out,indent=2,ensure_ascii=False)); print("[saved]",a.out)

if __name__=="__main__": main()
