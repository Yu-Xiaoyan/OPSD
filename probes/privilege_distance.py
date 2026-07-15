#!/usr/bin/env python3
"""特权距离探针：只给答案能不能把分布拉近到"给完整解"?
三个分布同一 base、同一固定 rollout 上 teacher-force,只差 prompt 特权内容:
  P_CoT = teacher(problem + 完整 solution)
  P_Ans = teacher(problem + 只 \boxed{答案})
  P_S   = student(problem only)
比 mean JSD(P_Ans,P_CoT) vs mean JSD(P_S,P_CoT)。若≈ -> 光有答案≈什么都不给(离solution-teacher一样远)。
不吃 teacher-forcing 反事实的坑(无篡改/无腐蚀,只是三个真实条件的分布距离)。
用法: python probes/privilege_distance.py --limit 60 --tok_cap 768
"""
import argparse, json, os, sys
import numpy as np, torch
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0,HERE); sys.path.insert(0,ROOT)
from scoring import build_teacher_prompt_text, build_student_prompt_text, forward_rollout_logits
from divergence import token_jsd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.expanduser("~/models/Qwen3-1.7B"))
    ap.add_argument("--rollouts", default=os.path.join(HERE,"data","rollouts_ckpt50_max4096.jsonl"))
    ap.add_argument("--limit", type=int, default=60)       # 各桶
    ap.add_argument("--tok_cap", type=int, default=768)
    ap.add_argument("--out", default=os.path.join(HERE,"analysis","privilege_distance.json"))
    a=ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("HF_DATASETS_OFFLINE","1")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    tok=AutoTokenizer.from_pretrained(a.base)
    model=AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2").to("cuda").eval()
    tr=load_dataset("siyanzhao/Openthoughts_math_30k_opsd", split="train")
    recs=[json.loads(l) for l in open(a.rollouts)]
    agg={b:{"ans_cot":[], "s_cot":[], "ans_s":[], "ntok":0, "nr":0} for b in ("correct","wrong")}
    cnt={"correct":0,"wrong":0}
    for r in recs:
        b=r.get("bucket")
        if b not in agg: continue
        if cnt[b]>=a.limit: continue
        ids=r["completion_token_ids"][:a.tok_cap]
        if len(ids)<8: continue
        problem=r["problem"]; gt=str(r["gt_answer"]); sol=tr[r["problem_id"]]["solution"]
        if not sol or not gt: continue
        ans_ref=f"\\boxed{{{gt}}}"                          # answer-only 特权段
        with torch.no_grad():
            Pcot,_,_=forward_rollout_logits(model, tok, build_teacher_prompt_text(tok,problem,sol), ids)
            Pans,_,_=forward_rollout_logits(model, tok, build_teacher_prompt_text(tok,problem,ans_ref), ids)
            Ps,_,_  =forward_rollout_logits(model, tok, build_student_prompt_text(tok,problem), ids)
        T=min(Pcot.shape[0],Pans.shape[0],Ps.shape[0])
        Pcot,Pans,Ps=Pcot[:T].cuda(),Pans[:T].cuda(),Ps[:T].cuda()
        agg[b]["ans_cot"].append(float(token_jsd(Pans,Pcot).mean()))
        agg[b]["s_cot"].append(float(token_jsd(Ps,Pcot).mean()))
        agg[b]["ans_s"].append(float(token_jsd(Pans,Ps).mean()))
        agg[b]["ntok"]+=T; agg[b]["nr"]+=1; cnt[b]+=1
        del Pcot,Pans,Ps
        if (cnt["correct"]+cnt["wrong"])%10==0:
            torch.cuda.empty_cache(); print(f"  ...c={cnt['correct']} w={cnt['wrong']}", flush=True)
    out={}
    for b in ("correct","wrong"):
        d=agg[b]
        if d["nr"]==0: out[b]={"nr":0}; continue
        m_ac=float(np.mean(d["ans_cot"])); m_sc=float(np.mean(d["s_cot"])); m_as=float(np.mean(d["ans_s"]))
        out[b]={"nr":d["nr"],"ntok":d["ntok"],
                "mean_JSD_Ans_CoT":m_ac, "mean_JSD_S_CoT":m_sc, "mean_JSD_Ans_S":m_as,
                "ratio_AnsCoT_over_SCoT": m_ac/m_sc if m_sc>0 else None,
                "closeness_gain_frac": 1 - m_ac/m_sc if m_sc>0 else None}  # 答案把分布拉近了多少(相对S)
    # 池化两桶
    allac=agg["correct"]["ans_cot"]+agg["wrong"]["ans_cot"]
    allsc=agg["correct"]["s_cot"]+agg["wrong"]["s_cot"]
    if allac:
        mac=float(np.mean(allac)); msc=float(np.mean(allsc))
        out["pooled"]={"nr":len(allac),"mean_JSD_Ans_CoT":mac,"mean_JSD_S_CoT":msc,
                       "ratio":mac/msc if msc>0 else None,"closeness_gain_frac":1-mac/msc if msc>0 else None}
    os.makedirs(os.path.dirname(a.out),exist_ok=True)
    json.dump(out,open(a.out,"w"),indent=2,ensure_ascii=False)
    print(json.dumps(out,indent=2,ensure_ascii=False)); print("[saved]",a.out)

if __name__=="__main__": main()
