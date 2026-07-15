#!/usr/bin/env python3
"""方向轴侦察：B 臂崩塌主因归属的四问（预注册见 docs/direction_axis_recon.md）。

wrong 桶 rollout 上，teacher=冻结 base，同一 rollout 三套 forward：
  π_T   = base(problem + 完整参考解)
  π_T̃  = base(problem + 腐蚀参考解)   （corrupt_solution）
  π_ref = base(参考解, 去掉题目)        （Purified OPSD reference-only teacher）

Q-a 冲突位置(p2/p1≥thr)占比; Q-b 冲突位置上 δ 正向质量在 epistemic 表的富集;
Q-c 冲突∩腐蚀(c_t>ρ) 重叠(Jaccard+双向 P/R); Q-d δ vs PMI残差 Δit 的 Spearman(全词表)+top质量重叠。

存储纪律 §4：只落派生标量，绝不存 [T,V]。
用法: python probes/direction_axis.py --limit 50 --tok_cap 768
"""
import argparse, json, os, sys
import numpy as np, torch
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0,HERE); sys.path.insert(0,ROOT)
from scoring import build_teacher_prompt_text, forward_rollout_logits, _transition_prompt
from divergence import token_jsd
from corrupt_solution import corrupt_solution

# --- Q-b epistemic 预注册集合（用户 Q-b 逐字 8 词；TRD 16-token 全表待核，见 doc 口径限定） ---
EPISTEMIC = {"wait","actually","perhaps","maybe","but","however","check","reconsider"}
CONFLICT_THRS = [0.2, 0.3, 0.5]     # 主判据 0.3
RHO = 0.0007                         # 腐蚀敏感阈（diag2x2_shard0 90 分位）
TOPK = 64                            # Q-d top 质量重叠
CHUNK = 128                          # Q-d 全词表 argsort 分块位置数


def build_reference_only_prompt_text(tok, solution, teacher_thinking=True):
    """build_teacher_prompt_text 去掉 'Problem: {problem}\\n\\n' 段 = Purified π_ref。"""
    tp = _transition_prompt(tok)
    msg = (f"Here is a reference solution to this problem:\n"
           f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
           f"{tp}\n"
           f"Please reason step by step, and put your final answer within \\boxed{{}}.")
    return tok.apply_chat_template([{"role":"user","content":msg}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=teacher_thinking)


def epistemic_ids(tok):
    ids=[]
    vocab=tok.get_vocab()
    for t,i in vocab.items():
        core="".join(ch for ch in t.lstrip(" \t▁Ġ").lower() if ch.isalpha())
        if core in EPISTEMIC:
            ids.append(i)
    return torch.tensor(sorted(set(ids)), dtype=torch.long)


def spearman_fullvocab(a, b):
    """逐行(位置)全词表 Spearman，a/b: [T,V] fp32 -> [T] 相关系数。分块由调用方控。"""
    ra = a.argsort(dim=-1).argsort(dim=-1).float()
    rb = b.argsort(dim=-1).argsort(dim=-1).float()
    ra = ra - ra.mean(dim=-1, keepdim=True)
    rb = rb - rb.mean(dim=-1, keepdim=True)
    num = (ra*rb).sum(dim=-1)
    den = torch.sqrt((ra*ra).sum(dim=-1) * (rb*rb).sum(dim=-1)).clamp_min(1e-12)
    return num/den


def topmass_overlap(a, b, k):
    """各取正向 top-k 的 overlap 系数 |∩|/k，a/b: [T,V] -> [T]。"""
    ta = a.topk(k, dim=-1).indices
    tb = b.topk(k, dim=-1).indices
    out=[]
    for i in range(a.shape[0]):
        out.append(len(set(ta[i].tolist()) & set(tb[i].tolist()))/k)
    return torch.tensor(out)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.expanduser("~/models/Qwen3-1.7B"))
    ap.add_argument("--rollouts", default=os.path.join(HERE,"data","rollouts_ckpt50_max4096.jsonl"))
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--tok_cap", type=int, default=768)
    ap.add_argument("--out", default=os.path.join(HERE,"analysis","direction_axis.json"))
    a=ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("HF_DATASETS_OFFLINE","1")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    tok=AutoTokenizer.from_pretrained(a.base)
    model=AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2").to("cuda").eval()
    tr=load_dataset("siyanzhao/Openthoughts_math_30k_opsd", split="train")
    epi=epistemic_ids(tok).to("cuda"); V=model.config.vocab_size
    base_rate=len(epi)/V

    recs=[json.loads(l) for l in open(a.rollouts)]
    wrong=[r for r in recs if r["bucket"]=="wrong"]

    # 累加器
    qa={thr:{"conf":0,"tot":0} for thr in CONFLICT_THRS}      # Q-a 冲突占比
    qb={"epi_mass":0.0,"tot_mass":0.0}                         # Q-b 富集(thr=0.3)
    qc={"inter":0,"union":0,"conf":0,"corr":0}                 # Q-c 重叠(thr=0.3)
    qd={"conf":[], "all":[], "conf_ov":[], "all_ov":[]}        # Q-d Spearman/overlap
    nr=0
    for r in wrong:
        if nr>=a.limit: break
        ids=r["completion_token_ids"][:a.tok_cap]
        if len(ids)<8: continue
        pid=r["problem_id"]; problem=r["problem"]; gt=r["gt_answer"]
        sol=tr[pid]["solution"]
        corrs=r.get("corrupted_answers") or []
        if not corrs: continue
        csol,nrep=corrupt_solution(sol, gt, corrs[0])
        if nrep==0: continue
        with torch.no_grad():
            pT,_,_ = forward_rollout_logits(model, tok, build_teacher_prompt_text(tok,problem,sol), ids)
            pTt,_,_= forward_rollout_logits(model, tok, build_teacher_prompt_text(tok,problem,csol), ids)
            pRef,_,_=forward_rollout_logits(model, tok, build_reference_only_prompt_text(tok,sol), ids)
        T=min(pT.shape[0],pTt.shape[0],pRef.shape[0])
        pT,pTt,pRef=pT[:T].cuda(),pTt[:T].cuda(),pRef[:T].cuda()
        logPT=torch.log_softmax(pT,-1); logPTt=torch.log_softmax(pTt,-1); logPref=torch.log_softmax(pRef,-1)
        prob=logPT.exp()
        top2=prob.topk(2,dim=-1).values
        ratio=top2[:,1]/top2[:,0].clamp_min(1e-12)             # [T] p2/p1
        ct=token_jsd(pT,pTt)                                   # [T] 腐蚀敏感
        delta=logPT-logPTt                                     # [T,V] 腐蚀差分 δ
        dit=logPT-logPref                                      # [T,V] PMI 残差 Δit

        # Q-a
        for thr in CONFLICT_THRS:
            m=(ratio>=thr)
            qa[thr]["conf"]+=int(m.sum()); qa[thr]["tot"]+=T
        conf=(ratio>=0.3)                                      # 主判据
        corr=(ct>RHO)

        # Q-b (冲突位置, δ 正向质量在 epi 富集)
        if int(conf.sum())>0:
            dpos=delta[conf].clamp_min(0)                      # [Nc,V]
            qb["epi_mass"]+=float(dpos.index_select(1,epi).sum())
            qb["tot_mass"]+=float(dpos.sum())

        # Q-c
        inter=int((conf&corr).sum()); uni=int((conf|corr).sum())
        qc["inter"]+=inter; qc["union"]+=uni
        qc["conf"]+=int(conf.sum()); qc["corr"]+=int(corr.sum())

        # Q-d 全词表 Spearman + top 质量重叠, 分块
        sp=torch.empty(T); ov=torch.empty(T)
        for s in range(0,T,CHUNK):
            e=min(s+CHUNK,T)
            sp[s:e]=spearman_fullvocab(delta[s:e], dit[s:e]).cpu()
            ov[s:e]=topmass_overlap(delta[s:e], dit[s:e], TOPK)
        confc=conf.cpu()
        qd["all"].append(sp); qd["all_ov"].append(ov)
        if int(confc.sum())>0:
            qd["conf"].append(sp[confc]); qd["conf_ov"].append(ov[confc])
        nr+=1
        del pT,pTt,pRef,logPT,logPTt,logPref,prob,delta,dit
        if nr%10==0: torch.cuda.empty_cache(); print(f"  ...{nr} rollouts", flush=True)

    def catmean(xs):
        if not xs: return None
        v=torch.cat(xs); return float(v.mean()), int(v.numel())
    sp_all=catmean(qd["all"]); sp_conf=catmean(qd["conf"])
    ov_all=catmean(qd["all_ov"]); ov_conf=catmean(qd["conf_ov"])
    out={
      "n_rollout":nr, "tok_cap":a.tok_cap, "rho":RHO,
      "epistemic_set":sorted(EPISTEMIC), "epi_token_ids":len(epi), "epi_base_rate":base_rate,
      "Q_a_conflict_frac":{str(thr):qa[thr]["conf"]/max(1,qa[thr]["tot"]) for thr in CONFLICT_THRS},
      "Q_b_provisional":{
          "epi_mass_share": qb["epi_mass"]/max(1e-12,qb["tot_mass"]),
          "base_rate": base_rate,
          "enrichment_ratio": (qb["epi_mass"]/max(1e-12,qb["tot_mass"]))/max(1e-12,base_rate),
          "note":"epistemic set = 用户 Q-b 逐字 8 词; TRD 16-token 全表待核, 结果 provisional",
      },
      "Q_c_overlap":{
          "jaccard": qc["inter"]/max(1,qc["union"]),
          "P_conflict_given_corruption": qc["inter"]/max(1,qc["corr"]),
          "P_corruption_given_conflict": qc["inter"]/max(1,qc["conf"]),
          "n_conflict":qc["conf"], "n_corruption":qc["corr"], "n_inter":qc["inter"],
      },
      "Q_d_orthogonality":{
          "spearman_conflict": sp_conf[0] if sp_conf else None,
          "spearman_all": sp_all[0] if sp_all else None,
          "topmass_overlap_conflict": ov_conf[0] if ov_conf else None,
          "topmass_overlap_all": ov_all[0] if ov_all else None,
          "n_conflict_pos": sp_conf[1] if sp_conf else 0, "n_all_pos": sp_all[1] if sp_all else 0,
          "topk": TOPK, "spearman_domain":"full_vocab",
          "verdict_rule":">=0.7 覆盖弃C / <=0.4 正交存活C / 中间人判",
      },
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out,"w"), indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("[saved]",a.out)

if __name__=="__main__": main()
