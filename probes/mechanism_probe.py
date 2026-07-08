#!/usr/bin/env python3
"""机制验证 M1/M2/M3 (+M4 剂量) —— 1 卡 probe，wrong 桶 rollout 上。
见 docs/mechanism_validation.md。取舍：teacher 用直接特权(problem+参考解+transition)
作为 bimodality 的可行代理；V(t)/ΔV 用 student 视图(与 teacher 分布独立)。
结果无论方向如实写 probes/analysis/mechanism_probe.json。"""
import os, sys, json, argparse, random
import numpy as np, torch, torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from answer_likelihood import answer_likelihood_probe, auto_checkpoints
from opsd_trainer import OPSDTrainer
JSD=OPSDTrainer.generalized_jsd_loss

def spearman(x,y):
    x=np.asarray(x); y=np.asarray(y)
    rx=np.argsort(np.argsort(x)); ry=np.argsort(np.argsort(y))
    if rx.std()==0 or ry.std()==0: return float('nan')
    return float(np.corrcoef(rx,ry)[0,1])
def auc(labels,scores):
    labels=np.asarray(labels); scores=np.asarray(scores)
    P=labels.sum(); N=len(labels)-P
    if P==0 or N==0: return float('nan')
    order=np.argsort(scores); ranks=np.empty(len(scores)); ranks[order]=np.arange(1,len(scores)+1)
    return float((ranks[labels==1].sum()-P*(P+1)/2)/(P*N))

def build_prompts(tok, problem, solution):
    su=f"Problem: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    sp=tok.apply_chat_template([{"role":"user","content":su}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    tu=(f"Problem: {problem}\n\nHere is a reference solution to this problem:\n"
        f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
        f"\n\nAfter reading the reference solution above, make sure you truly understand the reasoning "
        f"behind each step — do not copy or paraphrase it. Now, using your own words and independent "
        f"reasoning, derive the same final answer to the problem above.\n"
        f"Please reason step by step, and put your final answer within \\boxed{{}}.")
    tp=tok.apply_chat_template([{"role":"user","content":tu}],tokenize=False,add_generation_prompt=True,enable_thinking=True)
    return sp, tp

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",default=os.path.expanduser("~/models/Qwen3-1.7B"))
    ap.add_argument("--adapter",default=os.path.expanduser("~/opsd_outputs/qwen31b_v0_s1/checkpoint-100"))
    ap.add_argument("--rollouts",default="probes/data/rollouts_ckpt50_max1024.jsonl")
    ap.add_argument("--n_wrong",type=int,default=20)
    ap.add_argument("--n_heldout",type=int,default=20)
    ap.add_argument("--tau",type=float,default=7.63)
    ap.add_argument("--lr",type=float,default=1e-4)
    ap.add_argument("--wrong_tok_frac",type=float,default=0.078)  # M4: token-级 wrong 占比
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--out",default="probes/analysis/mechanism_probe.json")
    a=ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev="cuda"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok=AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    print("[load] base+adapter", a.adapter, flush=True)
    base=AutoModelForCausalLM.from_pretrained(a.base,torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2").to(dev)
    student=PeftModel.from_pretrained(base, a.adapter, is_trainable=True)  # student=trained; teacher=disable_adapter
    student.eval()
    # dataset problem->solution
    os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("HF_DATASETS_OFFLINE","1")
    from datasets import load_dataset
    ds=load_dataset("siyanzhao/Openthoughts_math_30k_opsd",split="train")
    sol={str(x["problem"])[:200]: x["solution"] for x in ds}
    # wrong rollouts
    W=[]
    for line in open(a.rollouts):
        r=json.loads(line)
        if r.get("bucket")=="wrong":
            k=str(r["problem"])[:200]
            if k in sol: r["solution"]=sol[k]; W.append(r)
        if len(W)>=a.n_wrong: break
    print(f"[data] {len(W)} wrong rollouts", flush=True)
    # held-out anchors: dataset problems not used above
    used={str(r["problem"])[:200] for r in W}
    anchors=[x for x in ds if str(x["problem"])[:200] not in used][:a.n_heldout]

    tau=a.tau
    m1_w, m1_top2, m1_ent = [], [], []      # per-token: v0 weight, teacher top2-ratio, teacher entropy
    dose_num=0.0; dose_den=0.0
    m2_coh_uni=[]; m2_coh_v0=[]
    for r in W:
        ids=r["completion_token_ids"]; problem=r["problem"]; gt=str(r["gt_answer"])
        if len(ids)<8: continue
        sp,tp=build_prompts(tok,problem,r["solution"])
        sp_ids=tok(sp,return_tensors="pt").input_ids[0].tolist()
        tp_ids=tok(tp,return_tensors="pt").input_ids[0].tolist()
        s_in=torch.tensor([sp_ids+ids],device=dev); t_in=torch.tensor([tp_ids+ids],device=dev)
        # teacher forward (disable adapter, no grad)
        with torch.no_grad(), student.disable_adapter():
            t_logits=student(t_in).logits[:, len(tp_ids)-1:-1, :].float()  # [1,T,V] over rollout tokens
        # student forward (grad on for M2)
        s_logits_full=student(s_in).logits
        s_logits=s_logits_full[:, len(sp_ids)-1:-1, :].float()
        T=min(t_logits.shape[1], s_logits.shape[1]); t_logits=t_logits[:,:T]; s_logits=s_logits[:,:T]
        # M1: teacher distribution shape
        tp_prob=F.softmax(t_logits[0],dim=-1)  # [T,V]
        top2=torch.topk(tp_prob,2,dim=-1).values  # [T,2]
        top2_ratio=(top2[:,1]/(top2[:,0]+1e-9)).detach().cpu().numpy()
        ent=(-(tp_prob*torch.log(tp_prob+1e-9)).sum(-1)).detach().cpu().numpy()
        # ΔV -> v0 weight (student-view)
        cps=auto_checkpoints(tok, ids, max_points=16)
        with torch.no_grad(), student.disable_adapter():
            vt=answer_likelihood_probe(student, tok, problem, ids, gt, checkpoint_positions=cps)
        # broadcast dV to tokens
        dv=np.zeros(len(ids)); pos=vt["checkpoint_positions"]; V=vt["V"]
        for j in range(1,len(pos)):
            lo=min(pos[j-1],len(ids)); hi=min(pos[j],len(ids)); dv[lo:hi]=V[j]-V[j-1]
        w=1/(1+np.exp(-dv/tau))  # sigmoid(dV/tau)
        w=w[:T]
        m1_w+=list(w); m1_top2+=list(top2_ratio); m1_ent+=list(ent)
        # M4 dose: sum(1-w) over these wrong tokens
        dose_num+=float((1-w).sum()); dose_den+=len(w)
        # M2: per-token logit grad of jsd; coherence uni vs v0
        labels=torch.tensor([ids[:T]],device=dev)
        jsd=JSD(student_logits=s_logits, teacher_logits=t_logits, labels=labels, beta=0.0,
                temperature=1.0, top_k=None, token_clip=0.05, reduction="none")  # [n_valid,K]
        per_tok=jsd.sum(-1)  # [T]
        g=torch.autograd.grad(per_tok.sum(), s_logits, retain_graph=False)[0][0]  # [T,V] = d(sum jsd)/d logit
        g=g.float()
        gn=g.norm(dim=-1)+1e-9  # [T]
        wt=torch.tensor(w,device=dev,dtype=g.dtype)
        # coherence = ||sum w g|| / sum w||g||
        coh_uni=float(g.sum(0).norm()/gn.sum())
        coh_v0=float((wt[:,None]*g).sum(0).norm()/(wt*gn).sum())
        m2_coh_uni.append(coh_uni); m2_coh_v0.append(coh_v0)
        student.zero_grad(set_to_none=True)

    # M1 stats
    m1={"spearman_w_vs_top2": spearman(m1_w,m1_top2),
        "spearman_w_vs_entropy": spearman(m1_w,m1_ent),
        "auc_lowW_predicts_highTop2": auc((np.asarray(m1_top2)>np.median(m1_top2)).astype(int), -np.asarray(m1_w)),
        "n_tokens": len(m1_w)}
    m4={"effective_removed_frac_within_wrong": dose_num/max(1,dose_den),
        "effective_removed_frac_of_total": (dose_num/max(1,dose_den))*a.wrong_tok_frac,
        "wrong_tok_frac_used": a.wrong_tok_frac}
    m2={"coh_uniform_mean": float(np.mean(m2_coh_uni)), "coh_v0_mean": float(np.mean(m2_coh_v0)),
        "coh_improved_frac": float(np.mean([v>u for u,v in zip(m2_coh_uni,m2_coh_v0)]))}
    out={"config":vars(a),"M1":m1,"M2":m2,"M4_dose":m4}
    print("[M1]",json.dumps(m1),flush=True)
    print("[M2]",json.dumps(m2),flush=True)
    print("[M4_dose]",json.dumps(m4),flush=True)
    # M3 在单独函数(单步更新), 见下
    out["M3"]=run_m3(student, tok, W, anchors, tau, a.lr, dev)
    print("[M3]",json.dumps(out["M3"]),flush=True)
    os.makedirs(os.path.dirname(a.out),exist_ok=True)
    json.dump(out, open(a.out,"w"), indent=2)
    print("[saved]",a.out,flush=True)

def _anchor_V(model, tok, items, dev):
    vals=[]
    for x in items:
        prob=x["problem"] if "problem" in x else x.get("problem","")
        gt=str(x.get("gt_answer", x.get("Answer","")))
        ids=x.get("completion_token_ids")
        with torch.no_grad(), model.disable_adapter() if False else torch.no_grad():
            # 用 student(当前 adapter)对答案似然打分: V(end)=log p(y*|student_prompt+rollout or problem)
            pass
    return vals

def run_m3(student, tok, W, anchors, tau, lr, dev):
    # 双锚: in-batch(W 自身 rollout 的 V) + held-out(anchors 的 V). 两组: OPSD(uniform) vs v0(gated) 单步更新.
    import copy
    def measure(model):
        # in-batch: W rollouts 的 V(end); held-out: anchors 的 V(end)(用其 Answer, 无 rollout 则跳过打分位置=末端)
        ib=[]; ho=[]
        with torch.no_grad(), model.disable_adapter():
            for r in W:
                v=answer_likelihood_probe(model,tok,r["problem"],r["completion_token_ids"],str(r["gt_answer"]),
                                          checkpoint_positions=[len(r["completion_token_ids"])])
                ib.append(v["V"][-1])
            for x in anchors[:12]:
                # held-out: 无 rollout, 用空前缀答案似然 V(0)=log p(y*|problem) 作锚
                v=answer_likelihood_probe(model,tok,x["problem"],[],str(x["Answer"]),checkpoint_positions=[0])
                ho.append(v["V"][-1])
        return float(np.mean(ib)), float(np.mean(ho))
    base_ib, base_ho = measure(student)
    res={"anchor_in_batch_pre":base_ib,"anchor_heldout_pre":base_ho}
    from opsd_trainer import OPSDTrainer as OT
    for name, gated in [("OPSD_uniform",False),("v0_gated",True)]:
        # 快照 adapter 参数
        snap={k:v.detach().clone() for k,v in student.named_parameters() if v.requires_grad}
        opt=torch.optim.SGD([p for p in student.parameters() if p.requires_grad], lr=lr)
        opt.zero_grad()
        for r in W:
            ids=r["completion_token_ids"]
            if len(ids)<8: continue
            sp,tp=build_prompts(tok,r["problem"],r["solution"])
            sp_ids=tok(sp,return_tensors="pt").input_ids[0].tolist(); tp_ids=tok(tp,return_tensors="pt").input_ids[0].tolist()
            s_in=torch.tensor([sp_ids+ids],device=dev); t_in=torch.tensor([tp_ids+ids],device=dev)
            with torch.no_grad(), student.disable_adapter():
                t_logits=student(t_in).logits[:, len(tp_ids)-1:-1, :].float()
            s_logits=student(s_in).logits[:, len(sp_ids)-1:-1, :].float()
            T=min(t_logits.shape[1],s_logits.shape[1]); t_logits=t_logits[:,:T]; s_logits=s_logits[:,:T]
            labels=torch.tensor([ids[:T]],device=dev)
            jsd=OT.generalized_jsd_loss(student_logits=s_logits,teacher_logits=t_logits,labels=labels,beta=0.0,temperature=1.0,token_clip=0.05,reduction="none")
            per_tok=jsd.sum(-1)
            if gated:
                cps=auto_checkpoints(tok,ids,max_points=16)
                with torch.no_grad(), student.disable_adapter():
                    vt=answer_likelihood_probe(student,tok,r["problem"],ids,str(r["gt_answer"]),checkpoint_positions=cps)
                dv=np.zeros(len(ids)); pos=vt["checkpoint_positions"]; V=vt["V"]
                for j in range(1,len(pos)):
                    lo=min(pos[j-1],len(ids)); hi=min(pos[j],len(ids)); dv[lo:hi]=V[j]-V[j-1]
                wv=torch.tensor(1/(1+np.exp(-dv[:T]/tau)),device=dev,dtype=per_tok.dtype)
            else:
                wv=torch.ones_like(per_tok)
            loss=(per_tok*wv).sum()/wv.sum().clamp_min(1.0)
            loss.backward()
        opt.step()
        ib,ho=measure(student)
        res[f"{name}_in_batch_dV"]=ib-base_ib; res[f"{name}_heldout_dV"]=ho-base_ho
        # 恢复快照
        with torch.no_grad():
            for k,v in student.named_parameters():
                if k in snap: v.copy_(snap[k])
    return res

if __name__=="__main__":
    main()
