#!/usr/bin/env python3
"""裁决三 (c): truncated 桶 V(t) 截断点读数 区分"后续正轨/歪轨" AUC。
预注册替代操作化见 docs/triage_rationale_recon.md：续写 138 条(cap+4096)+verifier-v2+AUC。

- 前缀 = rollout["prompt"]+completion_text（truncated 到 1024 的原文）；ckpt150 续写(锁定协议)。
- label: 续写后 bucket_rollout_v2 判 correct→1(正轨); wrong 或 仍 truncated→0(歪轨)。
- 读数 = 截断点 V(t) = diag_truncated 对应行 V_values[-1]（按 problem_id 有序配对）。
- AUC(读数, label)，判据 ≥0.65。存储纪律 §4：只落 AUC/标量。
用法: python probes/triage_truncated_vt.py --ckpt ~/opsd_outputs/qwen31b_repro_3xh200_gb30/checkpoint-150
"""
import argparse, json, os, sys
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0,HERE); sys.path.insert(0,ROOT)
from verify_answer import bucket_rollout_v2

def auc_mannwhitney(scores, labels):
    pos=[s for s,l in zip(scores,labels) if l==1]
    neg=[s for s,l in zip(scores,labels) if l==0]
    if not pos or not neg: return None
    c=0.0
    for p in pos:
        for n in neg:
            c += 1.0 if p>n else (0.5 if p==n else 0.0)
    return c/(len(pos)*len(neg))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.expanduser("~/models/Qwen3-1.7B"))
    ap.add_argument("--ckpt", default=os.path.expanduser("~/opsd_outputs/qwen31b_repro_3xh200_gb30/checkpoint-150"))
    ap.add_argument("--src", default=os.path.join(HERE,"data","rollouts_ckpt50_max1024.jsonl"))
    ap.add_argument("--vt", default=os.path.join(HERE,"data","diag_truncated.jsonl"))
    ap.add_argument("--cap", type=int, default=4096)
    ap.add_argument("--out", default=os.path.join(HERE,"analysis","triage_c_truncated.json"))
    a=ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("HF_DATASETS_OFFLINE","1")

    trunc=[json.loads(l) for l in open(a.src) if json.loads(l).get("bucket")=="truncated"]
    vt=[json.loads(l) for l in open(a.vt)]
    assert len(trunc)==len(vt), f"count mismatch {len(trunc)} vs {len(vt)}"
    # 有序配对核验（problem_id）
    n_match=sum(1 for t,v in zip(trunc,vt) if t["problem_id"]==v["problem_id"])
    assert n_match==len(trunc), f"pid order mismatch {n_match}/{len(trunc)}"
    Vtrunc=[float(v["V_values"][-1]) for v in vt]

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm=LLM(model=a.base, enable_lora=True, max_lora_rank=64, max_loras=1, max_cpu_loras=1,
            max_model_len=40960, gpu_memory_utilization=0.9, trust_remote_code=True,
            enforce_eager=True, distributed_executor_backend="mp")
    sp=SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0,
                      presence_penalty=0.0, max_tokens=a.cap)
    lora=LoRARequest("ckpt150", 1, a.ckpt)
    prompts=[t["prompt"]+t["completion_text"] for t in trunc]
    print(f"[c] 续写 {len(prompts)} truncated rollouts (cap +{a.cap})", flush=True)
    outs=llm.generate(prompts, sp, lora_request=lora)

    labels=[]; nbuck={"correct":0,"wrong":0,"truncated":0}
    for t,o in zip(trunc,outs):
        cont=o.outputs[0].text
        full=t["completion_text"]+cont
        b,_=bucket_rollout_v2(full, t["gt_answer"], t.get("problem",""))
        nbuck[b]+=1
        labels.append(1 if b=="correct" else 0)   # 正轨=1; wrong/still-truncated=0
    auc=auc_mannwhitney(Vtrunc, labels)
    npos=sum(labels); nneg=len(labels)-npos
    c_ok = (auc is not None and auc>=0.65)
    verdict = ("(c) 未定义: label 单一类 -> 不成立" if auc is None
               else f"(c) {'PASS' if c_ok else 'FAIL'}: AUC={auc:.3f} (>=0.65)")
    out={"n":len(labels),"n_ontrack":npos,"n_offtrack":nneg,"continue_buckets":nbuck,
         "auc":auc,"c_pass":bool(c_ok),"cap":a.cap,
         "Vtrunc_mean":sum(Vtrunc)/len(Vtrunc),"verdict":verdict,
         "note":"读数=截断点V(t)=V_values[-1]; label=续写后correct→1,wrong/仍truncated→0"}
    os.makedirs(os.path.dirname(a.out),exist_ok=True)
    json.dump(out,open(a.out,"w"),indent=2,ensure_ascii=False)
    print(json.dumps(out,indent=2,ensure_ascii=False)); print("[saved]",a.out)

if __name__=="__main__": main()
