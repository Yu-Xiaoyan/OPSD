#!/usr/bin/env python3
"""从 base 模型生成一批 on-policy rollout(student prompt),verifier-v2 分桶,写 jsonl。
供 privilege_distance.py 打分用(step-0 对齐版:teacher=student=base,无 ckpt/无漂移,全规模通用)。
生成参数沿训练口径(temp1.1/top_p0.95/top_k20/TM-off),max_tokens 可控。
用法: python probes/gen_base_rollouts.py --model ~/models/Qwen3-8B --out probes/data/rollouts_base_8b.jsonl --n 120 --cap 1024
"""
import argparse, json, os, sys
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0,HERE); sys.path.insert(0,ROOT)
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("HF_DATASETS_OFFLINE","1")
from transformers import AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams
from scoring import build_student_prompt_text
from verify_answer import bucket_rollout_v2

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--cap", type=int, default=1024)
    ap.add_argument("--dataset", default="siyanzhao/Openthoughts_math_30k_opsd")
    a=ap.parse_args()
    tok=AutoTokenizer.from_pretrained(a.model)
    tr=load_dataset(a.dataset, split="train")
    N=min(a.n, len(tr))
    prompts=[build_student_prompt_text(tok, tr[i]["problem"], student_thinking=False) for i in range(N)]
    llm=LLM(model=a.model, max_model_len=8192, gpu_memory_utilization=0.9,
            trust_remote_code=True, enforce_eager=True)
    sp=SamplingParams(temperature=1.1, top_p=0.95, top_k=20, max_tokens=a.cap, seed=1234)
    outs=llm.generate(prompts, sp)
    from collections import Counter
    nb=Counter(); n=0
    with open(a.out,"w") as f:
        for i,o in enumerate(outs):
            comp=o.outputs[0]; text=comp.text; ids=list(comp.token_ids)
            gt=str(tr[i]["Answer"]); problem=tr[i]["problem"]
            b,_=bucket_rollout_v2(text, gt, problem); nb[b]+=1
            f.write(json.dumps({"problem_id":i,"problem":problem,"completion_token_ids":ids,
                                "gt_answer":gt,"bucket":b}, ensure_ascii=False)+"\n"); n+=1
    print(f"[gen_base] {n} rollouts -> {a.out}  buckets={dict(nb)}")

if __name__=="__main__": main()
