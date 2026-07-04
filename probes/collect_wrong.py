"""Task 3a: extra wrong-bucket collection (vLLM) on unused training problems.

Samples one 4096-token rollout per problem from checkpoint-50 (exact training
gen params) on dataset problems starting at START_ID (>= the 0..199 already used
by collect_rollouts), keeps only wrong-with-boxed rollouts until TARGET are
gathered. Records the total sampled (for the difficulty/hit-rate report).
Schema matches rollouts_ckpt50_max*.jsonl (+ corrupted answers).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from transformers import AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scoring import build_student_prompt_text  # noqa: E402
from answer_likelihood import auto_checkpoints  # noqa: E402
from verify_answer import bucket_rollout  # noqa: E402
from corrupt_answers import corrupt_answer, seed_from  # noqa: E402

BASE = os.path.expanduser("~/models/Qwen3-1.7B")
CKPT = os.path.expanduser("~/opsd_outputs/qwen31b_repro_3xh200_gb30/checkpoint-50")
DATASET = "siyanzhao/Openthoughts_math_30k_opsd"
OUT = os.path.join(_HERE, "data", "wrong_extra.jsonl")
GEN = {"temperature": 1.1, "top_p": 0.95, "top_k": 20, "max_tokens": 4096,
       "student_thinking": False, "seed": 1234}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_id", type=int, default=200)
    ap.add_argument("--n_probe", type=int, default=700)
    ap.add_argument("--target", type=int, default=150)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    tr = load_dataset(DATASET)["train"]
    ids = list(range(args.start_id, min(args.start_id + args.n_probe, len(tr))))
    prompts = [build_student_prompt_text(tok, tr[i]["problem"], student_thinking=False)
               for i in ids]

    llm = LLM(model=BASE, enable_lora=True, max_lora_rank=64, max_loras=1,
              max_cpu_loras=1, tensor_parallel_size=1, trust_remote_code=True,
              gpu_memory_utilization=0.90, max_model_len=8192, enforce_eager=True)
    sp = SamplingParams(temperature=GEN["temperature"], top_p=GEN["top_p"],
                        top_k=GEN["top_k"], max_tokens=GEN["max_tokens"], n=1,
                        seed=GEN["seed"])
    print(f"sampling {len(ids)} problems (id {ids[0]}..{ids[-1]}) @4096 ...")
    outs = llm.generate(prompts, sp, lora_request=LoRARequest("ckpt50", 1, CKPT),
                        use_tqdm=True)

    kept = 0
    total = 0
    buckets = {"correct": 0, "wrong": 0, "truncated": 0}
    with open(OUT, "w", encoding="utf-8") as f:
        for i, out in zip(ids, outs):
            total += 1
            o = out.outputs[0]
            text, token_ids = o.text, list(o.token_ids)
            gt = str(tr[i]["Answer"])
            bucket, student = bucket_rollout(text, gt)
            buckets[bucket] += 1
            if bucket != "wrong" or kept >= args.target:
                continue
            avoid = [gt] + ([student] if student else [])
            corrupted = corrupt_answer(gt, n=3, avoid=avoid, seed=seed_from(i, gt))
            rec = {"problem_id": i, "problem": tr[i]["problem"],
                   "prompt": prompts[ids.index(i)], "completion_text": text,
                   "completion_token_ids": token_ids, "bucket": "wrong",
                   "student_answer": student, "gt_answer": gt,
                   "corrupted_answers": corrupted,
                   "checkpoint_positions": auto_checkpoints(tok, token_ids) if token_ids else [0],
                   "gen_params": GEN, "source_checkpoint": CKPT}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"\nkept {kept} wrong-with-boxed from {total} sampled")
    print(f"bucket distribution over sampled: {buckets} "
          f"(wrong rate {100*buckets['wrong']/total:.1f}%)")
    print(f"wrote {OUT}")
    summ = {"kept_wrong": kept, "total_sampled": total, "buckets": buckets,
            "start_id": args.start_id, "gen_params": GEN}
    with open(os.path.join(_HERE, "data", "wrong_extra_summary.json"), "w") as fp:
        json.dump(summ, fp, indent=2)


if __name__ == "__main__":
    main()
