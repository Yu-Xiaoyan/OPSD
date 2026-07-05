"""0.3 diagnostic rollout collection (vLLM) for the OPSD probe pipeline.

Samples one rollout per problem from checkpoint-50 with the EXACT training
generation params (temp 1.1, top_p 0.95, top_k 20, TM-off student prompt), at
two max_tokens (1024 = training length, 4096 = clean bucketing), buckets each by
the verifier (correct / wrong / truncated), attaches 3 corrupted answers and the
V(t) checkpoint positions, and writes jsonl to probes/data/.

STORAGE DISCIPLINE (CLAUDE.md §4): stores derived artifacts only — completion
token ids + text, buckets, corrupted answers, checkpoint positions. Never
[T,V] logits.

Run under pbs/collect_rollouts.pbs (vLLM, 1 GPU).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

# repo root (data_collator/opsd_trainer) + probes dir (local modules) on path
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from scoring import build_student_prompt_text
from answer_likelihood import auto_checkpoints
from verify_answer import bucket_rollout
from corrupt_answers import corrupt_answer, seed_from

BASE = os.path.expanduser("~/models/Qwen3-1.7B")
CKPT = os.path.expanduser("~/opsd_outputs/qwen31b_repro_3xh200_gb30/checkpoint-50")
DATASET = "siyanzhao/Openthoughts_math_30k_opsd"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

GEN = {"temperature": 1.1, "top_p": 0.95, "top_k": 20, "student_thinking": False,
       "seed": 1234}
MAX_TOKENS = [1024, 4096]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="number of problems")
    ap.add_argument("--max_tokens", type=int, nargs="+", default=MAX_TOKENS)
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    ap.add_argument("--base", type=str, default=BASE)
    ap.add_argument("--ckpt", type=str, default=CKPT)
    ap.add_argument("--tag", type=str, default="ckpt50",
                    help="output filename tag: rollouts_{tag}_max{M}.jsonl")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    tr = load_dataset(DATASET)["train"]
    n = min(args.n, len(tr))
    problems = [tr[i]["problem"] for i in range(n)]
    gts = [str(tr[i]["Answer"]) for i in range(n)]
    prompts = [build_student_prompt_text(tokenizer, p, student_thinking=False)
               for p in problems]

    llm = LLM(model=args.base, enable_lora=True, max_lora_rank=64, max_loras=1,
              max_cpu_loras=1, tensor_parallel_size=1, trust_remote_code=True,
              gpu_memory_utilization=args.gpu_mem, max_model_len=8192,
              enforce_eager=True)
    lora = LoRARequest(args.tag, 1, args.ckpt)

    for max_tokens in args.max_tokens:
        gen_params = {**GEN, "max_tokens": max_tokens}
        sp = SamplingParams(temperature=GEN["temperature"], top_p=GEN["top_p"],
                            top_k=GEN["top_k"], max_tokens=max_tokens, n=1,
                            seed=GEN["seed"])
        print(f"\n=== generating {n} rollouts @ max_tokens={max_tokens} ===")
        outputs = llm.generate(prompts, sp, lora_request=lora, use_tqdm=True)

        out_path = os.path.join(OUT_DIR, f"rollouts_{args.tag}_max{max_tokens}.jsonl")
        buckets = Counter()
        with open(out_path, "w", encoding="utf-8") as f:
            for i, out in enumerate(outputs):
                o = out.outputs[0]
                text = o.text
                token_ids = list(o.token_ids)
                bucket, student = bucket_rollout(text, gts[i])
                buckets[bucket] += 1
                avoid = [gts[i]] + ([student] if bucket == "wrong" and student else [])
                corrupted = corrupt_answer(
                    gts[i], n=3, avoid=avoid, seed=seed_from(i, gts[i]))
                cps = auto_checkpoints(tokenizer, token_ids) if token_ids else [0]
                rec = {
                    "problem_id": i,
                    "problem": problems[i],
                    "prompt": prompts[i],
                    "completion_text": text,
                    "completion_token_ids": token_ids,
                    "bucket": bucket,
                    "student_answer": student,
                    "gt_answer": gts[i],
                    "corrupted_answers": corrupted,
                    "checkpoint_positions": cps,
                    "gen_params": gen_params,
                    "source_checkpoint": args.ckpt,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        total = sum(buckets.values())
        print(f"[{max_tokens}] buckets: " + ", ".join(
            f"{k}={v} ({100*v/total:.1f}%)" for k, v in buckets.most_common()))
        print(f"[{max_tokens}] wrote {total} -> {out_path}")

        # tiny summary sidecar (git-committable, no bulk data)
        summ = {"max_tokens": max_tokens, "n": total, "buckets": dict(buckets),
                "gen_params": gen_params, "source_checkpoint": args.ckpt}
        with open(os.path.join(OUT_DIR, f"summary_{args.tag}_max{max_tokens}.json"), "w") as f:
            json.dump(summ, f, indent=2)


if __name__ == "__main__":
    main()
