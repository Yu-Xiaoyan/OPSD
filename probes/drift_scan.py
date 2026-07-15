"""Task 2: six-checkpoint drift scan (GPU).

Tests whether the OPSD "peak-by-100-steps then plateau" is driven by LoRA drift
swamping the privileged teaching signal (effective target -> KL-to-init).

For each checkpoint k in {25,50,75,100,125,150}, on the fixed 4096 diagnostic
rollouts (correct+wrong, 187):
  S_k    = adapter-k on, student prompt
  S0     = base, student prompt          (k-independent, computed once)
  T_S    = base, teacher (privileged) prompt (k-independent, computed once)
  drift_k = JSD(S_k, S0), teach = JSD(T_S, S0), total_k = JSD(T_S, S_k)

Dumps per-token derived quantities only (no [T,V]). 3-way shardable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from datasets import load_dataset  # noqa: E402
from peft import PeftModel  # noqa: E402

from scoring import (build_student_prompt_text, build_teacher_prompt_text,  # noqa: E402
                     forward_rollout_logits)
from divergence import token_jsd, token_classifier  # noqa: E402

BASE = os.path.expanduser("~/models/Qwen3-1.7B")
CKPT_DIR = os.path.expanduser("~/opsd_outputs/qwen31b_repro_3xh200_gb30")
DATA = os.path.join(_HERE, "data")
DATASET = "siyanzhao/Openthoughts_math_30k_opsd"
STEPS = [25, 50, 75, 100, 125, 150]
ROLLOUT_CAP = 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ckpt_dir", type=str, default=CKPT_DIR,
                    help="run dir holding checkpoint-{25..150} (default: seed42 repro)")
    ap.add_argument("--out_tag", type=str, default="",
                    help="output: drift{out_tag}_shard{shard}.jsonl (e.g. _s1, _s2)")
    args = ap.parse_args()
    ckpt_dir = args.ckpt_dir

    tok = AutoTokenizer.from_pretrained(BASE, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2").cuda().eval()
    # load all six adapters, switch with set_adapter; base via disable_adapter()
    model = PeftModel.from_pretrained(
        base, os.path.join(ckpt_dir, f"checkpoint-{STEPS[0]}"),
        adapter_name=str(STEPS[0])).eval()
    for k in STEPS[1:]:
        model.load_adapter(os.path.join(ckpt_dir, f"checkpoint-{k}"), adapter_name=str(k))

    tr = load_dataset(DATASET)["train"]
    recs = [json.loads(l) for l in open(os.path.join(DATA, "rollouts_ckpt50_max4096.jsonl"))]
    cw = [r for r in recs if r["bucket"] in ("correct", "wrong")]
    cw = [r for i, r in enumerate(cw) if i % args.nshards == args.shard]
    if args.limit:
        cw = cw[:args.limit]
    out_path = os.path.join(DATA, f"drift{args.out_tag}_shard{args.shard}.jsonl")
    print(f"[drift] {len(cw)} rollouts x {len(STEPS)} ckpts (shard {args.shard}) "
          f"ckpt_dir={ckpt_dir} -> {out_path}")

    with open(out_path, "w", encoding="utf-8") as f:
        for j, r in enumerate(cw):
            pid = r["problem_id"]
            rollout = r["completion_token_ids"][:ROLLOUT_CAP]
            T = len(rollout)
            if T < 4:
                continue
            problem = r["problem"]
            solution = tr[pid]["solution"]
            sp = build_student_prompt_text(tok, problem, student_thinking=False)
            tp = build_teacher_prompt_text(tok, problem, solution, teacher_thinking=True)
            with model.disable_adapter():
                S0, _, _ = forward_rollout_logits(model, tok, sp, rollout)
                T_S, _, _ = forward_rollout_logits(model, tok, tp, rollout)
            teach = token_jsd(T_S, S0).tolist()
            per_k = {}
            for k in STEPS:
                model.set_adapter(str(k))
                S_k, _, _ = forward_rollout_logits(model, tok, sp, rollout)
                per_k[str(k)] = {"drift": token_jsd(S_k, S0).tolist(),
                                 "total": token_jsd(T_S, S_k).tolist()}
                del S_k
            del S0, T_S
            toks = tok.convert_ids_to_tokens(rollout)
            rec = {"problem_id": pid, "bucket": r["bucket"], "T": T,
                   "tokens": toks, "categories": token_classifier(toks),
                   "answer_span": r.get("answer_span"),   # may be absent
                   "teach": teach, "per_k": per_k}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (j + 1) % 20 == 0:
                print(f"  ...{j+1}/{len(cw)}")
                torch.cuda.empty_cache()
    print(f"[drift] wrote {out_path}")


if __name__ == "__main__":
    main()
