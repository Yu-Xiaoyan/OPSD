"""2x2 corruption diagnostic scan (GPU) — framework.md axis-1 corruption probe.

For each rollout: forward the true-solution teacher (T_S), 3 corrupted-solution
teachers (T_S̃), the student (S), and — for wrong rollouts — a teacher corrupted
with the student's OWN wrong answer (right-bottom cell). Dumps per-token DERIVED
quantities to jsonl (NEVER [T,V] logits, CLAUDE.md §4):
  jsd_corruption[3] = JSD(T_S, T_S̃_k), jsd_corruption_studentwrong (wrong only),
  jsd_teacher_student = JSD(T_S, S), clipped_kl (training view), token category,
  answer-span mask, relative position.

correct/wrong from the 4096 collection (clean buckets); truncated handled in a
separate pass (T_S vs S + V(t), no corruption).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from datasets import load_dataset  # noqa: E402
from peft import PeftModel  # noqa: E402

from scoring import (score_with_privilege, teacher_mode,  # noqa: E402
                     build_student_prompt_text, forward_rollout_logits)
from divergence import token_jsd, clipped_kl_view, token_classifier  # noqa: E402
from corrupt_solution import corrupt_solution  # noqa: E402
from answer_likelihood import answer_likelihood_probe  # noqa: E402

BASE = os.path.expanduser("~/models/Qwen3-1.7B")
CKPT = os.path.expanduser("~/opsd_outputs/qwen31b_repro_3xh200_gb30/checkpoint-50")
DATA = os.path.join(_HERE, "data")
DATASET = "siyanzhao/Openthoughts_math_30k_opsd"
ROLLOUT_CAP = 1024


def answer_span_mask(tokenizer, rollout_ids):
    """Bool[T]: tokens whose chars fall inside a \\boxed{...} of the completion."""
    T = len(rollout_ids)
    text = tokenizer.decode(rollout_ids)
    spans = []
    for m in re.finditer(r"\\boxed\{", text):
        i = m.end() - 1
        depth = 0
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    spans.append((m.start(), i + 1))
                    break
            i += 1
    if not spans:
        return [False] * T
    ends = [len(tokenizer.decode(rollout_ids[:i])) for i in range(1, T + 1)]
    starts = [0] + ends[:-1]
    mask = []
    for t in range(T):
        c0, c1 = starts[t], ends[t]
        mask.append(any(not (c1 <= s or c0 >= e) for s, e in spans))
    return mask


def _teacher_logits(model, tok, problem, rollout, solution):
    with teacher_mode(model):
        return score_with_privilege(model, tok, problem, rollout, solution,
                                    teacher_thinking=True)["logits"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--base", type=str, default=BASE)
    ap.add_argument("--ckpt", type=str, default=CKPT)
    ap.add_argument("--input", type=str, default="rollouts_ckpt50_max4096.jsonl",
                    help="input rollouts jsonl (in probes/data/)")
    ap.add_argument("--out_tag", type=str, default="",
                    help="output tag: diag2x2{tag}_shard{n}.jsonl")
    ap.add_argument("--skip_truncated", action="store_true",
                    help="skip the truncated-bucket pass (8B corruption-null run)")
    args = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.base, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2").cuda().eval()
    model = PeftModel.from_pretrained(base, args.ckpt).eval()
    tr = load_dataset(DATASET)["train"]

    def solution_of(pid):
        return tr[pid]["solution"]

    # ---------- correct / wrong: full 2x2 (4096 collection) ----------
    recs = [json.loads(l) for l in open(os.path.join(DATA, args.input))]
    cw = [r for r in recs if r["bucket"] in ("correct", "wrong")]
    cw = [r for i, r in enumerate(cw) if i % args.nshards == args.shard]
    if args.limit:
        cw = cw[:args.limit]
    out_path = os.path.join(DATA, f"diag2x2{args.out_tag}_shard{args.shard}.jsonl")
    print(f"[2x2] {len(cw)} correct/wrong rollouts (shard {args.shard}/{args.nshards})")
    with open(out_path, "w", encoding="utf-8") as f:
        for j, r in enumerate(cw):
            pid = r["problem_id"]
            rollout = r["completion_token_ids"][:ROLLOUT_CAP]
            T = len(rollout)
            if T < 4:
                continue
            problem, gt, sol = r["problem"], r["gt_answer"], solution_of(pid)
            corrs = (r["corrupted_answers"] or [])[:3]
            T_S = _teacher_logits(model, tok, problem, rollout, sol)
            jsd_corr, nrep = [], []
            for c in corrs:
                csol, n = corrupt_solution(sol, gt, c)
                nrep.append(n)
                Tk = _teacher_logits(model, tok, problem, rollout, csol)
                jsd_corr.append(token_jsd(T_S, Tk).tolist())
                del Tk
            jsd_sw = None
            if r["bucket"] == "wrong" and r.get("student_answer"):
                csol, _ = corrupt_solution(sol, gt, r["student_answer"])
                Tw = _teacher_logits(model, tok, problem, rollout, csol)
                jsd_sw = token_jsd(T_S, Tw).tolist()
                del Tw
            sp = build_student_prompt_text(tok, problem, student_thinking=False)
            S, _, _ = forward_rollout_logits(model, tok, sp, rollout)       # adapter ON
            with teacher_mode(model):                                        # base, student prompt
                S0, _, _ = forward_rollout_logits(model, tok, sp, rollout)
            jsd_drift = token_jsd(S, S0).tolist()   # LoRA drift (S vs S0), for gate-A cross-tab
            del S0
            jsd_ts = token_jsd(T_S, S).tolist()
            clip = clipped_kl_view(T_S, S, clip=0.05).tolist()
            del T_S, S
            toks = tok.convert_ids_to_tokens(rollout)
            rec = {
                "problem_id": pid, "bucket": r["bucket"], "T": T,
                "tokens": toks, "categories": token_classifier(toks),
                "answer_span": answer_span_mask(tok, rollout),
                "rel_pos": [t / T for t in range(T)],
                "jsd_corruption": jsd_corr,
                "jsd_corruption_studentwrong": jsd_sw,
                "jsd_teacher_student": jsd_ts,
                "jsd_drift": jsd_drift,
                "clipped_kl": clip,
                "n_replacements": nrep,
                "gt_answer": gt, "student_answer": r.get("student_answer"),
                "corrupted_answers": corrs,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (j + 1) % 20 == 0:
                print(f"  ...{j+1}/{len(cw)}")
                torch.cuda.empty_cache()
    print(f"[2x2] wrote {out_path}")

    # ---------- truncated: T_S vs S + V(t) (1024 collection) ----------
    if args.shard == 0 and not args.skip_truncated:
        recs1 = [json.loads(l) for l in open(os.path.join(DATA, "rollouts_ckpt50_max1024.jsonl"))]
        trunc = [r for r in recs1 if r["bucket"] == "truncated"]
        if args.limit:
            trunc = trunc[:args.limit]
        tpath = os.path.join(DATA, "diag_truncated.jsonl")
        print(f"[trunc] {len(trunc)} truncated rollouts")
        with open(tpath, "w", encoding="utf-8") as f:
            for j, r in enumerate(trunc):
                pid = r["problem_id"]
                rollout = r["completion_token_ids"][:ROLLOUT_CAP]
                T = len(rollout)
                if T < 4:
                    continue
                problem, gt, sol = r["problem"], r["gt_answer"], solution_of(pid)
                T_S = _teacher_logits(model, tok, problem, rollout, sol)
                sp = build_student_prompt_text(tok, problem, student_thinking=False)
                S, _, _ = forward_rollout_logits(model, tok, sp, rollout)
                jsd_ts = token_jsd(T_S, S).tolist()
                clip = clipped_kl_view(T_S, S, clip=0.05).tolist()
                del T_S, S
                vt = answer_likelihood_probe(model, tok, problem, rollout, gt,
                                             bridge_variant=0)
                toks = tok.convert_ids_to_tokens(rollout)
                rec = {
                    "problem_id": pid, "T": T, "tokens": toks,
                    "categories": token_classifier(toks),
                    "answer_span": answer_span_mask(tok, rollout),
                    "rel_pos": [t / T for t in range(T)],
                    "jsd_teacher_student": jsd_ts, "clipped_kl": clip,
                    "V_positions": vt["checkpoint_positions"], "V_values": vt["V"],
                    "gt_answer": gt,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if (j + 1) % 20 == 0:
                    torch.cuda.empty_cache()
        print(f"[trunc] wrote {tpath}")


if __name__ == "__main__":
    main()
