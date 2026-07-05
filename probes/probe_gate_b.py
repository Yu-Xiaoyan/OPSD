"""Task 3c: gate-B final review (GPU) — V(t) hit table + ΔV distributions +
V(t) format-noise robustness. Descriptive (no threshold verdict).

Uses 1.7B + ckpt-50 (the model the annotated wrong_extra was sampled from).

Outputs probes/analysis/gate_b.md + figures.
"""
from __future__ import annotations

import bisect
import json
import os
import re
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import PeftModel  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from answer_likelihood import answer_likelihood_probe, auto_checkpoints  # noqa: E402

BASE = os.path.expanduser("~/models/Qwen3-1.7B")
CKPT = os.path.expanduser("~/opsd_outputs/qwen31b_repro_3xh200_gb30/checkpoint-50")
DATA = os.path.join(_HERE, "data")
ANALYSIS = os.path.join(_HERE, "analysis")
# full rollout: human t* was annotated on the FULL completion (median ~75 steps),
# so V(t) must cover it. max_points caps V(t) batch memory ([K, seq, vocab]).
ROLLOUT_CAP = 4096
MAX_CP = 24
# borderline verdict flips for the "version B" re-classification
FLIP = {440: "true_wrong_clear_tstar", 477: "pseudo_wrong_format",
        297: "vacuous_proof", 424: "pseudo_wrong_format"}


def step_start_tokens(tok, rollout_ids):
    """Token index at the start of each \\n\\n-delimited step."""
    text = tok.decode(rollout_ids)
    ends = [len(tok.decode(rollout_ids[:i])) for i in range(1, len(rollout_ids) + 1)]
    starts_char = [0] + [m.end() for m in re.finditer(r"\n\s*\n", text)]
    return [bisect.bisect_left(ends, sc) for sc in starts_char]


def most_negative_dv_start(out):
    """Token position at the start of the most-negative ΔV checkpoint segment."""
    pos = out["checkpoint_positions"]
    dv = out["delta_V"]
    if not dv:
        return pos[-1], 0.0
    k = int(np.argmin(dv))               # segment pos[k] -> pos[k+1]
    return pos[k], float(dv[k])


def _auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return float(wins) / (len(pos) * len(neg))


def main():
    os.makedirs(ANALYSIS, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(BASE, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2").cuda().eval()
    model = PeftModel.from_pretrained(base, CKPT).eval()

    ann = {r["pid"]: r for r in (json.loads(l) for l in
           open(os.path.join(DATA, "tstar_annotations.jsonl")))}
    extra = {r["problem_id"]: r for r in (json.loads(l) for l in
             open(os.path.join(DATA, "wrong_extra.jsonl")))}
    # correct reference: 30 from the 1.7B 4096 collection (v2 not needed; boxed=correct)
    corr = [r for r in (json.loads(l) for l in
            open(os.path.join(DATA, "rollouts_ckpt50_max4096.jsonl")))
            if r["bucket"] == "correct"]
    rng = np.random.RandomState(0)
    corr = [corr[i] for i in rng.choice(len(corr), size=min(30, len(corr)), replace=False)]

    def vend(rec):
        rollout = rec["completion_token_ids"][:ROLLOUT_CAP]
        if len(rollout) < 4:
            return None
        cps = auto_checkpoints(tok, rollout, max_points=MAX_CP)
        out = answer_likelihood_probe(model, tok, rec["problem"], rollout,
                                      rec["gt_answer"], checkpoint_positions=cps,
                                      bridge_variant=0)
        return out

    # ---- V(t) per annotated wrong rollout ----
    cache = {}
    for pid, r in ann.items():
        e = extra[pid]
        out = vend(e)
        cache[pid] = out

    # ---- clear hit table (both versions) ----
    def clear_set(flip):
        s = []
        for pid, r in ann.items():
            v = FLIP[pid] if (flip and pid in FLIP) else r["verdict"]
            if v == "true_wrong_clear_tstar" and str(r["tstar"]).isdigit():
                s.append(pid)
        return s

    def hit_rows(pids):
        rows = []
        for pid in pids:
            out = cache[pid]
            e = extra[pid]
            tstar = int(ann[pid]["tstar"])
            rollout = e["completion_token_ids"][:ROLLOUT_CAP]
            sst = step_start_tokens(tok, rollout)
            dv_tok, dv_val = most_negative_dv_start(out)
            step_vt = bisect.bisect_right(sst, dv_tok) - 1
            hit = abs(step_vt - tstar) <= 1
            rows.append((pid, tstar, step_vt, dv_tok, round(dv_val, 3), hit))
        return rows

    L = []
    A = L.append
    A("# Gate B — V(t) final review (task 3c)\n")
    A("Descriptive evidence only (no threshold verdict). Model: 1.7B + ckpt-50. "
      "t* = human first-error step; V(t) most-negative-ΔV segment mapped back to "
      "a \\n\\n step; hit = |step_V - t*| <= 1.\n")

    for flip, name in [(False, "A (original verdicts)"), (True, "B (borderline reclassified)")]:
        pids = clear_set(flip)
        rows = hit_rows(pids)
        nhit = sum(r[5] for r in rows)
        A(f"## Clear-set hit table — version {name}  (n={len(rows)})\n")
        A("| pid | human t* | V(t) step | ΔV-min tok | ΔV | hit(±1) |")
        A("|--:|--:|--:|--:|--:|:--:|")
        for pid, ts, sv, dt, dvv, h in rows:
            A(f"| {pid} | {ts} | {sv} | {dt} | {dvv} | {'✓' if h else '✗'} |")
        A(f"\n**hits: {nhit}/{len(rows)} (±1 step)**")
        miss = [r for r in rows if not r[5]]
        if miss:
            A("Miss modes:")
            for pid, ts, sv, dt, dvv, h in miss:
                A(f"- pid={pid}: V(t) drop at step {sv} vs t*={ts} — "
                  f"{'V drop later (post-error collapse)' if sv>ts else 'V drop earlier than annotated error'}.")
        A("")

    # ---- clear vs diffuse most-negative ΔV distribution (proxy threshold δ) ----
    def dv_group(verdict_name, flip=False):
        vals = []
        for pid, r in ann.items():
            v = FLIP[pid] if (flip and pid in FLIP) else r["verdict"]
            if v == verdict_name and cache[pid] is not None:
                vals.append(most_negative_dv_start(cache[pid])[1])
        return np.array(vals)

    clear_dv = dv_group("true_wrong_clear_tstar")
    diff_dv = dv_group("true_wrong_diffuse")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(clear_dv, bins=12, alpha=0.6, label=f"clear (n={len(clear_dv)})", color="#d62728")
    ax.hist(diff_dv, bins=12, alpha=0.6, label=f"diffuse (n={len(diff_dv)})", color="#1f77b4")
    delta = None
    if len(clear_dv) and len(diff_dv):
        delta = float((np.median(clear_dv) + np.median(diff_dv)) / 2)
        ax.axvline(delta, ls="--", color="k", label=f"proxy δ={delta:.2f}")
    ax.set_xlabel("most-negative segment ΔV (nats)")
    ax.set_title("Gate B: clear vs diffuse most-negative ΔV")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS, "gate_b_dv_dist.png"), dpi=120)
    plt.close(fig)
    A("## Proxy threshold δ (training-time clear/diffuse split)\n")
    A(f"- clear most-neg ΔV: median={np.median(clear_dv):.3f} (n={len(clear_dv)})")
    A(f"- diffuse most-neg ΔV: median={np.median(diff_dv):.3f} (n={len(diff_dv)})")
    A(f"- **proxy δ = {delta:.3f}** (midpoint of medians); see gate_b_dv_dist.png. "
      "Training-time rule: most-neg segment ΔV < δ -> treat as clear (t* hard "
      "mechanism), else diffuse (ΔV soft weighting).\n")

    # ---- format robustness: pseudo vs true_wrong vs correct V(end) ----
    def vend_group(verdict_names, flip=False):
        vals = []
        for pid, r in ann.items():
            v = FLIP[pid] if (flip and pid in FLIP) else r["verdict"]
            if v in verdict_names and cache[pid] is not None:
                vals.append(cache[pid]["V"][-1])
        return np.array(vals)

    for flip, name in [(False, "A"), (True, "B")]:
        pseudo = vend_group({"pseudo_wrong_format"}, flip)
        twrong = vend_group({"true_wrong_clear_tstar", "true_wrong_diffuse"}, flip)
        cvend = np.array([o["V"][-1] for o in
                          (vend(c) for c in corr) if o is not None])
        auc_pt = _auc(pseudo, twrong)
        auc_ct = _auc(cvend, twrong)
        if flip:
            continue  # correct group same; report once but AUCs for both
        A("## Format-noise robustness (V(end))\n")
        A(f"- pseudo_wrong_format (human=correct): mean={pseudo.mean():.2f} n={len(pseudo)}")
        A(f"- true_wrong (clear+diffuse): mean={twrong.mean():.2f} n={len(twrong)}")
        A(f"- correct (reference, n={len(cvend)}): mean={cvend.mean():.2f}")
        A(f"- **AUC(pseudo vs true_wrong) = {auc_pt:.3f}** (>=0.75 => V(t) robust to "
          f"reward format noise)")
        A(f"- **AUC(correct vs true_wrong) = {auc_ct:.3f}** (clean-label discriminability; "
          f"cf. prior 0.812)")
        # figure
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for arr, lab, col in [(cvend, "correct", "#2ca02c"),
                              (pseudo, "pseudo (fmt-misjudged)", "#ff7f0e"),
                              (twrong, "true_wrong", "#1f77b4")]:
            ax.hist(arr, bins=12, alpha=0.55, label=f"{lab} (n={len(arr)})", color=col)
        ax.set_xlabel("V(end) = log p(answer | full rollout)  (nats)")
        ax.set_title("Gate B: V(end) by audited verdict")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(ANALYSIS, "gate_b_robustness.png"), dpi=120)
        plt.close(fig)
        # version B AUCs (borderline)
        pseudo_b = vend_group({"pseudo_wrong_format"}, True)
        twrong_b = vend_group({"true_wrong_clear_tstar", "true_wrong_diffuse"}, True)
        A(f"- version B (borderline): AUC(pseudo vs true_wrong)="
          f"{_auc(pseudo_b, twrong_b):.3f}, AUC(correct vs true_wrong)="
          f"{_auc(cvend, twrong_b):.3f}\n")

    with open(os.path.join(ANALYSIS, "gate_b.md"), "w") as f:
        f.write("\n".join(L))
    print("wrote gate_b.md")
    print("clear hits A:", sum(r[5] for r in hit_rows(clear_set(False))),
          "/", len(clear_set(False)))
    print("proxy delta:", delta)


if __name__ == "__main__":
    main()
