"""v0 gating: trajectory triage + per-token weight synthesis.

Turns a batch of on-policy student rollouts into per-token distillation weights,
per the FROZEN framework v0 loss (docs/framework.md, "阶段 1 裁决与 v0 冻结"):

  - correct bucket   -> gate-A light distill (uniform w_correct)
  - wrong bucket     -> unified ΔV soft weighting  w ∝ σ(ΔV/τ)
                        (gate-B verdict: hard t* unlikelihood degraded to soft
                        weighting because ±1 hit was 0/10; residual v2-missed
                        pseudo is second-order protected by the soft weight)
  - truncated bucket -> V(t)-health split (scaffold: soft-weight like wrong)

Training-time ΔV is APPROXIMATED by a proxy (no extra V(t) forward per step):
the per-token teacher-student divergence, per-example z-scored. This is a
stand-in for the real V(t)=log p(answer|prefix) drop; wiring the real V(t)
teacher-forcing probe is a TODO (see answer_likelihood.py). The proxy keeps the
scaffold runnable and the plumbing (triage -> weight -> stats) exercised.

Verifier bucketing reuses probes/verify_answer.py::bucket_rollout_v2 (any-boxed +
option letter<->value mapping) — the same v2 the 3a/3d audit validated.
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from dataclasses import dataclass

import torch

# reuse the audited v2 verifier (probes/verify_answer.py)
_PROBES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probes")
if _PROBES not in sys.path:
    sys.path.insert(0, _PROBES)
from verify_answer import bucket_rollout_v2  # noqa: E402


@dataclass
class GateConfig:
    w_correct: float = 0.5        # gate-A light distill weight
    w_wrong_base: float = 1.0     # wrong-bucket base weight
    w_trunc: float = 0.7          # truncated-bucket base weight
    tau: float = 1.0              # softness of the ΔV sigmoid
    direction: float = -1.0       # -1: down-weight high-divergence (V-drop) tokens
    # v1 ablation ONLY; NOT implemented in v0 synth_weights. Gate-B verdict
    # (2026-07-05, gate_b.md): t* ±1 hit was 0/10 -> hard unlikelihood degraded
    # to unified ΔV soft weighting. Must stay 0 in v0; a nonzero value is a
    # misconfiguration and raises (see __post_init__) rather than being silently
    # ignored.
    lambda_unlik: float = 0.0

    def __post_init__(self):
        if self.lambda_unlik != 0.0:
            raise NotImplementedError(
                "hard t* unlikelihood is a v1 ablation and is NOT implemented in "
                "v0 (gate-B verdict 2026-07-05: t* ±1 hit 0/10 -> wrong bucket = "
                "unified ΔV soft weighting). Keep lambda_unlik=0; do not train a "
                "gate-B-retired branch.")


def bucket_batch(sampled_token_ids, shifted_labels, gt_answers, tokenizer,
                 problems=None):
    """Verifier-bucket each rollout in the batch: list[str] in
    {'correct','wrong','truncated'}. Decodes the non-masked generation region."""
    buckets = []
    B = sampled_token_ids.shape[0]
    for i in range(B):
        m = shifted_labels[i] != -100
        ids = sampled_token_ids[i][m].tolist()
        text = tokenizer.decode(ids, skip_special_tokens=True)
        gt = str(gt_answers[i]) if gt_answers is not None and i < len(gt_answers) else ""
        prob = problems[i] if problems is not None and i < len(problems) else ""
        b, _ = bucket_rollout_v2(text, gt, prob)
        buckets.append(b)
    return buckets


def synth_weights(per_tok, ex_ids, buckets, cfg):
    """Per-token distillation weights, aligned to the flat masked token order.

    per_tok:  [n_valid] detached per-token divergence (ΔV proxy).
    ex_ids:   [n_valid] example index each valid token belongs to.
    buckets:  list[str] length B.
    returns:  [n_valid] weights.
    """
    W = torch.empty_like(per_tok)
    for i, b in enumerate(buckets):
        sel = ex_ids == i
        if sel.sum() == 0:
            continue
        if b == "correct":
            W[sel] = cfg.w_correct
            continue
        base = cfg.w_wrong_base if b == "wrong" else cfg.w_trunc
        d = per_tok[sel]
        # ΔV proxy soft weight: z-score divergence within the rollout, sigmoid.
        # direction=-1 -> high divergence (proxy V-drop) tokens get LESS weight.
        z = (d - d.mean()) / (d.std() + 1e-6)
        W[sel] = base * torch.sigmoid(cfg.direction * z / cfg.tau)
    return W


def gating_stats(buckets, W):
    """WandB-loggable gating stats for one batch."""
    c = Counter(buckets)
    n = max(1, len(buckets))
    return {
        "gate/frac_correct": c.get("correct", 0) / n,
        "gate/frac_wrong": c.get("wrong", 0) / n,
        "gate/frac_truncated": c.get("truncated", 0) / n,
        "gate/weight_mean": float(W.mean()) if W.numel() else 0.0,
        "gate/weight_std": float(W.std()) if W.numel() > 1 else 0.0,
        "gate/n_examples": float(n),
    }
