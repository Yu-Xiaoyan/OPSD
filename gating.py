"""v0 gating: trajectory triage + per-token weight synthesis (framework-frozen).

Implements the FROZEN framework v0 loss (docs/framework.md, "v0 loss 完整定义"):

  - triage: verifier v2 bucket (correct / wrong / truncated) per rollout.
  - wrong bucket:   unified ΔV soft weighting  w = σ(ΔV/τ)  — V(t) drop (ΔV<0)
    down-weights, pre-drop distills normally. NO unlikelihood (gate-B verdict).
  - correct bucket: gated light distillation — structural tokens (markup/step/
    section) down-weighted; corruption gate OFF (gate-D verdict, v0 ablation).
  - truncated bucket: split by V(end) batch median — healthy (high V(end)) uses
    the wrong-bucket pre-drop logic (σ(ΔV/τ)); degraded (low V(end)) down-weighted.

ΔV here is the REAL V(t)=log p(answer|prefix) segment change (teacher-forcing),
broadcast to tokens by the caller (v0_trainer). τ is set from the 3c gate_b.md
ΔV distribution (see GateConfig.tau provenance).

Verifier bucketing reuses probes/verify_answer.py::bucket_rollout_v2.
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from dataclasses import dataclass

import torch

_PROBES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probes")
if _PROBES not in sys.path:
    sys.path.insert(0, _PROBES)
from verify_answer import bucket_rollout_v2  # noqa: E402


@dataclass
class GateConfig:
    # ΔV soft-weight temperature. Provenance: 3c gate_b.md — clear/diffuse
    # most-negative ΔV distribution midpoint δ=-7.629 (gate_b_dv_dist.png); τ=|δ|.
    tau: float = 7.63
    w_correct: float = 0.5          # correct, non-structural: light distill
    w_correct_struct: float = 0.3   # correct, structural token (markup/step): downweight
    w_wrong_base: float = 1.0       # wrong bucket base
    w_trunc_healthy: float = 1.0    # truncated healthy segment (high V(end))
    w_trunc_degraded: float = 0.3   # truncated degraded segment (low V(end))
    lambda_unlik: float = 0.0       # v1 ablation only; MUST be 0 in v0 (see below)
    corruption_gate: bool = False   # gate-D verdict: corruption gate OFF in v0

    def __post_init__(self):
        if self.lambda_unlik != 0.0:
            raise NotImplementedError(
                "hard t* unlikelihood is a v1 ablation and is NOT implemented in "
                "v0 (gate-B verdict 2026-07-05: t* ±1 hit 0/10 -> wrong bucket = "
                "unified ΔV soft weighting). Keep lambda_unlik=0.")


def bucket_batch(sampled_token_ids, shifted_labels, gt_answers, tokenizer,
                 problems=None):
    """Verifier-bucket each rollout: list[str] in {correct,wrong,truncated}."""
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


def synth_weights(buckets, ex_ids, dv_per_tok, vend_per_ex, struct_mask, cfg):
    """Per-token distillation weights, flat masked-token order.

    buckets:      list[str] length B.
    ex_ids:       [n_valid] example index each valid token belongs to.
    dv_per_tok:   [n_valid] REAL per-token ΔV (V(t) segment change broadcast to
                  tokens); used for wrong + truncated-healthy. 0 where undefined.
    vend_per_ex:  [B] V(end) per example (NaN allowed); truncated split uses the
                  batch median over finite values.
    struct_mask:  [n_valid] bool, structural token (correct-bucket downweight).
    returns:      [n_valid] weights.
    """
    device = dv_per_tok.device
    W = torch.ones_like(dv_per_tok)
    # truncated healthy/degraded threshold = batch median of finite V(end)
    finite = vend_per_ex[torch.isfinite(vend_per_ex)]
    # quantile(0.5) = true median (torch.median returns the LOWER median for an
    # even count, which would misclassify the split boundary).
    vmed = float(finite.quantile(0.5)) if finite.numel() else 0.0

    for i, b in enumerate(buckets):
        sel = ex_ids == i
        if sel.sum() == 0:
            continue
        if b == "correct":
            w = torch.full_like(dv_per_tok[sel], cfg.w_correct)
            w[struct_mask[sel]] = cfg.w_correct_struct
            W[sel] = w
        elif b == "wrong":
            W[sel] = cfg.w_wrong_base * torch.sigmoid(dv_per_tok[sel] / cfg.tau)
        elif b == "truncated":
            healthy = torch.isfinite(vend_per_ex[i]) and float(vend_per_ex[i]) >= vmed
            if healthy:
                W[sel] = cfg.w_trunc_healthy * torch.sigmoid(dv_per_tok[sel] / cfg.tau)
            else:
                W[sel] = cfg.w_trunc_degraded
        else:  # unknown bucket -> neutral
            W[sel] = cfg.w_wrong_base
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
