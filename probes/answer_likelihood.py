"""Answer-likelihood probe V(t) for the OPSD probe pipeline (framework.md axis 2).

V(t) = log p(y* | x, s_1:t): how well the rollout prefix up to token t supports
the ground-truth answer y*. Scored under the STUDENT view (no privileged
solution in the prompt) — this measures the prefix's own support for the answer,
the outcome-alignment signal used for failure localization and DeltaV process
advantage (docs/framework.md, V(t) probe).

For each checkpoint t we build [student_prompt + rollout[:t] + bridge + y* + "}"]
and teacher-force only the y* tokens, summing their log-probs. All checkpoints
are batched into a single forward.

STORAGE DISCIPLINE (CLAUDE.md §4): in-memory only; no [T,V] logits persisted.
"""
from __future__ import annotations

import bisect
import re

import numpy as np
import torch

from scoring import build_student_prompt_text

# bridge-string variants: "... \boxed{" (answer + "}" appended after)
BRIDGES = [
    "\n\nTherefore, the final answer is \\boxed{",
    "\n\nThe final answer is \\boxed{",
    "\n\nSo the answer is \\boxed{",
]


def auto_checkpoints(tokenizer, rollout_ids, grid: int = 64,
                     max_points: int = 32) -> list[int]:
    """Reasoning-step boundaries: token positions after '\\n\\n' in the decoded
    completion, unioned with a `grid`-token fallback and the end T. Capped to
    `max_points` (uniform subsample) to bound the batch size."""
    T = len(rollout_ids)
    if T == 0:
        return [0]
    pts = {T, min(grid, T)}
    text = tokenizer.decode(rollout_ids)
    nn = [m.end() for m in re.finditer(r"\n\n", text)]
    if nn:
        # char end position of each token prefix (T decodes; T<=1024)
        ends = [len(tokenizer.decode(rollout_ids[:i])) for i in range(1, T + 1)]
        for cp in nn:
            t = bisect.bisect_left(ends, cp)
            if 1 <= t <= T:
                pts.add(t)
    for t in range(grid, T, grid):
        pts.add(t)
    pts = sorted(p for p in pts if 1 <= p <= T)
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).round().astype(int)
        pts = sorted({pts[i] for i in idx})
    return pts


def answer_likelihood_probe(model, tokenizer, problem, rollout_token_ids,
                            gt_answer, checkpoint_positions=None,
                            bridge_variant: int = 0) -> dict:
    """Compute V(t) over checkpoints for one rollout. See module docstring.

    Returns dict:
      checkpoint_positions [K], V [K] (sum log-prob of y*, nats, <=0),
      delta_V [K-1], answer_token_logprobs [K][A] (per-answer-token breakdown),
      bridge_variant, num_answer_tokens A, answer_ids.
    """
    device = next(model.parameters()).device
    rollout = list(rollout_token_ids)
    T = len(rollout)

    if checkpoint_positions is None:
        checkpoint_positions = auto_checkpoints(tokenizer, rollout)
    else:
        checkpoint_positions = sorted(
            {int(t) for t in checkpoint_positions if 0 <= int(t) <= T})

    student_text = build_student_prompt_text(tokenizer, problem,
                                             student_thinking=False)
    student_ids = tokenizer(student_text, return_tensors="pt").input_ids[0].tolist()
    bridge_ids = tokenizer(BRIDGES[bridge_variant],
                           add_special_tokens=False).input_ids
    answer_ids = tokenizer(str(gt_answer), add_special_tokens=False).input_ids
    close_ids = tokenizer("}", add_special_tokens=False).input_ids
    A = len(answer_ids)

    seqs, starts = [], []
    for t in checkpoint_positions:
        seq = student_ids + rollout[:t] + bridge_ids + answer_ids + close_ids
        seqs.append(seq)
        starts.append(len(student_ids) + t + len(bridge_ids))  # y* start index

    maxlen = max(len(s) for s in seqs)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        attn[i, :len(s)] = 1
    input_ids, attn = input_ids.to(device), attn.to(device)

    # 存储纪律 §4（内存版）：禁止整块 [ckpts × T × V] 的 float 物化。
    # 逐 checkpoint 流式：forward -> 仅在答案位置切片取 log-prob -> 立即释放 logits。
    # 每条序列按其真实长度裁掉 padding，峰值内存 = 单条 [1, L_i, V]（模型 dtype）。
    ans_t = torch.tensor(answer_ids, device=device)
    V, per_tok = [], []
    with torch.no_grad():
        for i, start in enumerate(starts):
            Li = len(seqs[i])
            lg = model(input_ids=input_ids[i:i + 1, :Li],
                       attention_mask=attn[i:i + 1, :Li]).logits
            lp = torch.log_softmax(lg[0, start - 1:start - 1 + A, :].float(), dim=-1)
            tok_lp = lp.gather(-1, ans_t.unsqueeze(-1)).squeeze(-1)  # [A]
            V.append(float(tok_lp.sum()))
            per_tok.append([float(x) for x in tok_lp])
            del lg, lp, tok_lp

    V = np.asarray(V)
    return {
        "checkpoint_positions": list(checkpoint_positions),
        "V": V.tolist(),
        "delta_V": np.diff(V).tolist(),
        "answer_token_logprobs": per_tok,
        "bridge_variant": bridge_variant,
        "num_answer_tokens": A,
        "answer_ids": answer_ids,
    }
