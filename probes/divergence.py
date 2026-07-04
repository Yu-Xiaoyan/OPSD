"""Token-level divergences for the OPSD probe pipeline (stage 2).

Numerically stable, forward-only, all in memory. Inputs are `[T, V]` fp32
logits (one row per rollout token); outputs are `[T]` per-token divergences.

STORAGE DISCIPLINE (CLAUDE.md §4): these functions operate purely on in-memory
tensors and never write anything. Do NOT persist `[T, V]` logits anywhere —
re-run the forward pass if you need the full distribution again. Probe analysis
keeps only the derived `[T]` sequences produced here.

Default analysis uses the TRUE (un-clipped) divergences `token_jsd` / `token_kl`.
`clipped_kl_view` reproduces the trainer's clipped loss surface and exists only
for "training-view" comparison, not for measuring real divergence.

Convention (matches opsd_trainer.py:441-473, GKD `generalized_jsd_loss`):
`p_logits` is the reference/teacher, `q_logits` is the student. Forward KL is
`KL(p || q) = Σ p·(log p − log q)` (the beta=0 branch of the trainer). All
values are in nats; JSD is bounded above by log 2 ≈ 0.693 nats.
"""
from __future__ import annotations

import math
import os
import re

import torch
import torch.nn.functional as F

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _log_probs(logits: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(torch.as_tensor(logits, dtype=torch.float32), dim=-1)


def token_jsd(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """Per-token Jensen-Shannon divergence, `[T, V] x [T, V] -> [T]` (nats).

    JSD(p, q) = ½·KL(p‖m) + ½·KL(q‖m),  m = ½(p + q).

    The mixture is formed in log-space via logsumexp for stability:
    log m = logsumexp([log p, log q]) + log(½). Symmetric; in [0, log 2].
    """
    lp = _log_probs(p_logits)
    lq = _log_probs(q_logits)
    lm = torch.logsumexp(torch.stack([lp, lq], dim=0), dim=0) + math.log(0.5)
    kl_pm = (lp.exp() * (lp - lm)).sum(dim=-1)
    kl_qm = (lq.exp() * (lq - lm)).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def token_kl(p_logits: torch.Tensor, q_logits: torch.Tensor,
             direction: str = "forward") -> torch.Tensor:
    """Per-token KL divergence, `[T, V] x [T, V] -> [T]` (nats).

    direction="forward" -> KL(p‖q) = Σ p·(log p − log q)  (trainer beta=0)
    direction="reverse" -> KL(q‖p) = Σ q·(log q − log p)  (trainer beta=1)
    """
    lp = _log_probs(p_logits)
    lq = _log_probs(q_logits)
    if direction == "forward":
        return (lp.exp() * (lp - lq)).sum(dim=-1)
    if direction == "reverse":
        return (lq.exp() * (lq - lp)).sum(dim=-1)
    raise ValueError(f"direction must be 'forward' or 'reverse', got {direction!r}")


def clipped_kl_view(p_logits: torch.Tensor, q_logits: torch.Tensor,
                    clip: float = 0.05) -> torch.Tensor:
    """The trainer's "post-surgery" forward KL: `[T, V] x [T, V] -> [T]`.

    Reproduces the loss surface of opsd_trainer.py's beta=0 branch with
    `jsd_token_clip`: the per-**element** (per token × per vocab) forward-KL
    contribution `p·(log p − log q)` is `clamp(max=clip)`-ed BEFORE the vocab
    sum (element-wise granularity — see docs/archaeology.md §5). This is NOT a
    true divergence (element clamping breaks the KL identity) and is provided
    only for training-view comparison. For real analysis use `token_kl` /
    `token_jsd`.

    With `clip=inf` (or None) this reduces exactly to `token_kl(..., "forward")`.
    """
    lp = _log_probs(p_logits)
    lq = _log_probs(q_logits)
    elem = lp.exp() * (lp - lq)          # [T, V] forward-KL contribution
    if clip is not None:
        elem = elem.clamp(max=clip)
    return elem.sum(dim=-1)


# ---------------------------------------------------------------------------
# Token classification (math / style / other)
# ---------------------------------------------------------------------------

_DEFAULT_CATEGORIES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "token_categories.yaml")

# leading BPE/sentencepiece space markers to strip before word matching
_SPACE_MARKERS = " \t▁Ġ"          # ' ', tab, '▁', 'Ġ'
# math signal: a digit, a \latex-command, or a bare math operator/bracket
_MATH_RE = re.compile(r"[0-9]|\\[a-zA-Z]+|[=+\-*/^_<>(){}\[\]|]")


def load_token_categories(path: str | None = None) -> dict:
    """Load the style/math keyword config; falls back to a built-in style set."""
    path = path or _DEFAULT_CATEGORIES_PATH
    if yaml is not None and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:  # minimal fallback if pyyaml/file is unavailable
        cfg = {"style_keywords": ["wait", "hmm", "so", "let", "think",
                                  "okay", "well", "therefore", "thus"],
               "math_extra": ["boxed"]}
    return {
        "style": {str(w).lower() for w in cfg.get("style_keywords", [])},
        "math_extra": {str(w).lower() for w in cfg.get("math_extra", [])},
    }


def _classify_one(tok: str, style: set, math_extra: set) -> str:
    s = tok.lstrip(_SPACE_MARKERS)
    word = re.sub(r"[^a-z]", "", s.lower())   # letters-only core for word match
    if word and word in style:
        return "style"
    if word and word in math_extra:
        return "math"
    if _MATH_RE.search(s):
        return "math"
    return "other"


def token_classifier(token_strs, categories: dict | None = None) -> list[str]:
    """Coarse per-token category: 'math' | 'style' | 'other'.

    - math : digits, operators, LaTeX commands (`\\frac`, `\\boxed`, ...).
    - style: connective/reflective words from `token_categories.yaml`
             (wait / think / so / let / ...), matched on the letters-only core.
    - other: everything else (ordinary prose words, punctuation).

    `categories` may be a preloaded dict from `load_token_categories`; if None,
    the default yaml is loaded. Leading space markers (▁ / Ġ / space) are
    stripped before matching, so ' wait' and 'wait' classify the same.
    """
    if categories is None:
        categories = load_token_categories()
    style, math_extra = categories["style"], categories["math_extra"]
    return [_classify_one(t, style, math_extra) for t in token_strs]
