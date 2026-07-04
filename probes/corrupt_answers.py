"""Deterministic corrupted-answer generator for the corruption probe.

Given a ground-truth answer, produce up to n corrupted variants by perturbing
numbers (±1/±2, ×2), flipping sign, or taking a reciprocal — keeping the LaTeX
shape valid. Variants are != gt and != a supplied student wrong answer.
Deterministic given `seed` (derive it per-problem for reproducibility).

Some answers (single letters, pure prose) have no perturbable number; then fewer
than n variants are returned (reported honestly, not padded with junk).
"""
from __future__ import annotations

import hashlib
import random
import re

_NUM_RE = re.compile(r"-?\d+\.?\d*")
_FRAC_RE = re.compile(r"\\[dt]?frac\{(.+?)\}\{(.+?)\}")


def seed_from(*parts) -> int:
    h = hashlib.sha256("||".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16)


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _braces_balanced(s: str) -> bool:
    depth = 0
    for ch in s:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _valid(cand: str) -> bool:
    return bool(cand) and "\\boxed" not in cand and _braces_balanced(cand)


def _perturb_number(tok: str) -> list[str]:
    out: list[str] = []
    try:
        if "." in tok:
            f = float(tok)
            for g in (f + 1, f - 1, -f, 2 * f):
                out.append(f"{g:g}")
        else:
            i = int(tok)
            for g in (i + 1, i - 1, -i, 2 * i, i + 2):
                if g == i:
                    continue                       # skip no-ops (e.g. -0, 2*0)
                out.append(str(g))
    except ValueError:
        pass
    return out


def _num_val(s: str):
    try:
        return float(s)
    except ValueError:
        return None


def corrupt_answer(gt_answer: str, n: int = 3, avoid=None, seed: int = 0) -> list[str]:
    gt = gt_answer.strip()
    avoid_norm = {_norm(a) for a in (avoid or []) if a} | {_norm(gt)}
    rng = random.Random(seed)

    cands: list[str] = []

    # 1) perturb each number occurrence with each transform
    for m in _NUM_RE.finditer(gt):
        for pert in _perturb_number(m.group()):
            cands.append(gt[:m.start()] + pert + gt[m.end():])

    # 2) whole-expression sign flip
    if gt.startswith("-"):
        cands.append(gt[1:])
    else:
        cands.append("-" + gt)

    # 3) reciprocal-ish transforms
    fm = _FRAC_RE.search(gt)
    if fm:
        # swap numerator/denominator of the first \frac
        swapped = gt[:fm.start()] + f"\\frac{{{fm.group(2)}}}{{{fm.group(1)}}}" + gt[fm.end():]
        cands.append(swapped)
    elif re.fullmatch(r"-?\d+", gt):
        cands.append(f"\\frac{{1}}{{{gt}}}")

    # dedup + filter (deterministic shuffle for variety)
    gtv = _num_val(gt)
    rng.shuffle(cands)
    seen: set[str] = set()
    out: list[str] = []
    for c in cands:
        c = c.strip()
        cn = _norm(c)
        if cn in avoid_norm or cn in seen or not _valid(c):
            continue
        cv = _num_val(c)                       # drop numeric duplicates of gt (e.g. -0)
        if gtv is not None and cv is not None and cv == gtv:
            continue
        seen.add(cn)
        out.append(c)
        if len(out) >= n:
            break
    return out
