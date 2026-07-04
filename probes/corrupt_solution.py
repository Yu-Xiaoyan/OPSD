"""Solution corruptor for the 2x2 corruption diagnostic (framework.md axis 1).

corrupt_solution(solution, gt_answer, corrupted_answer) -> (corrupted_solution,
n_replacements): replaces occurrences of the gt answer in the reference solution
with a corrupted answer — inside \\boxed{...} (matched by normalized content) and
in the body (numeric with word boundaries, LaTeX/expr by exact substring).

Records the replacement count; n==0 means the answer form did not match (skip),
reported honestly rather than forcing a change.
"""
from __future__ import annotations

import re

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def corrupt_solution(solution: str, gt_answer: str,
                     corrupted_answer: str) -> tuple[str, int]:
    gt = gt_answer.strip()
    gt_norm = _norm(gt)
    n = 0
    out = solution

    if re.fullmatch(r"-?\d+\.?\d*", gt):
        # numeric: a single word-boundary pass already covers \boxed{343} (the
        # '{' / '}' are non-word boundaries), so we do NOT run the boxed pass —
        # otherwise a boxed replacement carrying the gt digits would be matched
        # again. Function replacement inserts the corruption literally (a string
        # repl would interpret backslashes, e.g. '\frac' -> bad escape).
        pat = re.compile(r"(?<![\w.])" + re.escape(gt) + r"(?![\w.])")
        out, k = pat.subn(lambda m: corrupted_answer, out)
        n += k
    elif gt:
        # LaTeX / expression: replace inside \boxed{...} (normalized match) and
        # exact substrings in the body.
        def _boxed_sub(m):
            nonlocal n
            if _norm(m.group(1)) == gt_norm:
                n += 1
                return "\\boxed{" + corrupted_answer + "}"
            return m.group(0)

        out = _BOXED_RE.sub(_boxed_sub, out)
        k = out.count(gt)
        if k:
            out = out.replace(gt, corrupted_answer)
            n += k
    return out, n
