"""Answer extraction + grading, reused from eval/evaluate_math.py.

Kept as a small standalone module (no vLLM import) so the probe pipeline can
bucket rollouts as correct / wrong / truncated without pulling the eval stack.
Logic mirrors eval/evaluate_math.py::extract_boxed_answer / grade_answer.
"""
from __future__ import annotations


def extract_boxed_answer(text: str):
    """Return the content of the last \\boxed{...}, or None (= no boxed output)."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = idx
    num_left = 0
    right = None
    while i < len(text):
        if text[i] == "{":
            num_left += 1
        if text[i] == "}":
            num_left -= 1
            if num_left == 0:
                right = i
                break
        i += 1
    if right is None:
        return None
    boxed = text[idx:right + 1]
    if boxed.startswith("\\boxed{") and boxed.endswith("}"):
        return boxed[7:-1].strip()
    return None


def grade_answer(predicted, ground_truth) -> bool:
    """True if predicted matches ground_truth (math_verify, string fallback)."""
    if predicted is None:
        return False
    try:
        from math_verify import parse, verify
        p = f"${predicted}$" if "$" not in predicted else predicted
        g = f"${ground_truth}$" if "$" not in ground_truth else ground_truth
        return bool(verify(parse(g, fallback_mode="no_fallback"),
                           parse(p, fallback_mode="no_fallback"),
                           timeout_seconds=5))
    except Exception:
        pn = predicted.replace("$", "").replace(" ", "").lower().strip()
        gn = str(ground_truth).replace("$", "").replace(" ", "").lower().strip()
        return pn == gn


def bucket_rollout(completion_text: str, gt_answer: str) -> tuple[str, str | None]:
    """Classify a rollout: ('correct'|'wrong'|'truncated', student_answer|None).

    truncated = no \\boxed output (proxy for an unfinished / cut-off rollout).
    """
    student = extract_boxed_answer(completion_text)
    if student is None:
        return "truncated", None
    return ("correct" if grade_answer(student, gt_answer) else "wrong"), student


# ---------------------------------------------------------------------------
# verifier v2 (task-3 upgrade): fixes format-only false-wrongs found by the
# 49-condition human audit — (a) multi-boxed (any boxed matching gt), and
# (b) multiple-choice letter<->value mapping. Residual false-wrongs (statement /
# interval semantic equivalence) still need a semantic judge and are NOT fixed.
# ---------------------------------------------------------------------------
import re as _re

_OPT_RE = _re.compile(r"\(([A-E])\)\s*(.+?)(?=\s*\([A-E]\)|$)", _re.S)
_OPT_RE2 = _re.compile(r"(?:^|\n)\s*([A-E])[.)]\s*(.+?)(?=(?:\n\s*[A-E][.)])|$)", _re.S)


def extract_all_boxed(text: str) -> list[str]:
    """All \\boxed{...} contents (not just the last)."""
    out, i = [], 0
    while True:
        idx = text.find("\\boxed", i)
        if idx < 0:
            break
        j, depth, right = idx, 0, None
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    right = j
                    break
            j += 1
        if right is None:
            break
        if text[idx:idx + 7] == "\\boxed{":
            out.append(text[idx + 7:right].strip())
        i = right + 1
    return out


def parse_options(problem: str) -> dict:
    opts = {m.group(1): m.group(2).strip() for m in _OPT_RE.finditer(problem)}
    if not opts:
        opts = {m.group(1): m.group(2).strip() for m in _OPT_RE2.finditer(problem)}
    return opts


def grade_answer_v2(boxes: list[str], gt_answer: str, problem: str = "") -> bool:
    """True if ANY boxed answer matches gt, with letter<->value option mapping."""
    gt = str(gt_answer).strip()
    opts = parse_options(problem) if problem else {}
    gt_is_letter = gt.upper() in ("A", "B", "C", "D", "E")
    cand_gt = [gt]
    if gt_is_letter and gt.upper() in opts:      # gt letter -> its option value
        cand_gt.append(opts[gt.upper()])
    for b in boxes:
        b = b.strip()
        for g in cand_gt:
            if grade_answer(b, g):
                return True
        # student gave a letter; gt is a letter
        if gt_is_letter and b.upper() == gt.upper():
            return True
        # student gave a value; gt is a letter -> compare against that option
        if gt_is_letter and gt.upper() in opts and grade_answer(b, opts[gt.upper()]):
            return True
    return False


def bucket_rollout_v2(completion_text: str, gt_answer: str,
                      problem: str = "") -> tuple[str, str | None]:
    """Upgraded bucketing: any-boxed + option-mapping. truncated = no boxed."""
    boxes = extract_all_boxed(completion_text)
    if not boxes:
        return "truncated", None
    ok = grade_answer_v2(boxes, gt_answer, problem)
    return ("correct" if ok else "wrong"), boxes[-1]
