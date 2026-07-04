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
