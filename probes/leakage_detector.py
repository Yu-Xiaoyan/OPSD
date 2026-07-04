"""Behavioral leakage detector v1 (OPSD).

Two behavioral probes over TM-off student rollouts (the `completion` text dumped
during training). This is a *coarse* behavioral screen meant to provide a
ground-truth signal for aligning the corruption probe (see docs/framework.md,
diagnostic gates). Low or zero hit rate is a valid result — the matching here is
deliberately conservative and is NOT loosened to manufacture hits.

Probe 1 — keyword: does the completion verbally cite privileged information
("reference solution", "the given solution", ...)? Seed list in
probes/leakage_keywords.txt (case-insensitive, whitespace-collapsed substring).

Probe 2 — answer early emission: does the ground-truth Answer (recovered by
linking the prompt's problem text back to siyanzhao/Openthoughts_math_30k_opsd)
appear very early in the completion (relative position < threshold) with little
preceding reasoning? Answer emitted at the top with almost no prefix is a
"jump to the answer with no reasoning support" signal.

Importable library + a small __main__ that runs over one dump for debugging.
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional

# ---------------------------------------------------------------------------
# Prompt / dataset linking
# ---------------------------------------------------------------------------

# Student prompt shape (from data_collator.py):
#   "<|im_start|>user\nProblem: {problem}\n\nPlease reason step by step, and put
#    your final answer within \boxed{}.<|im_end|>\n<|im_start|>assistant\n..."
_PROBLEM_RE = re.compile(
    r"Problem:\s*(.*?)\s*\n\nPlease reason step by step", re.DOTALL
)

PROBLEM_KEY_CHARS = 200

# A problem that asks to prove/show something: the ground-truth Answer IS the
# statement to be proved, so an early occurrence of that statement is the model
# restating the question, never leakage. Such items are excluded from the clean
# early-emission rate (the answer-early probe is undefined for them).
_PROOF_RE = re.compile(r"\b(prove|show that|demonstrate that)\b", re.IGNORECASE)


def is_proof_problem(problem: Optional[str]) -> bool:
    return bool(problem) and bool(_PROOF_RE.search(problem))


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_problem_from_prompt(prompt: str) -> Optional[str]:
    """Pull the raw problem statement out of a student prompt, or None."""
    m = _PROBLEM_RE.search(prompt)
    if m:
        return m.group(1).strip()
    return None


def problem_key(problem: str) -> str:
    """Normalized prefix key used to link a problem back to the dataset."""
    return _collapse_ws(problem)[:PROBLEM_KEY_CHARS]


def build_answer_index(dataset) -> dict:
    """Map problem_key -> ground-truth Answer string over the training set.

    dataset: a HF Dataset with 'problem' and 'Answer' columns.
    """
    index: dict[str, str] = {}
    problems = dataset["problem"]
    answers = dataset["Answer"]
    for p, a in zip(problems, answers):
        if p is None or a is None:
            continue
        index[problem_key(p)] = str(a)
    return index


# ---------------------------------------------------------------------------
# Keyword probe
# ---------------------------------------------------------------------------

def load_keywords(path: str) -> list[str]:
    kws = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            kws.append(line)
    return kws


def _norm_kw_with_map(text: str) -> tuple[str, list[int]]:
    """Lowercase + collapse whitespace, keeping a map norm_idx -> orig_idx."""
    norm: list[str] = []
    idx: list[int] = []
    prev_space = True  # trims leading space
    for i, ch in enumerate(text):
        if ch.isspace():
            if not prev_space:
                norm.append(" ")
                idx.append(i)
                prev_space = True
        else:
            norm.append(ch.lower())
            idx.append(i)
            prev_space = False
    # strip a trailing space
    if norm and norm[-1] == " ":
        norm.pop()
        idx.pop()
    return "".join(norm), idx


def detect_keywords(completion: str, keywords: list[str]) -> list[dict]:
    """Return [{keyword, char_pos}] for every seed phrase present."""
    norm, idx = _norm_kw_with_map(completion)
    hits = []
    for kw in keywords:
        kw_norm = _collapse_ws(kw).lower()
        j = norm.find(kw_norm)
        if j >= 0:
            hits.append({"keyword": kw, "char_pos": idx[j]})
    return hits


# ---------------------------------------------------------------------------
# Answer early-emission probe
# ---------------------------------------------------------------------------

# spacing / wrapper macros removed (do not change the mathematical value)
_MATH_DROP = ["\\left", "\\right", "\\qquad", "\\quad", "\\,", "\\!", "\\;",
              "\\:", "\\ ", "$"]
_MATH_REPLACE = {"\\dfrac": "\\frac", "\\tfrac": "\\frac"}


def _norm_math_with_map(text: str) -> tuple[str, list[int]]:
    """Remove whitespace + spacing macros; normalize \\dfrac->\\frac.

    Case-preserving (math is case-sensitive). Returns (norm, idx_map) with
    idx_map[k] = original char index of norm[k].
    """
    norm: list[str] = []
    idx: list[int] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        replaced = False
        for src, dst in _MATH_REPLACE.items():
            if text.startswith(src, i):
                for c in dst:
                    norm.append(c)
                    idx.append(i)
                i += len(src)
                replaced = True
                break
        if replaced:
            continue
        dropped = False
        for m in _MATH_DROP:
            if text.startswith(m, i):
                i += len(m)
                dropped = True
                break
        if dropped:
            continue
        norm.append(ch)
        idx.append(i)
        i += 1
    return "".join(norm), idx


def _norm_math(text: str) -> str:
    return _norm_math_with_map(text)[0]


def answer_variants(answer: str) -> list[str]:
    """Conservative normalized variants of the ground-truth answer.

    Variants: raw; RHS of a leading 'var =' assignment; \\text{} units dropped;
    \\text{X} content kept (e.g. \\text{No} -> No). Each normalized via
    _norm_math; variants shorter than 2 chars are dropped (single tokens like
    'B'/'a' have no discriminative power).
    """
    raw = answer.strip()
    cands = [raw]

    m = re.match(r"^\s*[A-Za-z]\w*\s*=\s*(.+)$", raw)
    if m:
        cands.append(m.group(1).strip())

    drop_units = re.sub(r"\\text\{[^}]*\}", "", raw).strip()
    if drop_units and drop_units != raw:
        cands.append(drop_units)

    keep_text = re.sub(r"\\text\{([^}]*)\}", r"\1", raw).strip()
    if keep_text and keep_text != raw:
        cands.append(keep_text)

    out: list[str] = []
    seen = set()
    for c in cands:
        nc = _norm_math(c)
        if len(nc) < 2:
            continue
        if nc not in seen:
            seen.add(nc)
            out.append(nc)
    return out


def _is_alnum(s: str) -> bool:
    return s.isalnum()


def find_answer_emission(completion: str, variants: list[str]) -> dict:
    """Earliest occurrence of any answer variant in the completion.

    Returns {found, char_pos, pos_ratio, prefix_word_count, matched_variant}.
    Purely alphanumeric variants (e.g. '20', '343', 'No') are matched on the RAW
    completion with alphanumeric word boundaries — so '20' will not match inside
    '2024' but will match '20' whether it is followed by a space or a brace.
    Variants containing symbols (LaTeX expressions) are matched in the
    whitespace/spacing-macro-normalized space and mapped back to a raw position.
    """
    result = {"found": False, "char_pos": None, "pos_ratio": None,
              "prefix_word_count": None, "matched_variant": None}
    if not variants or not completion:
        return result

    norm = idx = None  # lazily built for symbol variants
    best_orig = None
    best_variant = None
    for v in variants:
        orig_pos = None
        if _is_alnum(v):
            pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])")
            m = pat.search(completion)
            if m:
                orig_pos = m.start()
        else:
            if norm is None:
                norm, idx = _norm_math_with_map(completion)
            j = norm.find(v)
            if j >= 0:
                orig_pos = idx[j]
        if orig_pos is not None and (best_orig is None or orig_pos < best_orig):
            best_orig = orig_pos
            best_variant = v

    if best_orig is not None:
        result.update(
            found=True,
            char_pos=best_orig,
            pos_ratio=best_orig / len(completion),
            prefix_word_count=len(completion[:best_orig].split()),
            matched_variant=best_variant,
        )
    return result


# ---------------------------------------------------------------------------
# Per-sample driver
# ---------------------------------------------------------------------------

@dataclass
class SampleResult:
    step: int
    linked: bool = False              # problem linked back to dataset
    gt_answer: Optional[str] = None
    answer_detectable: bool = False   # linked AND has usable (>=2 char) variant
    # keyword probe
    keyword_hit: bool = False
    keyword_hits: list = field(default_factory=list)
    # answer emission probe
    answer_found: bool = False
    answer_pos_ratio: Optional[float] = None
    answer_prefix_words: Optional[int] = None
    answer_char_pos: Optional[int] = None
    matched_variant: Optional[str] = None
    # True when the gt answer already appears in the prompt/problem statement
    # (proof identities, floor-of-expression problems, ...). Early emission is
    # then just restating the question, NOT leakage — excluded from the clean
    # early-emission rate.
    answer_in_prompt: bool = False
    is_proof: bool = False            # problem asks to prove/show (answer = statement)
    completion_len: int = 0


def detect_sample(sample: dict, keywords: list[str], answer_index: dict,
                  step: int) -> SampleResult:
    completion = sample.get("completion", "") or ""
    prompt = sample.get("prompt", "") or ""
    res = SampleResult(step=step, completion_len=len(completion))

    # keyword probe (always applicable)
    kw_hits = detect_keywords(completion, keywords)
    res.keyword_hits = kw_hits
    res.keyword_hit = len(kw_hits) > 0

    # link problem -> gt answer
    problem = extract_problem_from_prompt(prompt)
    if problem is not None:
        res.is_proof = is_proof_problem(problem)
        gt = answer_index.get(problem_key(problem))
        if gt is not None:
            res.linked = True
            res.gt_answer = gt
            variants = answer_variants(gt)
            if variants:
                res.answer_detectable = True
                res.answer_in_prompt = find_answer_emission(prompt, variants)["found"]
                emit = find_answer_emission(completion, variants)
                res.answer_found = emit["found"]
                res.answer_pos_ratio = emit["pos_ratio"]
                res.answer_prefix_words = emit["prefix_word_count"]
                res.answer_char_pos = emit["char_pos"]
                res.matched_variant = emit["matched_variant"]
    return res


def load_dump(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("generations", [])


if __name__ == "__main__":
    # Debug: python probes/leakage_detector.py <dump.json> [keywords.txt]
    dump = sys.argv[1]
    kw_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(__file__), "leakage_keywords.txt")
    keywords = load_keywords(kw_path)
    samples = load_dump(dump)
    n = len(samples)
    kw = sum(1 for s in samples if detect_keywords(s.get("completion", ""), keywords))
    print(f"{dump}: {n} samples, {kw} keyword hits "
          f"(answer probe needs the dataset index; use analyze_leakage.py)")
