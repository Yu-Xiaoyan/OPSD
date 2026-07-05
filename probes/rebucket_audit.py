"""Task 3 (pre-3d): full v2 rebucket + 49-verdict confusion matrix.

Re-buckets every collected rollout with verifier v2 (any-boxed + option
letter<->value mapping), reports the v1->v2 bucket migration (i.e. the
format-only false-wrong rate that v2 recovers), and tests v2 against the 49
human verdicts as a confusion matrix (does v2 recover the 11 pseudo-wrong-format
cases while keeping true_wrong wrong?).

Login-node / CPU only (no torch, no vLLM). Writes probes/analysis/rebucket_audit.md.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from verify_answer import bucket_rollout, bucket_rollout_v2  # noqa: E402

DATA = os.path.join(_HERE, "data")
ANALYSIS = os.path.join(_HERE, "analysis")

# full-collection sources to rebucket (v1 bucket field present in each record)
SOURCES = [
    ("1.7B ckpt-50 @4096", "rollouts_ckpt50_max4096.jsonl"),
    ("wrong_extra (3a)",    "wrong_extra.jsonl"),
    ("8B ckpt-50 @4096",    "rollouts_8b_ckpt50_max4096.jsonl"),
]


def load(name):
    return [json.loads(l) for l in open(os.path.join(DATA, name))]


def main():
    os.makedirs(ANALYSIS, exist_ok=True)
    L = []
    A = L.append
    A("# v2 rebucket audit + 49-verdict confusion (task 3, pre-3d)\n")
    A("verifier v2 = any-\\boxed match + multiple-choice letter<->value mapping. "
      "Compares against v1 (last-boxed only). Confusion matrix tests v2 bucket "
      "against the 49 human verdicts.\n")

    # ---- 1. full v1 -> v2 migration ----
    A("## v1 -> v2 bucket migration (full collections)\n")
    A("| source | n | v1 wrong | v2 wrong | recovered (wrong->correct) | recover% |")
    A("|---|--:|--:|--:|--:|--:|")
    total_recover = 0
    total_v1wrong = 0
    migrations = defaultdict(Counter)  # per source
    for label, fn in SOURCES:
        rs = load(fn)
        n = len(rs)
        v1w = v2w = recovered = 0
        for r in rs:
            b1, _ = bucket_rollout(r["completion_text"], r["gt_answer"])
            b2, _ = bucket_rollout_v2(r["completion_text"], r["gt_answer"],
                                      r.get("problem", ""))
            migrations[label][(b1, b2)] += 1
            if b1 == "wrong":
                v1w += 1
                if b2 == "correct":
                    recovered += 1
            if b2 == "wrong":
                v2w += 1
        pct = 100 * recovered / v1w if v1w else 0.0
        A(f"| {label} | {n} | {v1w} | {v2w} | {recovered} | {pct:.1f}% |")
        total_recover += recovered
        total_v1wrong += v1w
    pct = 100 * total_recover / total_v1wrong if total_v1wrong else 0.0
    A(f"| **all** | | **{total_v1wrong}** | | **{total_recover}** | **{pct:.1f}%** |")
    A("")
    A("Non-trivial off-diagonal (v1!=v2) transitions:")
    for label, _ in SOURCES:
        for (b1, b2), c in sorted(migrations[label].items()):
            if b1 != b2:
                A(f"- [{label}] {b1} -> {b2}: {c}")
    A("")

    # ---- 2. 49-verdict confusion matrix ----
    ann = {r["pid"]: r for r in load("tstar_annotations.jsonl")}
    extra = {r["problem_id"]: r for r in load("wrong_extra.jsonl")}
    verdict_order = ["true_wrong_clear_tstar", "true_wrong_diffuse",
                     "pseudo_wrong_format", "vacuous_proof", "no_reasoning"]
    cols = ["correct", "wrong", "truncated"]
    conf = {v: Counter() for v in verdict_order}
    for pid, r in ann.items():
        e = extra[pid]
        b2, _ = bucket_rollout_v2(e["completion_text"], e["gt_answer"],
                                  e.get("problem", ""))
        conf[r["verdict"]][b2] += 1

    A("## Confusion matrix: human verdict (rows) x v2 bucket (cols)\n")
    A("| human verdict \\ v2 | " + " | ".join(cols) + " | n |")
    A("|---|" + "|".join(["--:"] * (len(cols) + 1)) + "|")
    for v in verdict_order:
        row = conf[v]
        nrow = sum(row.values())
        A(f"| {v} | " + " | ".join(str(row[c]) for c in cols) + f" | {nrow} |")
    A("")
    # key numbers
    pseudo_recovered = conf["pseudo_wrong_format"]["correct"]
    pseudo_n = sum(conf["pseudo_wrong_format"].values())
    tw = Counter()
    for v in ("true_wrong_clear_tstar", "true_wrong_diffuse"):
        tw.update(conf[v])
    tw_n = sum(tw.values())
    A(f"- **pseudo_wrong_format recovered by v2**: {pseudo_recovered}/{pseudo_n} "
      f"-> correct (format false-wrong fix).")
    A(f"- **true_wrong kept wrong**: {tw['wrong']}/{tw_n} (v2 did not leak "
      f"true errors into correct: {tw['correct']} leaked).")
    A(f"- vacuous_proof: {dict(conf['vacuous_proof'])}; no_reasoning: "
      f"{dict(conf['no_reasoning'])} (semantic-only; v2 not expected to fix).")
    A("")
    A("**v2 wrong bucket (post-rebucket) = true_wrong + residual semantic "
      "(vacuous/no_reasoning) - format-recovered pseudo.** This is the bucket "
      "gate C (3d) re-reviews.")

    out = os.path.join(ANALYSIS, "rebucket_audit.md")
    with open(out, "w") as f:
        f.write("\n".join(L))
    print("wrote", out)
    print(f"total v1 wrong={total_v1wrong} recovered={total_recover} ({pct:.1f}%)")
    print(f"pseudo recovered {pseudo_recovered}/{pseudo_n}; "
          f"true_wrong leaked {tw['correct']}/{tw_n}")


if __name__ == "__main__":
    main()
