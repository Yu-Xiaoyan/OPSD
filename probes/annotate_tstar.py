"""Task 3b helper: present wrong rollouts as numbered reasoning steps for t*
annotation, and merge annotations back.

Modes:
  dump   : print pid/gt/student + \\n\\n-split steps (truncated) for a slice.
  stats  : step-count distribution.
  merge  : given a small python-dict of {pid: (tstar, reason)}, write jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "wrong_extra.jsonl")
OUT = os.path.join(HERE, "data", "tstar_annotations.jsonl")


_MARK = re.compile(r"\b(wait|contradiction|mistake|recompute|hold on|"
                   r"let me correct|correction|error|oops|but that|hmm,? no|"
                   r"actually,? no|re-?evaluate|flawed|wrong)\b", re.I)


def steps_of(text):
    # split on blank lines; drop the leading <think> marker noise
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return parts


def first_marker(steps):
    for i, s in enumerate(steps):
        if _MARK.search(s):
            return i
    return None


def load():
    return [json.loads(l) for l in open(SRC, encoding="utf-8")]


SAMPLE_FILE = os.path.join(HERE, "data", "tstar_sample.json")


def stratified_sample(recs, n=50, seed=42):
    import numpy as np
    import random
    rng = random.Random(seed)
    meta = []
    for r in recs:
        st = steps_of(r["completion_text"])
        meta.append((r["problem_id"], len(st), first_marker(st) is not None))
    lens = [m[1] for m in meta]
    t1, t2 = np.quantile(lens, 1 / 3), np.quantile(lens, 2 / 3)

    def lbin(x):
        return "short" if x <= t1 else ("long" if x > t2 else "med")

    strata = {}
    for pid, ns, hm in meta:
        strata.setdefault((lbin(ns), hm), []).append((pid, ns, hm))
    total = len(meta)
    picked = []
    for key, items in sorted(strata.items()):
        quota = max(1, round(n * len(items) / total))
        rng.shuffle(items)
        for pid, ns, hm in items[:quota]:
            picked.append({"pid": pid, "stratum": f"{key[0]}/{'mark' if key[1] else 'nomark'}",
                           "n_steps": ns, "has_marker": hm})
    rng.shuffle(picked)
    picked = picked[:n]
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dump", "stats", "sample"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--steptrunc", type=int, default=160)
    ap.add_argument("--sample", action="store_true", help="dump over the sampled 50")
    args = ap.parse_args()
    recs = load()

    if args.mode == "stats":
        import numpy as np
        ns = [len(steps_of(r["completion_text"])) for r in recs]
        print(f"n_rollouts={len(recs)} | steps: min={min(ns)} "
              f"med={int(np.median(ns))} max={max(ns)} mean={np.mean(ns):.1f}")
        print(f"rollouts with >40 steps: {sum(1 for x in ns if x>40)}")
        return

    if args.mode == "sample":
        from collections import Counter
        picked = stratified_sample(recs, n=50, seed=42)
        json.dump(picked, open(SAMPLE_FILE, "w"), indent=1)
        print("strata:", dict(Counter(p["stratum"] for p in picked)))
        print(f"wrote {len(picked)} sampled pids -> {SAMPLE_FILE}")
        return

    if args.sample:
        picked = json.load(open(SAMPLE_FILE))
        order = [p["pid"] for p in picked]
        recs = sorted([r for r in recs if r["problem_id"] in order],
                      key=lambda r: order.index(r["problem_id"]))

    for r in recs[args.start:args.start + args.n]:
        steps = steps_of(r["completion_text"])
        fm = first_marker(steps)
        # focus window: up to first self-correction marker + 2 (else all if short)
        hi = min(len(steps), (fm + 3) if fm is not None else len(steps))
        print(f"\n===== pid={r['problem_id']} | gt={r['gt_answer']!r} | "
              f"student={r['student_answer']!r} | n_steps={len(steps)} | "
              f"first_marker={fm} =====")
        print(f"PROBLEM: {r['problem'][:240]}")
        for i in range(hi):
            s1 = steps[i].replace('\n', ' ')
            if len(s1) > args.steptrunc:
                s1 = s1[:args.steptrunc] + "…"
            tag = " <<<MARK" if fm == i else ""
            print(f"[{i}] {s1}{tag}")
        if hi < len(steps):
            print(f"... ({len(steps)-hi} more steps after the marker)")


if __name__ == "__main__":
    main()
