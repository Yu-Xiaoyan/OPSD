#!/usr/bin/env python3
"""Summarize OPSD reproduction eval results against the official numbers.

Reads every results/repro_eval/*.json produced by eval/evaluate_math.py and
prints a checkpoint x benchmark table of avg@12 accuracy, side by side with the
official README figures.

Stdlib only; run with any python:  python scripts/summarize_eval.py
"""
import glob
import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "repro_eval")

# Official README numbers, avg@12 (%). None = not reported at that step.
OFFICIAL = {
    "aime24": {"base": 51.5, "50": 52.8, "100": 57.2},
    "aime25": {"base": 36.7, "50": 43.9, "100": 41.1},
}

# filename tag -> display label / step key used in OFFICIAL
TAG_TO_STEP = {"base": "base", "ckpt50": "50", "ckpt100": "100"}
TAG_ORDER = ["base", "ckpt50", "ckpt100"]
TAG_LABEL = {"base": "base", "ckpt50": "ckpt-50", "ckpt100": "ckpt-100"}
BENCHMARKS = ["aime24", "aime25"]


def load_results():
    """Return {(tag, dataset): summary_dict} for every result file found."""
    out = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        name = os.path.splitext(os.path.basename(path))[0]  # e.g. ckpt100_aime24
        # dataset is the trailing token; tag is everything before it
        for ds in BENCHMARKS:
            if name.endswith("_" + ds):
                tag = name[: -(len(ds) + 1)]
                try:
                    with open(path) as f:
                        out[(tag, ds)] = json.load(f)
                except Exception as e:  # noqa: BLE001
                    print(f"  [warn] could not read {path}: {e}")
                break
    return out


def fmt(x):
    return f"{x:5.1f}" if isinstance(x, (int, float)) else f"{x:>5}"


def main():
    if not os.path.isdir(RESULTS_DIR):
        print(f"No results directory yet: {RESULTS_DIR}")
        return
    res = load_results()
    if not res:
        print(f"No result JSONs found under {RESULTS_DIR}")
        return

    for ds in BENCHMARKS:
        print(f"\n{ds.upper()}  (avg@12, %)")
        print(f"  {'point':<9}{'repro':>7}{'official':>10}{'delta':>8}   N")
        print("  " + "-" * 42)
        for tag in TAG_ORDER:
            step = TAG_TO_STEP[tag]
            official = OFFICIAL.get(ds, {}).get(step)
            summary = res.get((tag, ds))
            if summary is None:
                repro = "-"
                delta = "-"
                n = "-"
            else:
                repro = summary.get("average_at_n_pct")
                n = summary.get("total_solutions", "?")
                if isinstance(repro, (int, float)) and isinstance(official, (int, float)):
                    delta = f"{repro - official:+.1f}"
                else:
                    delta = "-"
            off_s = fmt(official) if official is not None else "  -  "
            print(f"  {TAG_LABEL[tag]:<9}{fmt(repro):>7}{off_s:>10}{delta:>8}   {n}")

    # completeness note
    missing = [(t, d) for t in TAG_ORDER for d in BENCHMARKS if (t, d) not in res]
    if missing:
        print("\nMissing (not yet finished / failed):")
        for t, d in missing:
            print(f"  - {TAG_LABEL[t]} x {d}")
    print()


if __name__ == "__main__":
    main()
