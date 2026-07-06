#!/usr/bin/env python3
"""Summarize v0 (gated) eval against the OPSD baseline, side by side.

Metric: avg@12 accuracy (`average_at_n_pct`) on AIME24 / AIME25, under the
locked protocol (temp 1.0 / top_p 1.0 / top_k -1 / min_p 0 / avg@12).

Sources (always includes the untrained base as reference — project rule):
  - base : results/repro_eval/base_{ds}.json          (model-independent)
  - OPSD : results/repro_eval/ckpt{STEP}_{ds}.json    (baseline reproduction)
  - v0   : results/v0_eval/v0ckpt{STEP}_{ds}.json      (this branch)

Also asserts the eval lock: any loaded file whose temperature != 1.0 or
top_p != 1.0 is flagged loudly (the handoff's temp 0.6 was wrong).

Stdlib only:  python scripts/summarize_v0.py
"""
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPRO = os.path.join(ROOT, "results", "repro_eval")
V0 = os.path.join(ROOT, "results", "v0_eval")
BENCHMARKS = ["aime24", "aime25"]
STEPS = [50, 100, 150]


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] could not read {path}: {e}")
        return None


def _acc(d):
    return d.get("average_at_n_pct") if d else None


def _check_lock(d, label, warns):
    if not d:
        return
    t, p = d.get("temperature"), d.get("top_p")
    if (t is not None and abs(t - 1.0) > 1e-9) or (p is not None and abs(p - 1.0) > 1e-9):
        warns.append(f"  !! {label}: temp={t} top_p={p} (LOCK is temp 1.0 / top_p 1.0)")


def fmt(x):
    return f"{x:6.1f}" if isinstance(x, (int, float)) else f"{'-':>6}"


def main():
    warns = []
    for ds in BENCHMARKS:
        base = _load(os.path.join(REPRO, f"base_{ds}.json"))
        _check_lock(base, f"base/{ds}", warns)
        base_acc = _acc(base)
        base_s = f"{base_acc:.1f}" if isinstance(base_acc, (int, float)) else "-"

        print(f"\n{ds.upper()}  (avg@12, %)   base = {base_s}")
        print(f"  {'step':<7}{'OPSD':>8}{'v0':>8}{'Δ(v0-OPSD)':>12}{'N':>6}")
        print("  " + "-" * 41)
        for step in STEPS:
            opsd = _load(os.path.join(REPRO, f"ckpt{step}_{ds}.json"))
            v0 = _load(os.path.join(V0, f"v0ckpt{step}_{ds}.json"))
            _check_lock(opsd, f"OPSD ckpt{step}/{ds}", warns)
            _check_lock(v0, f"v0 ckpt{step}/{ds}", warns)
            oa, va = _acc(opsd), _acc(v0)
            if isinstance(oa, (int, float)) and isinstance(va, (int, float)):
                delta = f"{va - oa:+.1f}"
            else:
                delta = "-"
            n = (v0 or opsd or {}).get("total_solutions", "-")
            print(f"  {step:<7}{fmt(oa):>8}{fmt(va):>8}{delta:>12}{str(n):>6}")

    if warns:
        print("\nPROTOCOL-LOCK VIOLATIONS (results NOT comparable):")
        for w in warns:
            print(w)

    have_v0 = glob.glob(os.path.join(V0, "v0ckpt*_*.json"))
    if not have_v0:
        print(f"\n(no v0 results yet under {V0} — run scripts/run_v0_eval.sh once "
              "qwen31b_v0_main checkpoints exist)")
    print()


if __name__ == "__main__":
    main()
