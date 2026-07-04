"""Experiment A analysis: leakage behavior — guard vs no-guard training runs.

Runs the leakage_detector's two probes (main + question-visibility filter) on
the training rollout dumps of BOTH runs, step by step, and appends a comparison
to probes/analysis/leakage_over_training.md.

  guard   = qwen31b_repro_3xh200_gb30   (official teacher prompt)
  noguard = qwen31b_noguard_3xh200_gb30 (transition guard replaced by a neutral
            connective; reference-solution segment kept)

Hypothesis: removing the 'do not copy / use your own words' guard raises
leakage-like behavior in the student rollouts. CPU only.
"""
from __future__ import annotations

import glob
import os
import re

from datasets import load_dataset

import leakage_detector as ld

GUARD_DIR = os.path.expanduser(
    "~/opsd_outputs/qwen31b_repro_3xh200_gb30/generations")
NOGUARD_DIR = os.path.expanduser(
    "~/opsd_outputs/qwen31b_noguard_3xh200_gb30/generations")
MD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis",
                  "leakage_over_training.md")
KW_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "leakage_keywords.txt")
DATASET = "siyanzhao/Openthoughts_math_30k_opsd"
MARKER = "<!-- NOGUARD-COMPARE -->"
EARLY_POS = 0.30


def _step(path):
    return int(re.search(r"generations_step_(\d+)\.json", path).group(1))


def aggregate_run(gen_dir, keywords, answer_index):
    rows = {}
    for path in sorted(glob.glob(os.path.join(gen_dir, "generations_step_*.json")),
                       key=_step):
        step = _step(path)
        n = kw = det_c = early_c = 0
        for s in ld.load_dump(path):
            r = ld.detect_sample(s, keywords, answer_index, step)
            n += 1
            if r.keyword_hit:
                kw += 1
            if r.answer_detectable and not (r.answer_in_prompt or r.is_proof):
                det_c += 1
                if (r.answer_found and r.answer_pos_ratio is not None
                        and r.answer_pos_ratio < EARLY_POS):
                    early_c += 1
        rows[step] = {"n": n, "kw": kw, "det_c": det_c, "early_c": early_c}
    return rows


def _tot(rows, key):
    return sum(r[key] for r in rows.values())


def main():
    keywords = ld.load_keywords(KW_PATH)
    tr = load_dataset(DATASET)["train"]
    index = ld.build_answer_index(tr)

    print("aggregating guard run ...")
    guard = aggregate_run(GUARD_DIR, keywords, index)
    print("aggregating noguard run ...")
    noguard = aggregate_run(NOGUARD_DIR, keywords, index)
    if not noguard:
        raise SystemExit(f"no noguard dumps found in {NOGUARD_DIR} "
                         "(has the run finished?)")

    steps = sorted(set(guard) & set(noguard))

    def rate(rows, step, num, den):
        r = rows[step]
        return 100 * r[num] / r[den] if r[den] else 0.0

    lines = [MARKER, "", "## Experiment A — leakage: guard vs no-guard", "",
             "Leakage probes on the two runs' training rollouts (TM-off), "
             "clean metric (question-visibility filtered). guard = "
             "`qwen31b_repro_3xh200_gb30` (official prompt); noguard = "
             "`qwen31b_noguard_3xh200_gb30` (guard replaced by neutral "
             "connective, reference-solution segment kept). Appended by "
             "`probes/compare_noguard_leakage.py`.", ""]

    # totals
    for label, rows in (("guard", guard), ("noguard", noguard)):
        n, kw, det_c, ec = (_tot(rows, "n"), _tot(rows, "kw"),
                            _tot(rows, "det_c"), _tot(rows, "early_c"))
        lines.append(f"- **{label}**: {len(rows)} steps, {n} samples | "
                     f"keyword {kw} ({100*kw/n if n else 0:.2f}%) | "
                     f"answer-early-clean {ec}/{det_c} "
                     f"({100*ec/det_c if det_c else 0:.2f}%)")
    lines += ["", "| step | guard kw% | noguard kw% | guard early% | noguard early% |",
              "|--:|--:|--:|--:|--:|"]
    for st in steps:
        lines.append(
            f"| {st} | {rate(guard,st,'kw','n'):.2f} | "
            f"{rate(noguard,st,'kw','n'):.2f} | "
            f"{rate(guard,st,'early_c','det_c'):.2f} | "
            f"{rate(noguard,st,'early_c','det_c'):.2f} |")
    lines += ["", "**Read**: if noguard columns are systematically higher, the "
              "guard instruction was suppressing leakage-like behavior; if the "
              "two are within noise, the guard has no measurable behavioral "
              "effect at this scale (consistent with the near-zero absolute "
              "rates from the main scan).", ""]
    section = "\n".join(lines) + "\n"

    md = ""
    if os.path.exists(MD):
        md = open(MD, encoding="utf-8").read()
        if MARKER in md:
            md = md[:md.index(MARKER)].rstrip() + "\n\n"
    with open(MD, "w", encoding="utf-8") as f:
        f.write(md + section)

    # console summary
    print("\n== TOTALS (clean) ==")
    for label, rows in (("guard", guard), ("noguard", noguard)):
        n, kw, det_c, ec = (_tot(rows, "n"), _tot(rows, "kw"),
                            _tot(rows, "det_c"), _tot(rows, "early_c"))
        print(f"{label:8}: kw {100*kw/n if n else 0:.2f}% | "
              f"early-clean {100*ec/det_c if det_c else 0:.2f}% "
              f"({ec}/{det_c})")
    print(f"appended comparison to {MD}")


if __name__ == "__main__":
    main()
