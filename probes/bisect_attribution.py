"""Suppressor bisection attribution: keyword + early-emission per variant.

Reads each variant's generation dumps directly, computes the extended-window
(105-195) keyword-citation rate and clean answer-early-emission rate, one row per
variant plus the Tier-1 baseline. CPU. Corruption column filled separately (GPU).
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from datasets import load_dataset  # noqa: E402
import leakage_detector as ld  # noqa: E402

DATASET = "siyanzhao/Openthoughts_math_30k_opsd"
LO, HI = 105, 195           # extended window (variants dump to 195)
EARLY_POS = 0.30

RUNS = [
    ("Tier1 (baseline)", "qwen31b_paper_opsd_v1"),
    ("+clip",            "qwen31b_bisect_clip"),
    ("+guard",           "qwen31b_bisect_guard"),
    ("+length",          "qwen31b_bisect_length"),
    ("+tmoff",           "qwen31b_bisect_tmoff"),
]


def window_rates(gen_dir, kws, ans_idx):
    n = kw = det_clean = early_clean = 0
    for p in glob.glob(os.path.join(gen_dir, "generations_step_*.json")):
        st = int(re.search(r"step_(\d+)", p).group(1))
        if not (LO <= st <= HI):
            continue
        for s in json.load(open(p))["generations"]:
            n += 1
            r = ld.detect_sample(s, kws, ans_idx, st)
            if r.keyword_hit:
                kw += 1
            visible = r.answer_in_prompt or r.is_proof
            if r.answer_detectable and not visible:
                det_clean += 1
                if (r.answer_found and r.answer_pos_ratio is not None
                        and r.answer_pos_ratio < EARLY_POS):
                    early_clean += 1
    return {"n": n,
            "kw_rate": 100 * kw / n if n else 0.0,
            "early_clean_rate": 100 * early_clean / det_clean if det_clean else 0.0}


def main():
    kws = ld.load_keywords(os.path.join(_HERE, "leakage_keywords.txt"))
    tr = load_dataset(DATASET)["train"]
    ans_idx = ld.build_answer_index(tr)
    out = [f"window {LO}-{HI} (extended)",
           f"{'variant':<20} {'n':>5} {'keyword%':>9} {'early_clean%':>13}"]
    for label, rc in RUNS:
        gd = os.path.expanduser(f"~/opsd_outputs/{rc}/generations")
        if not os.path.isdir(gd):
            out.append(f"{label:<20} (missing)")
            continue
        r = window_rates(gd, kws, ans_idx)
        out.append(f"{label:<20} {r['n']:>5} {r['kw_rate']:>8.2f}% "
                   f"{r['early_clean_rate']:>12.2f}%")
    text = "\n".join(out)
    with open(os.path.join(_HERE, "analysis", "bisect_attribution.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
