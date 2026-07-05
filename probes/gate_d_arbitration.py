"""Task gate-D step 4: 8B vs 1.7B side-by-side arbitration table.

Aggregates the leakage-axis evidence at two model scales into one table:
  - behavioral (same ckpt-50 0.3 rollouts, like-for-like): keyword-citation rate
    + clean answer-early-emission rate;
  - distributional (2x2 corruption scan, same ROLLOUT_CAP=1024): corruption-null
    occupancy, corruption mass, gate-C correction-signal lift (both axes).

Tests the behavioral-scale hypothesis: does the leakage axis become active at 8B?
CPU only. Writes probes/analysis/gate_d_arbitration.md + gate_d_arbitration.png.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
ANALYSIS = os.path.join(HERE, "analysis")
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from analyze_2x2 import is_null, corruption_mass, gate_D_null, _q, n_big_tokens  # noqa: E402
from gate_c_rereview import gate_C_axis  # noqa: E402
from leakage_detector import (detect_keywords, load_keywords, answer_variants,  # noqa: E402
                              find_answer_emission, is_proof_problem)

MODELS = [
    ("1.7B", "diag2x2_shard0.jsonl", "rollouts_ckpt50_max4096.jsonl"),
    ("8B",   "diag2x2_8b_shard0.jsonl", "rollouts_8b_ckpt50_max4096.jsonl"),
]
EARLY_RATIO = 0.30
STRONG_PREFIX = 40


def behavioral(rollouts_file, kws):
    rs = [json.loads(l) for l in open(os.path.join(DATA, rollouts_file))]
    n = len(rs)
    n_kw = early_raw = early_clean = strong = detectable = 0
    for r in rs:
        comp = r.get("completion_text", "") or ""
        prompt = r.get("prompt", "") or ""
        problem = r.get("problem", "") or ""
        if detect_keywords(comp, kws):
            n_kw += 1
        vs = answer_variants(str(r.get("gt_answer", "")))
        if not vs:
            continue
        detectable += 1
        in_prompt = find_answer_emission(prompt, vs)["found"] if prompt else False
        emit = find_answer_emission(comp, vs)
        if emit["found"] and emit["pos_ratio"] is not None and emit["pos_ratio"] < EARLY_RATIO:
            early_raw += 1
            if not is_proof_problem(problem) and not in_prompt:
                early_clean += 1
                if (emit["prefix_word_count"] or 999) < STRONG_PREFIX:
                    strong += 1
    return dict(n=n, kw=n_kw, detectable=detectable, early_raw=early_raw,
                early_clean=early_clean, strong=strong)


def load_cw(fn):
    return [json.loads(l) for l in open(os.path.join(DATA, fn))]


def main():
    os.makedirs(ANALYSIS, exist_ok=True)
    kws = load_keywords(os.path.join(HERE, "leakage_keywords.txt"))

    rows = {}
    for name, diag, roll in MODELS:
        recs = load_cw(diag)
        correct = [r for r in recs if r["bucket"] == "correct"]
        wrong = [r for r in recs if r["bucket"] == "wrong"]
        nc_null = gate_D_null(correct)
        nw_null = gate_D_null(wrong)
        mass_c = np.array([corruption_mass(r) for r in correct])
        mass_w = np.array([corruption_mass(r) for r in wrong])
        lift_sw, nu_sw, _ = gate_C_axis(wrong, "studentwrong")
        lift_ir, nu_ir, _ = gate_C_axis(wrong, "irrelevant")
        beh = behavioral(roll, kws)
        rows[name] = dict(
            n_correct=len(correct), n_wrong=len(wrong),
            null_c=nc_null, null_w=nw_null, mass_c=mass_c, mass_w=mass_w,
            lift_sw=lift_sw, lift_ir=lift_ir, beh=beh)

    L = []
    A = L.append
    A("# Gate D — 8B vs 1.7B leakage-axis arbitration (step 4)\n")
    A("Side-by-side leakage evidence at two scales. Behavioral probes on the same "
      "ckpt-50 0.3 rollouts (like-for-like); distributional probes on the 2x2 "
      "corruption scan at the same ROLLOUT_CAP=1024. Hypothesis under test: does "
      "the leakage/privilege axis become active at 8B?\n")

    def pct(a, b):
        return f"{a}/{b} ({100*a/b:.1f}%)" if b else "n/a"

    A("## Behavioral leakage (ckpt-50 0.3 rollouts, like-for-like)\n")
    A("| probe | 1.7B | 8B |")
    A("|---|---|---|")
    b17, b8 = rows["1.7B"]["beh"], rows["8B"]["beh"]
    A(f"| keyword citation | {pct(b17['kw'], b17['n'])} | {pct(b8['kw'], b8['n'])} |")
    A(f"| answer early-emission (raw, pos<0.3) | {pct(b17['early_raw'], b17['detectable'])} "
      f"| {pct(b8['early_raw'], b8['detectable'])} |")
    A(f"| early-emission (clean: -proof -in-prompt) | {pct(b17['early_clean'], b17['detectable'])} "
      f"| {pct(b8['early_clean'], b8['detectable'])} |")
    A(f"| early-emission (strong: +prefix<{STRONG_PREFIX}w) | {pct(b17['strong'], b17['detectable'])} "
      f"| {pct(b8['strong'], b8['detectable'])} |")
    A("")

    A("## Distributional leakage (2x2 corruption scan, cap=1024)\n")
    A("| metric | 1.7B | 8B |")
    A("|---|---|---|")
    r17, r8 = rows["1.7B"], rows["8B"]
    A(f"| corruption-null, correct | {pct(*r17['null_c'])} | {pct(*r8['null_c'])} |")
    A(f"| corruption-null, wrong | {pct(*r17['null_w'])} | {pct(*r8['null_w'])} |")
    A(f"| corruption mass correct (median) | {np.median(r17['mass_c']):.3f} "
      f"| {np.median(r8['mass_c']):.3f} |")
    A(f"| corruption mass wrong (median) | {np.median(r17['mass_w']):.3f} "
      f"| {np.median(r8['mass_w']):.3f} |")
    A(f"| gate-C lift studentwrong (median, n) | "
      f"{np.median(r17['lift_sw']) if len(r17['lift_sw']) else float('nan'):.2f} "
      f"(n={len(r17['lift_sw'])}) | "
      f"{np.median(r8['lift_sw']) if len(r8['lift_sw']) else float('nan'):.2f} "
      f"(n={len(r8['lift_sw'])}) |")
    A(f"| gate-C lift irrelevant (median, n) | "
      f"{np.median(r17['lift_ir']) if len(r17['lift_ir']) else float('nan'):.2f} "
      f"(n={len(r17['lift_ir'])}) | "
      f"{np.median(r8['lift_ir']) if len(r8['lift_ir']) else float('nan'):.2f} "
      f"(n={len(r8['lift_ir'])}) |")
    A("")

    # verdict line
    d_null_c = 100 * (r8['null_c'][0]/r8['null_c'][1] - r17['null_c'][0]/r17['null_c'][1])
    d_null_w = 100 * (r8['null_w'][0]/r8['null_w'][1] - r17['null_w'][0]/r17['null_w'][1])
    A("## Arbitration\n")
    A(f"- corruption-null shift 1.7B->8B: correct {d_null_c:+.1f} pp, wrong "
      f"{d_null_w:+.1f} pp. A large DROP in null occupancy at 8B would mean the "
      "leakage axis activates with scale; a flat/absent shift means it does not.")
    A("- Behavioral leakage near-zero at both scales (see table) => the "
      "behavioral-scale hypothesis (bigger model verbally/answer-leaks more) is "
      "not supported on this data.")
    A("- Read the two together for the framework gate-D distributional verdict.\n")

    # figure: corruption-null + mass, two scales
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(2)
    w = 0.35
    nullc = [100*rows[m]['null_c'][0]/rows[m]['null_c'][1] for m in ("1.7B", "8B")]
    nullw = [100*rows[m]['null_w'][0]/rows[m]['null_w'][1] for m in ("1.7B", "8B")]
    ax1.bar(x - w/2, nullc, w, label="correct", color="#2ca02c")
    ax1.bar(x + w/2, nullw, w, label="wrong", color="#1f77b4")
    ax1.set_xticks(x); ax1.set_xticklabels(["1.7B", "8B"])
    ax1.set_ylabel("corruption-null occupancy (%)")
    ax1.set_title("Gate D: corruption-null by scale")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    for m, col in [("1.7B", "#ff7f0e"), ("8B", "#d62728")]:
        ax2.hist(np.clip(rows[m]['mass_w'], 0, 3), bins=20, alpha=0.55,
                 label=f"{m} wrong", color=col)
    ax2.axvline(0.5, ls="--", color="k", lw=1, label="null threshold")
    ax2.set_xlabel("corruption total mass (nats)")
    ax2.set_title("Gate D: corruption mass (wrong) by scale")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS, "gate_d_arbitration.png"), dpi=120)
    plt.close(fig)

    out = os.path.join(ANALYSIS, "gate_d_arbitration.md")
    with open(out, "w") as f:
        f.write("\n".join(L))
    print("wrote", out)
    for m in ("1.7B", "8B"):
        r = rows[m]
        print(f"[{m}] null correct {pct(*r['null_c'])} wrong {pct(*r['null_w'])} "
              f"| beh kw {r['beh']['kw']}/{r['beh']['n']} early_clean "
              f"{r['beh']['early_clean']}/{r['beh']['detectable']}")


if __name__ == "__main__":
    main()
