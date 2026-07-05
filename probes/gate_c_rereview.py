"""Task 3d: gate-C re-review on the v2-rebucketed wrong bucket.

Same diag2x2 corruption data (no new GPU pass), but the wrong bucket is
re-defined by verifier v2 (format-recovered pseudo removed by pid-join to the
source rollouts). Recomputes correction-signal LIFT on the answer span for BOTH
corruption axes (irrelevant = mean over 3 corrupted answers; studentwrong =
teacher fed the student's own wrong answer), the corruption-null occupancy, and
the quality floor — v1 wrong vs v2 wrong side by side. no-reasoning rollouts are
excluded (REASONING_MIN). CPU only.

Writes probes/analysis/gate_c_rereview.md + gate_c_rereview.png.
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

from analyze_2x2 import (is_null, corruption_mass, n_big_tokens, jsd_corr_mean,  # noqa: E402
                         gate_D_null, _q, REASONING_MIN, load_cw)
from verify_answer import bucket_rollout_v2  # noqa: E402

SRC = "rollouts_ckpt50_max4096.jsonl"   # source with completion_text + problem


def gate_C_axis(wrong, axis):
    """LIFT = (answer-span JSD-mass frac)/(answer-span token frac), per axis.

    axis 'studentwrong' -> jsd_corruption_studentwrong; 'irrelevant' -> mean over
    the 3 irrelevant-answer corruptions. no-reasoning rollouts excluded.
    """
    lift = []
    n_usable = n_noreason = 0
    for r in wrong:
        if is_null(r):
            continue
        if axis == "studentwrong":
            sw = r.get("jsd_corruption_studentwrong")
            if sw is None:
                continue
            sw = np.asarray(sw, dtype=float)
        else:
            sw = jsd_corr_mean(r)
        span = np.asarray(r["answer_span"], dtype=bool)
        tot = sw.sum()
        if tot <= 0 or span.sum() == 0:
            continue
        n_usable += 1
        if (r["T"] - int(span.sum())) < REASONING_MIN:
            n_noreason += 1
            continue
        sf = span.sum() / r["T"]
        if sf > 0:
            lift.append(sw[span].sum() / tot / sf)
    return np.array(lift), n_usable, n_noreason


def main():
    os.makedirs(ANALYSIS, exist_ok=True)
    recs = load_cw()
    wrong_v1 = [r for r in recs if r["bucket"] == "wrong"]

    # v2 rebucket via pid-join to the source rollouts (completion_text + problem)
    src = {r["problem_id"]: r for r in
           (json.loads(l) for l in open(os.path.join(DATA, SRC)))}
    n_recovered = n_nojoin = 0

    def v2_wrong(r):
        nonlocal n_recovered, n_nojoin
        s = src.get(r["problem_id"])
        if s is None:
            n_nojoin += 1
            return True                       # cannot rebucket -> keep as wrong
        b, _ = bucket_rollout_v2(s["completion_text"], r["gt_answer"],
                                 s.get("problem", ""))
        if b != "wrong":
            n_recovered += 1
        return b == "wrong"

    wrong_v2 = [r for r in wrong_v1 if v2_wrong(r)]

    L = []
    A = L.append
    A("# Gate C re-review on v2 wrong bucket (task 3d)\n")
    A("Same diag2x2 corruption data; wrong bucket re-defined by verifier v2 "
      "(format-recovered pseudo removed via pid-join to source rollouts). LIFT = "
      "(answer-span JSD-mass frac)/(answer-span token frac); lift>1 = correction "
      "signal denser on the answer span than uniform. Corruption-null & "
      "no-reasoning excluded.\n")
    A(f"- wrong (v1): {len(wrong_v1)} | wrong (v2): {len(wrong_v2)} "
      f"(v2 removed {n_recovered} format-recovered; {n_nojoin} kept unjoined)\n")

    # ---- corruption-null occupancy (gate D read) v1 vs v2 ----
    A("## Corruption-null & quality floor (v1 vs v2 wrong)\n")
    A("| metric | v1 wrong | v2 wrong |")
    A("|---|---|---|")
    for label, sub in [("null occupancy", None), ("corruption mass", "mass"),
                       ("big tokens (>0.05)", "big")]:
        if label == "null occupancy":
            a1 = gate_D_null(wrong_v1)
            a2 = gate_D_null(wrong_v2)
            A(f"| {label} | {a1[0]}/{a1[1]} ({100*a1[0]/a1[1]:.1f}%) | "
              f"{a2[0]}/{a2[1]} ({100*a2[0]/a2[1]:.1f}%) |")
        elif label == "corruption mass":
            m1 = np.array([corruption_mass(r) for r in wrong_v1])
            m2 = np.array([corruption_mass(r) for r in wrong_v2])
            A(f"| {label} (nats) | {_q(m1)} | {_q(m2)} |")
        else:
            b1 = np.array([n_big_tokens(r) for r in wrong_v1])
            b2 = np.array([n_big_tokens(r) for r in wrong_v2])
            A(f"| {label} | {_q(b1)} | {_q(b2)} |")
    A("")

    # ---- LIFT both axes, v1 vs v2 ----
    A("## Correction-signal LIFT — both corruption axes (v1 vs v2 wrong)\n")
    A("| axis | bucket | no-reason excl | lift |")
    A("|---|---|---|---|")
    lifts = {}
    for axis in ("studentwrong", "irrelevant"):
        for name, sub in [("v1", wrong_v1), ("v2", wrong_v2)]:
            lift, nu, nnr = gate_C_axis(sub, axis)
            lifts[(axis, name)] = lift
            nr = f"{nnr}/{nu} ({100*nnr/nu if nu else 0:.1f}%)"
            A(f"| {axis} | {name} | {nr} | {_q(lift)} |")
    A("")
    A("Read: v2 removes format-recovered pseudo (student math correct) from the "
      "wrong bucket. If the v2 lift is >= v1 lift, the remaining true-wrong "
      "rollouts carry a correction signal at least as concentrated on the answer "
      "span — i.e. the pseudo cases were diluting, not driving, gate C.\n")

    # ---- figure: v1 vs v2 lift, studentwrong axis ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, axis in zip(axes, ("studentwrong", "irrelevant")):
        for name, col in [("v1", "#9467bd"), ("v2", "#2ca02c")]:
            lv = lifts[(axis, name)]
            if len(lv):
                ax.hist(np.clip(lv, 0, 40), bins=20, alpha=0.55,
                        label=f"{name} (n={len(lv)}, med={np.median(lv):.2f})",
                        color=col)
        ax.axvline(1.0, ls="--", color="k", lw=1, label="lift=1")
        ax.set_title(f"Gate C lift — {axis} corruption")
        ax.set_xlabel("lift (answer-span JSD-mass frac / token frac)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS, "gate_c_rereview.png"), dpi=120)
    plt.close(fig)

    out = os.path.join(ANALYSIS, "gate_c_rereview.md")
    with open(out, "w") as f:
        f.write("\n".join(L))
    print("wrote", out)
    print(f"wrong v1={len(wrong_v1)} v2={len(wrong_v2)} recovered={n_recovered}")
    for axis in ("studentwrong", "irrelevant"):
        print(f"[{axis}] v1 lift:", _q(lifts[(axis, 'v1')]))
        print(f"[{axis}] v2 lift:", _q(lifts[(axis, 'v2')]))


if __name__ == "__main__":
    main()
