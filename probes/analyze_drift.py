"""Task 2 aggregation: drift share over training -> drift_over_training.md + png.

drift share_k = sum(drift_k) / (sum(drift_k) + sum(teach)), per token category and
overall. Overlays training loss (parsed from the run log) and the AIME24 eval
points as a performance reference. Reads probes/data/drift_shard*.jsonl. CPU.
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
ANALYSIS = os.path.join(HERE, "analysis")
MD = os.path.join(ANALYSIS, "drift_over_training.md")
STEPS = [25, 50, 75, 100, 125, 150]
CAT4 = ["math", "style", "other", "structural"]
BIG = 0.05
_STRUCT_RE = re.compile(r"\*\*|-{3,}|#{2,}|\bStep\b|\bSection\b|\bPart\b|\bChapter\b|\bCase\b", re.I)
LOG_GLOB = os.path.expanduser("~/projects/OPSD/pbs/logs/opsd_repro_1b.*.log")
EVAL_AIME24 = {0: 49.2, 50: 52.5, 100: 55.0}   # avg@12, from results/repro_eval


def _cat(cats, toks):
    return ["structural" if _STRUCT_RE.search(t) else c for c, t in zip(cats, toks)]


def parse_loss():
    files = sorted(glob.glob(LOG_GLOB))
    if not files:
        return [], []
    txt = open(files[-1], encoding="utf-8", errors="ignore").read()
    losses = [float(m) for m in re.findall(r"'loss': (-?\d+\.\d+)", txt)]
    steps = [2 * (i + 1) for i in range(len(losses))]   # logging_steps=2
    return steps, losses


def main():
    recs = []
    for p in sorted(glob.glob(os.path.join(DATA, "drift_shard*.jsonl"))):
        recs += [json.loads(l) for l in open(p, encoding="utf-8")]
    print(f"loaded {len(recs)} rollouts")

    # accumulate per (step, category) drift and teach mass; overall; hi-drift frac
    share = {k: {c: [0.0, 0.0] for c in CAT4 + ["all"]} for k in STEPS}  # [drift, teach]
    total_dist = {k: [] for k in STEPS}
    hidrift_frac = {k: [] for k in STEPS}
    for r in recs:
        cats = _cat(r["categories"], r["tokens"])
        teach = np.asarray(r["teach"], float)
        for k in STEPS:
            dk = np.asarray(r["per_k"][str(k)]["drift"], float)
            tk = np.asarray(r["per_k"][str(k)]["total"], float)
            total_dist[k].append(float(tk.sum()))
            hidrift_frac[k].append(float((dk > BIG).mean()))
            share[k]["all"][0] += dk.sum()
            share[k]["all"][1] += teach.sum()
            for c in CAT4:
                m = np.array([cc == c for cc in cats])
                if m.any():
                    share[k][c][0] += dk[m].sum()
                    share[k][c][1] += teach[m].sum()

    def frac(k, c):
        d, t = share[k][c]
        return d / (d + t) if (d + t) > 0 else np.nan

    # figure
    fig, ax1 = plt.subplots(figsize=(9, 5))
    colors = {"all": "#000000", "math": "#1f77b4", "style": "#ff7f0e",
              "other": "#2ca02c", "structural": "#9467bd"}
    for c in ["all"] + CAT4:
        ax1.plot(STEPS, [100 * frac(k, c) for k in STEPS], "-o",
                 color=colors[c], label=f"drift share: {c}",
                 lw=2 if c == "all" else 1)
    ax1.set_xlabel("training step")
    ax1.set_ylabel("drift share = drift / (drift + teach)  (%)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=8)
    ax2 = ax1.twinx()
    ls, lv = parse_loss()
    if ls:
        ax2.plot(ls, lv, color="#7f7f7f", alpha=0.5, lw=1, label="train loss")
    ev_s = sorted(EVAL_AIME24)
    ax2.plot(ev_s, [EVAL_AIME24[s] for s in ev_s], "D--", color="#d62728",
             label="AIME24 avg@12")
    ax2.set_ylabel("train loss  /  AIME24 avg@12 (%)")
    ax2.legend(loc="upper right", fontsize=8)
    ax1.set_title("Drift share vs training step (Qwen3-1.7B OPSD)")
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS, "drift_over_training.png"), dpi=120)
    plt.close(fig)

    # slope / monotonicity read
    overall = [100 * frac(k, "all") for k in STEPS]
    diffs = np.diff(overall)
    monotone = bool(np.all(diffs > -0.5))
    max_slope_idx = int(np.argmax(diffs))     # segment STEPS[i]->STEPS[i+1]

    L = []
    A = L.append
    A("# Drift share over training (task 2)\n")
    A("drift_k = JSD(S_k, S0), teach = JSD(T_S, S0) (k-independent), "
      "total_k = JSD(T_S, S_k), on the fixed 4096 diagnostic rollouts "
      f"(n={len(recs)}). drift share = Σdrift / (Σdrift + Σteach). "
      "See `probes/drift_scan.py`.\n")
    A("## Drift share by step (%)\n")
    A("| step | all | math | style | other | structural | total_k mass (med) | hi-drift tok frac (med) |")
    A("|--:|--:|--:|--:|--:|--:|--:|--:|")
    for k in STEPS:
        A(f"| {k} | {100*frac(k,'all'):.1f} | {100*frac(k,'math'):.1f} | "
          f"{100*frac(k,'style'):.1f} | {100*frac(k,'other'):.1f} | "
          f"{100*frac(k,'structural'):.1f} | {np.median(total_dist[k]):.3f} | "
          f"{np.median(hidrift_frac[k]):.3f} |")
    A("")
    A("## Read\n")
    A(f"- overall drift share: {[round(x,1) for x in overall]} (steps {STEPS})")
    A(f"- monotone non-decreasing (tol 0.5pp): **{monotone}**")
    A(f"- steepest rise segment: **{STEPS[max_slope_idx]}→{STEPS[max_slope_idx+1]}** "
      f"(+{diffs[max_slope_idx]:.1f}pp)")
    A(f"- performance plateau (AIME24): base {EVAL_AIME24[0]} → step50 "
      f"{EVAL_AIME24[50]} → step100 {EVAL_AIME24[100]}; overlay in "
      f"`drift_over_training.png`.")
    A("- Mechanism read: if drift share rises monotonically and its steepest "
      "segment sits around 75–100 (the performance plateau), the privileged "
      "teaching signal is being progressively swamped by drift (target -> "
      "KL-to-init). Reported as-is; not forced.\n")
    with open(MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"wrote {MD}")
    print("overall drift share:", [round(x, 1) for x in overall])
    print(f"monotone={monotone} steepest={STEPS[max_slope_idx]}->{STEPS[max_slope_idx+1]}")


if __name__ == "__main__":
    main()
