"""Aggregate the 2x2 corruption diagnostic -> probes/analysis/diag_2x2.md + figs.

Reads probes/data/diag2x2_shard*.jsonl (correct/wrong) and diag_truncated.jsonl.
Produces the gate-A / gate-C readouts, corruption stability, layered position
curves, and the truncated-bucket section. CPU only; pure aggregation.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
ANALYSIS = os.path.join(HERE, "analysis")
MD = os.path.join(ANALYSIS, "diag_2x2.md")
N_BINS = 20
TOPQ = 90            # top-10% = >= 90th percentile
LAYERS = ["math", "style", "other", "answer_span"]


def load_cw():
    recs = []
    for p in sorted(glob.glob(os.path.join(DATA, "diag2x2_shard*.jsonl"))):
        recs += [json.loads(l) for l in open(p, encoding="utf-8")]
    return recs


def _arr(r, key):
    return np.asarray(r[key], dtype=float)


def jsd_corr_mean(r):
    return np.asarray(r["jsd_corruption"], dtype=float).mean(0)   # [T]


# ---------------------------------------------------------------------------
# (a)/(e) layered position curves
# ---------------------------------------------------------------------------
def position_layers(recs, value_fn):
    """Per layer -> (centers, mean[20]) binned by relative position."""
    bins = {L: [[] for _ in range(N_BINS)] for L in LAYERS}
    for r in recs:
        val = value_fn(r)
        rel = _arr(r, "rel_pos")
        cats = r["categories"]
        span = r["answer_span"]
        idx = np.minimum((rel * N_BINS).astype(int), N_BINS - 1)
        for t in range(len(val)):
            b = idx[t]
            if span[t]:
                bins["answer_span"][b].append(val[t])
            c = cats[t] if cats[t] in ("math", "style", "other") else "other"
            bins[c][b].append(val[t])
    centers = (np.arange(N_BINS) + 0.5) / N_BINS
    out = {}
    for L in LAYERS:
        m = np.array([np.mean(b) if b else np.nan for b in bins[L]])
        out[L] = m
    return centers, out


def plot_position(recs, title, path):
    centers, layers = position_layers(recs, jsd_corr_mean)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    styles = {"math": "-o", "style": "-s", "other": "-^", "answer_span": "-D"}
    colors = {"math": "#1f77b4", "style": "#ff7f0e", "other": "#7f7f7f",
              "answer_span": "#d62728"}
    for L in LAYERS:
        ax.plot(centers, layers[L], styles[L], color=colors[L], markersize=4,
                label=L)
    ax.set_xlabel("relative position t/T")
    ax.set_ylabel("mean JSD(T_S, T_S̃)  (corruption sensitivity, nats)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# (b) corruption stability
# ---------------------------------------------------------------------------
def _spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def stability(recs):
    jac, spr = [], []
    for r in recs:
        jc = np.asarray(r["jsd_corruption"], dtype=float)   # [3, T]
        if jc.shape[0] < 2 or jc.shape[1] < 10:
            continue
        T = jc.shape[1]
        k = max(1, T // 10)
        tops = [set(np.argsort(jc[i])[-k:]) for i in range(jc.shape[0])]
        for a in range(len(tops)):
            for b in range(a + 1, len(tops)):
                u = len(tops[a] | tops[b])
                jac.append(len(tops[a] & tops[b]) / u if u else 0.0)
                spr.append(_spearman(jc[a], jc[b]))
    jac = np.array(jac)
    spr = np.array([s for s in spr if not np.isnan(s)])
    return jac, spr


# ---------------------------------------------------------------------------
# (c) gate A  (correct): divergence mass outside corruption-sensitive tokens
# ---------------------------------------------------------------------------
def gate_A(correct):
    sens_ts, insens_ts, insens_frac = [], [], []
    for r in correct:
        jcm = jsd_corr_mean(r)
        jts = _arr(r, "jsd_teacher_student")
        if len(jcm) < 10:
            continue
        thr = np.percentile(jcm, TOPQ)
        sens = jcm >= thr
        if sens.sum() == 0 or (~sens).sum() == 0:
            continue
        sens_ts.append(jts[sens].mean())
        insens_ts.append(jts[~sens].mean())
        tot = jts.sum()
        insens_frac.append(jts[~sens].sum() / tot if tot else np.nan)
    return (np.array(sens_ts), np.array(insens_ts),
            np.array([f for f in insens_frac if not np.isnan(f)]))


# ---------------------------------------------------------------------------
# (d) gate C  (wrong x student-wrong): correction-signal concentration on answer
# ---------------------------------------------------------------------------
def gate_C(wrong):
    conc = []
    for r in wrong:
        sw = r.get("jsd_corruption_studentwrong")
        if sw is None:
            continue
        sw = np.asarray(sw, dtype=float)
        span = np.asarray(r["answer_span"], dtype=bool)
        tot = sw.sum()
        if tot <= 0 or span.sum() == 0:
            continue
        conc.append(sw[span].sum() / tot)
    return np.array(conc)


def _q(a):
    if len(a) == 0:
        return "n/a"
    return (f"mean={a.mean():.4f} median={np.median(a):.4f} "
            f"q25={np.quantile(a,.25):.4f} q75={np.quantile(a,.75):.4f} n={len(a)}")


# ---------------------------------------------------------------------------
# truncated bucket
# ---------------------------------------------------------------------------
def analyze_truncated():
    path = os.path.join(DATA, "diag_truncated.jsonl")
    if not os.path.exists(path):
        return None
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    if not recs:
        return None
    # jsd_ts position curve
    centers, layers = position_layers(recs, lambda r: _arr(r, "jsd_teacher_student"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for L in LAYERS:
        ax1.plot(centers, layers[L], "-o", markersize=3, label=L)
    ax1.set_title("truncated: JSD(T_S, S) by position")
    ax1.set_xlabel("relative position t/T")
    ax1.set_ylabel("teacher-student JSD (nats)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    # V(t) trajectory bundle (normalized position)
    for r in recs[:60]:
        pos = np.asarray(r["V_positions"], float)
        pos = pos / pos[-1] if pos[-1] else pos
        ax2.plot(pos, r["V_values"], "-", alpha=0.25, color="#1f77b4")
    ax2.set_title(f"truncated: V(t) bundle (n={min(60,len(recs))})")
    ax2.set_xlabel("relative checkpoint position")
    ax2.set_ylabel("V(t) = log p(answer | prefix)")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS, "diag_truncated.png"), dpi=120)
    plt.close(fig)
    v_end = np.array([r["V_values"][-1] for r in recs])
    return {"n": len(recs), "v_end_mean": float(v_end.mean()),
            "v_end_median": float(np.median(v_end))}


def main():
    os.makedirs(ANALYSIS, exist_ok=True)
    recs = load_cw()
    correct = [r for r in recs if r["bucket"] == "correct"]
    wrong = [r for r in recs if r["bucket"] == "wrong"]
    print(f"loaded correct={len(correct)} wrong={len(wrong)}")

    plot_position(correct, "Top-left cell: correct x irrelevant corruption",
                  os.path.join(ANALYSIS, "diag_2x2_position_correct.png"))
    plot_position(wrong, "Bottom-left cell: wrong x irrelevant corruption",
                  os.path.join(ANALYSIS, "diag_2x2_position_wrong.png"))

    jac, spr = stability(correct + wrong)
    sA, iA, fracA = gate_A(correct)
    concC = gate_C(wrong)

    # gate-A figure
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(iA, bins=20, alpha=0.6, label="insensitive tokens", color="#1f77b4")
    ax.hist(sA, bins=20, alpha=0.6, label="corruption-sensitive tokens", color="#d62728")
    ax.set_title("Gate A: teacher-student JSD, sensitive vs insensitive")
    ax.set_xlabel("mean JSD(T_S, S) per rollout (nats)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS, "diag_2x2_gateA.png"), dpi=120)
    plt.close(fig)
    # gate-C figure
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(concC, bins=20, color="#2ca02c")
    ax.set_title("Gate C: correction-signal concentration on answer span")
    ax.set_xlabel("fraction of JSD(T_S, T_S̃_studentwrong) mass on the answer span")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS, "diag_2x2_gateC.png"), dpi=120)
    plt.close(fig)

    trunc = analyze_truncated()

    L = []
    A = L.append
    A("# 2x2 corruption diagnostic (stage 1)\n")
    A("Per-token teacher-corruption sensitivity `JSD(T_S, T_S̃)` and "
      "teacher-student divergence `JSD(T_S, S)` on ckpt-50 rollouts "
      "(`probes/run_2x2.py`). correct/wrong from the 4096 collection.\n")
    A(f"- correct rollouts: {len(correct)} | wrong rollouts: {len(wrong)}\n")

    A("## (b) corruption stability (3 corrupted versions)\n")
    A(f"- top-10% token-set **Jaccard** (pairwise): {_q(jac)}")
    A(f"- token-level **Spearman** (pairwise jsd_corruption): {_q(spr)}\n")

    A("## (c) Gate A — divergence mass outside corruption-sensitive tokens (correct)\n")
    A(f"- mean JSD(T_S,S) on **corruption-sensitive** (top-10%) tokens: {_q(sA)}")
    A(f"- mean JSD(T_S,S) on **insensitive** tokens: {_q(iA)}")
    A(f"- **insensitive tokens' share of total teacher-student divergence mass**: {_q(fracA)}")
    A("  - Read: a high insensitive-share means substantial student-teacher "
      "divergence lives OUTSIDE copying positions (competence-driven), which "
      "argues for a non-trivial correct-branch treatment; a low share means "
      "divergence is mostly at corruption-sensitive (privilege) tokens.\n")

    A("## (d) Gate C — correction-signal concentration on answer span (wrong x student-wrong)\n")
    A(f"- concentration (answer-span JSD mass / total): {_q(concC)}")
    A("  - Compares against the prior observation that the correction signal "
      "sits almost entirely at the answer position. concentration→1 supports it.\n")

    A("## (a)/(e) position curves\n")
    A("- `diag_2x2_position_correct.png` (top-left), "
      "`diag_2x2_position_wrong.png` (bottom-left): corruption sensitivity by "
      "relative position, layered by token category + answer span.\n")

    if trunc:
        A("## Truncated bucket (1024 collection, no corruption)\n")
        A(f"- n={trunc['n']} truncated rollouts; teacher-student divergence "
          f"position curve + V(t) bundle in `diag_truncated.png`.")
        A(f"- V(end) on truncated: mean={trunc['v_end_mean']:.3f} "
          f"median={trunc['v_end_median']:.3f} nats — the training signal on the "
          f"dominant (69% @1024) truncated bucket, which the verifier cannot "
          f"score.\n")

    with open(MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"wrote {MD}")
    print("gateA insensitive-share:", _q(fracA))
    print("gateC concentration:", _q(concC))
    print("stability Jaccard:", _q(jac))


if __name__ == "__main__":
    main()
