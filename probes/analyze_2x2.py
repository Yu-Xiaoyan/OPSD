"""Aggregate the 2x2 corruption diagnostic -> probes/analysis/diag_2x2.md + figs.

Reads probes/data/diag2x2_shard*.jsonl (correct/wrong) and diag_truncated.jsonl.
Produces the gate-A / gate-C readouts, corruption stability, layered position
curves, and the truncated-bucket section. CPU only; pure aggregation.
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
MD = os.path.join(ANALYSIS, "diag_2x2.md")
N_BINS = 20
TOPQ = 90            # top-10% = >= 90th percentile
LAYERS = ["math", "style", "other", "answer_span"]
CAT4 = ["math", "style", "other", "structural"]
MASS_MIN = 0.5      # corruption total mass (nats) below this -> corruption-null
BIG_TOK = 0.05      # per-token corruption threshold for the ">0.05 token" count
_STRUCT_RE = re.compile(r"\*\*|-{3,}|#{2,}|\bStep\b|\bSection\b|\bPart\b|\bChapter\b|\bCase\b", re.I)


def corruption_mass(r):
    return float(np.asarray(r["jsd_corruption"], float).mean(0).sum())


def n_big_tokens(r):
    return int((np.asarray(r["jsd_corruption"], float).mean(0) > BIG_TOK).sum())


def is_null(r):
    return corruption_mass(r) < MASS_MIN


def relabel_struct(cats, tokens):
    return ["structural" if _STRUCT_RE.search(t) else c for c, t in zip(cats, tokens)]


def last_span_start(span):
    idxs = [i for i, v in enumerate(span) if v]
    if not idxs:
        return None
    s = set(idxs)
    start = idxs[-1]
    while start - 1 in s:
        start -= 1
    return start


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
        if is_null(r):
            continue
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
    """insensitive-share of teacher-student divergence, plus a category breakdown
    (math/style/other/structural) and a privilege x drift 2x2 mass table.
    Excludes corruption-null rollouts."""
    sens_ts, insens_ts, insens_frac = [], [], []
    cat_mass = {c: 0.0 for c in CAT4}          # insensitive jsd_ts mass by category
    pd_mass = {"sens_hidrift": 0.0, "sens_lodrift": 0.0,
               "insens_hidrift": 0.0, "insens_lodrift": 0.0}
    used = 0
    for r in correct:
        if is_null(r):
            continue
        jcm = jsd_corr_mean(r)
        jts = _arr(r, "jsd_teacher_student")
        if len(jcm) < 10:
            continue
        thr = np.percentile(jcm, TOPQ)
        sens = jcm >= thr
        if sens.sum() == 0 or (~sens).sum() == 0:
            continue
        used += 1
        sens_ts.append(jts[sens].mean())
        insens_ts.append(jts[~sens].mean())
        tot = jts.sum()
        if tot > 0:
            insens_frac.append(jts[~sens].sum() / tot)
        cats = relabel_struct(r["categories"], r["tokens"])
        for i in np.where(~sens)[0]:
            c = cats[i] if cats[i] in CAT4 else "other"
            cat_mass[c] += jts[i]
        if "jsd_drift" in r:
            jdr = _arr(r, "jsd_drift")
            hi = jdr >= np.percentile(jdr, TOPQ)
            pd_mass["sens_hidrift"] += jts[sens & hi].sum()
            pd_mass["sens_lodrift"] += jts[sens & ~hi].sum()
            pd_mass["insens_hidrift"] += jts[~sens & hi].sum()
            pd_mass["insens_lodrift"] += jts[~sens & ~hi].sum()
    return {"sens_ts": np.array(sens_ts), "insens_ts": np.array(insens_ts),
            "insens_frac": np.array(insens_frac), "cat_mass": cat_mass,
            "pd_mass": pd_mass, "used": used}


def gate_D_null(recs):
    n = len(recs)
    null = sum(1 for r in recs if is_null(r))
    return null, n


def commit_point(correct):
    """Distance (tokens) of the top-5 corruption-JSD tokens from the start of the
    last answer segment. Negative => before the answer segment. Excludes null."""
    dists = []
    for r in correct:
        if is_null(r):
            continue
        astart = last_span_start(r["answer_span"])
        if astart is None:
            continue
        jcm = jsd_corr_mean(r)
        for t in np.argsort(jcm)[-5:]:
            dists.append(int(t) - astart)
    return np.array(dists)


def gate_A_crosstab(correct):
    """Full privilege x drift x category mass table over teacher-student
    divergence (correct, non-null). rows = category, cols = sens/insens x
    hi/lo-drift. Values are fractions of total jsd_ts mass."""
    cols = ["sens_hi", "sens_lo", "insens_hi", "insens_lo"]
    tab = {c: {col: 0.0 for col in cols} for c in CAT4}
    total = 0.0
    for r in correct:
        if is_null(r) or "jsd_drift" not in r:
            continue
        jcm = jsd_corr_mean(r)
        jts = _arr(r, "jsd_teacher_student")
        jdr = _arr(r, "jsd_drift")
        if len(jcm) < 10:
            continue
        sens = jcm >= np.percentile(jcm, TOPQ)
        hi = jdr >= np.percentile(jdr, TOPQ)
        cats = relabel_struct(r["categories"], r["tokens"])
        for i in range(len(jcm)):
            c = cats[i] if cats[i] in CAT4 else "other"
            col = ("sens_" if sens[i] else "insens_") + ("hi" if hi[i] else "lo")
            tab[c][col] += jts[i]
            total += jts[i]
    return tab, total, cols


# ---------------------------------------------------------------------------
# (d) gate C  (wrong x student-wrong): correction-signal concentration on answer
# ---------------------------------------------------------------------------
REASONING_MIN = 20   # rollouts with fewer non-answer tokens => no-reasoning bucket


def gate_C(wrong):
    """lift = (answer-span JSD mass fraction) / (answer-span token fraction).

    lift>1 => correction signal is DENSER on the answer span than uniform.
    Rollouts whose reasoning body (non-answer tokens) < REASONING_MIN are moved
    to a no-reasoning sub-bucket and excluded from gate C.
    """
    lift, conc, spanfrac = [], [], []
    n_usable = n_noreason = 0
    for r in wrong:
        if is_null(r):
            continue
        sw = r.get("jsd_corruption_studentwrong")
        if sw is None:
            continue
        sw = np.asarray(sw, dtype=float)
        span = np.asarray(r["answer_span"], dtype=bool)
        tot = sw.sum()
        if tot <= 0 or span.sum() == 0:
            continue
        n_usable += 1
        if (r["T"] - int(span.sum())) < REASONING_MIN:
            n_noreason += 1
            continue
        c = sw[span].sum() / tot
        sf = span.sum() / r["T"]
        conc.append(c)
        spanfrac.append(sf)
        lift.append(c / sf if sf > 0 else np.nan)
    return {"lift": np.array([x for x in lift if not np.isnan(x)]),
            "conc": np.array(conc), "spanfrac": np.array(spanfrac),
            "n_usable": n_usable, "n_noreason": n_noreason,
            "n_gateC": len(conc)}


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
    gA = gate_A(correct)
    gc = gate_C(wrong)
    cp = commit_point(correct)
    tab, tab_total, tab_cols = gate_A_crosstab(correct)
    nc_null, nc = gate_D_null(correct)
    nw_null, nw = gate_D_null(wrong)
    mass_c = np.array([corruption_mass(r) for r in correct])
    mass_w = np.array([corruption_mass(r) for r in wrong])
    big_c = np.array([n_big_tokens(r) for r in correct])
    big_w = np.array([n_big_tokens(r) for r in wrong])

    # gate-A figure
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(gA["insens_ts"], bins=20, alpha=0.6, label="insensitive tokens", color="#1f77b4")
    ax.hist(gA["sens_ts"], bins=20, alpha=0.6, label="corruption-sensitive tokens", color="#d62728")
    ax.set_title("Gate A: teacher-student JSD, sensitive vs insensitive (non-null)")
    ax.set_xlabel("mean JSD(T_S, S) per rollout (nats)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS, "diag_2x2_gateA.png"), dpi=120)
    plt.close(fig)
    # gate-C figure (lift)
    fig, ax = plt.subplots(figsize=(7, 4))
    if len(gc["lift"]):
        ax.hist(np.clip(gc["lift"], 0, 40), bins=20, color="#2ca02c")
    ax.axvline(1.0, ls="--", color="k", lw=1, label="lift=1 (uniform)")
    ax.set_title("Gate C: correction-signal lift on answer span (wrong x student-wrong)")
    ax.set_xlabel("lift = (answer-span JSD mass frac) / (answer-span token frac)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS, "diag_2x2_gateC.png"), dpi=120)
    plt.close(fig)
    # commit-point figure
    fig, ax = plt.subplots(figsize=(7, 4))
    if len(cp):
        ax.hist(np.clip(cp, -200, 200), bins=40, color="#9467bd")
    ax.axvline(0, ls="--", color="k", lw=1, label="answer-segment start")
    ax.set_title("Commit point: top-5 corruption tokens vs answer-segment start (correct)")
    ax.set_xlabel("token distance (negative = before the answer segment)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS, "diag_2x2_commit.png"), dpi=120)
    plt.close(fig)

    trunc = analyze_truncated()

    L = []
    A = L.append
    A("# 2x2 corruption diagnostic (stage 1)\n")
    A("Per-token teacher-corruption sensitivity `JSD(T_S, T_S̃)`, teacher-student "
      "divergence `JSD(T_S, S)`, and LoRA drift `JSD(S, S0)` (S0 = base on the "
      "student prompt) on ckpt-50 rollouts (`probes/run_2x2.py`). correct/wrong "
      "from the 4096 collection.\n")
    A(f"- correct rollouts: {len(correct)} | wrong rollouts: {len(wrong)}\n")

    A("## Quality floor & corruption-null (gate D read)\n")
    A(f"- corruption total mass (nats), correct: {_q(mass_c)}")
    A(f"- corruption total mass (nats), wrong: {_q(mass_w)}")
    A(f"- tokens with per-token corruption > {BIG_TOK}, correct: {_q(big_c)}")
    A(f"- tokens with per-token corruption > {BIG_TOK}, wrong: {_q(big_w)}")
    A(f"- **corruption-null** (total mass < {MASS_MIN} nats), **correct**: "
      f"{nc_null}/{nc} ({100*nc_null/nc if nc else 0:.1f}%)")
    A(f"- **corruption-null**, **wrong**: {nw_null}/{nw} "
      f"({100*nw_null/nw if nw else 0:.1f}%)")
    A("  - Gate D read: a high corruption-null fraction = the teacher barely "
      "reacts to the answer being corrupted, i.e. the privilege/leakage axis is "
      "weak at this scale. All concentration/lift metrics below EXCLUDE null "
      "rollouts.\n")

    A("## (b) corruption stability (3 corrupted versions, non-null)\n")
    A(f"- top-10% token-set **Jaccard** (pairwise): {_q(jac)}")
    A(f"- token-level **Spearman** (pairwise jsd_corruption): {_q(spr)}\n")

    A("## (c) Gate A — divergence outside corruption-sensitive tokens (correct, non-null)\n")
    A(f"- mean JSD(T_S,S) on corruption-sensitive (top-10%) tokens: {_q(gA['sens_ts'])}")
    A(f"- mean JSD(T_S,S) on insensitive tokens: {_q(gA['insens_ts'])}")
    A(f"- **insensitive tokens' share of total teacher-student divergence**: {_q(gA['insens_frac'])}\n")
    A("### Three-way mass table: privilege x drift x category")
    A("Fraction of total teacher-student divergence mass (%, correct non-null). "
      "Columns: sensitive/insensitive (privilege) x hi/lo LoRA-drift.")
    A("| category | sens·hi-drift | sens·lo-drift | insens·hi-drift | insens·lo-drift | row |")
    A("|---|--:|--:|--:|--:|--:|")
    for c in CAT4:
        vals = [tab[c][col] for col in tab_cols]
        row = sum(vals)
        cells = " | ".join(f"{100*v/tab_total if tab_total else 0:.1f}" for v in vals)
        A(f"| {c} | {cells} | {100*row/tab_total if tab_total else 0:.1f} |")
    colsum = [sum(tab[c][col] for c in CAT4) for col in tab_cols]
    A(f"| **col** | " + " | ".join(f"{100*v/tab_total if tab_total else 0:.1f}" for v in colsum) + " | 100 |")
    A("  - Read: mass in **insensitive x any-drift** = divergence not explained "
      "by privilege (copying); mass in **hi-drift** columns = attributable to "
      "LoRA drift; the **structural** row isolates markup/section tokens.\n")

    A("## (d) Gate C — correction-signal LIFT on answer span (wrong x student-wrong, non-null)\n")
    nr_frac = (100 * gc["n_noreason"] / gc["n_usable"]) if gc["n_usable"] else 0
    A(f"- no-reasoning sub-bucket (reasoning body < {REASONING_MIN} tokens, "
      f"excluded): {gc['n_noreason']}/{gc['n_usable']} ({nr_frac:.1f}%)")
    A(f"- **lift** (answer-span JSD-mass frac / answer-span token frac), "
      f"n={gc['n_gateC']}: {_q(gc['lift'])}")
    A(f"- raw concentration (answer-span JSD mass / total): {_q(gc['conc'])}")
    A(f"- answer-span token fraction: {_q(gc['spanfrac'])}")
    A("  - lift>1 = correction signal DENSER on the answer span than uniform. "
      "See `samples/prefix_failure_micro.html` (pid=19) for the token-level "
      "mechanism.\n")

    A("## Commit-point hypothesis (correct, non-null)\n")
    if len(cp):
        within = 100 * np.mean((cp >= -30) & (cp <= 0))
        before = 100 * np.mean(cp < 0)
        A(f"- top-5 corruption-token distance from the answer-segment start "
          f"(tokens): {_q(cp)}")
        A(f"- fraction in a [-30, 0] pre-answer window: {within:.1f}%; "
          f"fraction strictly before the answer segment: {before:.1f}%")
        A("  - If the top corruption tokens cluster just before the final answer "
          "segment, the teacher 'commits' to the answer in a small window — the "
          "commit-point mechanism. See `diag_2x2_commit.png` and 3 HTML stamps "
          "`samples/commit_point_*.html`.\n")

    A("## (a)/(e) position curves\n")
    A("- `diag_2x2_position_correct.png` (top-left), "
      "`diag_2x2_position_wrong.png` (bottom-left): corruption sensitivity by "
      "relative position, layered by token category + answer span.\n")

    if trunc:
        A("## Truncated bucket (1024 collection, no corruption)\n")
        A(f"- n={trunc['n']} truncated rollouts; JSD(T_S,S) position curve + V(t) "
          f"bundle in `diag_truncated.png`.")
        A(f"- V(end): mean={trunc['v_end_mean']:.3f} median={trunc['v_end_median']:.3f} "
          f"nats — training signal on the dominant (69%@1024) truncated bucket "
          f"the verifier cannot score.")
        A("- HTML stamps mark the most-negative delta_V checkpoint segment "
          "(delta_V is broadcast at checkpoint-segment granularity).\n")

    with open(MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"wrote {MD}")
    print(f"corruption-null: correct {nc_null}/{nc}, wrong {nw_null}/{nw}")
    print("gateA insensitive-share:", _q(gA["insens_frac"]))
    print(f"gateC no-reasoning: {gc['n_noreason']}/{gc['n_usable']} | lift:", _q(gc["lift"]))
    print("commit-point dist:", _q(cp))
    print("stability Jaccard:", _q(jac))


if __name__ == "__main__":
    main()
