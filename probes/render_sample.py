"""Card 4: single-rollout HTML visualization of the 2x2 diagnostic.

Renders a rollout's tokens colored by a channel (jsd_corruption mean /
jsd_teacher_student / delta_V), hover shows value + category, answer-span tokens
are boxed. Auto-picks 3 representative rollouts per cell (highest / median /
lowest answer-span concentration) and writes standalone HTML to
probes/analysis/samples/. CPU only.
"""
from __future__ import annotations

import glob
import html
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "analysis", "samples")


def _tok_text(t):
    return t.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\n")


def _last_span_start(span):
    idxs = [i for i, v in enumerate(span) if v]
    if not idxs:
        return None
    s = set(idxs)
    start = idxs[-1]
    while start - 1 in s:
        start -= 1
    return start


def _corr_mass(r):
    return float(np.asarray(r["jsd_corruption"], float).mean(0).sum())


def _delta_v_per_token(rec, T):
    pos = rec.get("V_positions")
    vals = rec.get("V_values")
    if not pos or not vals or len(pos) < 2:
        return None
    dv = np.zeros(T)
    prev_p, prev_v = pos[0], vals[0]
    for p, v in zip(pos[1:], vals[1:]):
        lo, hi = min(prev_p, T), min(p, T)
        dv[lo:hi] = v - prev_v
        prev_p, prev_v = p, v
    return dv.tolist()


def _channel_values(rec, T):
    ch = {}
    if "jsd_corruption" in rec and rec["jsd_corruption"]:
        ch["jsd_corruption"] = np.asarray(rec["jsd_corruption"], float).mean(0).tolist()
    if rec.get("jsd_corruption_studentwrong"):
        ch["jsd_corruption_studentwrong"] = rec["jsd_corruption_studentwrong"]
    if "jsd_teacher_student" in rec:
        ch["jsd_teacher_student"] = rec["jsd_teacher_student"]
    dv = _delta_v_per_token(rec, T)
    if dv is not None:
        ch["delta_V"] = dv
    return ch


def _color(v, vmax, vmin=0.0):
    if vmax <= vmin:
        a = 0.0
    else:
        a = (v - vmin) / (vmax - vmin)
    a = max(0.0, min(1.0, a))
    if v < 0:                      # diverging for delta_V (blue negative)
        return f"rgba(31,119,180,{min(1.0, abs(a)):.3f})"
    return f"rgba(214,39,40,{a:.3f})"


def render_html(rec, path, title, note=""):
    toks = rec["tokens"]
    cats = rec["categories"]
    span = rec["answer_span"]
    T = len(toks)
    channels = _channel_values(rec, T)
    parts = [f"<h2>{html.escape(title)}</h2>"]
    if note:
        parts.append(f"<p style='background:#fffbdd;border:1px solid #e0d060;"
                     f"padding:8px'><b>Note:</b> {html.escape(note)}</p>")
    parts += [f"<p><b>bucket</b>: {rec.get('bucket','truncated')} | "
              f"<b>gt</b>: {html.escape(str(rec.get('gt_answer')))} | "
              f"<b>student</b>: {html.escape(str(rec.get('student_answer')))} | "
              f"<b>T</b>: {T}</p>",
             "<style>.tok{padding:1px 0;border-radius:2px}"
             ".ans{outline:2px solid #111;outline-offset:1px}"
             ".dvdrop{outline:3px dashed #1f77b4;outline-offset:1px}"
             ".seg{font-family:monospace;line-height:2.1;white-space:pre-wrap;"
             "border:1px solid #ccc;padding:10px;margin:6px 0}</style>"]
    for name, vals in channels.items():
        v = np.asarray(vals, float)
        vmax = float(np.nanmax(np.abs(v))) if len(v) else 1.0
        vmin = float(np.nanmin(v)) if len(v) else 0.0
        extra = (" (broadcast per checkpoint segment; dashed box = "
                 "most-negative delta_V segment)") if name == "delta_V" else ""
        parts.append(f"<h3>channel: {name} (max|v|={vmax:.3f}){extra}</h3><div class='seg'>")
        for t in range(T):
            cls = "tok ans" if span[t] else "tok"
            if name == "delta_V" and v[t] == vmin and vmin < 0:
                cls += " dvdrop"
            col = _color(v[t], vmax)
            title_attr = html.escape(f"{toks[t]} | {cats[t]} | {name}={v[t]:.4f}")
            parts.append(
                f"<span class='{cls}' style='background:{col}' "
                f"title='{title_attr}'>{html.escape(_tok_text(toks[t]))}</span>")
        parts.append("</div>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("<!doctype html><meta charset='utf-8'>" + "".join(parts))


def _concentration(rec, value_key):
    if value_key == "jsd_corruption":
        v = np.asarray(rec["jsd_corruption"], float).mean(0)
    else:
        v = rec.get(value_key)
        if v is None:
            return None
        v = np.asarray(v, float)
    span = np.asarray(rec["answer_span"], bool)
    tot = v.sum()
    if tot <= 0 or span.sum() == 0:
        return None
    return float(v[span].sum() / tot)


def pick3(recs, score_fn):
    scored = [(s, r) for r in recs if (s := score_fn(r)) is not None]
    if not scored:
        return []
    scored.sort(key=lambda x: x[0])
    return [("hi", *scored[-1]), ("mid", *scored[len(scored) // 2]),
            ("lo", *scored[0])]


def main():
    os.makedirs(OUT, exist_ok=True)
    cw = []
    for p in sorted(glob.glob(os.path.join(DATA, "diag2x2_shard*.jsonl"))):
        cw += [json.loads(l) for l in open(p, encoding="utf-8")]
    correct = [r for r in cw if r["bucket"] == "correct"]
    wrong = [r for r in cw if r["bucket"] == "wrong"]
    trunc = []
    tp = os.path.join(DATA, "diag_truncated.jsonl")
    if os.path.exists(tp):
        trunc = [json.loads(l) for l in open(tp, encoding="utf-8")]

    def conc_fn(vkey):
        return lambda r: _concentration(r, vkey)

    def reasoning_ok(r):    # exclude no-reasoning rollouts (reasoning body < 20)
        return (r["T"] - sum(r["answer_span"])) >= 20

    vend = lambda r: r["V_values"][-1] if r.get("V_values") else None

    # 5 curated samples: cross-cell pid-dedup, no-reasoning excluded for answer cells
    slots = [
        ("gateA_high_sensitivity_on_answer", correct, conc_fn("jsd_corruption"),
         "hi", True,
         "Gate A extreme: corruption sensitivity coincides with the answer span."),
        ("gateA_competence_spread", correct, conc_fn("jsd_corruption"), "lo", True,
         "Gate A typical: teacher-student divergence is spread over the reasoning "
         "body, not the (corruption-insensitive) answer — competence-driven."),
        ("gateC_correction_on_answer", wrong,
         conc_fn("jsd_corruption_studentwrong"), "hi", True,
         "Gate C: student-wrong correction signal concentrated on the answer span "
         "(high lift)."),
        ("wrong_irrelevant_corruption", wrong, conc_fn("jsd_corruption"), "hi", True,
         "Bottom-left: irrelevant corruption on a wrong rollout (descriptive)."),
        ("truncated_unsupported", trunc, vend, "lo", False,
         "Truncated: V(t) never supports the answer; delta_V channel shown."),
    ]
    used, written = set(), []
    for name, recs, score_fn, tag, excl_nr, note in slots:
        cands = [r for r in recs if r["problem_id"] not in used
                 and (not excl_nr or (reasoning_ok(r) and _corr_mass(r) >= 0.5))]
        chosen = next((x for x in pick3(cands, score_fn) if x[0] == tag), None)
        if not chosen:
            continue
        _, score, rec = chosen
        used.add(rec["problem_id"])
        path = os.path.join(OUT, f"{name}.html")
        render_html(rec, path, f"{name} [score={score:.3f}] pid={rec['problem_id']}",
                    note=note)
        written.append(name)
        print(f"{name:38}: pid={rec['problem_id']} score={score:.3f}")

    # pid=19 mechanism stamp
    p19 = next((r for r in wrong if r["problem_id"] == 19), None)
    if p19:
        note19 = ("Mechanism micro-example (prefix failure): the student-wrong "
                  "corruption JSD concentrates on the answer's first fork tokens "
                  "and drops to ~0 afterwards — a token-level demonstration that "
                  "the correction force does NOT propagate back through the wrong "
                  "prefix. Look at the 'jsd_corruption_studentwrong' channel.")
        path = os.path.join(OUT, "prefix_failure_micro.html")
        render_html(p19, path,
                    f"prefix_failure_micro pid=19 gt={p19['gt_answer']}", note=note19)
        print(f"stamp: prefix_failure_micro.html (pid=19, "
              f"gt={p19['gt_answer']!r} student={p19['student_answer']!r})")

    # commit-point stamps: correct, non-null, top-5 corruption just before answer
    cand = []
    for r in correct:
        if _corr_mass(r) < 0.5:
            continue
        astart = _last_span_start(r["answer_span"])
        if astart is None:
            continue
        jcm = np.asarray(r["jsd_corruption"], float).mean(0)
        top5 = np.argsort(jcm)[-5:]
        d = float(np.median([int(t) - astart for t in top5]))
        if d <= 5 and r["problem_id"] not in used:
            cand.append((abs(d), d, r))
    cand.sort(key=lambda x: x[0])
    for i, (_, d, rec) in enumerate(cand[:3], 1):
        note = (f"Commit-point stamp: the top-5 corruption tokens sit a median "
                f"{d:.0f} tokens from the answer-segment start — the teacher "
                f"commits to the answer in a small pre-answer window.")
        render_html(rec, os.path.join(OUT, f"commit_point_{i}.html"),
                    f"commit_point_{i} pid={rec['problem_id']} median_dist={d:.0f}",
                    note=note)
        print(f"commit_point_{i}: pid={rec['problem_id']} median_dist={d:.0f}")

    print(f"\nwrote {len(written)} curated + stamps to {OUT}")


if __name__ == "__main__":
    main()
