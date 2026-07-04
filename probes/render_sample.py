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


def render_html(rec, path, title):
    toks = rec["tokens"]
    cats = rec["categories"]
    span = rec["answer_span"]
    T = len(toks)
    channels = _channel_values(rec, T)
    parts = [f"<h2>{html.escape(title)}</h2>",
             f"<p><b>bucket</b>: {rec.get('bucket','truncated')} | "
             f"<b>gt</b>: {html.escape(str(rec.get('gt_answer')))} | "
             f"<b>student</b>: {html.escape(str(rec.get('student_answer')))} | "
             f"<b>T</b>: {T}</p>",
             "<style>.tok{padding:1px 0;border-radius:2px}"
             ".ans{outline:2px solid #111;outline-offset:1px}"
             ".seg{font-family:monospace;line-height:2.1;white-space:pre-wrap;"
             "border:1px solid #ccc;padding:10px;margin:6px 0}</style>"]
    for name, vals in channels.items():
        v = np.asarray(vals, float)
        vmax = float(np.nanmax(np.abs(v))) if len(v) else 1.0
        parts.append(f"<h3>channel: {name} (max|v|={vmax:.3f})</h3><div class='seg'>")
        for t in range(T):
            cls = "tok ans" if span[t] else "tok"
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

    # 2x2 cells: score = answer-span concentration; truncated: score = V(end)
    def conc_fn(vkey):
        return lambda r: _concentration(r, vkey)

    cells = [
        ("top_left_correct", correct, conc_fn("jsd_corruption"), "concentration"),
        ("bottom_left_wrong", wrong, conc_fn("jsd_corruption"), "concentration"),
        ("bottom_right_wrong_studentwrong", wrong,
         conc_fn("jsd_corruption_studentwrong"), "concentration"),
        ("truncated", trunc,
         lambda r: r["V_values"][-1] if r.get("V_values") else None, "V(end)"),
    ]
    written = []
    for cell, recs, score_fn, sname in cells:
        for tag, score, rec in pick3(recs, score_fn):
            path = os.path.join(OUT, f"{cell}_{tag}.html")
            render_html(rec, path,
                        f"{cell} [{tag} {sname}={score:.3f}] pid={rec['problem_id']}")
            written.append((cell, tag, score, path))
            print(f"{cell:38} {tag}: {sname}={score:.3f} -> {os.path.basename(path)}")
    print(f"\nwrote {len(written)} HTML samples to {OUT}")


if __name__ == "__main__":
    main()
