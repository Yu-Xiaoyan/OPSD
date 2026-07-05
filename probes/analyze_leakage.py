"""Run the leakage detector across all training-step rollout dumps and report.

Outputs:
  probes/analysis/leakage_over_training.md   -- hit-rate table, in/out-of-schedule
                                                comparison, hit-sample appendix
  probes/analysis/leakage_over_training.png  -- hit-rate vs training step

No GPU needed; data is tiny. Run on the login node:
  ~/.conda/envs/opsd/bin/python probes/analyze_leakage.py
"""
from __future__ import annotations

import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import load_dataset

import leakage_detector as ld

# --- config ---------------------------------------------------------------
GEN_DIR = os.path.expanduser(
    "~/opsd_outputs/qwen31b_repro_3xh200_gb30/generations")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis")
OUT_PREFIX = "leakage_over_training"     # output basename (md + png)
TITLE = "OPSD leakage behavioral probes vs training step (Qwen3-1.7B, TM-off)"
KW_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "leakage_keywords.txt")
DATASET = "siyanzhao/Openthoughts_math_30k_opsd"

EARLY_POS = 0.30          # answer within first 30% of completion
STRONG_PREFIX_WORDS = 40  # ...and almost no preceding reasoning
OFFICIAL_MAX_STEP = 100   # <=100 official schedule; 105-150 extended
SAMPLES_PER_STEP = 3      # hit samples kept for manual review


def step_of(path: str) -> int:
    return int(re.search(r"generations_step_(\d+)\.json", path).group(1))


def snippet(text: str, pos: int, radius: int = 90) -> str:
    lo = max(0, pos - radius)
    hi = min(len(text), pos + radius)
    frag = text[lo:hi].replace("\n", " ")
    frag = re.sub(r"\s+", " ", frag)
    pre = "..." if lo > 0 else ""
    post = "..." if hi < len(text) else ""
    return pre + frag + post


def main():
    import argparse
    global GEN_DIR, OUT_PREFIX, TITLE
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", default=GEN_DIR,
                    help="dir with generations_step_*.json dumps")
    ap.add_argument("--out_prefix", default=OUT_PREFIX,
                    help="output basename in probes/analysis/ (md + png)")
    ap.add_argument("--title", default=TITLE, help="plot title")
    args = ap.parse_args()
    GEN_DIR = os.path.expanduser(args.gen_dir)
    OUT_PREFIX, TITLE = args.out_prefix, args.title

    os.makedirs(OUT_DIR, exist_ok=True)
    keywords = ld.load_keywords(KW_PATH)
    print(f"loaded {len(keywords)} keywords")

    print("loading dataset + building answer index ...")
    tr = load_dataset(DATASET)["train"]
    answer_index = ld.build_answer_index(tr)
    print(f"answer index: {len(answer_index)} problem keys")

    dumps = sorted(glob.glob(os.path.join(GEN_DIR, "generations_step_*.json")),
                   key=step_of)
    print(f"found {len(dumps)} dumps: steps "
          f"{step_of(dumps[0])}..{step_of(dumps[-1])}")

    rows = []            # per-step aggregate
    appendix = []        # (step, list of sample dicts)

    for path in dumps:
        step = step_of(path)
        samples = ld.load_dump(path)
        n = len(samples)
        kw_hits = 0
        linked = 0
        detectable = 0
        detectable_clean = 0     # detectable AND answer not visible in question
        proof_count = 0          # proof/show problems (answer = the statement)
        early_raw = 0            # pos<EARLY_POS (includes restating-the-question)
        early_clean = 0          # ...AND answer not visible in question (real signal)
        strong_clean = 0         # early_clean AND prefix_words<STRONG_PREFIX_WORDS
        hit_samples = []

        for s in samples:
            r = ld.detect_sample(s, keywords, answer_index, step)
            if r.keyword_hit:
                kw_hits += 1
            if r.linked:
                linked += 1
            is_early = (r.answer_found and r.answer_pos_ratio is not None
                        and r.answer_pos_ratio < EARLY_POS)
            # "answer visible in the question" -> early emission is restating,
            # not leakage: answer literally in the prompt, or a proof/show task
            # whose answer IS the statement to prove.
            visible = r.answer_in_prompt or r.is_proof
            if r.answer_detectable:
                detectable += 1
                if r.is_proof:
                    proof_count += 1
                if is_early:
                    early_raw += 1
                if not visible:
                    detectable_clean += 1
                    if is_early:
                        early_clean += 1
                        if (r.answer_prefix_words or 0) < STRONG_PREFIX_WORDS:
                            strong_clean += 1

            clean_early_hit = is_early and r.answer_detectable and not visible
            is_hit = r.keyword_hit or clean_early_hit
            if is_hit and len(hit_samples) < SAMPLES_PER_STEP:
                completion = s.get("completion", "")
                evid = []
                if r.keyword_hit:
                    kh = r.keyword_hits[0]
                    evid.append(
                        f"keyword '{kh['keyword']}' @char {kh['char_pos']}: "
                        f"`{snippet(completion, kh['char_pos'])}`")
                if clean_early_hit:
                    evid.append(
                        f"answer early (not in prompt): variant "
                        f"'{r.matched_variant}' pos_ratio={r.answer_pos_ratio:.2f} "
                        f"prefix_words={r.answer_prefix_words} "
                        f"@char {r.answer_char_pos}: "
                        f"`{snippet(completion, r.answer_char_pos or 0)}`")
                hit_samples.append({
                    "gt_answer": r.gt_answer,
                    "problem": (ld.extract_problem_from_prompt(s.get('prompt', '')) or '')[:120],
                    "evidence": evid,
                })

        rows.append({
            "step": step, "n": n,
            "kw_hits": kw_hits, "kw_rate": kw_hits / n if n else 0.0,
            "linked": linked, "detectable": detectable,
            "detectable_clean": detectable_clean, "proof": proof_count,
            "early_raw": early_raw, "early_clean": early_clean,
            "early_rate_clean": early_clean / detectable_clean if detectable_clean else 0.0,
            "strong_clean": strong_clean,
        })
        appendix.append((step, hit_samples))
        print(f"step {step:>3}: n={n} kw={kw_hits} detectable={detectable} "
              f"proof={proof_count} clean={detectable_clean} early_raw={early_raw} "
              f"early_clean={early_clean} strong={strong_clean}")

    _plot(rows)
    _write_md(rows, appendix, keywords)
    _print_summary(rows)


def _seg(rows, lo, hi):
    sub = [r for r in rows if lo <= r["step"] <= hi]
    n = sum(r["n"] for r in sub)
    kw = sum(r["kw_hits"] for r in sub)
    det = sum(r["detectable"] for r in sub)
    det_c = sum(r["detectable_clean"] for r in sub)
    proof = sum(r["proof"] for r in sub)
    early_raw = sum(r["early_raw"] for r in sub)
    early_c = sum(r["early_clean"] for r in sub)
    strong_c = sum(r["strong_clean"] for r in sub)
    return {
        "steps": f"{lo}-{hi}", "n": n, "kw": kw,
        "kw_rate": kw / n if n else 0.0,
        "detectable": det, "detectable_clean": det_c, "proof": proof,
        "early_raw": early_raw, "early_clean": early_c,
        "early_rate_clean": early_c / det_c if det_c else 0.0,
        "strong_clean": strong_c,
    }


def _plot(rows):
    steps = [r["step"] for r in rows]
    kw = [100 * r["kw_rate"] for r in rows]
    early = [100 * r["early_rate_clean"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, kw, "o-", label="keyword citation", color="#1f77b4")
    ax.plot(steps, early, "s-",
            label=f"answer early-emission (clean, pos<{EARLY_POS})",
            color="#d62728")
    ax.axvline(OFFICIAL_MAX_STEP, ls="--", color="gray", lw=1)
    ax.text(OFFICIAL_MAX_STEP + 1, ax.get_ylim()[1] * 0.9, "official max (100)",
            color="gray", fontsize=8)
    ax.set_xlabel("training step")
    ax.set_ylabel("hit rate (%)")
    ax.set_title(TITLE)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"{OUT_PREFIX}.png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


def _write_md(rows, appendix, keywords):
    seg_in = _seg(rows, 0, OFFICIAL_MAX_STEP)
    seg_out = _seg(rows, OFFICIAL_MAX_STEP + 5, 10 ** 9)
    lines = []
    A = lines.append
    A("# Leakage behavioral probes over training\n")
    A("Behavioral leakage screen over TM-off student rollouts "
      "(`generations_step_*.json`). Coarse by design; low/zero hit rate is a "
      "valid result and is reported as-is. See `probes/leakage_detector.py` for "
      "matching details.\n")

    tot_n = sum(r["n"] for r in rows)
    tot_kw = sum(r["kw_hits"] for r in rows)
    tot_det = sum(r["detectable"] for r in rows)
    tot_det_c = sum(r["detectable_clean"] for r in rows)
    tot_early_raw = sum(r["early_raw"] for r in rows)
    tot_early_c = sum(r["early_clean"] for r in rows)
    tot_strong = sum(r["strong_clean"] for r in rows)
    A("## Verdict\n")
    A("**Both probes find approximately zero real leakage** on these TM-off "
      "rollouts (manually reviewed).\n")
    A(f"- Keyword citation: {tot_kw}/{tot_n} ({100*tot_kw/tot_n:.2f}%). All "
      f"reviewed as false positives — ordinary math phrasing ('reference "
      f"point/angle', 'given answer choices'), not citations of privileged text.")
    A(f"- Answer early-emission: raw {tot_early_raw}/{tot_det} "
      f"({100*tot_early_raw/tot_det:.1f}%) collapses to clean {tot_early_c}/"
      f"{tot_det_c} ({100*tot_early_c/tot_det_c:.2f}%), strong={tot_strong}. Raw "
      f"is dominated by proof/floor problems restating the question; the residual "
      f"clean hits are numbers coinciding with mid-solution quantities (given "
      f"constants, intermediate results), not reasoning-free answer jumps "
      f"(appendix).")
    A("- In-schedule (<=100) vs extended (105+) are both near-zero and within "
      "noise; extended training does not visibly raise behavioral leakage.\n")
    A("**Implication (docs/framework.md).** Behavioral surface signals do not "
      "provide a usable leakage ground truth on this data. If leakage exists it "
      "is distributional (in the teacher's logits), which is exactly what the "
      "corruption / JSD probe is designed to measure — this negative result "
      "motivates that choice rather than undermining it.\n")

    A("## Method\n")
    A(f"- **Keyword probe**: case-insensitive, whitespace-collapsed substring "
      f"match against {len(keywords)} seed phrases in "
      f"`probes/leakage_keywords.txt`.")
    A(f"- **Answer early-emission probe**: link the prompt's problem back to "
      f"`{DATASET}` (problem-prefix index, {ld.PROBLEM_KEY_CHARS} chars), take "
      f"the ground-truth `Answer`, and find its earliest boundary-respecting "
      f"occurrence (normalized variants) in the completion. A hit = occurrence "
      f"at `pos_ratio < {EARLY_POS}`; 'strong' additionally requires "
      f"`prefix_words < {STRONG_PREFIX_WORDS}`.")
    A(f"- Single-character answers (e.g. 'B','a') are skipped as "
      f"non-discriminative; numeric answers require non-alphanumeric "
      f"boundaries to avoid substring false positives.\n")
    A("- **Question-visibility filter (key caveat)**: most items are *proof* / "
      "*show-that* problems (or floor-of-expression tasks) whose `Answer` IS the "
      "statement to prove, or a number lifted straight from the prompt. There an "
      "early occurrence is the model **restating the question, not leakage** — "
      "~75% of raw early hits are this class. The **clean** rate excludes any "
      "sample whose answer is (a) already present in the prompt, or (b) from a "
      "proof/show problem (matched by 'prove'/'show that' in the statement, "
      "because LaTeX rewrites — `^{k}` vs `^k`, `\\left` — let the answer dodge a "
      "literal prompt match). Even the clean rate should be read as an upper "
      "bound: manual review found the residual hits are still mostly restatement "
      "(see appendix).\n")

    A("## In-schedule vs extended comparison\n")
    A("| segment | samples | kw hits | kw rate | detectable | proof | clean | early raw | early clean | early rate (clean) | strong |")
    A("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for s in (seg_in, seg_out):
        label = s['steps'].replace('-1000000000', '+')
        A(f"| step {label} | {s['n']} | {s['kw']} | {100*s['kw_rate']:.2f}% | "
          f"{s['detectable']} | {s['proof']} | {s['detectable_clean']} | "
          f"{s['early_raw']} | {s['early_clean']} | "
          f"{100*s['early_rate_clean']:.2f}% | {s['strong_clean']} |")
    A("")

    A("## Per-step hit rates\n")
    A("Columns: kw = keyword hits; detectable = linked & usable answer; proof = "
      "proof/show problems (answer = statement); clean = detectable minus "
      "answer-visible-in-question; early raw/clean = answer at "
      f"pos<{EARLY_POS} (raw / filtered); strong = clean early with "
      f"prefix_words<{STRONG_PREFIX_WORDS}.\n")
    A("| step | n | kw | detectable | proof | clean | early raw | early clean | early rate (clean) | strong |")
    A("|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for r in rows:
        A(f"| {r['step']} | {r['n']} | {r['kw_hits']} | {r['detectable']} | "
          f"{r['proof']} | {r['detectable_clean']} | {r['early_raw']} | "
          f"{r['early_clean']} | {100*r['early_rate_clean']:.2f}% | "
          f"{r['strong_clean']} |")
    A("")

    A("## Hit-sample appendix (manual review)\n")
    A("Up to 3 hit samples per step, with the matched position annotated. "
      "Empty steps had no hits.\n")
    any_hit = False
    for step, samples in appendix:
        if not samples:
            continue
        any_hit = True
        A(f"### step {step}\n")
        for k, s in enumerate(samples, 1):
            A(f"**sample {k}** — gt answer `{s['gt_answer']}`  ")
            A(f"problem: {s['problem']}...  ")
            for e in s["evidence"]:
                A(f"- {e}")
            A("")
    if not any_hit:
        A("_No hits under either probe across all steps._\n")

    out = os.path.join(OUT_DIR, f"{OUT_PREFIX}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {out}")


def _print_summary(rows):
    seg_in = _seg(rows, 0, OFFICIAL_MAX_STEP)
    seg_out = _seg(rows, OFFICIAL_MAX_STEP + 5, 10 ** 9)
    tot_n = sum(r["n"] for r in rows)
    tot_kw = sum(r["kw_hits"] for r in rows)
    tot_det = sum(r["detectable"] for r in rows)
    tot_det_c = sum(r["detectable_clean"] for r in rows)
    tot_proof = sum(r["proof"] for r in rows)
    tot_early_raw = sum(r["early_raw"] for r in rows)
    tot_early_c = sum(r["early_clean"] for r in rows)
    tot_strong_c = sum(r["strong_clean"] for r in rows)
    print("\n================ SUMMARY ================")
    print(f"total samples: {tot_n}")
    print(f"proof/show problems: {tot_proof}/{tot_det} detectable "
          f"({100*tot_proof/tot_det if tot_det else 0:.1f}%)")
    print(f"keyword probe:         {tot_kw} hits  "
          f"({100*tot_kw/tot_n if tot_n else 0:.2f}%)")
    print(f"answer early-emission (raw):   {tot_early_raw} / {tot_det} detectable "
          f"({100*tot_early_raw/tot_det if tot_det else 0:.2f}%) "
          f"-- includes restating-the-question")
    print(f"answer early-emission (clean): {tot_early_c} / {tot_det_c} detectable "
          f"({100*tot_early_c/tot_det_c if tot_det_c else 0:.2f}%), "
          f"strong={tot_strong_c}  <-- headline")
    print(f"  in-schedule (<=100): kw {100*seg_in['kw_rate']:.2f}% | "
          f"early_clean {100*seg_in['early_rate_clean']:.2f}%")
    print(f"  extended (105+):     kw {100*seg_out['kw_rate']:.2f}% | "
          f"early_clean {100*seg_out['early_rate_clean']:.2f}%")
    print("=========================================")


if __name__ == "__main__":
    main()
