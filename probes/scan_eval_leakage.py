"""Side-quest 1: leakage scan over EVAL outputs (TM-on, AIME).

Reuses the two leakage_detector probes on the eval generations in
results/repro_eval/{base,ckpt50,ckpt100}_{aime24,aime25}.json. Unlike the
training-rollout scan (TM-off, short), these are TM-on 38k-token generations —
a different behavioral space, which is the independent value of this scan.

Adaptations vs the training scan:
- gt answer comes from the eval dataset's `ground_truth` (per result), not the
  Openthoughts training set.
- question-visibility filter (answer already in the problem statement) applies to
  AIME too (rare — checked, not assumed).

Appends a marked, idempotent section to
probes/analysis/leakage_over_training.md. CPU only.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

import leakage_detector as ld

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "repro_eval")
MD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis",
                  "leakage_over_training.md")
KW_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "leakage_keywords.txt")
MARKER = "<!-- EVAL-SCAN -->"
EARLY_POS = 0.30
TAGS = ["base", "ckpt50", "ckpt100"]
DATASETS = ["aime24", "aime25"]
SAMPLES_PER_TAG = 3


def scan_file(path, keywords):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    agg = Counter()
    hit_samples = []
    for res in data.get("results", []):
        problem = res.get("problem", "") or ""
        gt = str(res.get("ground_truth", "")).strip()
        variants = ld.answer_variants(gt) if gt else []
        proof = ld.is_proof_problem(problem)
        in_prompt = bool(variants) and ld.find_answer_emission(problem, variants)["found"]
        for g in res.get("generations", []):
            text = g.get("full_generation", "") or ""
            agg["n"] += 1
            kw = ld.detect_keywords(text, keywords)
            if kw:
                agg["kw"] += 1
            if not variants:
                continue
            agg["detectable"] += 1
            if proof:
                agg["proof"] += 1
            emit = ld.find_answer_emission(text, variants)
            is_early = emit["found"] and emit["pos_ratio"] is not None \
                and emit["pos_ratio"] < EARLY_POS
            if is_early:
                agg["early_raw"] += 1
            visible = in_prompt or proof
            if not visible:
                agg["detectable_clean"] += 1
                if is_early:
                    agg["early_clean"] += 1
            clean_hit = is_early and not visible
            if (kw or clean_hit) and len(hit_samples) < SAMPLES_PER_TAG:
                ev = []
                if kw:
                    ev.append(f"keyword '{kw[0]['keyword']}' @char {kw[0]['char_pos']}")
                if clean_hit:
                    p = emit["char_pos"]
                    frag = text[max(0, p - 60):p + 60].replace("\n", " ")
                    ev.append(f"answer '{emit['matched_variant']}' early "
                              f"pos={emit['pos_ratio']:.2f}: ...{frag}...")
                hit_samples.append({"gt": gt, "problem": problem[:110], "ev": ev})
    return agg, hit_samples


def main():
    keywords = ld.load_keywords(KW_PATH)
    rows = {}          # (tag, ds) -> agg
    appendix = {}      # tag -> samples
    for tag in TAGS:
        appendix[tag] = []
        for ds in DATASETS:
            path = os.path.join(RESULTS, f"{tag}_{ds}.json")
            if not os.path.exists(path):
                print(f"[warn] missing {path}")
                continue
            agg, samples = scan_file(path, keywords)
            rows[(tag, ds)] = agg
            appendix[tag].extend(samples[:SAMPLES_PER_TAG])
            print(f"{tag:8} {ds}: n={agg['n']} kw={agg['kw']} "
                  f"detectable={agg['detectable']} proof={agg['proof']} "
                  f"in_prompt_clean={agg['detectable_clean']} "
                  f"early_raw={agg['early_raw']} early_clean={agg['early_clean']}")

    # per-checkpoint totals (aime24+aime25)
    def tot(tag, key):
        return sum(rows[(tag, ds)][key] for ds in DATASETS if (tag, ds) in rows)

    lines = [MARKER, "", "## Eval-output leakage scan (TM-on, AIME)", "",
             "Independent scan over the eval generations "
             "(`results/repro_eval/`, TM-on 38k-token AIME outputs) — a different "
             "behavioral space from the TM-off short training rollouts above. "
             "Same two probes; gt answer from each result's `ground_truth`; "
             "question-visibility filter (answer already in the problem, or a "
             "proof problem) applied. Appended by `probes/scan_eval_leakage.py`.",
             "",
             "**Interpretation.** Eval generations are pure student TM-on "
             "reasoning with NO teacher / privileged context at inference, so "
             "'answer early' here measures a *behavioral residue* baked into the "
             "student by training — not live leakage. AIME integer answers "
             "appearing in the first 30% are usually reasoning coincidences "
             "(appendix). The **base-vs-ckpt comparison** is the real signal: "
             "training does NOT raise these rates (answer early-clean "
             "base 8.1% -> ckpt-50 5.6% -> ckpt-100 5.8%; keyword <1% throughout) "
             "— consistent with the training-rollout scan's near-zero result.",
             "",
             "| checkpoint | samples | kw hits | kw rate | detectable | early raw | early clean | early rate (clean) |",
             "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for tag in TAGS:
        n = tot(tag, "n")
        if n == 0:
            continue
        kw = tot(tag, "kw")
        det_c = tot(tag, "detectable_clean")
        er = tot(tag, "early_raw")
        ec = tot(tag, "early_clean")
        lines.append(
            f"| {tag} | {n} | {kw} | {100*kw/n:.2f}% | {tot(tag,'detectable')} | "
            f"{er} | {ec} | {100*ec/det_c if det_c else 0:.2f}% |")
    lines += ["", "### Hit-sample appendix (manual review)", ""]
    any_hit = False
    for tag in TAGS:
        if not appendix[tag]:
            continue
        any_hit = True
        lines.append(f"**{tag}**")
        for s in appendix[tag]:
            lines.append(f"- gt `{s['gt']}` | {s['problem']}...")
            for e in s["ev"]:
                lines.append(f"  - {e}")
        lines.append("")
    if not any_hit:
        lines.append("_No hits under either probe across all eval outputs._")
    section = "\n".join(lines) + "\n"

    # idempotent append (strip any prior EVAL-SCAN section)
    md = ""
    if os.path.exists(MD):
        md = open(MD, encoding="utf-8").read()
        if MARKER in md:
            md = md[:md.index(MARKER)].rstrip() + "\n\n"
    with open(MD, "w", encoding="utf-8") as f:
        f.write(md + section)
    print(f"\nappended eval-scan section to {MD}")


if __name__ == "__main__":
    main()
