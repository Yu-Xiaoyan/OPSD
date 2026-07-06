"""Bisect attribution — corruption column (config-level).

Third column of the suppressor attribution table: config-level corruption mass
Σ JSD(T_S, T_S̃) per variant, vs Tier-1 and the repo baseline (0.47).

Key simplification: the teacher is the FIXED base model (fixed_teacher is kept in
Tier 1 and all four variants), so corruption sensitivity depends only on each
variant's ROLLOUT content, not on any adapter. One base model scores rollouts
pooled from each run's own generation dumps (extended window). No high/low split
(variants may have ~zero keyword hits) — report the config-level median over a
random sample.

GPU, 1 card. Stores only derived per-rollout corruption mass (CLAUDE.md §4).
Writes probes/analysis/bisect_corruption.txt.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from datasets import load_dataset  # noqa: E402

from scoring import score_with_privilege  # noqa: E402
from divergence import token_jsd  # noqa: E402
from corrupt_solution import corrupt_solution  # noqa: E402
from corrupt_answers import corrupt_answer, seed_from  # noqa: E402
import leakage_detector as ld  # noqa: E402

BASE = os.path.expanduser("~/models/Qwen3-1.7B")
DATASET = "siyanzhao/Openthoughts_math_30k_opsd"
ANALYSIS = os.path.join(_HERE, "analysis")
CAP = 2048
LO, HI = 150, 195       # high-leakage extended window to sample rollouts from
N = 30
REPO_BASELINE = 0.469   # repo-OPSD correct-bucket corruption mass median (diag_2x2.md)

RUNS = [
    ("Tier1",   "qwen31b_paper_opsd_v1"),
    ("+clip",   "qwen31b_bisect_clip"),
    ("+guard",  "qwen31b_bisect_guard"),
    ("+length", "qwen31b_bisect_length"),
    ("+tmoff",  "qwen31b_bisect_tmoff"),
]


def main():
    os.makedirs(ANALYSIS, exist_ok=True)
    tr = load_dataset(DATASET)["train"]
    ans_idx = ld.build_answer_index(tr)
    sol_idx = {ld.problem_key(p): s for p, s in zip(tr["problem"], tr["solution"])}

    tok = AutoTokenizer.from_pretrained(BASE, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2").cuda().eval()

    def corr_mass(rec):
        rollout = tok(rec["completion"], add_special_tokens=False).input_ids[:CAP]
        if len(rollout) < 8:
            return None
        T_S = score_with_privilege(base, tok, rec["problem"], rollout,
                                   rec["solution"], teacher_thinking=True)["logits"]
        cans = corrupt_answer(rec["gt"], n=3, avoid=[rec["gt"]],
                              seed=seed_from(rec["problem"], rec["gt"]))
        masses = []
        for c in cans:
            csol, _ = corrupt_solution(rec["solution"], rec["gt"], c)
            Tk = score_with_privilege(base, tok, rec["problem"], rollout,
                                      csol, teacher_thinking=True)["logits"]
            masses.append(float(token_jsd(T_S, Tk).sum()))
            del Tk
        del T_S
        torch.cuda.empty_cache()
        return float(np.mean(masses))

    def sample(rc):
        gd = os.path.expanduser(f"~/opsd_outputs/{rc}/generations")
        recs = []
        for p in glob.glob(os.path.join(gd, "generations_step_*.json")):
            st = int(re.search(r"step_(\d+)", p).group(1))
            if not (LO <= st <= HI):
                continue
            for s in json.load(open(p))["generations"]:
                prob = ld.extract_problem_from_prompt(s.get("prompt", ""))
                if not prob:
                    continue
                sol = sol_idx.get(ld.problem_key(prob))
                gt = ans_idx.get(ld.problem_key(prob))
                if sol is None or gt is None:
                    continue
                recs.append({"problem": prob, "solution": sol, "gt": str(gt),
                             "completion": s.get("completion", "")})
        rng = np.random.RandomState(0)
        rng.shuffle(recs)
        return recs[:N]

    out = ["config-level corruption mass  Σ JSD(T_S, T_S̃)  (teacher=base; "
           f"rollouts from window {LO}-{HI})",
           f"repo-OPSD baseline (correct bucket median): {REPO_BASELINE}",
           f"{'run':<10} {'median':>8} {'mean':>8} {'vs_repo':>8}  n"]
    results = {}
    for name, rc in RUNS:
        recs = sample(rc)
        masses = [m for m in (corr_mass(r) for r in recs) if m is not None]
        a = np.array(masses)
        results[name] = a
        med = float(np.median(a)) if len(a) else float("nan")
        out.append(f"{name:<10} {med:>8.3f} {a.mean():>8.3f} "
                   f"{med/REPO_BASELINE:>7.1f}x  {len(a)}")
        print(f"[{name}] median={med:.3f} n={len(a)}", flush=True)

    text = "\n".join(out)
    with open(os.path.join(ANALYSIS, "bisect_corruption.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
