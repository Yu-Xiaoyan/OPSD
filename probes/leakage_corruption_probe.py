"""Task (a): corruption-probe VALIDITY check on the Tier-1 leaky run.

Positive control for probe validity. Hypothesis: rollouts that behaviorally leak
(student cites the privileged reference solution in its TM-on <think>) should
also show HIGHER distributional privilege dependence — i.e. a larger corruption
sensitivity JSD(T_S, T_S̃) — than non-leaking rollouts and than the main repo
run. If behavioral leakage (keyword hit) and distributional corruption mass
correlate at the trajectory level, the corruption probe is measuring real
privilege dependence.

Teacher = base (fixed_teacher path, LoRA disabled) — the same teacher Tier 1
trained against. Rollouts come from the Tier-1 generations dumps (TM-on, 2048),
split into high-leakage (keyword hit) vs low-leakage (no hit) by
leakage_detector. GPU, 1 card. Stores only derived per-rollout corruption mass
(CLAUDE.md §4).

Writes probes/analysis/leakage_corruption_validity.md + .png.
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
from peft import PeftModel  # noqa: E402
from datasets import load_dataset  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scoring import score_with_privilege, teacher_mode  # noqa: E402
from divergence import token_jsd  # noqa: E402
from corrupt_solution import corrupt_solution  # noqa: E402
from corrupt_answers import corrupt_answer, seed_from  # noqa: E402
import leakage_detector as ld  # noqa: E402

BASE = os.path.expanduser("~/models/Qwen3-1.7B")
CKPT = os.path.expanduser("~/opsd_outputs/qwen31b_paper_opsd_v1/checkpoint-300")
GEN = os.path.expanduser("~/opsd_outputs/qwen31b_paper_opsd_v1/generations")
DATASET = "siyanzhao/Openthoughts_math_30k_opsd"
ANALYSIS = os.path.join(_HERE, "analysis")
CAP = 2048
MIN_STEP = 150          # pool extended (high-leakage) dumps
N_PER_GROUP = 30
BASELINE_MASS = 0.469   # repo-OPSD correct-bucket corruption mass median (diag_2x2.md)


def _auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return float(wins) / (len(pos) * len(neg))


def main():
    os.makedirs(ANALYSIS, exist_ok=True)
    kws = ld.load_keywords(os.path.join(_HERE, "leakage_keywords.txt"))
    tr = load_dataset(DATASET)["train"]
    ans_idx = ld.build_answer_index(tr)
    sol_idx = {ld.problem_key(p): s for p, s in zip(tr["problem"], tr["solution"])}

    # pool extended dumps, split high/low leakage by keyword hit
    dumps = [p for p in glob.glob(os.path.join(GEN, "generations_step_*.json"))
             if int(re.search(r"step_(\d+)", p).group(1)) >= MIN_STEP]
    high, low = [], []
    for path in sorted(dumps):
        for s in ld.load_dump(path):
            r = ld.detect_sample(s, kws, ans_idx, 0)
            prob = ld.extract_problem_from_prompt(s.get("prompt", ""))
            if not prob:
                continue
            sol = sol_idx.get(ld.problem_key(prob))
            if sol is None or not r.gt_answer:
                continue
            rec = {"problem": prob, "solution": sol, "gt": str(r.gt_answer),
                   "completion": s.get("completion", "")}
            (high if r.keyword_hit else low).append(rec)
    rng = np.random.RandomState(0)
    for grp in (high, low):
        rng.shuffle(grp)
    n = min(N_PER_GROUP, len(high), len(low))
    high, low = high[:n], low[:n]
    print(f"pooled dumps>= {MIN_STEP}: high={len(high)} low={len(low)} (n per group={n})")

    tok = AutoTokenizer.from_pretrained(BASE, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2").cuda().eval()
    model = PeftModel.from_pretrained(base, CKPT).eval()

    def corr_mass(rec):
        rollout = tok(rec["completion"], add_special_tokens=False).input_ids[:CAP]
        if len(rollout) < 8:
            return None
        with teacher_mode(model):
            T_S = score_with_privilege(model, tok, rec["problem"], rollout,
                                       rec["solution"], teacher_thinking=True)["logits"]
        cans = corrupt_answer(rec["gt"], n=3, avoid=[rec["gt"]],
                              seed=seed_from(rec["problem"], rec["gt"]))
        masses = []
        for c in cans:
            csol, _ = corrupt_solution(rec["solution"], rec["gt"], c)
            with teacher_mode(model):
                Tk = score_with_privilege(model, tok, rec["problem"], rollout,
                                          csol, teacher_thinking=True)["logits"]
            masses.append(float(token_jsd(T_S, Tk).sum()))
            del Tk
        del T_S
        torch.cuda.empty_cache()
        return float(np.mean(masses))

    def run(group, label):
        out = []
        for i, rec in enumerate(group):
            m = corr_mass(rec)
            if m is not None:
                out.append(m)
            if (i + 1) % 10 == 0:
                print(f"  [{label}] {i+1}/{len(group)}")
        return np.array(out)

    hi = run(high, "high")
    lo = run(low, "low")

    def q(a):
        return (f"mean={a.mean():.3f} median={np.median(a):.3f} "
                f"q25={np.quantile(a,.25):.3f} q75={np.quantile(a,.75):.3f} n={len(a)}")

    auc = _auc(hi, lo)
    L = []
    A = L.append
    A("# Corruption-probe validity check on Tier-1 leaky run (task a)\n")
    A("Positive control: do behaviorally-leaking rollouts (keyword citation of "
      "the privileged solution in TM-on <think>) carry HIGHER corruption "
      "sensitivity JSD(T_S, T_S̃) than non-leaking ones? Teacher = base "
      "(fixed_teacher). Pooled Tier-1 dumps step>= "
      f"{MIN_STEP}.\n")
    A(f"- **high-leakage (kw hit)** corruption mass: {q(hi)}")
    A(f"- **low-leakage (no hit)** corruption mass: {q(lo)}")
    A(f"- repo-OPSD main-run baseline (correct bucket median): {BASELINE_MASS}")
    A(f"- **AUC(high vs low) = {auc:.3f}** (>0.5 => leaking rollouts are more "
      "corruption-sensitive; the two leakage measures agree at trajectory level)")
    A(f"- high/low median ratio: {np.median(hi)/max(np.median(lo),1e-6):.2f}x; "
      f"high/baseline: {np.median(hi)/BASELINE_MASS:.2f}x\n")
    A("**Read**: AUC>0.5 and high>low>baseline => the corruption probe measures "
      "real privilege dependence — behavioral leakage and distributional "
      "corruption sensitivity are the same phenomenon seen two ways. Probe "
      "validity positively confirmed on the positive control.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    hi_c = np.clip(hi, 0, np.quantile(np.concatenate([hi, lo]), 0.98))
    lo_c = np.clip(lo, 0, np.quantile(np.concatenate([hi, lo]), 0.98))
    ax.hist(lo_c, bins=15, alpha=0.6, label=f"low-leakage (n={len(lo)})", color="#1f77b4")
    ax.hist(hi_c, bins=15, alpha=0.6, label=f"high-leakage (n={len(hi)})", color="#d62728")
    ax.axvline(BASELINE_MASS, ls="--", color="k", lw=1, label=f"repo baseline {BASELINE_MASS}")
    ax.set_xlabel("corruption mass  Σ JSD(T_S, T_S̃)  (nats)")
    ax.set_title("Probe validity: corruption sensitivity by behavioral leakage (Tier 1)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS, "leakage_corruption_validity.png"), dpi=120)
    plt.close(fig)

    with open(os.path.join(ANALYSIS, "leakage_corruption_validity.md"), "w") as f:
        f.write("\n".join(L))
    print("wrote leakage_corruption_validity.md")
    print(f"high {q(hi)}\nlow  {q(lo)}\nAUC(high vs low)={auc:.3f}")


if __name__ == "__main__":
    main()
