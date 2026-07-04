"""GPU test for probes/answer_likelihood.py — V(t) separation + bridge stability.

- Pull ~12 correct + ~12 wrong rollouts from the training generations dumps
  (linked to gt answers, graded by the verifier).
- V(end) separation: assert correct-group mean V(end) > wrong-group mean, and
  PRINT mean diff + AUC (no hard threshold — human judgement).
- Bridge sensitivity: Spearman correlation of V(end) across bridge variants
  0/1/2 must be > 0.8.
- Saves 3 example V(t) curves (correct / wrong / truncated) to probes/analysis/.

Runs under pbs/probe_vt_test.pbs (same template as the golden test).
"""
import os
import sys
import unittest

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_PROBES = os.path.dirname(_HERE)
for p in (_REPO, _PROBES):
    if p not in sys.path:
        sys.path.insert(0, p)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from datasets import load_dataset  # noqa: E402
from peft import PeftModel  # noqa: E402

from answer_likelihood import answer_likelihood_probe, BRIDGES  # noqa: E402
from verify_answer import bucket_rollout  # noqa: E402
from leakage_detector import (extract_problem_from_prompt, build_answer_index,  # noqa: E402
                              problem_key, load_dump)
import plotting  # noqa: E402

BASE = os.path.expanduser("~/models/Qwen3-1.7B")
CKPT = os.path.expanduser("~/opsd_outputs/qwen31b_repro_3xh200_gb30/checkpoint-50")
GEN_DIR = os.path.expanduser("~/opsd_outputs/qwen31b_repro_3xh200_gb30/generations")
DATASET = "siyanzhao/Openthoughts_math_30k_opsd"
ANALYSIS = os.path.join(_PROBES, "analysis")
N_PER_BUCKET = 12
ROLLOUT_CAP = 600


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def _auc(pos, neg):
    """P(V_correct > V_wrong) via Mann-Whitney (ties counted 0.5)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return float(wins) / (len(pos) * len(neg))


class TestAnswerLikelihood(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert torch.cuda.is_available(), "V(t) test requires a GPU"
        cls.device = torch.device("cuda")
        cls.tokenizer = AutoTokenizer.from_pretrained(BASE, padding_side="left")
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            BASE, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2").to(cls.device).eval()
        cls.model = PeftModel.from_pretrained(base, CKPT).eval()

        tr = load_dataset(DATASET)["train"]
        index = build_answer_index(tr)

        import glob
        import re
        dumps = sorted(glob.glob(os.path.join(GEN_DIR, "generations_step_*.json")),
                       key=lambda p: int(re.search(r"step_(\d+)", p).group(1)))
        cls.correct, cls.wrong, cls.truncated = [], [], []
        for path in dumps:
            for s in load_dump(path):
                if (len(cls.correct) >= N_PER_BUCKET
                        and len(cls.wrong) >= N_PER_BUCKET
                        and len(cls.truncated) >= 2):
                    break
                prob = extract_problem_from_prompt(s.get("prompt", ""))
                if prob is None:
                    continue
                gt = index.get(problem_key(prob))
                if gt is None:
                    continue
                comp = s.get("completion", "")
                bucket, student = bucket_rollout(comp, gt)
                ids = cls.tokenizer(comp, add_special_tokens=False).input_ids[:ROLLOUT_CAP]
                if len(ids) < 16:
                    continue
                rec = {"problem": prob, "gt": gt, "rollout": ids, "student": student}
                if bucket == "correct" and len(cls.correct) < N_PER_BUCKET:
                    cls.correct.append(rec)
                elif bucket == "wrong" and len(cls.wrong) < N_PER_BUCKET:
                    cls.wrong.append(rec)
                elif bucket == "truncated" and len(cls.truncated) < 4:
                    cls.truncated.append(rec)
        print(f"\n[setup] collected correct={len(cls.correct)} "
              f"wrong={len(cls.wrong)} truncated={len(cls.truncated)}")
        cls._Vend_cache = {}

    def _v_end(self, rec, bridge=0):
        key = (id(rec), bridge)
        if key not in self._Vend_cache:
            out = answer_likelihood_probe(
                self.model, self.tokenizer, rec["problem"], rec["rollout"],
                rec["gt"], bridge_variant=bridge)
            self._Vend_cache[key] = (out["V"][-1], out)
        return self._Vend_cache[key]

    def test_separation(self):
        self.assertGreaterEqual(len(self.correct), 5)
        self.assertGreaterEqual(len(self.wrong), 5)
        vc = [self._v_end(r)[0] for r in self.correct]
        vw = [self._v_end(r)[0] for r in self.wrong]
        mc, mw = float(np.mean(vc)), float(np.mean(vw))
        auc = _auc(vc, vw)
        print(f"\n[SEP] V(end) correct: mean={mc:.3f} std={np.std(vc):.3f} "
              f"(n={len(vc)})")
        print(f"[SEP] V(end) wrong:   mean={mw:.3f} std={np.std(vw):.3f} "
              f"(n={len(vw)})")
        print(f"[SEP] mean diff (correct-wrong) = {mc - mw:.3f} nats")
        print(f"[SEP] AUC P(V_correct > V_wrong) = {auc:.3f}  "
              f"(0.5=no separation, 1.0=perfect)")
        self.assertGreater(mc, mw, "correct group V(end) not above wrong group")

    def test_bridge_spearman(self):
        recs = (self.correct + self.wrong)[:20]
        v = {b: [self._v_end(r, b)[0] for r in recs] for b in (0, 1, 2)}
        pairs = [(0, 1), (0, 2), (1, 2)]
        rhos = {p: _spearman(v[p[0]], v[p[1]]) for p in pairs}
        for p, rho in rhos.items():
            print(f"[BRIDGE] Spearman V(end) bridge{p[0]} vs bridge{p[1]} = {rho:.3f}")
        self.assertGreater(min(rhos.values()), 0.8,
                           f"bridge variants not stable: {rhos}")

    def test_plot_examples(self):
        os.makedirs(ANALYSIS, exist_ok=True)
        curves = []
        picks = [("correct", self.correct), ("wrong", self.wrong),
                 ("truncated", self.truncated)]
        for label, group in picks:
            if not group:
                continue
            _, out = self._v_end(group[0])
            curves.append((out["checkpoint_positions"], out["V"], label))
            print(f"[PLOT] {label}: T={out['checkpoint_positions'][-1]} "
                  f"K={len(out['V'])} V(end)={out['V'][-1]:.3f} "
                  f"answer_tokens={out['num_answer_tokens']}")
        path = os.path.join(ANALYSIS, "vt_examples.png")
        plotting.plot_vt_curves(curves, path,
                                title="V(t) examples (Qwen3-1.7B ckpt-50)")
        print(f"[PLOT] wrote {path}")
        self.assertTrue(os.path.getsize(path) > 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
