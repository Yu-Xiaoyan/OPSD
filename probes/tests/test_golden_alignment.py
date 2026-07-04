"""Golden alignment test for probes/scoring.py + probes/divergence.py (GPU).

Three levels, run as one PBS job (pbs/probe_golden_test.pbs):
  1. logits alignment  : probe path vs official (collator + trainer offset).
  2. loss alignment    : clipped_kl_view vs OPSDTrainer.generalized_jsd_loss
                         (beta=0, clip=0.05), with a real student != teacher via
                         a checkpoint-50 LoRA adapter (adapter off = teacher).
  3. classifier spot-check: print token_classifier on real Qwen3 tokens.

Also prints single-forward latency + peak memory (cost baseline). Failures print
prompt-text/length diffs; tolerances are NOT loosened to pass.

Run:  ~/.conda/envs/opsd/bin/python probes/tests/test_golden_alignment.py
"""
import os
import sys
import time
import unittest

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

from data_collator import SelfDistillationDataCollator  # noqa: E402
from opsd_trainer import OPSDTrainer  # noqa: E402
from scoring import (  # noqa: E402
    score_with_privilege, build_teacher_prompt_text, build_student_prompt_text,
    forward_rollout_logits, teacher_mode,
)
from divergence import clipped_kl_view, token_classifier  # noqa: E402

BASE = os.path.expanduser("~/models/Qwen3-1.7B")
CKPT = os.path.expanduser(
    "~/opsd_outputs/qwen31b_repro_3xh200_gb30/checkpoint-50")
DATASET = "siyanzhao/Openthoughts_math_30k_opsd"
CLIP = 0.05
N_ROLLOUT = 96


class TestGoldenAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert torch.cuda.is_available(), "golden test requires a GPU"
        cls.device = torch.device("cuda")

        cls.tokenizer = AutoTokenizer.from_pretrained(BASE, padding_side="left")
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2").to(cls.device).eval()
        # LoRA adapter -> student != teacher (adapter off = base = teacher)
        cls.model = PeftModel.from_pretrained(base_model, CKPT).eval()

        ds = load_dataset(DATASET)["train"]
        cls.problem = ds[0]["problem"]
        cls.solution = ds[0]["solution"]
        print(f"\n[setup] problem[:80]: {cls.problem[:80]!r}")
        print(f"[setup] solution len: {len(cls.solution)} chars")

        # rollout: greedy-generate from the student prompt (adapter on = policy)
        sp = build_student_prompt_text(cls.tokenizer, cls.problem,
                                       student_thinking=False)
        pids = cls.tokenizer(sp, return_tensors="pt").input_ids.to(cls.device)
        with torch.no_grad():
            gen = cls.model.generate(pids, max_new_tokens=N_ROLLOUT,
                                     do_sample=False,
                                     pad_token_id=cls.tokenizer.pad_token_id)
        cls.rollout_ids = gen[0, pids.shape[1]:].tolist()
        # drop trailing pad/eos padding if any
        eos = cls.tokenizer.eos_token_id
        while len(cls.rollout_ids) > 8 and cls.rollout_ids[-1] in (
                eos, cls.tokenizer.pad_token_id):
            cls.rollout_ids.pop()
        cls.T = len(cls.rollout_ids)
        print(f"[setup] rollout tokens T = {cls.T}")

    # ---- Level 1: logits alignment -------------------------------------
    def test_level1_logits_alignment(self):
        tok, model, device = self.tokenizer, self.model, self.device

        # official path: collator teacher prompt + trainer-style concat/offset
        collator = SelfDistillationDataCollator(
            tok, max_length=20000, reason_first=False,
            student_thinking=False, teacher_thinking=True)
        batch = collator([{"problem": self.problem, "solution": self.solution}])
        off_prompt = batch["teacher_prompts"].to(device)          # [1, P_t]
        P_off = int(batch["teacher_prompt_length"])
        rollout_t = torch.tensor(self.rollout_ids, device=device).unsqueeze(0)
        full = torch.cat([off_prompt, rollout_t], dim=1)
        with torch.no_grad():
            off_logits = model(
                input_ids=full, attention_mask=torch.ones_like(full)
            ).logits[:, P_off - 1:-1, :].float().squeeze(0)

        # probe path
        res = score_with_privilege(model, tok, self.problem, self.rollout_ids,
                                   self.solution, teacher_thinking=True)
        pr_logits = res["logits"]
        P_pr = res["teacher_prompt_length"]

        if P_pr != P_off:
            # diagnostic before failing: compare prompt texts / ids
            probe_text = build_teacher_prompt_text(
                tok, self.problem, self.solution, True)
            probe_ids = tok(probe_text, return_tensors="pt").input_ids[0].tolist()
            off_ids = off_prompt[0].tolist()
            print(f"[L1 DIAG] P_probe={P_pr} P_official={P_off}")
            print(f"[L1 DIAG] probe_ids[:20]={probe_ids[:20]}")
            print(f"[L1 DIAG] offic_ids[:20]={off_ids[:20]}")
        self.assertEqual(P_pr, P_off, "teacher_prompt_length mismatch")
        self.assertEqual(pr_logits.shape, off_logits.shape)

        # per-position argmax must match exactly (strong condition)
        argmax_match = (pr_logits.argmax(-1) == off_logits.argmax(-1))
        print(f"[L1] argmax match {int(argmax_match.sum())}/{self.T}")
        self.assertTrue(bool(argmax_match.all()), "argmax mismatch")
        self.assertTrue(torch.allclose(pr_logits, off_logits, rtol=1e-3,
                                       atol=1e-2), "logits not allclose")

        # bit-level determinism: same input twice
        res2 = score_with_privilege(model, tok, self.problem, self.rollout_ids,
                                    self.solution, teacher_thinking=True)
        self.assertTrue(torch.equal(pr_logits, res2["logits"]),
                        "probe forward is not bit-identical across runs")
        print("[L1] PASS: length, argmax, allclose, bit-determinism")

    # ---- Level 2: loss alignment ---------------------------------------
    def test_level2_loss_alignment(self):
        tok, model, device = self.tokenizer, self.model, self.device

        # teacher logits: adapter OFF (base = fixed teacher), privileged prompt
        with teacher_mode(model):
            tp = build_teacher_prompt_text(tok, self.problem, self.solution, True)
            t_logits, _, _ = forward_rollout_logits(model, tok, tp,
                                                    self.rollout_ids)
        # student logits: adapter ON, student prompt
        sp = build_student_prompt_text(tok, self.problem, student_thinking=False)
        s_logits, _, _ = forward_rollout_logits(model, tok, sp, self.rollout_ids)

        # sanity: student != teacher
        max_abs = (t_logits - s_logits).abs().max().item()
        print(f"[L2] max|teacher-student logit| = {max_abs:.4f} (must be > 0)")
        self.assertGreater(max_abs, 0.0, "teacher==student; adapter not applied")

        labels = torch.zeros(1, self.T, dtype=torch.long, device=device)  # all valid
        official = OPSDTrainer.generalized_jsd_loss(
            student_logits=s_logits.unsqueeze(0),
            teacher_logits=t_logits.unsqueeze(0),
            labels=labels, beta=0, temperature=1.0, token_clip=CLIP).item()

        probe = clipped_kl_view(t_logits, s_logits, clip=CLIP).mean().item()
        print(f"[L2] official generalized_jsd_loss = {official:.8f}")
        print(f"[L2] probe clipped_kl_view.mean()  = {probe:.8f}")
        print(f"[L2] abs diff = {abs(official - probe):.2e}")
        self.assertTrue(
            torch.allclose(torch.tensor(official), torch.tensor(probe),
                           rtol=1e-4, atol=1e-6),
            f"loss mismatch: official={official} probe={probe}")
        print("[L2] PASS: loss-level alignment")

    # ---- Level 3: classifier spot-check (no assert) --------------------
    def test_level3_classifier_spotcheck(self):
        toks = self.tokenizer.convert_ids_to_tokens(self.rollout_ids[:80])
        labels = token_classifier(toks)
        print("\n[L3] token_classifier on first 80 real rollout tokens "
              "(tok -> label):")
        line = []
        for t, l in zip(toks, labels):
            line.append(f"{t!r}:{l}")
            if len(line) == 5:
                print("   " + "  ".join(line))
                line = []
        if line:
            print("   " + "  ".join(line))
        from collections import Counter
        print("[L3] label counts:", dict(Counter(labels)))

    # ---- cost baseline --------------------------------------------------
    def test_cost_baseline(self):
        tok, model = self.tokenizer, self.model
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        _ = score_with_privilege(model, tok, self.problem, self.rollout_ids,
                                 self.solution, teacher_thinking=True)
        torch.cuda.synchronize()
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n[COST] single privileged teacher forward: {dt*1000:.1f} ms, "
              f"peak CUDA mem {peak:.2f} GB (T={self.T}, "
              f"teacher_prompt_len from setup)")

    # ---- regression: transition_prompt_override (experiment A) ----------
    def test_transition_override_regression(self):
        """Default collator (override=None) must stay byte-for-byte official;
        the neutral override must apply cleanly (drops the guard, keeps the
        reference-solution segment and the boxed instruction)."""
        tok = self.tokenizer
        feat = {"problem": self.problem, "solution": self.solution}
        c_def = SelfDistillationDataCollator(
            tok, max_length=20000, reason_first=False,
            student_thinking=False, teacher_thinking=True)
        ids_def = c_def([feat])["teacher_prompts"][0].tolist()
        probe_text = build_teacher_prompt_text(tok, self.problem, self.solution, True)
        probe_ids = tok(probe_text, return_tensors="pt").input_ids[0].tolist()
        self.assertEqual(ids_def, probe_ids,
                         "default collator drifted from official teacher prompt")

        NEUTRAL = "\n\nNow, derive the final answer to the problem above."
        c_ng = SelfDistillationDataCollator(
            tok, max_length=20000, reason_first=False, student_thinking=False,
            teacher_thinking=True, transition_prompt_override=NEUTRAL)
        ids_ng = c_ng([feat])["teacher_prompts"][0].tolist()
        self.assertNotEqual(ids_def, ids_ng, "override had no effect")
        txt = tok.decode(ids_ng)
        self.assertIn("Now, derive the final answer to the problem above.", txt)
        self.assertNotIn("do not copy or paraphrase", txt)
        self.assertIn("=== Reference Solution Begin ===", txt)
        self.assertIn("put your final answer within \\boxed{}", txt)
        print(f"[REGRESSION] default P_t={len(ids_def)} byte-exact vs probe "
              f"builder; noguard P_t={len(ids_ng)}, guard removed cleanly")


if __name__ == "__main__":
    unittest.main(verbosity=2)
