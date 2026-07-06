"""OPSDGatedTrainer (v0) — gated on-policy self-distillation.

Subclass of OPSDTrainer that keeps the entire OPSD machinery (vLLM on-policy
rollout, reason-first teacher, fixed/EMA teacher, JSD/clip loss) and injects the
frozen v0 gating decided in stage 1 (docs/framework.md):

  compute_loss:
    1. student + teacher forward (same as OPSD),
    2. per-token JSD via generalized_jsd_loss(reduction="none"),
    3. trajectory triage (verifier v2 bucket per rollout),
    4. per-token weight synthesis (gate-A light / wrong ΔV-soft / trunc soft),
    5. weighted, weight-normalized loss,
    6. wandb gating stats.

Only the JSD path is gated; the thinking-machines (reverse-KL policy-gradient)
path falls back to the parent. ΔV is the REAL V(t)=log p(answer|prefix) segment
change (teacher-forcing on the fixed base teacher, no grad), broadcast to tokens
by checkpoint segment; τ from 3c gate_b.md. Corruption gate OFF (gate-D verdict).
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from accelerate.utils import is_peft_model

from opsd_trainer import OPSDTrainer
from gating import GateConfig, bucket_batch, synth_weights, gating_stats


class OPSDGatedTrainer(OPSDTrainer):
    def __init__(self, *args, gate_config: GateConfig | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate_config = gate_config or GateConfig()
        print(f"[OPSDGatedTrainer] gating enabled: {self.gate_config}")

    def _set_signature_columns_if_needed(self):
        # keep "Answer" so the gated collator can verifier-bucket rollouts
        super()._set_signature_columns_if_needed()
        if self._signature_columns is not None and "Answer" not in self._signature_columns:
            self._signature_columns.append("Answer")

    def _teacher_context(self, model):
        if self.use_ema_teacher:
            return self._ema_teacher_context(model)
        if self.fixed_teacher and is_peft_model(model):
            return self.accelerator.unwrap_model(model).disable_adapter()
        return nullcontext()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # TM path is not gated in v0 — defer to parent.
        if self.use_thinking_machines_loss:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        student_prompt_len = inputs["student_prompt_length"]
        teacher_prompt_len = inputs["teacher_prompt_length"]
        sampled_token_ids = inputs["student_input_ids"][:, student_prompt_len:]
        shifted_labels = inputs["labels"][:, student_prompt_len:]

        # === STUDENT FORWARD ===
        outputs_student = model(
            input_ids=inputs["student_input_ids"],
            attention_mask=inputs["student_attention_mask"],
        )
        student_logits_for_loss = outputs_student.logits[:, student_prompt_len - 1 : -1, :]
        del outputs_student
        torch.cuda.empty_cache()

        # === TEACHER FORWARD ===
        with torch.no_grad(), self._teacher_context(model):
            outputs_teacher = model(
                input_ids=inputs["teacher_input_ids"],
                attention_mask=inputs["teacher_attention_mask"],
            )
            teacher_logits_for_loss = outputs_teacher.logits[:, teacher_prompt_len - 1 : -1, :]
            del outputs_teacher
            torch.cuda.empty_cache()

        # === PER-TOKEN JSD (reduction="none" returns jsd[mask] = [n_valid, K]) ===
        jsd_masked = self.generalized_jsd_loss(
            student_logits=student_logits_for_loss,
            teacher_logits=teacher_logits_for_loss,
            labels=shifted_labels,
            beta=self.beta,
            temperature=self.temperature,
            top_k=self.top_k_loss,
            token_clip=self.jsd_token_clip,
            reduction="none",
        )
        del student_logits_for_loss, teacher_logits_for_loss
        per_tok = jsd_masked.sum(-1)            # [n_valid], carries grad
        del jsd_masked

        # flat example-id for each valid token (same mask order as jsd[mask])
        mask = shifted_labels != -100           # [B, gen_len]
        B = mask.shape[0]
        ex_ids = torch.arange(B, device=mask.device)[:, None].expand_as(mask)[mask]

        # === TRIAGE (verifier v2) ===
        gt_answers = inputs.get("gt_answers")
        problems = inputs.get("problems")
        buckets = bucket_batch(sampled_token_ids, shifted_labels, gt_answers,
                               self.processing_class, problems)

        # === REAL V(t) ΔV + structural mask (per wrong/truncated rollout) ===
        # teacher-forcing V(t)=log p(answer|prefix) on the fixed base teacher, no
        # grad (weight synthesis only). ΔV broadcast to tokens by segment.
        n_valid = per_tok.shape[0]
        dev = per_tok.device
        dv_per_tok = torch.zeros(n_valid, device=dev)
        struct_mask = torch.zeros(n_valid, dtype=torch.bool, device=dev)
        vend = torch.full((B,), float("nan"), device=dev)
        tok = self.processing_class
        t_probe = 0.0
        with torch.no_grad(), self._teacher_context(model):
            for i in range(B):
                sel = ex_ids == i
                if int(sel.sum()) == 0:
                    continue
                ids_i = sampled_token_ids[i][mask[i]].tolist()
                toks_i = tok.convert_ids_to_tokens(ids_i)
                struct_mask[sel] = torch.tensor(_structural_mask(toks_i), device=dev)
                if buckets[i] in ("wrong", "truncated") and len(ids_i) >= 8:
                    t0 = time.time()
                    cps = auto_checkpoints(tok, ids_i, max_points=16)
                    vt = answer_likelihood_probe(
                        model, tok, problems[i] if problems else "", ids_i,
                        str(gt_answers[i]) if gt_answers else "",
                        checkpoint_positions=cps)
                    t_probe += time.time() - t0
                    vend[i] = float(vt["V"][-1])
                    dv_i = _broadcast_dv(vt["delta_V"], vt["checkpoint_positions"],
                                         len(ids_i))
                    dv_per_tok[sel] = torch.tensor(dv_i, device=dev)

        # === WEIGHT SYNTHESIS (frozen v0) ===
        W = synth_weights(buckets, ex_ids, dv_per_tok, vend, struct_mask,
                          self.gate_config)
        loss = (per_tok * W).sum() / W.sum().clamp_min(1.0)

        # === GATING STATS -> metrics ===
        mode = "train" if model.training else "eval"
        stats = gating_stats(buckets, W)
        stats["gate/probe_time_s"] = t_probe
        for k, v in stats.items():
            self._metrics[mode][k].append(v)

        torch.cuda.empty_cache()

        if return_outputs:
            class MinimalOutput:
                def __init__(self, loss):
                    self.loss = loss
            return (loss, MinimalOutput(loss))
        return loss
