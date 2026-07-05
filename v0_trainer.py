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
path falls back to the parent. ΔV is a proxy (per-token divergence z-score) —
wiring the real V(t) probe is a TODO (see gating.py).
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

        # === TRIAGE + WEIGHTS ===
        buckets = bucket_batch(
            sampled_token_ids, shifted_labels,
            inputs.get("gt_answers"), self.processing_class,
            inputs.get("problems"),
        )
        W = synth_weights(per_tok.detach(), ex_ids, buckets, self.gate_config)
        loss = (per_tok * W).sum() / W.sum().clamp_min(1.0)

        # === GATING STATS -> metrics (averaged in log()) ===
        mode = "train" if model.training else "eval"
        for k, v in gating_stats(buckets, W).items():
            self._metrics[mode][k].append(v)

        torch.cuda.empty_cache()

        if return_outputs:
            class MinimalOutput:
                def __init__(self, loss):
                    self.loss = loss
            return (loss, MinimalOutput(loss))
        return loss
