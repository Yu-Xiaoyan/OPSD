"""Privileged teacher scoring for the OPSD corruption probe (stage 1).

Score a rollout under the teacher's privileged (solution-conditioned) context,
producing the per-token logits the corruption probe compares across true vs
corrupted solutions (docs/framework.md, axis-1 groundedness).

Faithful to the official training path:
- teacher prompt built verbatim from data_collator.py:96-109 (the
  `transition_prompt` string is IMPORTED from SelfDistillationDataCollator, not
  hand-copied);
- rollout appended and forwarded once, logits sliced with the per-sample
  prompt-length offset `logits[:, P-1:-1]` (opsd_trainer.py:633-645 / :689);
- fixed_teacher semantics via `teacher_mode(model)` (opsd_trainer.py:676-687).

INTENTIONAL DIFFERENCE vs training: single sample, ZERO padding. The offset is
the sample's true prompt length, sidestepping the trainer's batch-max offset +
right-padding fragility (docs/archaeology.md §3).

STORAGE DISCIPLINE (CLAUDE.md §4): returned `[T, V]` logits are in-memory only.
Never write them to disk — re-run this forward if the full distribution is
needed again.
"""
from __future__ import annotations

from contextlib import nullcontext

import torch

from data_collator import SelfDistillationDataCollator

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover
    PeftModel = None


def teacher_mode(model):
    """Context replicating opsd_trainer.py:676-687 fixed_teacher branch.

    For a PEFT model, disables LoRA adapters so the forward uses the base
    (step-0) weights = the fixed teacher. For a plain model, a no-op.
    """
    if PeftModel is not None and isinstance(model, PeftModel):
        return model.disable_adapter()
    return nullcontext()


_TRANSITION_CACHE: dict[int, str] = {}


def _transition_prompt(tokenizer) -> str:
    """Return the exact `transition_prompt` string from the official collator.

    Imported (not hand-copied) so it can never drift from data_collator.py.
    Instantiating the collator mutates `tokenizer.padding_side`; we restore it.
    """
    key = id(tokenizer)
    if key not in _TRANSITION_CACHE:
        orig_side = tokenizer.padding_side
        collator = SelfDistillationDataCollator(tokenizer=tokenizer)
        _TRANSITION_CACHE[key] = collator.transition_prompt
        tokenizer.padding_side = orig_side
    return _TRANSITION_CACHE[key]


def build_teacher_prompt_text(tokenizer, problem: str, solution: str,
                              teacher_thinking: bool = True) -> str:
    """Verbatim replica of the teacher prompt in data_collator.py:96-109."""
    transition_prompt = _transition_prompt(tokenizer)
    teacher_user_message = (
        f"Problem: {problem}\n\n"
        f"Here is a reference solution to this problem:\n"
        f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
        f"{transition_prompt}\n"
        f"Please reason step by step, and put your final answer within \\boxed{{}}."
    )
    messages = [{"role": "user", "content": teacher_user_message}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=teacher_thinking,
    )


def build_student_prompt_text(tokenizer, problem: str,
                              student_thinking: bool = False) -> str:
    """Verbatim replica of the student prompt in data_collator.py:66-74."""
    student_user_message = (
        f"Problem: {problem}\n\nPlease reason step by step, and put your final "
        f"answer within \\boxed{{}}."
    )
    messages = [{"role": "user", "content": student_user_message}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=student_thinking,
    )


def forward_rollout_logits(model, tokenizer, prompt_text: str,
                           rollout_token_ids) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Single-sample, zero-padding teacher-forcing forward.

    Concatenates `prompt_text` tokens with `rollout_token_ids`, forwards once
    (no_grad, model's own dtype), and slices `logits[:, P-1:-1]` so row t is the
    distribution predicting rollout token t (opsd_trainer.py:633-645/:689).

    Returns (logits[T, V] fp32, log_probs_on_rollout[T] fp32, prompt_length P).
    """
    device = next(model.parameters()).device
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    P = int(prompt_ids.shape[1])
    rollout = torch.tensor(rollout_token_ids, dtype=torch.long,
                           device=device).unsqueeze(0)
    input_ids = torch.cat([prompt_ids, rollout], dim=1)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, P - 1:-1, :].float().squeeze(0)          # [T, V]
    log_probs = torch.log_softmax(logits, dim=-1)
    lp_on = log_probs.gather(-1, rollout.squeeze(0).unsqueeze(-1)).squeeze(-1)  # [T]
    return logits, lp_on, P


def score_with_privilege(model, tokenizer, problem: str, rollout_token_ids,
                         privileged_solution: str,
                         teacher_thinking: bool = True) -> dict:
    """Score a rollout under the privileged (solution-conditioned) teacher.

    fixed_teacher semantics are the CALLER's responsibility: wrap in
    `with teacher_mode(model):` to score with adapters disabled (base teacher).

    Returns {"logits": [T, V] fp32, "log_probs_on_rollout": [T],
             "teacher_prompt_length": int}. See module docstring for the
    storage-discipline constraint on the returned logits.
    """
    prompt_text = build_teacher_prompt_text(
        tokenizer, problem, privileged_solution, teacher_thinking)
    logits, lp_on, P = forward_rollout_logits(
        model, tokenizer, prompt_text, rollout_token_ids)
    return {
        "logits": logits,
        "log_probs_on_rollout": lp_on,
        "teacher_prompt_length": P,
    }
