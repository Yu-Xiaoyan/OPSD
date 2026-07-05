"""GatedDataCollator (v0) — SelfDistillationDataCollator + triage inputs.

Adds the ground-truth answer and raw problem text to each batch so the gated
trainer can verifier-bucket the on-policy rollouts at loss time. Everything else
(student/teacher prompt construction, padding, lengths) is unchanged, so a v0
run is byte-for-byte identical to OPSD except for the two extra list[str] keys.

The extra keys are non-tensor and survive HF `_prepare_inputs`; the trainer adds
"Answer" to its signature columns so `remove_unused_columns` keeps it.
"""
from __future__ import annotations

from data_collator import SelfDistillationDataCollator


class GatedDataCollator(SelfDistillationDataCollator):
    def __call__(self, features):
        result = super().__call__(features)
        result["gt_answers"] = [str(f.get("Answer", "")) for f in features]
        result["problems"] = [f.get("problem", "") for f in features]
        return result
