# v0 main run — pre-registered predictions & reading protocol

**Pre-registration date: 2026-07-06**, before the v0 main run (`qwen31b_v0_main`)
has produced any eval. This is a TRUE pre-registration (unlike the bisection
corruption predictions, which were confirmatory because the data already existed).

## Run under test
`qwen31b_v0_main`: OPSDGatedTrainer, repro-identical to the OPSD baseline
(1.7B, global batch 30, 150 steps, save 25, TM-off, 1024, fixed_teacher,
clip 0.05, seed as main repro) with ONE change — `--gated` (frozen v0 gating:
verifier-v2 triage + real V(t) ΔV soft weighting on wrong, structural-token
downweight on correct, V(end) split on truncated; no unlikelihood; corruption
gate OFF). Single seed first; multi-seed only after this passes.

## Reading protocol (fixed before results)
Eval: AIME24 / AIME25 avg@12 at ckpt 50 / 100 / 150, top_p 1.0,
`summarize_eval` side-by-side with the main-repro OPSD baseline.

- **Primary metric**: gain over base (untrained Qwen3-1.7B) for v0 vs the gain
  for OPSD. OPSD's gains are **+5.8 (AIME24) / +8.1 (AIME25)**. v0 is judged on
  whether its gain over base beats OPSD's.
- **Secondary metric**: peak step + post-peak stability.

## Pre-registered prediction (drift-erosion hypothesis)
The drift-erosion hypothesis holds that OPSD's ~100-step plateau/saturation comes
from LoRA drift swamping the privileged signal. v0's gating (down-weighting
low-value / V-drop tokens, structural tokens, and degraded-truncated segments)
should **mitigate that saturation**. Concretely, pre-registered:

> **P: v0's peak step is LATER than OPSD's, and/or v0's post-peak stability is
> better than OPSD's** (smaller drop from peak by step 150).

Falsification: if v0 peaks no later than OPSD AND degrades as fast or faster
post-peak, the drift-erosion mitigation claim is not supported by this run.

This is a *secondary*, mechanism-flavored prediction; the primary
accept/reject is the gain-vs-OPSD comparison above. Reported as-is either way.
