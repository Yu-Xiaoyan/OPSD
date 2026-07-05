# t* annotation set (task 3b) — verifier audit ground truth

`tstar_annotations.jsonl` — 49 stratified wrong rollouts (length-tertile x
self-correction marker, seed 42), hand-annotated by Claude Code.

Fields: pid, tstar (int step | "uncertain"), error_type
{calculation/logic/misread/uncertain}, has_marker, confidence {high/low/medium},
verdict {true_wrong_clear_tstar / true_wrong_diffuse / pseudo_wrong_format /
vacuous_proof / no_reasoning}, borderline (bool), reason.

## Verdict distribution (49)
clear 10 (20%) | diffuse 20 (41%) | pseudo 11 (22%) | vacuous 5 (10%) | no_reasoning 3 (6%)

## Reliability
Round-2 re-annotation of 15 (seed 7): 100% agreement (tstar ±1 step + verdict).
CAVEAT: single annotator + memory -> this is self-consistency, NOT inter-rater
reliability; treat as optimistic. Gate-B uses it with this caveat.

## Known limitations (documented per gate)
- **Window blind spot**: dump was marker-focused (up to first self-correction
  marker + 2 steps). When the marker fires early but the true error is later
  (pid 229/485/494/462/577/369), the exact first-error step is beyond the window
  and is marked tstar="uncertain" (verdict still assigned). A full-dump re-pass
  would tighten these.
- **Suspicious gt** (EXCLUDED from gate-B/C use): pid 290 (gt=1 vs student's
  plausible count 66), pid 293 (gt=60 vs clean t=12). Likely dataset gt errors,
  not student errors.
- **borderline=true** (440/477/297/424): defensible either way; gate-B reports
  hit-rate under BOTH the original and the flipped verdict as a sensitivity check.

## Gate-B usable set
Hard t* localization is well-defined only for the 10 `true_wrong_clear_tstar`
(all conf=high, explicit step): pid 244/347/361/380/402/429/441/466/526/566.
