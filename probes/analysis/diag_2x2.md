# 2x2 corruption diagnostic (stage 1)

Per-token teacher-corruption sensitivity `JSD(T_S, T_S̃)` and teacher-student divergence `JSD(T_S, S)` on ckpt-50 rollouts (`probes/run_2x2.py`). correct/wrong from the 4096 collection.

- correct rollouts: 107 | wrong rollouts: 80

## (b) corruption stability (3 corrupted versions)

- top-10% token-set **Jaccard** (pairwise): mean=0.3835 median=0.3691 q25=0.2911 q75=0.4571 n=467
- token-level **Spearman** (pairwise jsd_corruption): mean=0.8472 median=0.8542 q25=0.8244 q75=0.8843 n=467

## (c) Gate A — divergence mass outside corruption-sensitive tokens (correct)

- mean JSD(T_S,S) on **corruption-sensitive** (top-10%) tokens: mean=0.0822 median=0.0813 q25=0.0687 q75=0.0906 n=107
- mean JSD(T_S,S) on **insensitive** tokens: mean=0.0227 median=0.0228 q25=0.0169 q75=0.0279 n=107
- **insensitive tokens' share of total teacher-student divergence mass**: mean=0.6967 median=0.7118 q25=0.6530 q75=0.7515 n=107
  - Read: a high insensitive-share means substantial student-teacher divergence lives OUTSIDE copying positions (competence-driven), which argues for a non-trivial correct-branch treatment; a low share means divergence is mostly at corruption-sensitive (privilege) tokens.

## (d) Gate C — correction-signal concentration on answer span (wrong x student-wrong)

- concentration (answer-span JSD mass / total): mean=0.3363 median=0.1704 q25=0.0192 q75=0.6761 n=22
  - Compares against the prior observation that the correction signal sits almost entirely at the answer position. concentration→1 supports it.

## (a)/(e) position curves

- `diag_2x2_position_correct.png` (top-left), `diag_2x2_position_wrong.png` (bottom-left): corruption sensitivity by relative position, layered by token category + answer span.

## Truncated bucket (1024 collection, no corruption)

- n=138 truncated rollouts; teacher-student divergence position curve + V(t) bundle in `diag_truncated.png`.
- V(end) on truncated: mean=-12.887 median=-9.801 nats — the training signal on the dominant (69% @1024) truncated bucket, which the verifier cannot score.
