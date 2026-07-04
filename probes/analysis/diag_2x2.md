# 2x2 corruption diagnostic (stage 1)

Per-token teacher-corruption sensitivity `JSD(T_S, T_S̃)`, teacher-student divergence `JSD(T_S, S)`, and LoRA drift `JSD(S, S0)` (S0 = base on the student prompt) on ckpt-50 rollouts (`probes/run_2x2.py`). correct/wrong from the 4096 collection.

- correct rollouts: 107 | wrong rollouts: 80

## Quality floor & corruption-null (gate D read)

- corruption total mass (nats), correct: mean=0.7621 median=0.4690 q25=0.2095 q75=1.0203 n=107
- corruption total mass (nats), wrong: mean=0.4535 median=0.2305 q25=0.1613 q75=0.4087 n=80
- tokens with per-token corruption > 0.05, correct: mean=2.2430 median=1.0000 q25=0.0000 q75=3.0000 n=107
- tokens with per-token corruption > 0.05, wrong: mean=1.0750 median=0.0000 q25=0.0000 q75=1.0000 n=80
- **corruption-null** (total mass < 0.5 nats), **correct**: 56/107 (52.3%)
- **corruption-null**, **wrong**: 62/80 (77.5%)
  - Gate D read: a high corruption-null fraction = the teacher barely reacts to the answer being corrupted, i.e. the privilege/leakage axis is weak at this scale. All concentration/lift metrics below EXCLUDE null rollouts.

## (b) corruption stability (3 corrupted versions, non-null)

- top-10% token-set **Jaccard** (pairwise): mean=0.4593 median=0.4468 q25=0.3691 q75=0.5224 n=181
- token-level **Spearman** (pairwise jsd_corruption): mean=0.8738 median=0.8723 q25=0.8486 q75=0.9155 n=181

## (c) Gate A — divergence outside corruption-sensitive tokens (correct, non-null)

- mean JSD(T_S,S) on corruption-sensitive (top-10%) tokens: mean=0.0883 median=0.0852 q25=0.0791 q75=0.0971 n=51
- mean JSD(T_S,S) on insensitive tokens: mean=0.0231 median=0.0224 q25=0.0179 q75=0.0282 n=51
- **insensitive tokens' share of total teacher-student divergence**: mean=0.6840 median=0.7022 q25=0.6484 q75=0.7290 n=51

### Three-way mass table: privilege x drift x category
Fraction of total teacher-student divergence mass (%, correct non-null). Columns: sensitive/insensitive (privilege) x hi/lo LoRA-drift.
| category | sens·hi-drift | sens·lo-drift | insens·hi-drift | insens·lo-drift | row |
|---|--:|--:|--:|--:|--:|
| math | 0.7 | 2.1 | 0.9 | 3.5 | 7.2 |
| style | 2.1 | 0.8 | 2.1 | 1.6 | 6.7 |
| other | 10.6 | 10.8 | 17.6 | 34.1 | 73.1 |
| structural | 1.9 | 1.1 | 5.0 | 5.0 | 13.0 |
| **col** | 15.4 | 14.7 | 25.7 | 44.2 | 100 |
  - Read: mass in **insensitive x any-drift** = divergence not explained by privilege (copying); mass in **hi-drift** columns = attributable to LoRA drift; the **structural** row isolates markup/section tokens.

## (d) Gate C — correction-signal LIFT on answer span (wrong x student-wrong, non-null)

- no-reasoning sub-bucket (reasoning body < 20 tokens, excluded): 1/9 (11.1%)
- **lift** (answer-span JSD-mass frac / answer-span token frac), n=8: mean=7.6441 median=3.2998 q25=0.5257 q75=11.8200 n=8
- raw concentration (answer-span JSD mass / total): mean=0.1562 median=0.0728 q25=0.0215 q75=0.2314 n=8
- answer-span token fraction: mean=0.0256 median=0.0275 q25=0.0090 q75=0.0396 n=8
  - lift>1 = correction signal DENSER on the answer span than uniform. See `samples/prefix_failure_micro.html` (pid=19) for the token-level mechanism.

## Commit-point hypothesis (correct, non-null)

- top-5 corruption-token distance from the answer-segment start (tokens): mean=-85.7185 median=-9.0000 q25=-67.0000 q75=-5.0000 n=135
- fraction in a [-30, 0] pre-answer window: 53.3%; fraction strictly before the answer segment: 84.4%
  - If the top corruption tokens cluster just before the final answer segment, the teacher 'commits' to the answer in a small window — the commit-point mechanism. See `diag_2x2_commit.png` and 3 HTML stamps `samples/commit_point_*.html`.

## (a)/(e) position curves

- `diag_2x2_position_correct.png` (top-left), `diag_2x2_position_wrong.png` (bottom-left): corruption sensitivity by relative position, layered by token category + answer span.

## Truncated bucket (1024 collection, no corruption)

- n=138 truncated rollouts; JSD(T_S,S) position curve + V(t) bundle in `diag_truncated.png`.
- V(end): mean=-12.887 median=-9.801 nats — training signal on the dominant (69%@1024) truncated bucket the verifier cannot score.
- HTML stamps mark the most-negative delta_V checkpoint segment (delta_V is broadcast at checkpoint-segment granularity).
