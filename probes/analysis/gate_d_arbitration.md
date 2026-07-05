# Gate D — 8B vs 1.7B leakage-axis arbitration (step 4)

Side-by-side leakage evidence at two scales. Behavioral probes on the same ckpt-50 0.3 rollouts (like-for-like); distributional probes on the 2x2 corruption scan at the same ROLLOUT_CAP=1024. Hypothesis under test: does the leakage/privilege axis become active at 8B?

## Behavioral leakage (ckpt-50 0.3 rollouts, like-for-like)

| probe | 1.7B | 8B |
|---|---|---|
| keyword citation | 1/200 (0.5%) | 1/200 (0.5%) |
| answer early-emission (raw, pos<0.3) | 24/158 (15.2%) | 25/158 (15.8%) |
| early-emission (clean: -proof -in-prompt) | 6/158 (3.8%) | 7/158 (4.4%) |
| early-emission (strong: +prefix<40w) | 0/158 (0.0%) | 1/158 (0.6%) |

## Distributional leakage (2x2 corruption scan, cap=1024)

| metric | 1.7B | 8B |
|---|---|---|
| corruption-null, correct | 56/107 (52.3%) | 79/136 (58.1%) |
| corruption-null, wrong | 62/80 (77.5%) | 34/51 (66.7%) |
| corruption mass correct (median) | 0.469 | 0.379 |
| corruption mass wrong (median) | 0.230 | 0.205 |
| gate-C lift studentwrong (median, n) | 3.30 (n=8) | 3.65 (n=5) |
| gate-C lift irrelevant (median, n) | 2.21 (n=8) | 5.68 (n=5) |

## Arbitration

- corruption-null shift 1.7B->8B: correct +5.8 pp, wrong -10.8 pp. A large DROP in null occupancy at 8B would mean the leakage axis activates with scale; a flat/absent shift means it does not.
- Behavioral leakage near-zero at both scales (see table) => the behavioral-scale hypothesis (bigger model verbally/answer-leaks more) is not supported on this data.
- Read the two together for the framework gate-D distributional verdict.
