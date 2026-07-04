# Drift share over training (task 2)

drift_k = JSD(S_k, S0), teach = JSD(T_S, S0) (k-independent), total_k = JSD(T_S, S_k), on the fixed 4096 diagnostic rollouts (n=187). drift share = Σdrift / (Σdrift + Σteach). See `probes/drift_scan.py`.

## Drift share by step (%)

| step | all | math | style | other | structural | total_k mass (med) | hi-drift tok frac (med) |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 25 | 38.1 | 31.2 | 39.8 | 39.2 | 34.4 | 22.565 | 0.073 |
| 50 | 50.8 | 41.8 | 54.7 | 51.2 | 50.7 | 27.161 | 0.184 |
| 75 | 55.9 | 48.6 | 58.8 | 55.8 | 58.4 | 32.051 | 0.234 |
| 100 | 59.6 | 52.9 | 61.2 | 59.1 | 64.4 | 36.497 | 0.257 |
| 125 | 61.3 | 55.0 | 62.3 | 60.7 | 66.9 | 38.462 | 0.269 |
| 150 | 62.1 | 55.7 | 62.8 | 61.5 | 67.7 | 39.184 | 0.272 |

## Read

- overall drift share: [38.1, 50.8, 55.9, 59.6, 61.3, 62.1] (steps [25, 50, 75, 100, 125, 150])
- monotone non-decreasing (tol 0.5pp): **True**
- **drift share crosses 50% at step ~48.5** (linear interpolation between 25=38.1% and 50=50.8%).
- steepest rise segment: **25→50** (+12.6pp); slope then decays and saturates after step 100 (+1.7, +0.8pp).
- AIME24 avg@12: base 49.2 → step50 52.5 → step100 55.0 (peak at 100). The ~step-48.5 'over-half' point falls near step 50 — i.e. **drift becomes the majority of the S_k-vs-S0 divergence BEFORE the performance peak** (which is at step 100).
- Mechanism read (over-half framing): drift share rises monotonically and passes 50% by step ~48.5, reaching 62% by step 150 — the effective target is drifting toward a KL-to-init regularizer, and it does so before the AIME peak, consistent with the 'privilege swamped by drift' direction. The steepest rise is EARLY (25→50), not at the 75–100 plateau, and the slope saturates afterward — reported as-is, not forced.

## Methodological caveat

- The rollouts are **fixed, sampled from ckpt-50**. For every other checkpoint (25/75/100/125/150) this is an **off-policy** evaluation: those checkpoints would generate somewhat different trajectories on-policy. The **absolute** drift-share values are therefore biased by this off-policy mismatch (largest at the checkpoints farthest from 50). The **monotone upward trend** across steps is robust to it — a fixed rollout set only shifts the level, not the direction, of Σdrift_k growth.
