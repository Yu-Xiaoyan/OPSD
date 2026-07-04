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

- overall drift share: [np.float64(38.1), np.float64(50.8), np.float64(55.9), np.float64(59.6), np.float64(61.3), np.float64(62.1)] (steps [25, 50, 75, 100, 125, 150])
- monotone non-decreasing (tol 0.5pp): **True**
- steepest rise segment: **25→50** (+12.6pp)
- performance plateau (AIME24): base 49.2 → step50 52.5 → step100 55.0; overlay in `drift_over_training.png`.
- Mechanism read: if drift share rises monotonically and its steepest segment sits around 75–100 (the performance plateau), the privileged teaching signal is being progressively swamped by drift (target -> KL-to-init). Reported as-is; not forced.
