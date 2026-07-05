# Gate B — V(t) final review (task 3c)

Descriptive evidence only (no threshold verdict). Model: 1.7B + ckpt-50. t* = human first-error step; V(t) most-negative-ΔV segment mapped back to a \n\n step; hit = |step_V - t*| <= 1.

## Clear-set hit table — version A (original verdicts)  (n=10)

| pid | human t* | V(t) step | ΔV-min tok | ΔV | hit(±1) |
|--:|--:|--:|--:|--:|:--:|
| 402 | 99 | 97 | 3136 | -16.035 | ✗ |
| 429 | 68 | 65 | 1426 | -8.421 | ✗ |
| 441 | 28 | 85 | 2295 | -6.703 | ✗ |
| 347 | 90 | 95 | 2876 | -5.008 | ✗ |
| 380 | 30 | 26 | 1112 | -13.896 | ✗ |
| 466 | 113 | 37 | 732 | -6.648 | ✗ |
| 566 | 124 | 152 | 3072 | -5.234 | ✗ |
| 361 | 16 | 5 | 99 | -7.865 | ✗ |
| 244 | 13 | 22 | 660 | -6.336 | ✗ |
| 526 | 18 | 25 | 704 | -11.345 | ✗ |

**hits: 0/10 (±1 step)**
Miss modes:
- pid=402: V(t) drop at step 97 vs t*=99 — V drop earlier than annotated error.
- pid=429: V(t) drop at step 65 vs t*=68 — V drop earlier than annotated error.
- pid=441: V(t) drop at step 85 vs t*=28 — V drop later (post-error collapse).
- pid=347: V(t) drop at step 95 vs t*=90 — V drop later (post-error collapse).
- pid=380: V(t) drop at step 26 vs t*=30 — V drop earlier than annotated error.
- pid=466: V(t) drop at step 37 vs t*=113 — V drop earlier than annotated error.
- pid=566: V(t) drop at step 152 vs t*=124 — V drop later (post-error collapse).
- pid=361: V(t) drop at step 5 vs t*=16 — V drop earlier than annotated error.
- pid=244: V(t) drop at step 22 vs t*=13 — V drop later (post-error collapse).
- pid=526: V(t) drop at step 25 vs t*=18 — V drop later (post-error collapse).

## Clear-set hit table — version B (borderline reclassified)  (n=11)

| pid | human t* | V(t) step | ΔV-min tok | ΔV | hit(±1) |
|--:|--:|--:|--:|--:|:--:|
| 402 | 99 | 97 | 3136 | -16.035 | ✗ |
| 429 | 68 | 65 | 1426 | -8.421 | ✗ |
| 441 | 28 | 85 | 2295 | -6.703 | ✗ |
| 347 | 90 | 95 | 2876 | -5.008 | ✗ |
| 380 | 30 | 26 | 1112 | -13.896 | ✗ |
| 440 | 38 | 17 | 384 | -5.893 | ✗ |
| 466 | 113 | 37 | 732 | -6.648 | ✗ |
| 566 | 124 | 152 | 3072 | -5.234 | ✗ |
| 361 | 16 | 5 | 99 | -7.865 | ✗ |
| 244 | 13 | 22 | 660 | -6.336 | ✗ |
| 526 | 18 | 25 | 704 | -11.345 | ✗ |

**hits: 0/11 (±1 step)**
Miss modes:
- pid=402: V(t) drop at step 97 vs t*=99 — V drop earlier than annotated error.
- pid=429: V(t) drop at step 65 vs t*=68 — V drop earlier than annotated error.
- pid=441: V(t) drop at step 85 vs t*=28 — V drop later (post-error collapse).
- pid=347: V(t) drop at step 95 vs t*=90 — V drop later (post-error collapse).
- pid=380: V(t) drop at step 26 vs t*=30 — V drop earlier than annotated error.
- pid=440: V(t) drop at step 17 vs t*=38 — V drop earlier than annotated error.
- pid=466: V(t) drop at step 37 vs t*=113 — V drop earlier than annotated error.
- pid=566: V(t) drop at step 152 vs t*=124 — V drop later (post-error collapse).
- pid=361: V(t) drop at step 5 vs t*=16 — V drop earlier than annotated error.
- pid=244: V(t) drop at step 22 vs t*=13 — V drop later (post-error collapse).
- pid=526: V(t) drop at step 25 vs t*=18 — V drop later (post-error collapse).

## Proxy threshold δ (training-time clear/diffuse split)

- clear most-neg ΔV: median=-7.284 (n=10)
- diffuse most-neg ΔV: median=-7.974 (n=20)
- **proxy δ = -7.629** (midpoint of medians); see gate_b_dv_dist.png. Training-time rule: most-neg segment ΔV < δ -> treat as clear (t* hard mechanism), else diffuse (ΔV soft weighting).

## Format-noise robustness (V(end))

- pseudo_wrong_format (human=correct): mean=-15.77 n=11
- true_wrong (clear+diffuse): mean=-25.26 n=30
- correct (reference, n=30): mean=-10.04
- **AUC(pseudo vs true_wrong) = 0.700** (>=0.75 => V(t) robust to reward format noise)
- **AUC(correct vs true_wrong) = 0.779** (clean-label discriminability; cf. prior 0.812)
- version B (borderline): AUC(pseudo vs true_wrong)=0.598, AUC(correct vs true_wrong)=0.772
