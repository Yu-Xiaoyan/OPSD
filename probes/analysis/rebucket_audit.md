# v2 rebucket audit + 49-verdict confusion (task 3, pre-3d)

verifier v2 = any-\boxed match + multiple-choice letter<->value mapping. Compares against v1 (last-boxed only). Confusion matrix tests v2 bucket against the 49 human verdicts.

## v1 -> v2 bucket migration (full collections)

| source | n | v1 wrong | v2 wrong | recovered (wrong->correct) | recover% |
|---|--:|--:|--:|--:|--:|
| 1.7B ckpt-50 @4096 | 200 | 80 | 71 | 9 | 11.2% |
| wrong_extra (3a) | 150 | 150 | 138 | 12 | 8.0% |
| 8B ckpt-50 @4096 | 200 | 51 | 39 | 11 | 21.6% |
| **all** | | **281** | | **32** | **11.4%** |

Non-trivial off-diagonal (v1!=v2) transitions:
- [1.7B ckpt-50 @4096] wrong -> correct: 9
- [wrong_extra (3a)] wrong -> correct: 12
- [8B ckpt-50 @4096] wrong -> correct: 11
- [8B ckpt-50 @4096] wrong -> truncated: 1

## Confusion matrix: human verdict (rows) x v2 bucket (cols)

| human verdict \ v2 | correct | wrong | truncated | n |
|---|--:|--:|--:|--:|
| true_wrong_clear_tstar | 0 | 10 | 0 | 10 |
| true_wrong_diffuse | 1 | 19 | 0 | 20 |
| pseudo_wrong_format | 3 | 8 | 0 | 11 |
| vacuous_proof | 0 | 5 | 0 | 5 |
| no_reasoning | 0 | 3 | 0 | 3 |

- **pseudo_wrong_format recovered by v2**: 3/11 -> correct (format false-wrong fix).
- **true_wrong kept wrong**: 29/30 (v2 did not leak true errors into correct: 1 leaked).
- vacuous_proof: {'wrong': 5}; no_reasoning: {'wrong': 3} (semantic-only; v2 not expected to fix).

**v2 wrong bucket (post-rebucket) = true_wrong + residual semantic (vacuous/no_reasoning) - format-recovered pseudo.** This is the bucket gate C (3d) re-reviews.