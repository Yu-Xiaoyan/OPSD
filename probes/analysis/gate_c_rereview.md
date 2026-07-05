# Gate C re-review on v2 wrong bucket (task 3d)

Same diag2x2 corruption data; wrong bucket re-defined by verifier v2 (format-recovered pseudo removed via pid-join to source rollouts). LIFT = (answer-span JSD-mass frac)/(answer-span token frac); lift>1 = correction signal denser on the answer span than uniform. Corruption-null & no-reasoning excluded.

- wrong (v1): 80 | wrong (v2): 71 (v2 removed 9 format-recovered; 0 kept unjoined)

## Corruption-null & quality floor (v1 vs v2 wrong)

| metric | v1 wrong | v2 wrong |
|---|---|---|
| null occupancy | 62/80 (77.5%) | 55/71 (77.5%) |
| corruption mass (nats) | mean=0.4535 median=0.2305 q25=0.1613 q75=0.4087 n=80 | mean=0.4610 median=0.2355 q25=0.1598 q75=0.4273 n=71 |
| big tokens (>0.05) | mean=1.0750 median=0.0000 q25=0.0000 q75=1.0000 n=80 | mean=1.1127 median=0.0000 q25=0.0000 q75=1.0000 n=71 |

## Correction-signal LIFT — both corruption axes (v1 vs v2 wrong)

| axis | bucket | no-reason excl | lift |
|---|---|---|---|
| studentwrong | v1 | 1/9 (11.1%) | mean=7.6441 median=3.2998 q25=0.5257 q75=11.8200 n=8 |
| studentwrong | v2 | 1/7 (14.3%) | mean=10.1100 median=6.7364 q25=2.8740 q75=16.8170 n=6 |
| irrelevant | v1 | 1/9 (11.1%) | mean=9.3106 median=2.2105 q25=0.4174 q75=10.4636 n=8 |
| irrelevant | v2 | 1/7 (14.3%) | mean=12.3921 median=4.2054 q25=1.7763 q75=20.7271 n=6 |

Read: v2 removes format-recovered pseudo (student math correct) from the wrong bucket. If the v2 lift is >= v1 lift, the remaining true-wrong rollouts carry a correction signal at least as concentrated on the answer span — i.e. the pseudo cases were diluting, not driving, gate C.
