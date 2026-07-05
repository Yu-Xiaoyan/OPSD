# Corruption-probe validity check on Tier-1 leaky run (task a)

Positive control: do behaviorally-leaking rollouts (keyword citation of the privileged solution in TM-on <think>) carry HIGHER corruption sensitivity JSD(T_S, T_S̃) than non-leaking ones? Teacher = base (fixed_teacher). Pooled Tier-1 dumps step>= 150.

- **high-leakage (kw hit)** corruption mass: mean=6.690 median=3.253 q25=0.956 q75=8.457 n=30
- **low-leakage (no hit)** corruption mass: mean=6.635 median=5.497 q25=0.666 q75=9.655 n=30
- repo-OPSD main-run baseline (correct bucket median): 0.469
- **AUC(high vs low) = 0.492** (>0.5 => leaking rollouts are more corruption-sensitive; the two leakage measures agree at trajectory level)
- high/low median ratio: 0.59x; high/baseline: 6.94x

**Read (corrected — the auto-generated optimistic line was wrong).**
- **Trajectory level: NULL.** AUC(high vs low)=0.49 ≈ chance; high median (3.25)
  is not above low (5.50). Keyword leakage does NOT predict corruption mass per
  rollout. This is expected once the subjects are separated: corruption
  sensitivity is a property of the **teacher's** dependence on the solution
  (T_S vs T_S̃ forward), whereas keyword leakage is whether the **student**
  verbalizes the reference. Different subjects → no trajectory-level correlation.
- **Config level: STRONG.** Both groups sit at ~14x the repo baseline (median
  3.3–5.5 vs 0.47). The leaky CONFIG (Tier-1: TM-on, 2048, paper guard) carries
  far higher distributional privilege dependence than the non-leaky repo config.
- **Validity verdict**: behavioral and distributional leakage correlate at the
  **config** level, not the **trajectory** level. The probe measures real
  privilege dependence of the config (14x lift), but a keyword-hit rollout is not
  individually the more corruption-sensitive one. Claim scoped accordingly; do
  not assert trajectory-level agreement.