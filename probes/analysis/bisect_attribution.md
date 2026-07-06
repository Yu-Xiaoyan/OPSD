# Suppressor attribution table (bisection main result)

Each variant = Tier 1 (paper-OPSD v1) with ONE knob restored to its repo value;
everything else held at Tier 1. Behavioral window 105-195, corruption window
150-195 (high-leakage). Corruption mass = config-level median Σ JSD(T_S, T_S̃),
teacher = fixed base. repo-OPSD correct-bucket baseline = 0.469.

| variant | keyword % | early_clean % | corruption median (×repo) |
|---|--:|--:|--:|
| **Tier 1 baseline** | 11.79 | 11.09 | 2.652 (5.7×) |
| **+clip** (0→0.05) | 0.32 | 5.47 | 0.609 (1.3×) |
| **+guard** (repo hardened) | 1.37 | 11.09 | 2.303 (4.9×) |
| **+length** (2048→1024) | 0.42 | 2.43 | 1.019 (2.2×) |
| **+tmoff** (thinking off) | 0.00† | 1.82 | 0.616 (1.3×) |

† keyword→0 is **expected channel closure** (citation lives inside `<think>`),
not suppression — tmoff is judged by early-emission + corruption per the bisection
instruction.

## Findings

- **No single minimal killing set — redundant suppression.** clip, length, and
  tmoff each *independently* cut early-emission (the real behavioral leakage:
  answer surfacing early, not confined to `<think>`) from 11% down to 1.8–5.5%,
  and cut config-level corruption toward the repo baseline.
- **guard is a SURFACE suppressor.** It cuts keyword phrasing to 1.37% but leaves
  early-emission **unchanged (11.09% = Tier 1)** AND config-level corruption
  essentially intact (4.9× vs Tier 1's 5.7×). The anti-copy guard teaches the
  student not to *say* "reference solution" — it does not reduce the student's
  actual dependence on it. Behavioral masking, not pathology removal.
- **clip and tmoff push corruption back to the repo baseline (1.3×).** They remove
  the *distributional* privilege dependence, not just the surface citation — the
  strongest true suppressors.
- **length is intermediate** (corruption 2.2×, early 2.43%).
- **True-suppression ranking** (early-emission + corruption, not keyword):
  **clip ≈ tmoff (strong, back to repo) > length (partial) > guard (surface-only,
  not a real suppressor despite killing the keyword metric).**

## Implication for the paper

The repo's silent mitigation set does not act through one mechanism, and the knob
that *looks* like the anti-leakage measure (the guard) is the **least** effective
— cosmetic at the corruption/early-emission level. The dissociation between the
keyword metric and the early-emission/corruption metrics is exactly why a
keyword-only leakage screen would **misrank** the suppressors: it would credit the
guard and could miss that clip/TM-off are what actually remove the dependence.
This is the case for measuring leakage distributionally (the corruption probe),
not just behaviorally.

Sources: `bisect_attribution.txt` (keyword + early-emission), `bisect_corruption.txt`
(corruption), `leakage_tier1_paper.md` (Tier-1 positive control).
