# OPSD version genealogy — paper-OPSD vs repo-OPSD

Provenance study for the leakage bridge experiment. Establishes what the
published OPSD paper actually describes versus what the released code does, and
where the five "leaky-config" knobs sit in that lineage. Feeds paper §3 / the
appendix version table and the framework gate-D annotation.

> **One-line finding.** Four of the five bridge knobs are genuine repo-vs-paper
> deltas (clip **added**, guard **strengthened**, student thinking **flipped
> off**, generation **shortened**); the fifth — a **fixed** initial-policy
> teacher — is **paper-original**, not a silent repo fix. The bare run's
> *dynamic* teacher therefore goes **beyond** paper-OPSD and is a max-leakage
> probe, not a paper replica.

---

## 1. Timeline (git-verifiable + external anchors)

| date | event | source |
|---|---|---|
| 2026-01 | paper v1 (arXiv 2601.18734) | external (user anchor) |
| 2026-03-03 | repo public first release | external (user anchor) — **not in git history** |
| **2026-03-18** | **git root commit `85d6bf9` "updated code"** (3639 lines, all knobs present) | `git log --reverse` |
| 2026-03-20 | paper last revised | arXiv metadata |
| 2026-04-03 | RLSD (observes strong behavioral leakage on a bare config) | external (user anchor) |
| 2026-04-08 | `0feada9` "save steps" | git |
| 2026-05-06 | `401b849` "Add non-thinking mode scripts" — `student_thinking` knob introduced | `git log -S` |
| 2026-07-03 | `9f47e1b` 3xH200 repro scripts (this cluster adaptation) | git |

**Archaeology limitation (report as-is).** The current git history's earliest
commit is `85d6bf9` (2026-03-18); `git show 85d6bf9^` is `fatal` — it is a root
commit. The **2026-03-03 first release is not represented in this git history**
(squashed / force-replaced / different repo), so the 3.03→3.18 knob delta
**cannot be established by git diff**. All paper-vs-repo claims below rest on
the paper *text* (arXiv HTML), not on a git diff against 3.03.

---

## 2. Per-knob table — paper-OPSD vs repo-OPSD

Paper column = arXiv 2601.18734 (method + Fig 2 prompt + Tables 5/6). Repo
column = git tree from `85d6bf9` (3.18) onward.

| knob | paper-OPSD (v1) | repo-OPSD (git 3.18+) | delta |
|---|---|---|---|
| per-token JSD clip | **absent** | `jsd_token_clip=0.05` (default) | **repo ADDED** |
| teacher prompt guard | mild: *"try to solve … using your own approach"* (Fig 2) | strong: *"do not copy or paraphrase … using your own words"* (`data_collator.py`) | **repo STRENGTHENED** |
| teacher policy | **fixed initial policy** (*"we fix the teacher policy to be the initial policy … acts as regularization"*) | fixed via LoRA-disabled base (`fixed_teacher`) | **same** (paper-original) |
| student rollout mode | **thinking ON** (Table 5) | thinking OFF default (`student_thinking=False`) | **repo FLIPPED off** |
| max generation length | **2048** (Table 6) | 1024 (main repro script) | **repo SHORTENED** |

Net: the repo silently **adds a divergence clip**, **hardens the anti-copy
guard**, **turns off student thinking**, and **halves the generation budget**,
while **keeping** the paper's fixed teacher. Each of the first four is a
plausible leakage-suppressor; the clip and guard in particular read as
after-the-fact mitigations absent from the published method.

---

## 3. The three guard texts (verbatim)

The transition/guard sentence exists in three distinct wordings; the bridge
experiment must not conflate them.

- **paper-OPSD v1** (arXiv 2601.18734, Fig 2, mild):
  > "After understanding the reference solution, please try to solve this problem using your own approach below:"
- **repo-OPSD** (`data_collator.py`, hardened):
  > "After reading the reference solution above, make sure you truly understand the reasoning behind each step — do not copy or paraphrase it. Now, using your own words and independent reasoning, derive the same final answer to the problem above. Think step by step, explore different approaches, and don't be afraid to backtrack or reconsider if something doesn't work out:"
- **experiment-A neutral** (`run_opsd_1b_3gpu_noguard.sh`, a *third* text, NOT used by the bridge):
  > "Now, derive the final answer to the problem above."

Tier 1 uses the **paper v1 mild** text verbatim via `--transition_prompt_override`.

---

## 4. Two-tier bridge design

Superseded the earlier single "bare" (4096, dynamic, neutral-guard) config once
the archaeology established that fixed-teacher is paper-original. The bridge is
now two runs differing by a single knob (the teacher axis), so a positive/negative
split cleanly attributes the dynamic feedback loop.

### Tier 1 — paper-OPSD v1 faithful reproduction (MAIN)
`scripts/run_tier1_paper_opsd.sh`, run_config `qwen31b_paper_opsd_v1`.

| knob | Tier 1 value | vs paper v1 | vs repo |
|---|---|---|---|
| clip | 0 | = paper | undoes repo add |
| guard | paper v1 mild (verbatim, §3) | = paper | undoes repo hardening |
| student thinking | ON | = paper (Table 5) | undoes repo flip |
| generation | **2048** / 300 steps | = paper (Table 6) | longer than repo (1024) |
| teacher | **fixed** initial policy | = paper (kept) | = repo |

LoRA is retained (paper trains the same way). **LoRA is therefore NOT a
suppressor candidate**: it is common to paper and repo, and the RLSD-attacked
paper version has LoRA yet leaks, so it cannot explain the paper-vs-repo
difference. The suppressor-candidate set is the four repo deltas (clip, guard,
thinking, length) plus the absent dynamic teacher (Tier 2), not LoRA.

**Batch note.** Per-device micro-batch is 2 (down from repro's 5, for the 2048
memory budget) but grad-accum is raised to 5, so the effective batch is 2×5×3 =
**30 — identical to the repro's 5×2×3 = 30**. Only the micro-batch differs
(math-equivalent for LoRA; it changes memory/speed, not the optimizer step). And
even if the effective batch did differ, leakage is a *behavioral-emergence*
observation, not a performance benchmark, so a batch delta would not confound a
leak/no-leak read.

**Science question: does the version RLSD attacked leak on our model/data?**

### Tier 2 — dynamic-teacher variant (aggressive probe, BEYOND paper)
`scripts/run_tier2_dynamic.sh`, run_config `qwen31b_tier2_dynamic`. **Exactly
Tier 1 with `--fixed_teacher` removed** → teacher = current student + privileged
context (`compute_loss` nullcontext path = teacher-student feedback loop). Run
**only if Tier 1 is negative**.

> **Scheduling note.** An earlier 4096/dynamic/neutral-guard "bare" run (job
> 30189) was submitted before this two-tier revision and **OOM'd** (batch-4 @4096
> full-vocab JSD forward hit 136/140 GiB). It is retired, not revived: its 4096
> length and neutral guard no longer match the Tier-2 definition (2048, paper
> guard). Tier 2 is the correct single-knob sibling of Tier 1.

---

## 5. Decision tree & identity

The bridge is a **version-genealogy evaluation**, not a new method. Tier 1 tests
the published method; Tier 2 tests the dynamic-loop hypothesis. Four outcomes:

| outcome | reading |
|---|---|
| **T1 positive** | RLSD leakage reproduces on paper-OPSD. The four repo deltas (clip, guard, thinking, length) are the suppressor-candidate set → bisection (restore one repo value at a time, 150-step short runs) localizes the minimal killing set. |
| **T1 negative, T2 positive** | Leakage needs the dynamic teacher-student feedback loop; both paper and repo are protected by the **fixed teacher** — fixed-teacher is then the single strongest suppressor. Bisection focuses on the remaining knobs *on top of* T2. |
| **both positive** | Follow the T1 branch; T2 serves as an intensity/upper-bound control. |
| **both negative** | Escalate to 8B (Tier-1 config first). If still clean, report honest non-reproduction; residual gap to RLSD (VL model / its data / its implementation) is the reproducibility boundary, tabled alongside this genealogy. |
