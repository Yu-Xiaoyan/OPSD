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

## 3. The bridge "bare" config in this lineage

`scripts/run_opsd_1b_3gpu_bare.sh` flips five knobs toward maximal leakage:

| knob | bare value | vs paper | vs repo |
|---|---|---|---|
| clip | 0 (off) | = paper (no clip) | undoes repo add |
| guard | neutral connective | ≈ paper (mild) | undoes repo hardening |
| student thinking | ON | = paper | undoes repo flip |
| generation | 4096 / 300 steps | ≥ paper (2048) | longer than repo |
| teacher | **dynamic** | **beyond paper (fixed)** | undoes repo fixed |

So the bare run is **not** a byte-accurate paper replica: on the teacher axis it
is *more* leak-prone than paper-OPSD (dynamic vs fixed). It is deliberately the
**upper bound** of the leakage surface — if leakage does not appear here, the
softer paper/repo configs will not show it either. If it *does* appear, the
suppressor bisection (restore one knob at a time) localizes the minimal killing
set, and the fixed-teacher restoration tests whether paper-OPSD itself would
have leaked.

---

## 4. Identity of the bridge experiment

The bridge experiment is a **version-genealogy evaluation**, not a new method:
the five knobs are the repo's (partly silent) mitigation set layered on top of
the published method. A positive control (leakage reproduced on the bare config)
plus a suppressor bisection turns "we saw zero leakage" into "leakage is real
but suppressed by knobs X, Y" — or, if the bare config is also clean, into an
honest non-reproduction reported alongside this table (the RLSD observation may
rest on model/data specifics outside this repo).
