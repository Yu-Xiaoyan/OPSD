# CLAUDE.md — OPSD project conventions

On-Policy Self-Distillation (OPSD) reproduction. The code is built on top of
`trl`'s experimental **GOLD** trainer. `main` mirrors the official release; this
repo is being adapted to run offline on the NTU PBS cluster.

> Terminology note: in this repo "OPSD" is the **official baseline method**
> (cot + a single privileged-context teacher). Do not describe it as a
> "two-teacher" or "selective-KL" method — that is a separate line of work.

---

## 1. Compute environment

- **Cluster:** NTU PBS (OpenPBS / `qsub`).
- **Hardware target:** a single node with **3 × H200** GPUs.
- **Queue:** `gpu_ded`. **Project code:** `ds_ccds_wei.lu`.
- Typical select line: `#PBS -l select=1:ncpus=12:ngpus=3`.
- Submit jobs with `qsub pbs/<name>.pbs`. Keep every PBS script under `pbs/`
  with a clear, descriptive name (see `pbs/smoke_test.pbs` as the template).
- `pbs/logs/` holds run logs and must not be committed.

## 2. Compute nodes have NO internet — everything must run offline

The GPU nodes cannot reach the network. Every script must be offline-safe:

- Export **`HF_HUB_OFFLINE=1`** and **`HF_DATASETS_OFFLINE=1`** and
  **`WANDB_MODE=offline`** in every training/eval job.
- Load models from **local paths under `~/models/`** (e.g.
  `~/models/Qwen3-1.7B`). The upstream scripts hard-code paths such as
  `/data0/shared/Qwen3-1.7B` — these must be repointed to `~/models/…` on
  `repro`.
- **Do not write code that can silently trigger a network download.** In
  particular, any `load_dataset("<hub-name>")` / `from_pretrained("<hub-name>")`
  must resolve from a pre-populated HF cache, and any `trust_remote_code=True`
  dataset (aime25, hmmt25) needs its loader script cached in advance. If a
  dataset/model is not already local, download it on a login node first, never
  from inside a job.
- No implicit calls to WandB servers, HF Hub, `trackio`, etc. from compute
  nodes.

## 3. Conda environment

```bash
module load anaconda/2025          # then follow its printed hook line:
eval "$(/usr/local/anaconda2025/bin/conda shell.bash hook)"
conda activate opsd
module load cuda/13.1              # as needed for builds
```

- Env spec lives in `environment.yml` (Python 3.10; torch 2.8.0,
  transformers 4.57.1, trl 0.26.0, vllm 0.11.0, deepspeed 0.18.2, peft 0.17.1,
  datasets 3.6.0). `flash-attn==2.8.3` is installed separately
  (`pip install flash-attn==2.8.3 --no-build-isolation`).
- **Pin these versions.** The trainer's teacher/student logit alignment depends
  on the exact Qwen3 chat template shipped with `transformers`, and on the
  vLLM/flash-attn build. Version drift silently corrupts the KL signal.
- PBS scripts must reproduce the module-load + activate sequence above; use
  `pbs/smoke_test.pbs` as the canonical template.
- **Every `python` example in docs and scripts must be explicitly prefixed with
  `conda activate opsd`, or use the absolute interpreter
  `~/.conda/envs/opsd/bin/python`.** A fresh login shell defaults to the `base`
  environment, so a bare `python …` runs against `base` and fails on missing
  packages. Never write a runnable `python`/`accelerate` example that assumes
  the `opsd` env is already active.

## 4. Storage discipline

- **The home volume (NFS) is the only storage.** Total ~932G, currently ~319G
  free, and there is **no separate scratch** partition. Everything —
  checkpoints, caches, dumps, analysis outputs — competes for this one pool.
- **Probe / analysis pipelines may only persist *derived statistics*:**
  per-token JSD, `V(t)` sequences, top-k logit summaries, aggregated curve data.
  **Never save full-vocabulary logits.** One rollout's full-vocab fp32 logits is
  ≈ 600 MB, so ~1000 rollouts is ~600 G and will blow the disk. If you need the
  full distribution again, **re-run the forward pass** rather than caching it.
- **Check free space before any large batch job:** `df -h ~`.

## 5. Branch conventions

- **`main`** — official upstream mirror. Keep it clean; do not adapt it to the
  cluster.
- **`repro`** — reproduction/cluster adaptation (offline paths, 3-GPU launch,
  PBS wrappers). This is the working branch.
- **`probes`** — probe / analysis development.

**Large files never go into git:** model checkpoints, LoRA adapters,
`*.jsonl` / `*.json` generation dumps, eval result JSON, logs, and WandB runs.
(`.gitignore` already covers `outputs/`, `checkpoints/`, `wandb/`, `logs/`,
`pbs/logs/`, `*.jsonl`, `*.log`, and `results/*.o[0-9]*` alongside the
`__pycache__`/`*.pyc` defaults — keep new large-artifact paths added there.)

## 6. Repo map (entry points)

| File | Role |
|---|---|
| `opsd_trainer.py` | `OPSDTrainer` (subclass of `trl` `SFTTrainer`) — the core |
| `data_collator.py` | `SelfDistillationDataCollator` — builds student vs teacher prompts |
| `opsd_train.py` | OPSD training entry point / arg parsing |
| `sft_train.py`, `grpo_train.py` | SFT / GRPO baselines |
| `eval/evaluate_math.py` | vLLM eval (AIME24/25, HMMT25, MATH500, …) |
| `scripts/*.sh` | upstream launch commands (paths need repointing) |
| `accelerate.yaml` | DeepSpeed ZeRO-2 + CPU optimizer offload, bf16 |

See `docs/archaeology.md` for a detailed walkthrough of the mechanism, loss
semantics, and reproduction risks.
