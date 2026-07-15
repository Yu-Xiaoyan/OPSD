# 数字账本（number ledger）—— Section 3 总核对

> 协议：每个数字答三件事——**脚本 / 输入文件 / commit**。溯源不到标 **DEAD**，正文禁用。
> 原值优先（报舍入前）。本 session 数字源 = `probes/analysis/*.json`；旧数字尽力溯源。

## A. 数字溯源表

| 数字（报值 / 原值） | 脚本 | 输入文件 | commit |
|---|---|---|---|
| **Q-a 冲突占比 22.9%**（0.22928；0.2 档 0.28341 / 0.5 档 0.14733） | `probes/direction_axis.py` | `probes/data/rollouts_ckpt50_max4096.jsonl`（wrong 桶 50 条） | `direction_axis.json`←`c6278a5`；probe `cdc43fe` |
| **R-b ΔP enrichment 33.5×**（33.4648） | `probes/direction_axis.py` | 同上 | `direction_axis_recompute.json`←`b39307a` |
| **R-b δ·P_T enrichment 32.5×**（32.5207） | 同上 | 同上 | `b39307a` |
| **R-c style 子集 17.7×**（17.7422，n_style=129，exploratory） | 同上 | 同上 | `b39307a` |
| **Q-c Jaccard 0.19**（0.18870；n_conf 8212 / n_corr 2894 / n_∩ 1763） | `probes/direction_axis.py` | 同上 | `c6278a5` |
| **Q-c P(冲突\|腐蚀) 0.61**（0.60919） | 同上 | 同上 | `c6278a5` |
| **Q-c P(腐蚀\|冲突) 0.22**（0.21469） | 同上 | 同上 | `c6278a5` |
| **Q-d Spearman 0.12 / 0.14 / top-mass 0.04**（0.11487 / 0.13961 / 0.04225） | `probes/direction_axis.py` | 同上 | `c6278a5` |
| **R-a top-K K=50 −0.214**（conflict −0.21409；all −0.23203；K20 −0.20827 / K100 −0.21850） | `probes/direction_axis.py` | 同上 | `b39307a` |
| **漂移 3-seed 18 格**（见下"漂移全表"） | `probes/drift_multiseed.py` + `drift_scan.py` | `drift_shard*` / `drift_s1_shard*` / `drift_s2_shard*` | `drift_multiseed.json`←`07cf38c` |
| **教学含量 1.27×**（⚠️ 舍入撞车，原值 s42=**1.2662** / s1=**1.2671** / s2=**1.2681**，池化 1.2672） | `probes/triage_teaching.py` | `drift_*shard*.jsonl`（per_k.total） | `triage_teaching.json`←`2a79bf1` |
| **漂移桶差 −0.3/−0.2/−0.3pp**（原值 s42=**−0.3406** / s1=**−0.1980** / s2=**−0.2757**；池化 −0.2714） | `probes/triage_drift.py` | `drift_*shard*.jsonl`（per_k.drift, teach） | `triage_a_drift.json`←`3e1b439` |
| **教学含量 correct 0.0279→0.0379 / wrong 0.0353→0.0470**（step50→150，池化） | `probes/triage_teaching.py` | 同上 | `2a79bf1` |
| **(c) AUC 0.648**（0.64790） | `probes/triage_truncated_vt.py` | `rollouts_ckpt50_max1024.jsonl`(truncated) + `diag_truncated.jsonl`(V(t)) + ckpt150 续写 | `triage_c_truncated.json`←`ea3194d` |
| **(c) 续写达对 49%**（68/138；歪轨 70 = wrong 61 + 仍截断 9） | 同上 | 同上 | `ea3194d` |
| **t\* ±1 命中率 0/10**（clear-tstar 桶，±1 步窗） | `probes/annotate_tstar.py` | `probes/data/tstar_annotations.jsonl`（n=49） | 载体 `docs/framework.md`（旧 session） |
| **pseudo 形态五类 20/41/22/10/6%**（原计数 n=49：true_wrong_diffuse 20→41%、true_wrong_clear_tstar 10→20%、pseudo_wrong_format 11→**22%**、vacuous_proof 5→10%、no_reasoning 3→6%） | verdict 字段计数（`probes/rebucket_audit.py` 读同源） | `probes/data/tstar_annotations.jsonl` | 数据文件在库；载体 `docs/framework.md` |
| **14×** | — | — | **DEAD**（全仓库 docs/logs 无踪；archaeology 仅有 style-token KL "6–15×"，非 14×） |
| **bisect 11.8%→0.32%**（0.3285） | bisect 运行 | `~/opsd_outputs/qwen31b_bisect_*` | **仅存于未提交日志** `pbs/logs/bisect_all.30247` / `bisect_corr.30333`；**无提交产物** → 正文引用前须补 committed 摘要，否则按 DEAD 处理 |
| **腐蚀质量 B/A 2.84× / 2.63×**（step100 B0.0015365/A0.0005412；step145 B0.0016269/A0.0006225） | `probes/corruption_quality.py` | `qwen31b_repro`/`declip_B` generations + 数据集 | `corruption_quality_{A,B}.json`←`49e7841`；载体 `v1_design.md`(`661c060`) |
| **M1 AUC 0.49 n=14449**（原值 **0.48678**；spearman_w_top2 0.0245 / entropy 0.0275） | `probes/mechanism_probe.py` | `rollouts_ckpt50_max1024.jsonl` + `qwen31b_v0_s1/checkpoint-100` | `mechanism_probe.json`←`49e7841`；载体 `mechanism_validation.md`(`c7c4e26`) |

### 漂移全表（overall drift share %，18 格 + 桶分解）
| step | s42 | s1 | s2 |
|--:|--:|--:|--:|
| 25 | 38.130 | 34.533 | 35.400 |
| 50 | 50.779 | 47.417 | 48.254 |
| 75 | 55.872 | 53.006 | 53.327 |
| 100 | 59.634 | 57.170 | 57.255 |
| 125 | 61.349 | 58.735 | 58.908 |
| 150 | 62.138 | 59.193 | 59.576 |

（源 `drift_multiseed.json`←`07cf38c`；桶分解见 `triage_a_drift.json`。）

## B. 代码回读（四件 verbatim）

### B1. V(t) 评分模型身份 —— **checkpoint 学生（adapter ON），非冻结 base**
`probes/answer_likelihood.py:109-113`（无 `teacher_mode`/`disable_adapter` 包裹，用传入 model 的当前 adapter 态）：
```
with torch.no_grad():
    for i, start in enumerate(starts):
        Li = len(seqs[i])
        lg = model(input_ids=input_ids[i:i + 1, :Li],
                   attention_mask=attn[i:i + 1, :Li]).logits
```
调用处 `probes/run_2x2.py:191` `vt = answer_likelihood_probe(model, ...)` 在 `teacher_mode` 块**之外** →
adapter ON = **checkpoint 学生**。M3 修复 `ecd2d20` 亦显式改为 adapter-on（此前误用 disable_adapter → 测成 base → ΔV=0）。

### B2. 腐蚀探针变体使用登记（`corrupt_solution` vs `corrupt_answers`）
- `corrupt_answers.py::corrupt_answer(gt, n, seed)` = **生成**无关腐蚀答案串；
  `corrupt_solution.py::corrupt_solution(sol, gt, corr)` = 把参考解里的 gt 答案**替换**为 corr。
- **de-clip C/D**（`declip_trainer.py:77-80`）：`corrupt_answer` 生成 + `corrupt_solution` 注入（在线，每 rollout seed=i）。
- **Q 系列 / triage**（`direction_axis.py`、`triage_delta.py`、`triage_truncated_vt.py`、`corruption_quality.py`）：
  用 rollout 自带 `corrupted_answers[0]`（预生成）+ `corrupt_solution` 注入（`nrep==0` 则跳过）。
- **2×2 诊断**（`run_2x2.py:124-131`）：rollout 自带 `corrupted_answers`（≤3 个）+ `corrupt_solution`，
  wrong 桶另加 student-own-wrong 腐蚀。

### B3. t\* 预注册容差窗口 —— **±1 step**
`probes/data/tstar_annotations_README.md:15`（verbatim）：
> Round-2 re-annotation of 15 (seed 7): 100% agreement (tstar ±1 step + verdict).
盲区注记（同文件 :20-22）：window 以 marker 为中心，部分 pid（229/485/494/462/577/369）真实首错步**超出窗口**。

### B4. 漂移份额公式（R0）
`probes/analyze_drift.py:3`（verbatim）：
> `drift share_k = sum(drift_k) / (sum(drift_k) + sum(teach))`，per token category and overall.
其中 `drift_k = JSD(S_k, S0)`、`teach = JSD(T_S, S0)`（k 无关）、`S0 = base 学生 prompt`、`T_S = base 特权 teacher`。

## DEAD 清单（正文禁用）
- **14×**：全仓库无源（仅 archaeology "6–15×" style-token KL，非此数）。
- **bisect 11.8%→0.32%**：仅未提交日志，无 committed 产物 → 正文引用前须补摘要，否则 DEAD。
