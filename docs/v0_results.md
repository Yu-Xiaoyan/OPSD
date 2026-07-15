# v0 main run — 结果与判读（按 v0_predictions.md 预注册口径）

**Eval 完成 2026-07-07**（job `30605.gaas`，exit 0）。锁定协议：AIME24/25
avg@12，temp 1.0 / top_p 1.0 / top_k -1 / min_p 0 / max_new 38912 / thinking on。
30 题满量，单 seed。base = 未训练 Qwen3-1.7B。判读口径在结果产生前已 pre-register
（`docs/v0_predictions.md`），此处如实报告，无论支持与否。

> 论文映射：本文件是论文草稿 §4.5 的仓库载体（草稿在作者处维护，见
> `docs/paper_map.md`）。

## 结果表（avg@12, %）

### AIME24  (base = 49.2)
| step | OPSD | v0 | Δ(v0−OPSD) | v0 gain/base |
|---|---|---|---|---|
| 50 | 52.5 | 55.0 | **+2.5** | +5.8 |
| 100 | 55.0 | 56.9 | **+1.9** | **+7.7** (peak) |
| 150 | 54.2 | 56.1 | **+1.9** | +6.9 |

### AIME25  (base = 35.0)
| step | OPSD | v0 | Δ(v0−OPSD) | v0 gain/base |
|---|---|---|---|---|
| 50 | 40.6 | 42.2 | **+1.7** | **+7.2** (peak) |
| 100 | 43.1 | 40.8 | **−2.2** | +5.8 |
| 150 | 42.8 | 41.7 | **−1.1** | +6.7 |

OPSD peak gain（参考，= ckpt100 − base，**单 seed**）：AIME24 +5.8 / AIME25 +8.1。
> ⚠️ **基准口径统一（2026-07-15）**：正文引用的"饱和前增益"改用 **T2 3-seed 表**（见下 §T2）：
> OPSD 相对 base @ ckpt100 = **AIME24 +6.3±1.1 / AIME25 +6.8±1.8 / MATH500 +1.4±0.3**（mean±std 报全，
> std 取 OPSD 列，base 为单点未训练）。上面的**单 seed +5.8/+8.1 降为"复现一致性佐证"**，不作正文主引用。
>
> **饱和操作定义（采纳 2026-07-15）**：saturation step = 最早 checkpoint c，使其后**所有评测点**相对 c
> 的提升 **≤ 1 pooled seed std**。**评测网格粒度 = ckpt {50, 100, 150}（步长 50）**，pooled seed std
> 由 3-seed 表估（AIME ~1–2pt、MATH500 ~0.3pt）。网格粗（仅 3 点）须正文注明——不足以定位 50 步内饱和点。

## 主判据（primary accept/reject）— v0 best gain/base vs OPSD

- **AIME24：v0 +7.7 (ckpt100) > OPSD +5.8 → ACCEPT。** 且逐 checkpoint 同步对比
  v0 全程领先（+2.5、+1.9），ckpt150 (56.1) 仍高于 OPSD 的 ckpt100 峰值。
- **AIME25：v0 +7.2 (ckpt50) < OPSD +8.1 → REJECT（略逊 0.9）。** OPSD 在 ckpt100
  冲到 43.1，v0 同步反而回落到 40.8（−2.2）。

**综合：分裂结果。** 两 benchmark 平均峰值增益 v0 7.45 vs OPSD 6.95（v0 略优），
但优势完全来自 AIME24；AIME25 上 OPSD 的 ckpt100 峰值 v0 未能追平。

## 预注册预言 P 判定

**P 原文**（`docs/v0_predictions.md`，结果产生前锁定）：
> v0's peak step is LATER than OPSD's, and/or v0's post-peak stability is better
> than OPSD's（到 step 150 从峰值的回落更小）。
> 证伪：若 v0 peaks no later than OPSD **且** post-peak degrade 同样快或更快。

### 判定：P 完全被证伪（前后两半均成立证伪条件）
- **峰值步 → 不更晚。** AIME24：v0 峰 = ckpt100 = OPSD 峰步（相同，不更晚）；
  AIME25：v0 峰 = ckpt50，比 OPSD（ckpt100）**更早**。证伪条件前半
  （"v0 peaks no later than OPSD"）在两个 benchmark 上均成立。
- **post-peak 对照 → 同样不支持 P**（OPSD ckpt150 已补测，job `30675`，
  avg@12 aime24=54.2 / aime25=42.8，协议锁定）。峰值→ckpt150 回落对照：
  - **AIME24**：OPSD 峰 ckpt100 55.0→54.2（**−0.8**）；v0 峰 ckpt100 56.9→56.1（**−0.8**）。**回落相同**，v0 不更稳。
  - **AIME25**：OPSD 峰 ckpt100 43.1→42.8（**−0.3**，很稳）；v0 峰 ckpt50 42.2→41.7（**−0.5**），且中途 ckpt100 掉到 40.8。**v0 不比 OPSD 稳**。
  → 证伪条件后半（"post-peak degrade 同样快或更快"）成立。

### post-hoc 判别（⚠️ 以下为事后分析，非预注册，不作确证）
- **机制层面 P 的失败可解释、且与 drift-erosion 主假设自洽。** v0 门控只作用于
  **监督信号的质量**（down-weight 低价值 / V(t) 骤降 / 结构 token），**不触及漂移
  本身**：teacher 全程冻结（fixed_teacher），student 的 LoRA 照常漂移，门控没有任何
  约束 adapter drift 的机制。
- **因此本轮排除的是"坏监督信号是饱和主因"这一竞争解释**：若坏信号为主因，改善
  信号质量的门控应当延后饱和——但没有（P 失败）。反过来，P 失败与"**LoRA 漂移
  才是饱和主因**"自洽——门控够不到的那一层，正是饱和发生的那一层。
- **指向（未来方向，非本轮结论）：teacher 刷新（EMA teacher）**是唯一能触及漂移
  层的杠杆。但它并非无代价：需引 **genealogy 的 sync-10 泄露反馈环发现**作为张力
  —— EMA 每次 sync 会把 student 侧的泄露 / 漂移反馈进 teacher，形成放大环，可能
  让"对抗漂移"反噬为"放大泄露"。此发现在作者处 genealogy 维护，登记于
  `docs/paper_map.md` 外部依赖，待补仓库数据链接。

## 注意事项 / 数据缺口

- **小样本 + 单 seed：** AIME 各 30 题，1 题 = 3.3pt。表中 1.7–2.5pt 的 Δ 落在
  单题量级，应读作趋势而非定论。预注册即规定单 seed 先行，multi-seed only after pass。
- **OPSD ckpt150 已补测**（job `30675`）：P 的 post-peak 对照已完成，见上节。
- **AIME24/AIME25 方向相反：** v0 在 AIME24 稳定占优、AIME25 峰值略逊，值得下一步
  排查是数据集特性还是 gating 在某类题上的副作用；全套 benchmark（MATH500 等）
  压低标准误后再判。

## 方差控制包（进行中）

- **全套 benchmark**（v0 + OPSD，ckpt 50/100/150 × MATH500/AMC23/Minerva/HMMT25 +
  AIME）：MATH500 的 500 题把标准误压到 AIME 的 ~1/4。job 见 `pbs/eval_matrix.pbs`。
- **multi-seed**：v0 + OPSD 各 3 seeds（seed 42 已有，补 seed 1/2），eval 先跑
  AIME24/25 + MATH500 @ ckpt 50/100/150，报 mean±std。过夜串行。
- **纪律**：结果全部落地前冻结 v0 一切超参与门控配置，不据本轮单 seed 数字调参。

## T2 — multi-seed 结果（3 seeds {42,1,2}）

Eval 完成 2026-07-08，锁定协议，完整性核验无 warn（val_n / temp / top_p 全对）。
数据：seed42+base → `results/v0_eval` `results/repro_eval`；seed1/2 → `results/multiseed_eval`。
复现：`python scripts/summarize_multiseed.py`。口径：AIME avg@12 / MATH500 avg@4，ckpt100+150，per-benchmark 不跨 N 平均。

| benchmark | ckpt | OPSD | v0 | Δ(v0−OPSD) | base |
|---|---|---|---|---|---|
| AIME24 (avg@12) | 100 | 55.5±1.1 | 57.6±1.9 | +2.1 | 49.2 |
| AIME24 (avg@12) | 150 | 56.1±1.9 | 56.8±0.7 | +0.6 | |
| AIME25 (avg@12) | 100 | 41.8±1.8 | 40.9±1.3 | −0.8 | 35.0 |
| AIME25 (avg@12) | 150 | 42.7±0.2 | 41.2±0.8 | −1.5 | |
| MATH500 (avg@4) | 100 | 92.2±0.3 | 92.5±0.2 | +0.2 | 90.8 |
| MATH500 (avg@4) | 150 | 92.0±0.6 | 92.6±0.1 | +0.6 | |

**判读（诚实）：v0 相对 OPSD 无稳健优势。** AIME24 v0 微领先（ckpt100 +2.1≈1.5σ、ckpt150 +0.6 落噪声内）；
AIME25 v0 微落后（−0.8 / −1.5，噪声量级）；MATH500 近乎并列（均 ~92%，base 90.8，接近天花板、判别力有限）。
**多 seed 削弱了单 seed 的"AIME24 赢 / AIME25 输"分裂叙事**——差异多落在 seed 方差内，n=3 不足以做强统计声明。
与预注册 P（已完全证伪）一致：v0 未展现机制性的持续增益。冻结纪律下，不据此调参。

## 难度分层分析（任务1，零训练；分支二主攻）

按 base avg@12 逐题 pass-rate 分层（AIME24+25 pool，60 题）：hard(base=0%,n=20) / mid(0–50%,n=18) / easy(>50%,n=22)。
各层内 v0 vs OPSD 逐题 pass-rate 的 3-seed mean±std。复现 `scripts/strat_analysis.py`，图 `docs/figs/strat_aime.png`。
**预注册预测**（结果前锁定）：Δ(v0−OPSD) 随层难度上升而增大。

| 层 | n | ckpt100 OPSD / v0 / Δ | ckpt150 OPSD / v0 / Δ |
|---|---|---|---|
| hard (base=0%) | 20 | 5.4 / 5.4 / **+0.0** | 5.0 / 4.9 / **−0.1** |
| mid (0–50%] | 18 | 43.7 / 45.1 / +1.4 | 46.3 / 45.2 / −1.1 |
| easy (>50%) | 22 | 91.9 / 92.6 / +0.6 | 92.3 / 92.2 / −0.1 |

**判定：预测不成立（两 ckpt 均非单调）。** 最难层（base=0%）v0≈OPSD（Δ≈0）——门控未"解锁"难题；
唯一非平凡 Δ 在 mid 层，但跨 ckpt 变号（+1.4→−1.1）、落 seed 噪声内。训练增益几乎全来自 mid 层，
难题层纹丝不动。与 T2 / 预注册 P（已证伪）一致：v0 无难度结构化增益。如实报告，无论支持与否。
