# Drift-takeover 分析：实施与结果

## 动机
OPSD 训练在 ~100 步后性能饱和/回落。问题：饱和是不是因为 **student 的 LoRA 持续漂离初始点**，
使得有效监督信号逐渐从"teacher 的特权教学"退化成"把模型往起点拽（KL-to-init 式正则）"？
本分析把师生分歧**三方分解**，量化"漂移成分"随训练步数的占比。

## 实施（`probes/drift_scan.py` + `probes/drift_multiseed.py`）

### 三个逐 token 分布（同一固定 rollout 上 teacher-force）
对每个 checkpoint k ∈ {25, 50, 75, 100, 125, 150}，在**固定的 4096-长度诊断 rollout 集**
（correct+wrong 两桶，n=187）上，用 teacher-forcing 取三个 next-token 分布：

| 记号 | 定义 | 说明 |
|---|---|---|
| `S0`  | base 模型 + **student prompt**（只题目） | 初始学生快照（step-0），k 无关，算一次 |
| `T_S` | base 模型 + **privileged teacher prompt**（题目+参考解） | 冻结特权 teacher，k 无关，算一次 |
| `S_k` | **adapter-k on** + student prompt | 训练到第 k 步的学生（唯一随 k 变） |

### 三个逐 token 散度（JSD ∈ [0, log2]）
- **漂移** `drift_k = JSD(S_k, S0)`：学生离初始点多远（漂移量）；
- **教学** `teach = JSD(T_S, S0)`：teacher 相对初始点携带的教学信号（k 无关）；
- `total_k = JSD(T_S, S_k)`：teacher 与当前学生的分歧。

### 漂移份额（"拽回起点"成分）
> **drift share_k = 100 · Σ_tok drift_k / (Σ_tok drift_k + Σ_tok teach)**  （全 token 聚合，%）

份额上升 = 学生的分布位移越来越由"漂离初始点"主导，而非"朝 teacher 学"。

### 数据与稳健性
- **3 seed**（s42 / s1 / s2），每 seed 187 rollout；`drift_scan.py` 每 seed 一份 shard，
  `drift_multiseed.py` 聚合并算 mean±std。
- **共用同一固定 rollout 集**（采样自 seed42-ckpt50）→ `S0`/`T_S`/`teach` 三 seed 相同，
  仅 `S_k` 变 → seed 效应干净隔离。
- **off-policy caveat**：固定 rollout 对 25/75/100/125/150 是 off-policy 评测，**绝对水平有偏**，
  但"随步上升的趋势/首尾差/过半"对此稳健（偏置只移动水平、不改趋势方向）。

## 结果

### 漂移份额随训练步（overall drift share，%）
| step | s42 | s1 | s2 | **mean±std** |
|--:|--:|--:|--:|--:|
| 25  | 38.13 | 34.53 | 35.40 | **36.0 ± 1.5** |
| 50  | 50.78 | 47.42 | 48.25 | **48.8 ± 1.4** |
| 75  | 55.87 | 53.01 | 53.33 | **54.1 ± 1.3** |
| 100 | 59.63 | 57.17 | 57.25 | **58.0 ± 1.1** |
| 125 | 61.35 | 58.73 | 58.91 | **59.7 ± 1.2** |
| 150 | 62.14 | 59.19 | 59.58 | **60.3 ± 1.3** |

（源 `probes/analysis/drift_multiseed.json`；三 seed 各 n=187。）

### 预注册预言（均 PASS）
- **(a) 上升**：三 seed 首尾差 = +24.0 / +24.7 / +24.2 pp（均 > 10）→ **PASS**。
- **(b) 过半**：150 步 = 62.1 / 59.2 / 59.6 %（均 > 50）→ **PASS**。
- **结论**：漂移份额**随训练单调上升、并在 50 步前越过 50%**，到 150 步达 ~60%——
  有效监督目标越来越退化成"回起点"分量；三 seed 一致，std ≤1.5pp。

### 漂移与 rollout 对错无关（新事实 F4）
把同一 3-seed 漂移数据按 verifier-v2 桶拆开（correct n=107 / wrong n=80 每 seed）：

| step | correct 桶 | wrong 桶 | Δ(c−w) |
|--:|--:|--:|--:|
| 25  | 36.3 | 35.7 | +0.5 |
| 100 | 58.0 | 58.1 | −0.1 |
| 150 | 60.2 | 60.5 | **−0.3** |

150 步 Δ(correct−wrong) = **−0.3pp**（per-seed −0.34 / −0.20 / −0.28，3-seed 一致）→
**correct 与 wrong 桶漂移份额几乎相同**（桶内对照，off-policy 偏置两桶同量抵消，null 稳健）。

- **新事实**：**漂移是全局权重位移（global weight displacement），与数据/rollout 对错无关（非数据条件性）**。

## 判读
- **漂移接管是 OPSD 饱和的机制候选**：有效目标随步退化为 KL-to-init，且**在性能峰值（~100 步）
  之前就已过半**，与"特权被漂移淹没"方向一致。
- **漂移是全局的**（F4）→ 任何"按数据对错重加权监督"的干预都**够不到漂移**；对症的干预须直接
  作用于漂移轴（诊断结论止于此；具体方法方向不在本诊断文档内展开）。
- 复现：`probes/drift_scan.py`（GPU，逐 seed）→ `probes/drift_multiseed.py`（CPU，聚合+判读）；
  桶分解 `probes/triage_drift.py`。数据 `probes/analysis/{drift_multiseed,triage_a_drift}.json`。

---

# 附：v2 方法 evaluation 结果（失败，已放弃）

> ⚠️ **与上文 drift 诊断是两回事，别混**：上文 60.3% 是"漂移**份额**"（分歧构成的机制诊断量）；
> 下文是"AIME **正确率**"（性能量）。都是百分数，但一个测"漂移占比"、一个测"做对题的比例"。

## 方法（v2 主案 = 目标手术）
在 OPSD 基线上改**蒸馏目标**（非重加权）：每 rollout 逐 token 重构 target
`log_tgt = log π_T + λ·max(δ,0)（C′-1，wrong支）− γ·log π_S0（B 全局漂移扣除，所有支）`，
renorm 后 clipped forward-KL 蒸馏；λ=1、γ=0.5、clip 0.05。三路分诊（correct/wrong/truncated）。

## 结果（v2main 臂，AIME avg@12；`results/v2_eval/`）
| ckpt | AIME24 | AIME25 |
|--:|--:|--:|
| v2main 50 | 41.7 | 36.1 |
| v2main 100 | **27.8** | **26.1** |
| v2main 150 | **28.9** | **26.1** |
| — base（未训练） | 49.2 | 35.0 |
| — 官方 OPSD 100 | 55.0 | 43.1 |

**训崩**：ckpt50 已掉到 base 以下，ckpt100/150 塌到 ~28/26 —— **比 base 低 ~20pt、比官方 OPSD 低 ~27pt**。

## 判定（战略层裁决 2026-07-15，入档）
- v2 主案（叠加配置 B γ=0.5 + C′-1 λ=1）**训崩**：AIME24/25 = 28.9/26.1，低于 base ~20pt。
- **按预注册，论文默认形态切换为 A 路线（纯诊断）**；**8/15 闸门规则不变**。
- **头号嫌疑**：B 的 PMI 减法 `−γ·log π_S0` 放大稀有 token → 重构 target 畸形 → 训崩；
  次疑：clip 加在重构 target 上的交互（clip 公式与原版一致，但作用于合成分布而非真 teacher）。
- **部件级死因待定位（#2）**：ablation 臂 checkpoint **在盘**（`qwen31b_v2_noCp`=纯B、
  `qwen31b_v2_noB`=纯C′，均有 ckpt-100）。只对两臂 ckpt100 跑 AIME24 avg@12 单点 eval 归因
  （判据见下"归因预注册"），**只为归因、不为翻案**。
- **教训（措辞修正）**：**未经单独验证的目标叠加是雷区**——B 与 C′ 直接叠加、无部件级单跑验证
  就上，是本次失败的方法论教训；**部件级死因待 #2 定位**。（原"目标重构是雷区"过度概括，降级。）

## 归因预注册（#2，跑前锁死）
- **对象**：`qwen31b_v2_noCp`（纯 B）、`qwen31b_v2_noB`（纯 C′），**仅 ckpt-100**，**AIME24 avg@12 单点**。
- **对照**：官方 OPSD ckpt100 = **55.0**；base = **49.2**。
- **三档判据**：某臂 <49.2（低于 base）= **崩** / ≈55 = **平** / >55 = **涨**。
  - 若 noCp 崩、noB 平/涨 → 死因主要在 **B**；若 noB 崩、noCp 平/涨 → 死因主要在 **C′**；
    若两臂都崩 → 叠加与各自均有份；两臂都平/涨 → 死因在**叠加交互**。
- **只归因，不翻案**：无论结果，A 路线裁决不变。
