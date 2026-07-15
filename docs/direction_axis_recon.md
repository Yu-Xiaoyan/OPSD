# 方向轴侦察：B 臂崩塌的主因归属（预注册，先落定后跑）

> 载体：de-clip 关账后的"主因归属"侦察（承 `docs/v1_design.md` B 臂三侧尸检的悬案——
> copying 为显著伴随、style/OOD 为主因候选，中间带不足以断定 copying 为主因）。
> **本文件写定即锁死判读，结果前不改口径。** 数据/输出：`probes/analysis/direction_axis.json`。

## 背景与目标

B（unclipped）崩塌，三侧尸检已入档（v1_design）：行为侧 early-emission strong 0→5、
性能侧 AIME25 −9pt、分布侧腐蚀质量 B/A≈2.7×。**崩塌幅度 ≫ 腐蚀信号量级** → copying/特权
依赖是显著伴随但未必主因。本侦察在 **wrong 桶 rollout** 的**冲突位置**上，把"特权敏感成分"
拆开看它到底长什么样、与既有方法（Purified OPSD 的 PMI 提纯）是否正交。

**数据源**：`probes/data/rollouts_ckpt50_max4096.jsonl` 的 **wrong 桶**（80 条；即
`diag2x2_shard0.jsonl` 所派生的同一批 rollout），补 forward 取原始分布（存储纪律 §4：
只落派生标量，绝不存 [T,V]）。**teacher = 冻结 base**（frozen regime 的特权信号本体）。

**三套 base forward / rollout**（同一 rollout token 序列上 teacher-forcing）：
- `π_T`  = base(problem + 完整参考解)     —— 正常特权 teacher；
- `π_T̃` = base(problem + 腐蚀参考解)     —— 参考解答案被无关腐蚀（复用 `corrupt_solution`）；
- `π_ref`= base(参考解，**去掉题目**)      —— Purified OPSD 的 reference-only teacher。
  实现 = `build_teacher_prompt_text` 去掉 `Problem: {problem}\n\n` 段（新函数
  `build_reference_only_prompt_text`，逐字记录）。

## 外部对标（先精读，供 Q-d/查重）

- **Purified OPSD**（Shen et al., arXiv:2607.02234，用户称 OPSD-PMI）：构造 reference-only
  teacher `π_ref`（参考、去题目），PMI 残差 `Δit(v)=log π_T(v)−log π_ref(v)`，提纯目标
  `P_PMI(v) ∝ P0(v)·exp((1/β)·Δit(v))`（P0=clean base=仅题目）。**其 π_ref 与本侦察 Q-d
  的 π_ref 定义逐字一致** → Q-d 直接检验"我方腐蚀差分 δ 是否被 Purified 的 PMI 残差覆盖"。
- **PACED**（Xu, Sang, Zhou, He, Wang, arXiv:2603.11178）：按 student pass-rate 加权**问题**
  （competence frontier / 难度轴），gradient-SNR 在 pass-rate→0/1 两端消失。**轴不同**——
  PACED 选"练哪道题"（数据/难度级），我方选"学哪个 token"（token 级腐蚀/冲突）；正交，
  非查重命中。查重结论详见文末。

## 冲突位置定义（预注册判据）

**冲突位置 = teacher 分布双峰性**：`π_T` 的 top-2 概率比 `p2/p1 ≥ 0.3`。
- 主判据 **0.3**；附 **0.2 / 0.5** 两档敏感性一并报。

## 四问（判据结果前锁死）

### Q-a 剂量
wrong 桶全体上，冲突位置（p2/p1≥0.3）**占 token 总数比例**。附 0.2/0.5 两档。

### Q-b 核心假设（特权敏感成分 ≈ 修正模式？）
冲突位置上算**腐蚀差分** `δ(v) = log π_T(v) − log π_T̃(v)`（逐 vocab）。检验其**正向质量**
是否集中于修正类 token。
- **判据**：epistemic 表上的正差分质量份额 ≥ **3×** 其词表基率 → 支持"特权敏感成分≈修正模式"。
  - share = Σ_{位置∈冲突} Σ_{v∈epi} max(δ,0) / Σ_{位置∈冲突} Σ_v max(δ,0)；
  - base_rate = |epi 词表 token id| / |V|；报 ratio = share / base_rate。
- **epistemic 表（预注册集合，⚠️ 见"口径限定"）**：`Wait, Actually, Perhaps, Maybe, But,
  However, check, reconsider`（用户 Q-b 逐字给出的 8 词）。matching = 去前导 ▁/Ġ/空格后
  letters-core 小写命中（覆盖 " Wait"/"Wait"/"wait" 等变体）。
  - **口径限定（诚实标注）**：用户口径为"**TRD 16-token** epistemic 表"，但该 16-token 逐字表
    **无法从可及文献复现**——Purified OPSD(2607.02234) 不含此表、无 "TRD" 缩写；repo
    `token_categories.yaml` 只有 ~80 词的 style 表（注称出自 OPSD paper 附录 C）。故本轮用
    用户 Q-b **自己逐字给出的 8 词**作预注册集合，**Q-b 判据结果标为 provisional**，待 TRD
    原表（16 token）核对；若原表不同，Q-b 一键重算（δ 派生量不需重跑 forward，见实现）。

### Q-c 轴重叠
冲突位置（p2/p1≥0.3）与**腐蚀敏感位置**（`c_t = JSD(π_T,π_T̃) > ρ=0.0007`，沿用
`diag2x2_shard0` 90 分位出处）的重叠：**Jaccard** + 双向 precision/recall
（P(冲突|腐蚀)、P(腐蚀|冲突)）。跨 rollout 聚合计数。

### Q-d 与 Purified 的正交性（生死问）
逐位置计算 `δ(v)` 与 PMI 残差 `Δit(v)=log π_T(v)−log π_ref(v)` 的：
- **Spearman 秩相关**（口径 = **全词表** V；位置分块以控显存），
- **top 质量重叠**（各取正向 top-K=64，overlap 系数 |∩|/K）。
- 分 **"冲突位置"** 与 **"全位置"** 两组报告均值。
- **预注册判读**：
  - 均值 Spearman **≥ 0.7** → δ 被 Purified PMI 残差实质覆盖 → **C 路线降级放弃**；
  - **≤ 0.4** → 正交增量 → **C 路线存活，进 `v2_design`**；
  - **0.4 < r < 0.7** → 中间带，人判。

## 实现与预算
- `probes/direction_axis.py`：单卡，base=`~/models/Qwen3-1.7B`，flash-attn2、bf16、no_grad。
  每 wrong rollout 3 次 base forward（π_T/π_T̃/π_ref），rollout token cap 768，limit 50 条。
- 显存：单 rollout 3×[T,V] fp32 + δ/Δit [T,V]，逐条 del；Q-d 全词表 argsort 按 128-位置分块。
- 存储纪律：只落 `direction_axis.json`（四问派生标量 + 敏感性档），**不存任何 [T,V]**。
- 预算：50 条 × 3 forward + 全词表秩相关，估 ~45–75min，1 卡短作业（< 2h，自主边界内）。

## 查重报告（PACED / Purified，先入档）
- **Purified OPSD (2607.02234)**：**高相关命中**——其 `π_ref`/PMI 残差与本侦察 δ/Δit 同源。
  Q-d 即为对其的**定量正交性检验**；判读按上（≥0.7 放弃 C / ≤0.4 存活）。
- **PACED (2603.11178)**：**非命中**——难度/competence 轴（选题），与 token 级腐蚀/冲突轴正交。
- 结论：C 路线（腐蚀敏感位精准干预）的存活与否，**由 Q-d 对 Purified 的正交性单点裁决**。

---

## 结果（job `33403`，exit 0；50 wrong-bucket rollout，tok_cap 768；`probes/analysis/direction_axis.json`）

| 问 | 量 | 值 |
|---|---|---|
| **Q-a 剂量** | 冲突位置占比 (p2/p1≥0.3) | **22.9%**（0.2→28.3% / 0.5→14.7%） |
| **Q-b 富集** | epi 正差分质量份额 / 基率 | **1.00×**（provisional，⚠️ 见 caveat 1） |
| **Q-c 轴重叠** | Jaccard(冲突,腐蚀) | **0.189** |
| | P(冲突\|腐蚀) | **0.609** |
| | P(腐蚀\|冲突) | **0.215**（n_conf=8212, n_corr=2894, n_∩=1763） |
| **Q-d 正交性** | Spearman(δ,Δit) 冲突/全 (全词表) | **0.115 / 0.140** |
| | top-mass overlap 冲突/全 (top-64) | **0.040 / 0.042** |

### 判读

- **Q-a**：teacher 在 ~23% token 双峰——冲突结构是**广谱**的。
- **Q-c**：腐蚀敏感位（copying）**61% 落在冲突位内**，但冲突位只有 **21.5%** 是腐蚀敏感 →
  **copying 是冲突结构的少数子集，冲突 ≫ copying（约 2.8×）**。与 B 臂尸检"copying 为少数
  伴随、style/OOD 为主因候选"自洽。
- **Q-d（生死问）**：Spearman 0.12–0.14、top-mass overlap 0.04，**均 ≤ 0.4** → 按预注册
  **δ 与 Purified 的 PMI 残差 Δit 基本正交 → C 路线存活，进 v2_design**。
  即：**腐蚀门方向不被 Purified OPSD 覆盖**，是独立增量。

### ⚠️ 方法学 caveat（诚实标注，影响判读强度）

1. **Q-b 疑似尾部支配伪影**：enrichment **恰好 = 1.000×**（epi 份额 0.0003686 ≈ 基率
   0.0003686，4 位有效数字重合）。预注册用的 `Σ max(δ,0)`（δ=对数比、全词表）被 ~15 万
   近零噪声尾 token 支配 → 任意固定小子集≈其基率份额。**故 Q-b 当前值不可解读为"修正模式
   假设被否"，只能判为无信息**。清洁重算建议：改**质量加权**（如 ΔP=P_T−P_T̃ 的正部，或
   δ 按 P_T 加权）；且 TRD 16-token 全表仍待核。**不据本轮 Q-b 下任何结论。**
2. **Q-d 全词表 Spearman 偏低有偏**：全词表秩相关同受尾部噪声压低（99% 词表在排噪声）。
   **但 top-mass overlap（top-64，尾部稳健）= 0.04 独立佐证低相关**——两指标一致 → "正交"
   判读在**尾稳健指标上成立**，非纯全词表伪影。仍建议若要定稿，补 **top-K 并集域** Spearman
   作三角互证。

### 待裁（不擅自拍）

- Q-d 生死问在**尾稳健的 top-mass overlap 上支持"C 存活/正交"**；是否据此**写入 v2_design**
  （C=腐蚀门作为 Purified 之外的独立轴），还是**先补清洁重算**（Q-b 质量加权 + Q-d top-K 并集域）
  再定稿——请你裁。**本文件先据预注册规则记 Q-d ≤0.4 → C 存活，caveat 一并在案。**
