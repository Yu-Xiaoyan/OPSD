# v2 设计 —— 只写设计与预算，不跑

> 论文映射：v2 设计载体（草稿在作者处，见 `docs/paper_map.md`）。**先写定，后开跑。**
> 本文件收录**两条候选线**：
> - **线 1｜EMA teacher（新鲜度轴）**——见下节。
> - **线 2｜腐蚀门（独立轴，C 路线）**——见文末，状态 **C 存活（初判）/ 稳健性确认 pending**
>   （待 `direction_axis_recon.md` R-a top-K 并集域 Spearman 确认；训练实验全部冻结）。

---

# 线 1｜EMA teacher（新鲜度轴）

## 动机（用新证据框定）

teacher 管理的**两个端点均被实测为衰变**：
- **frozen 端 → 漂移接管**：teacher 冻结在 step-0，student 的 LoRA 持续漂移；本环境实测
  漂移份额 **38% → 62%**（drift 扫描），门控/去 clip 都够不到这一层（M1–M4、P-a 三线确认）。
- **synced 端 → 特权塌缩为空洞**：teacher 与 student 同步刷新时，两者分布趋同，特权信息的
  增量→0（TRD Fig.3：synced 的 PPL gap 贴零 = teacher 相对 student 不再携带额外信息）。

**EMA 假设**：在 frozen（漂移失控）与 synced（特权塌空）**两个衰变端点之间存在甜点**——
EMA teacher（θ_T ← α·θ_T + (1−α)·θ_S，慢速跟随）既跟上 student 的分布（缓解漂移接管），
又保留足够的特权/新鲜度落差（不塌成空洞）。α 是新鲜度旋钮。

## 关键风险（预注册，先入档）
**EMA sync → 泄露反馈环**（genealogy sync-10 发现 + 本环境 P-b 证据）：EMA 每次把 student
侧的泄露/漂移拌进 teacher，可能形成放大环——"对抗漂移"反噬为"放大泄露"。故**护栏沿用**：
腐蚀质量**在线监测**（每 sync 后测 teacher 的 JSD(T_S,T_S̃)，若腐蚀质量随 sync 单调上升 →
泄露反馈环触发 → 止损）。泄露检测器（keyword+early-emission，每 5 步 dump）全程挂载。

## 设计（草案，待细化）
- 基底：repo 冻结口径（clip 0.05、1024、TM-off、solution、seed42、gb30）——**保留 clip**
  （P-a 已证本 regime clip 是净正，v2 不动它）。唯一变量 = teacher 刷新策略。
- 臂：{frozen（=A 基线，复用）} × {EMA α∈{0.999, 0.99}（2 档新鲜度）}，先 1.7B 单 seed 侦察。
- 实现：`use_ema_teacher` 开关在 opsd_trainer 已有雏形（need 核实），EMA 更新在 optimizer step 后；
  sync 频率与 α 记入 genealogy。
- 探针：腐蚀质量在线（护栏）+ 漂移份额曲线（验证 EMA 是否压住 frozen 端的 38→62% 爬升）。
- 评测：AIME24/25 avg@12 + MATH500 avg@4 @ ckpt100/150，锁定协议，阶段内对照。

## 预注册预言（草案，开跑前锁死）
- **Q-a 甜点存在**：某 α 下性能 > frozen 基线 A，且漂移份额曲线低于 A。
- **Q-b 反馈环护栏**：腐蚀质量随 sync **不**单调爆升（若爆升 → 泄露反馈环，止损）。
- **Q-c 端点对照**：α→1（≈frozen）与 α→0（≈synced，特权塌空）应分别复现两个衰变端点的劣化。

## 预算（自估，不跑）
frozen 复用 A；EMA 2 档 × 1.7B 单 seed = 2 训练 run（EMA forward 每步 +teacher 前向，成本按
96.6ms 基准）@ 1024 ~54min/run + 依赖 eval。≤ 一夜。4B 迁移待 1.7B 侦察后另议。
**先核实 `opsd_trainer` 的 `use_ema_teacher` 实现是否完整、EMA 更新点是否正确**，再开跑。

---

# 线 2｜三路分诊 + 错支目标手术（v2 主案）

> **状态：架构定案（探针阶段收官）/ 训练冻结，待 (c) + 发令。** triage 验证
> （`triage_rationale_recon.md`）：**(a) FAIL**（漂移份额桶间不变，全局位移）；**(b) PASS 贴线**
> （wrong 修正需求 1.55×）→ 错支 C′ 前提成立；**裁决二 (i) 教学含量 1.27×**（correct 略低 −21%，
> 不支持降权）→ **correct 支主案 ω=1（等权）+ ω 消融轴**；**(c) AUC=0.648<0.65 → truncated 支
> 保守按 wrong**。C 存活（R-a K=50=−0.214，正交/不相关，稳健确认）= **wrong 支 C′ 部件**。
> **架构三支全部定案；训练冻结，唯一解冻门槛 = 用户发令。**

## 方法形态（用户定义 2026-07-15）

v2 = **两类正交部件** 组合：**(A) 三路分诊 = 资源分配**（对错走不同 loss）+ **(B) 全局漂移扣除
= 去毒**（从所有支目标减去"回起点"分量）。二者正交，可独立开关、可叠加。

**新事实（F4，2026-07-15）**：漂移份额**与 rollout 对错无关**（triage (a)：桶间 Δ=−0.3pp）→
**漂移是全局权重位移，非数据条件性** → 故漂移扣除是**全局手术**（下 B），不属某一支。

### (A) 三路分诊（资源分配）
- **correct 支｜权重 ω（消融轴），主案 ω=1.0（等权）**。原"大幅降权/跳过"的两条依据均不成立：
  triage (a) 证伪"漂移更多"（桶间 Δ=−0.3pp）；裁决二 (i) 教学含量 wrong/correct **仅 1.27×**
  （correct 略低 −21%，3-seed 一致）**不支持跳过或大幅降权**。→ **主案 correct 支不动（ω=1）**，
  分诊差异化**全部由错支手术承担**；ω 作训练矩阵消融轴（见下），由训练实验直接裁决，不补代理探针。
- **wrong 支｜完整蒸馏 + 目标手术**。**C′（促修正目标）挂载此支**（子节见下）。triage (b)
  支持本支前提（wrong 修正需求 1.55×correct，**边际通过**，n=107/80；引用须连带裁决二 (ii) 复测）。
- **truncated 支｜保守按 wrong 处理（完整蒸馏）**。triage (c)：V(t) 截断点读数区分"后续正轨/歪轨"
  **AUC=0.648 < 0.65（贴线 FAIL）** → **不用 V(t) 分派，改保守默认（按 wrong 完整蒸馏）**。
  （续写显示 ~49% truncated 续后达对，弱预测力不足以支撑分派。）

### (B) 全局漂移扣除（去毒，作用于所有支）
第三参照 = 初始学生快照 `π_S0`；从蒸馏目标中减去"回起点（drift-to-init）"分量，**所有支通用**。
与分诊 (A) **正交**：可单列、可与 C′ 叠加（叠加时需消融拆解可加性）。

## C′ 子节：促修正目标（设计选项表，不定稿）

依据：Q-b 质量加权 **修正类富集 ~33×（≥3×）** → 腐蚀差分 δ 的正向质量集中于修正类标记 →
可据 δ 构造"促修正"目标。

**定稿（战略层 2026-07-15）：第一批矩阵采用 C′-1（δ 加权重构），依据 R-b 33.5× 富集、活动部件最少。**

| 选项 | 形态 | 批次 | 预期风险 |
|---|---|:--:|---|
| **C′-1 δ 加权目标重构** | 目标 logits = `log π_T + λ·max(δ,0)` renorm（放大修正类成分） | **第一批（定稿）** | δ 噪声放大；λ 超参 |
| C′-2 仅蒸馏高 δ 成分 | 目标 = π_T 在高 δ token，其余降权 | 备选 | 丢非修正正确信号 |
| C′-3 δ 叠漂移扣除 | (π_T−回起点) 再叠 δ | 备选 | 交互未知 |
| **C′-4 对比式** | 对比 π_T vs π_T̃ 的方向性目标（登记） | **第二批候选** | 待设计 |

- **升级路径**：C′-1 有效但幅度小 → 升级试 **C′-4（对比式）**；C′-1 无效 → 换药（C′-2/3）。
- **δ 加权强度 λ**：pinned **λ=1.0**（默认；见 `v2_launch_prereg.md`，可调）。

## 与 Purified OPSD 的四坐标差异（查重定位）

| 坐标 | Purified OPSD (2607.02234) | C′（本案 wrong 支手术） |
|---|---|---|
| **信号** | PMI 残差 `Δit=log π_T−log π_ref`（题目相对参考的增量） | **腐蚀差分** `δ=log π_T−log π_T̃`（参考**正确性**的增量） |
| **作用位置** | 全位置（提纯目标分布 P_PMI） | **wrong 支的冲突位置**（teacher 双峰 p2/p1≥0.3）为主 |
| **方向** | 过滤 reference 捷径（减 copying 项） | **促修正**（在腐蚀敏感位抬修正成分） |
| **regime** | 官方配方（其 clip/长度/teacher 设定） | repo 冻结口径（clip 0.05、1024、TM-off、frozen、solution） |

→ 正交且互补；R-a top-K 并集域 Spearman K=50=**−0.214（正交/不相关）**是"非 Purified 重复"的
定量依据（稳健性已确认；**措辞：表述为正交/不相关，不作"反向信号"解读**）。

## 侦察图景收录（Q-a / Q-c，作"腐蚀门 ≠ 防泄露门"论据）

- **Q-a**：冲突位（teacher 双峰）占 token **22.9%**——冲突是**广谱结构**。
- **Q-c**：腐蚀敏感位（copying）只占冲突位的 **22%**；反向，腐蚀敏感位 **61% 落在冲突位内**。
- → **"冲突是广谱结构，copying 是其少数子集"** → **C′ 作用对象 ≠ 泄露/copying 位**，命中的是
  冲突结构里"参考正确性敏感"那部分 → **C′ 是"促修正门"而非"防泄露门"**（防泄露由 clip 承担，
  见 v1_design B 臂尸检 clip 三重身份）。

## related work 划界（草稿）

- **vs PACED（2603.11178）**：PACED 在**题目级**给**标量权重**（按 student pass-rate，强度轴/难度轴，
  选"练哪道题"）；本案在 **rollout 级**做**目标切换**（按对错分诊 + 错支方向性目标手术，方向轴/结构轴，
  选"对某条 rollout 学什么信号"）。**权重 vs 目标、题目级 vs rollout 级、强度 vs 方向**，正交。
- **vs Purified OPSD（2607.02234）**：Purified **减**参考诱导（PMI 提纯剔除 reference 捷径）；本案 C′
  **保留并利用**特权诱导中的**修正**成分（δ 促修正）。R-a K=50=−0.214（正交/不相关）为"非重复"佐证。

### correctness-aware OPD 段（查重 2026-07-15，Uni-OPD 领衔）

- **Uni-OPD（2605.03677）**：双视角 = 数据平衡（离线上采样 mid-difficulty + 在线强制 rollout group
  内 correct/incorrect 平衡）+ outcome-guided margin calibration（锚 = **轨迹级 outcome reward 次序**）。
  **与本案正交**：(a) 其对错平衡是**数据采样/组成级**（选哪些 rollout 进 batch），本案分诊是
  **rollout 级 loss 目标切换**（每条走不同 target）；(b) 其锚 = outcome 次序，本案两锚 = **π_S0 漂移轴 /
  δ 特权差分轴**；(c) 其假设 frozen teacher、**不涉 staleness/drift**（本案 B 漂移扣除正补此缺）；
  (d) 其 mid-difficulty 上采样与本案难度分层"mid 层信息量高"**同向**，引为难度重加权先例。
  *划界句*：与 Uni-OPD 不同，本案在 **rollout 级切换蒸馏目标**（非 batch 级平衡采样），锚于
  **漂移（π_S0）与特权差分（δ）**（非轨迹级 outcome 次序），且显式处理 **frozen teacher 漂移**。
- **AOPD / Asymmetric OPD（2605.06387）**：非正优势区把 advantage-weighted PG **换成对 same-policy
  teacher（on-policy 前缀条件）的局部 forward-KL**，无特权/无腐蚀/无对比，**advantage/RL 触发**。
  **与 C′-4（对比式）不实质重叠**：同为"坏区域分布匹配"，但 AOPD 锚 = **advantage**、teacher =
  same-policy；C′-4 锚 = **腐蚀差分 δ**、teacher = 特权 correct vs 腐蚀对比，无 RL。→ **C′-4 保留**
  （不降级），related work 引 AOPD 划界锚差异；**C′-1 不受影响**（更远）。
- **Unmasking OPD（2605.10889）**：training-free 梯度对齐诊断（per token/question/teacher 的 gradient
  alignment score）。**独立佐证本案**：其"distillation 在 incorrect rollout 上对齐显著高于 correct
  （correct 处学生已会、teacher 信号变噪）"**呼应本案裁决二 (i)**（教学含量 wrong>correct 1.27×）与
  wrong-支手术 rationale；其"wrong demos 伤自蒸馏、hard math 例外"呼应本案难度分层。**不重叠**：其为
  **梯度轴遥测诊断**，本案为 **rollout 级校准级干预** + 漂移轴（drift share）量化，互补。
  *引用建议*：作**独立第三方佐证**入 related work 与 (i) 结论并列。

## 实验矩阵（**训练冻结**；ω 为消融轴，2026-07-15 终审）

- **基底**：repo 冻结口径（clip 0.05、1024、TM-off、solution、seed42、gb30）；ρ=0.0007（不调参）。
- **主案**：全局漂移扣除（B）+ 错支 C′ + **correct 支 ω=1.0（等权）** + truncated 分派（(c) 裁）。
- **消融轴**（各相对主案单变量关闭/改档）：
  1. **去全局漂移扣除**（B off）；
  2. **去 C′**（wrong 支仅完整蒸馏，无促修正）；
  3. **correct 支 ω=0.5**（降权档，与主案 ω=1 对照）；
  4. **（算力允许）ω=0 臂**——仅补剂量-响应曲线，**预期掉分**，作 clip 式承重验证。
- **护栏**：腐蚀质量在线 + 泄露检测器（每 5 步 dump）+ 漂移份额曲线。
- **评测**：锁定协议（见下），AIME24/25 avg@12 + MATH500 avg@4 @ ckpt100/150，阶段内对照。

## 训练放行方案（指令三，只写不跑）

- **解冻条件**：**(c) 已落地**（AUC=0.648<0.65 → truncated 支保守按 wrong，分派方案已定）；
  裁决二已收案（correct ω=1 主案 + ω 消融轴）。→ **剩余唯一门槛 = 用户明示发令。**
- **最小实验矩阵**（1.7B 单 seed）：

  | # | run | 训练 | 说明 |
  |--:|---|:--:|---|
  | 1 | 主案（B + C′-1 + ω=1，1024） | ✓ | 全局漂移扣除 + 错支 C′-1 + correct 等权 |
  | 2 | 去 B（C′-1 + ω=1） | ✓ | 漂移扣除消融 |
  | 3 | 去 C′（B + ω=1） | ✓ | 错支手术消融 |
  | 4 | ω=0.5（B + C′-1） | ✓ | correct 降权档消融 |
  | 5 | **max_len 2048（主案 + truncated 直判）** | ✓ | **必跑**：max_len 轴 + truncated 直判替代保守默认 |
  | 6 | ω=0（B + C′-1）〔算力允许〕 | ✓ | 剂量-响应端点，预期掉分 |
  | B1 | base（未训练 Qwen3-1.7B） | ✗ | 仅 eval 基线 |
  | B2 | 官方 OPSD（复用 `qwen31b_repro_3xh200_gb30`） | ✗ | 仅 eval 基线，不重训 |

- **预计总卡时（修订，含 max_len 臂）**（2 GPU/训练 run，gb30 保持 2×5×3）：
  - 训练：run 1–4 各 ~1.5h × 2 GPU = ~12 GPU-h；**run 5（2048）~2.5h × 2 = ~5 GPU-h**（生成翻倍）；
    加 ω=0（run 6）~3 GPU-h → **必跑 5 臂 ~17 GPU-h；含 ω=0 ~20 GPU-h**。
  - eval：必跑 5 训练 + B1 + B2 = 7 条 × 2 ckpt × (AIME24/25 avg@12 + MATH500 avg@4)，
    1 GPU/eval，~1.5h/条 ≈ **~10–12 GPU-h**（2048 臂 eval 略高）。
  - **修订合计 ≈ 27–32 GPU-h**（含 ω=0 ~30–35）；8 卡共享池 **≤ 一夜偏满**。4B 待 1.7B 后另议。
- **评测锁定协议**：temp 1.0 / top_p 1.0 / top_k −1 / min_p 0 / presence_penalty 0 /
  max_new_tokens 38912 / thinking ON；AIME avg@12、MATH500 avg@4。
- **当前不启跑**（§9：训练冻结中，待 (c) 落地 + 用户发令）。
