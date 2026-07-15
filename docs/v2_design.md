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

> **状态：架构 rationale 验证中 —— correct 支 rationale (a) 已 FAIL → 架构重审进行中。**
> triage 验证（`triage_rationale_recon.md`）：**(a) FAIL**（correct/wrong 漂移份额相同，150 步
> Δ=−0.3pp，3-seed 一致）→ correct-支降权失据；**(b) PASS 贴线**（wrong/correct 修正需求
> 1.55×≥1.5）→ wrong-支 C′ 前提成立（边际）；**(c) 待跑**。**半解冻条件未满足**（因 a）。
> C 存活（R-a K=50=−0.214，稳健确认）保留为 **wrong 支的 C′ 部件**。**训练全部冻结。**

## 方法形态（用户定义 2026-07-15）

按 rollout verifier-v2 对错**三路分诊**，各支走不同 loss：

- **correct 支｜大幅降权 / 跳过**。原 rationale = 学生已会、信号以漂移为主。
  ⚠️ **该 rationale 未获支撑**：triage (a) 实测 correct 桶漂移份额**不高于** wrong 桶
  （150 步 Δ=−0.3pp）→ "correct 桶漂移更多"不成立 → **降权失据，重审**。
  替代降权依据待另立探针（候选：teach 信号**绝对量** / "已正确=可学空间小"的直接度量，
  非漂移份额）。
- **wrong 支｜完整蒸馏 + 目标手术**。**C′（促修正目标）挂载此支**（子节见下）；
  **漂移扣除**（第三参照 = 初始学生快照 `π_S0`，从蒸馏目标中减去"回起点"分量）为**同支候选**，
  与 C′ **可对照可叠加**。triage (b) 支持本支前提（wrong 修正需求 1.55×correct，边际）。
- **truncated 支｜V(t) 分派**。verifier 失明；用 V(t) 截断点读数把 truncated rollout 分派到
  上两支的处理。默认由 triage (c) 裁：**(c) 不成立 → 保守按 wrong 处理（完整蒸馏）**。

## C′ 子节：促修正目标（设计选项表，不定稿）

依据：Q-b 质量加权 **修正类富集 ~33×（≥3×）** → 腐蚀差分 δ 的正向质量集中于修正类标记 →
可据 δ 构造"促修正"目标。**选项（列不选，各附风险）：**

| 选项 | 形态 | 预期风险 |
|---|---|---|
| **C′-1 δ 加权目标重构** | 蒸馏目标按 δ 正部重加权（放大修正类成分） | δ 噪声放大；需归一化，超参引入 |
| **C′-2 仅蒸馏高 δ 成分** | 目标 = π_T 在高 δ token 上的分布，其余位降权 | 丢失非修正的正确信号；可学面变窄 |
| **C′-3 δ 与漂移扣除叠加** | wrong 支目标 = (π_T − 回起点分量) 再叠 δ 促修正 | 两手术交互未知；需消融拆解可加性 |

- **对照臂（必含）**：**correct 支"轻量 clip 保护项" vs "完全跳过"**——防"完全跳过 correct 支"
  致格式/风格巩固功能流失（clip 的稳定器/防泄露身份仍需在 correct 支保留最小剂量）。

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

## 实验矩阵草案（**训练全部冻结**）

全部标注 **【冻结】**，仅列不跑：
- **【冻结】** 基底：repo 冻结口径（clip 0.05、1024、TM-off、solution、seed42、gb30）。
- **【冻结】** 主案：三路分诊（correct 降权/跳过 × wrong 完整+C′ × truncated 分派）。
- **【冻结】** 消融支数：{correct 支：轻量 clip 保护 vs 完全跳过} × {wrong 支：C′-1/2/3、漂移扣除单列/叠加}
  × {truncated 支：(c) 裁的默认}。
- **【冻结】** ρ=0.0007（不调参）；探针护栏：腐蚀质量在线 + 泄露检测器（每 5 步 dump）+ 漂移份额曲线。
- **【冻结】** 评测：**锁定协议 temp 1.0 / top_p 1.0 / top_k −1 / min_p 0，AIME24/25 avg@12 +
  MATH500 avg@4 @ ckpt100/150**，阶段内对照。

## 训练放行方案（指令三，只写不跑）

- **放行条件（二者缺一不可）**：
  1. triage **(a)(b) 通过**——当前 **(a) FAIL**，条件**未满足**；需先完成 correct-支 rationale 重审
     （替代降权依据成立）或架构调整（如取消 correct-支降权，改等权/轻降）。
  2. **用户明示发令**。
- **最小实验矩阵**：主案（三路分诊）vs 关键消融 —— {correct 支：轻量 clip 保护 ↔ 完全跳过}
  与 {wrong 支：C′ 挂载 ↔ 关闭} 两个 2-level 因子，1.7B 单 seed 侦察 = 主案 + 3 消融 ≈ 4 run。
- **算力估算**：1.7B 单 seed，每 run 含每步 +1 腐蚀 teacher forward（96.6ms 基准）@1024
  ~54–95min/run；4 run 串行 ~4–6.5h（≤ 一夜）+ 依赖 eval。4B 迁移待 1.7B 侦察后另议。
- **评测锁定协议**：temp 1.0 / top_p 1.0 / top_k −1 / min_p 0 / presence_penalty 0 /
  max_new_tokens 38912 / thinking ON；AIME avg@12、MATH500 avg@4。
- **当前不启跑**（§9：训练冻结中，待 (a) 重审通过 + 用户发令）。
