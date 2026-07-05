# 研究框架 v0：基于分歧归因的可信监督蒸馏

> 本文档是已收敛的 **v0 设计的唯一权威版本**。后续所有修正都直接更新到这份文档，
> 不另起副本。标注【caveat】/【来自代码考古】的条目表示与 `docs/archaeology.md`
> 的代码事实绑定，改动前需回查考古报告。

---

## 背景与问题

- OPSD 的两个已知病灶：
  - **privileged information leakage**：teacher 的置信来自它*看到了答案*，而非来自
    可复现的推理；student 在推理时看不到答案，无法复现这份置信。
  - **prefix failure**：在错误 rollout 上，teacher 是在*错误前提*下做局部修补，
    其稠密的 token 级信号未必指向"最终答对"。
- **统一归因视角**：token 级监督信号的可信度是不均匀的。现有方法用
  teacher–student 不一致性这**一个标量**来笼统加权，无法区分分歧究竟*来自哪里*，
  因而把不可信的信号也一并放大。

---

## 核心命题：可信度两轴

- **轴 1 — Groundedness**：teacher 的判断是否*基于 student 可见的前文*，
  而不是复读它被注入的特权信息。
- **轴 2 — Outcome-alignment**：在该位置模仿 teacher 分布是否*确实提高最终答对概率*。
- 原本设想的**第三轴（on-support）在 OPSD 中近似塌缩进轴 1**：师生分歧的来源
  = 特权信息注入 + LoRA 漂移。
  - 【caveat，来自代码考古】`fixed_teacher` 下 teacher 被冻结为 step-0 的 base 权重
    （关闭 LoRA adapter），student 的 adapter 随训练漂移。在**短训程 + LoRA 低秩**
    条件下漂移量小，塌缩论证近似成立。**论文中需如实说明此近似**，不可默认第三轴
    恒等于零。

---

## 重要事实修正（来自代码考古，覆盖此前讨论中的假设）

- **官方主设定 `beta=0` 是 forward KL：`KL(teacher‖student)`，mass-covering**，
  **不是**此前假设的 reverse KL。
  含义：loss 按 **teacher 概率**加权，teacher 在泄露位置的高置信会*放大*这些位置的
  训练信号 —— 也就是说 **leakage 在 forward KL 下被主动放大**。这一事实直接强化了
  本工作的 motivation。
- **官方特权信息 = 数据集 `solution` 字段（完整参考解答）**，而非仅短答案。
  数据集另有 `Answer`（短答案）与 `COT_Reason`（约 5k token 的长推理）字段，
  三者构成天然的**特权阶梯**（v1 素材，v0 不使用）。
- teacher prompt 中含**反复读指令**（"do not copy or paraphrase it … using your own
  words"）：说明原作者*已意识到* leakage 风险，并用 prompt 做**软性抑制**；但这种
  抑制**不可测量**。本工作的探针恰好补上"测量"这一环（intro 素材）。

---

## 两个探针（全部 forward-only，无额外采样）

### 腐蚀探针（测轴 1 · Groundedness）

- **操作**：把 teacher prompt 中 `=== Reference Solution Begin ===` /
  `=== Reference Solution End ===` 定界符之间的 `solution` 替换为**腐蚀版本**得到
  `T_Ã`，逐 token 计算 `JSD(T_A, T_Ã)`。
- **v0 腐蚀方式**：只替换 solution 中的**最终答案**（含 `\boxed{}` 内、以及推理文本中
  出现的答案表达式），**保留推理主体**。
- **腐蚀答案生成规则**：
  - LaTeX 表达式：改系数 / 改根号内数字 / 变号 / 取倒数；纯数字走数字扰动。
  - 约束：格式与真答案一致、复杂度相仿、**≠ 真答案**，且在**错误 rollout 场景下
    ≠ student 自己的错误答案**。
  - 每题生成 **2–3 个**腐蚀版本，取平均 / 交集，以过滤"察觉矛盾的困惑"噪声。
- **2×2 语义表**（同一个 JSD 在不同条件下语义不同，这是全框架最易混淆处）：

  | | 腐蚀源 = 无关干扰答案 | 腐蚀源 = student 自己的错误答案 |
  |---|---|---|
  | **正确 rollout** | **主战场**：跟随腐蚀而动的 token = copying 位置，用于门控降权 | 不适用 |
  | **错误 rollout** | 信号混杂 | 测"正确答案的修正力注入位置"；初步实验显示修正信号几乎只集中在**答案位置**，推理主体 ≈ 0 → **prefix failure 的探针式实锤**，motivation 核心证据 + 轨迹分诊统计量 |

### 答案似然探针 `V(t)`（测轴 2 · Outcome-alignment）

- **定义**：`V(t) = log p(y* | x, s_1:t)` —— 在 rollout 的**推理步边界**
  （以 `\n\n` 切分为主，每 64 token 为备选）通过 teacher-forcing 拼接
  **桥接串 + 答案**打分。
- **桥接串**：`"\n\nTherefore, the final answer is \boxed{" + Answer + "}"`，
  与数据集答案约定一致；保留 **2–3 个变体**做敏感性检查。
- **用途**：
  - 失败定位：`V` 骤降点 `t*`。
  - `ΔV(t)` 作为**免采样的过程级 advantage**。
  - 对/错信号的逐位置**注入通道**。
  - 对**截断未完成**的 rollout 同样适用（verifier 无法处理截断桶，`V(t)` 可以 ——
    **差异化卖点**）。

### 效度策略修订（2026-07 回填）

**行为级泄露 ground truth 在当前设定不可得**（1.7B / TM-off / 150 步 / 官方
prompt 下为干净负结果，见 `probes/analysis/leakage_over_training.md`）。因此
腐蚀探针的效度**不再依赖行为级对齐**，改由三支柱支撑：

1. **2×2 结构判别力**：腐蚀敏感位置的分布是否呈**可解释结构**——集中于中间
   结果 / 答案 span，而非均匀噪声。有结构 = 探针测到了真实的信息依赖。
2. **多腐蚀版本稳定性**：高敏感位置在**每题 3 个腐蚀版本**之间的交集 / 秩相关。
   稳定 = 信号不是单个腐蚀答案的偶然。
3. **后置因果验证**：阶段 2 的 **oracle masking 消融**——mask 掉高敏感 token
   后再训练，观察行为与最终性能的变化。因果闭环 = 敏感位置确实承载可干预的
   privilege 依赖。

（探针工具 `probes/scoring.py::score_with_privilege` 已通过 loss 级黄金对齐，
数值复刻训练目标；效度问题是"探针测到的分歧是否等于 leakage"，由上述三支柱
而非行为检测回答。）

---

## 门控设计（v0）

- **第一层 · 轨迹分诊**：verifier 判对错（零成本）+ 截断桶单独处理。
- **第二层 · token 门控**：
  - **正确 rollout**：待 2×2 左上格诊断结果裁决，**三选一** ——
    - (a) 不产生 loss；
    - (b) EMA-baseline 的轻量似然强化（REINFORCE with moving-average baseline，
      不需组内多样本）；
    - (c) 腐蚀探针门控的轻量蒸馏。
    - 背景：SRPO 发现对已正确样本蒸馏会导致掉点与坍缩；本框架可将其解释为
      —— 正确轨迹上 competence-driven 分歧趋零，**剩余分歧几乎全为
      privilege-driven**（轴 1 不可信成分）。
  - **错误 rollout**：用 `V(t)` 定位 `t*`；
    - `t*` 之前正常蒸馏；
    - `t*` 处对 **student 实际 token** 施加 unlikelihood；
    - `t*` 之后降权或 mask；
    - 若 `t*` 定位精度不足（命中率 `<~70%`）则退化为 `ΔV(t)` **软加权**。
- **第三层 · 目标分布**：v0 仅单一 teacher `T_A`（solution-conditioned，与官方一致），
  **无"选 teacher"问题**。

---

## 范式约束与生态位

- **单 rollout 红线**：**禁止新增采样轨迹**。额外开销仅为在*固定序列*上的
  teacher-forcing forward（`T_Ã` 一次 + `V(t)` 若干次短打分），可并行、成本确定，
  与 SDPO/SRPO 的额外*自回归采样*有**量级差异**（成本表素材）。
- **生态位**：在**低 pass-rate 难题**上，sibling 依赖方法（SDPO/SRPO）无监督源、
  GRPO 组内 advantage 全零；本方法的特权信息（ground-truth 答案 / solution）
  **可得性不依赖采样运气**。→ 主实验需**按 pass rate 分层评测**。

---

## 与近邻工作的切割

- **RLSD**：方向全交给环境 vs 本工作保留稠密 teacher 方向、探针门控。
- **TRD**：轨迹级重写 vs 本工作不重写、归因门控；`V(t)` 提供更便宜的失败定位。
- **TrOPD / TIP**：单一标量可信度 vs **分歧归因**。
- **SRPO**：轨迹级零阶路由（仅 outcome 一个比特选 loss）vs token 级一阶归因；
  **SRPO 可表述为本框架探针全部退化时的特例**。
- **multi-rollout peer distillation**：扰动 rollout vs **扰动信息集**。
- **多 teacher ensemble（老方向）**：扰动参数 vs 扰动同一模型的信息集
  （**反事实信息干预**）。

---

## 诊断阶段决策门（阶段 1 完成后回填结果）

**0.3 采集分桶（证据，ckpt-50，200 题/版，`probes/data/summary_max*.json`）**：

| max_tokens | truncated | correct | wrong | verifier 可判（correct+wrong） |
|---|--:|--:|--:|--:|
| **1024（训练长度）** | **69.0%** | 21.0% | 10.0% | **31.0%** |
| 4096（诊断） | 6.5% | 53.5% | 40.0% | 93.5% |

- **门 A（2×2 左上格）**：正确 rollout 上，腐蚀敏感位置**之外**是否存在分散于
  推理主体的师生分歧？→ 决定正确分支方案 **(a)/(b)/(c)**。
- **门 B（`V(t)` 定位精度 + 分诊主力）**：标注 100–200 条错误 rollout 的首个
  出错步，`t*` 命中率是否 **≥~70%**？→ **硬定位** vs **`ΔV` 软加权**。
  **关键背景**：在训练长度（1024）下 verifier 覆盖率**仅 31%**——69% 的 rollout
  被截断、无 `\boxed{}` 输出，verifier 无法判分（见上表）。因此 **`V(t)` 是主
  分诊工具**，不是 verifier 的补充：它对截断桶照样打分，覆盖了 verifier 完全
  失效的 2/3 训练分布。分离度已验证（correct vs wrong 组 V(末端) AUC 0.812，
  桥接变体 Spearman ≥0.98，见测试日志）。
- **门 C（2×2 右下格）**：修正信号集中于答案位置这一现象的**普遍性** →
  决定 **motivation 强度**。注意错误桶在 1024 下仅占 10%、在 4096 下占 40%——
  普遍性统计应在 **4096 诊断集**上做（桶划分干净），再回看 1024 训练分布。
- **门 D（leakage 轴活跃度与模型规模）**：行为级泄露检测在
  **1.7B / TM-off / 150 步 / 官方 prompt** 设定下为**干净负结果**
  （`probes/analysis/leakage_over_training.md`）；而 RLSD 的泄露观察在 **8B** 上。
  **假说**：泄露（行为级症状及 / 或分布级依赖强度）**随模型容量增长**。
  回填证据：(a) **实验 A**（去指令 1.7B，`qwen31b_noguard_3xh200_gb30`）的行为
  检测结果；(b) 2×2 左上格的**分布级判别力**；(c) 必要时 **8B 复现 run** 的同套
  检测。**裁决影响**：主实验战场选 1.7B 还是 8B；leakage 轴在论文中的戏份配比。

---

## 阶段 1 裁决与 v0 冻结

> **v0 设计冻结于本节。此后任何改动需在本节留显式修订记录（日期 + 变更 + 依据）。**
> 数据出处：`probes/analysis/diag_2x2.md`（2×2 诊断）、`leakage_over_training.md`
> （行为级泄露）、`gate_b.md`（V(t) 终审）、`drift_over_training.md`（漂移）、
> `tstar_annotations.jsonl`（49 条人工审计 = verifier ground truth）。

### 门 A 裁决（正确分支）
正确轨迹的师生分歧三分账（correct 非 null，`diag_2x2.md`）：**腐蚀敏感（特权）
token 只承载 ~30% 的分歧质量，insensitive×any-drift = 69.9%**（非 copying）；
其中 **hi-drift 列 41.1%** 可归因 LoRA 漂移，最大格 **insens×lo-drift×other =
34.1%** 是 competence-driven 核心；style/structural 单列（各 ~7%/13%）。
→ **正确分支 = 可配置 `{none / EMA-baseline 似然强化 / 门控轻蒸馏}`，v0 默认
门控轻蒸馏（结构 token 降权 ×（腐蚀门开启时）corruption-sensitive 降权），
三者进消融。**

### 门 B 裁决（wrong 分支定位机制）
**3c 终审已回填**（`gate_b.md`，1.7B ckpt-50，clear n=10 双版本）：
- **命中率 0/10（±1 步）**：V(t) 最负段与人工 t\* **相关但精度不足** —— 约 4/10
  落在 ±5 步内（402: t\*99/V97、429: 68/65、347: 90/95、380: 30/26），其余偏移大
  （441: 28→V85、466: 113→V37；post-error collapse / 早降各半）。
- **代理阈值 δ = -7.63**，但 **clear/diffuse 最负段 ΔV 分布几乎重叠**（median
  -7.28 vs -7.97）→ δ 区分 clear/diffuse **判别力弱**（`gate_b_dv_dist.png`）。
- **格式鲁棒性 AUC(pseudo vs true_wrong) = 0.700 < 0.75**（强线未达），但方向对
  （pseudo V(end)=-15.8 明显高于 true_wrong=-25.3，接近 correct -10.0）→ V(t) 对
  格式噪声**部分**鲁棒；AUC(correct vs true_wrong)=0.779（cf. 旧 0.812）。
- **裁决**：t\* 硬定位精度不足 ±1（命中 <7/10）→ **clear 型的 t\* 处 unlikelihood
  项退化为 ΔV 软加权**，**v0 wrong 分支统一 ΔV 软加权**（不用硬 t\* 定位）；
  V(t) **主用相对变化 ΔV**（骤降定位 + 过程 advantage），**降低对 V(end) 绝对值
  的依赖**（AUC 0.700 表明绝对值部分含格式信号）。代理分层 δ 保留但标注判别力弱，
  训练期 wrong 桶近似统一软加权。
- **verifier v2 修复力有限（`rebucket_audit.md`）**：全量 v1 wrong→correct 救回
  32/281（11.4%；8B 21.6% > 1.7B 11.2% > wrong_extra 8.0%）。49 条混淆矩阵中
  **pseudo 仅 3/11 被 v2 救回**（多-boxed / option 类），残 8/11 为语义等值
  （interval/set/statement）仍留 wrong 桶；true_wrong 泄漏 1/30（any-boxed 误匹配
  中间步）。故 wrong 桶仍含 ~16% 残留 pseudo，由 **V(t) 软加权二次保护**（pseudo
  V(end) 高于 true_wrong，软加权自动降其惩罚）—— 这是 wrong 桶统一软加权、
  不用硬 unlikelihood 的**第二独立理由**。
- **v0 修订记录（2026-07-05）**：原设计的"clear 型 t\* 硬 unlikelihood"因 ±1 命中
  0/10 退化为 ΔV 软加权；这是本节冻结后的第一条显式修订。

### 门 C 裁决（修正信号集中度）
**claim 软化**：门 C 为**弱信号 + 双峰**（`diag_2x2.md`：lift median ≈3.3 但
n 很小，corruption-null 排除后 gate-C usable 仅个位数）。部分双峰为退化样本
伪影。**最终 claim = "student-wrong 修正信号的集中度呈双峰、与出错位置相关"**，n 与口径
（lift、质量下限、no-reasoning 排除）如实标注，不做超样本量的强断言。pid=19 机制
标本（首分叉集中、不穿透错误前缀）作定性佐证。
- **3d v2 重审已回填**（`gate_c_rereview.md`）：v2 重分桶从 wrong 桶移除 9 条
  format-recovered pseudo 后，两腐化轴 lift **均上升**（studentwrong median
  3.30→6.74、irrelevant 2.21→4.21）→ **pseudo 是稀释而非驱动** gate C 信号，剩余
  true-wrong 的修正信号在答案位置更集中，弱信号 claim 方向稳健。**但 n 未扩量**
  （v2 usable 仍仅 n=6：corruption-null 占 77.5%、no-reasoning 再排除），故维持
  "弱信号、不超样本强断言"的口径；扩量需更大 wrong 采集（v1 规模所限）。
  corruption-null 占比 v1/v2 同为 77.5%（pseudo 不落在 null，移除不改 null 结构）。

### 门 D 裁决（leakage 轴活跃度与模型规模）
**1.7B 侧 leakage 轴疲软**（三条互证）：行为级近零命中（关键词 0.27%、答案
早现 clean 1.28%，`leakage_over_training.md`）；**分布级 corruption-null 占比高
（correct 52% / wrong 78%，`diag_2x2.md`）—— 过半轨迹 teacher 对答案被腐蚀
几乎无反应**；去指令 guard 消融行为无差异（early-clean 1.28% vs 0.88%）。
**8B 补充**：行为级 leakage 8B ≈ 1.7B ≈ 0（8B kw 0.07%），**行为级"随容量增长"
假说不成立**。三个候选抑制器并列：**(a) 模型容量**、**(b) prompt guard + clip
缓解**、**(c) LoRA 低秩**。**[PENDING — 8B 分布级裁决]**：8B 2×2 的 corruption-null
占比 vs 1.7B（若 8B 显著更低则支持容量假说）。

### v0 loss 完整定义（冻结）
- **三桶分诊**（rollout 生成后即时，零成本 verifier + V(t)）：
  - 训练长度（1024）下 **verifier 覆盖 31%**（correct+wrong-with-boxed）；
    **V(t) 主分诊剩余 69%**（含 truncated），verifier 失效处 V(t) 打分。
- **wrong 桶**：门 B 裁决后**统一 ΔV 软加权** `w ∝ σ(ΔV/τ)`（V(t) 骤降处降权、
  骤降前正常蒸馏）。原"clear 型 t\* 处 unlikelihood"因 ±1 命中 0/10 退化为软加权；
  t\* 硬 unlikelihood 项作为 **v1 消融**保留（默认关，λ_unlik=0）。
- **correct 桶**：门 A 三选一（v0 默认门控轻蒸馏）。
- **truncated 桶**：按 V(t) 走势二分 —— 健康段（V(end) 高分位）照 wrong 桶 t\*
  前逻辑蒸馏，恶化段（V(end) 低分位）照 t\* 后逻辑降权。
- **训练期探针预算**：每条 rollout **≤3 次额外 forward**（实测基准 96.6ms/次，
  峰值 4.63GB，`docs/archaeology.md` 复现节）——T_Ã 一次（腐蚀门，可关）+
  V(t) 批处理若干短打分。禁止新增自回归采样。
- **jsd_token_clip 兼具泄露抑制作用**（候选抑制器 b 的机制注释）：clip 封顶
  teacher 高置信位置的**梯度贡献**，而 copying / privilege 位置恰是 teacher
  置信最高处 → clip 隐式压制了泄露位置的训练信号。这可作为 **8B 上的一个
  消融方向记录（clip on/off × 泄露强度），不排期**。

## v1 推迟项（明确不做，防止范围蔓延）

- CoT teacher（`COT_Reason` 字段）
- 特权阶梯与三分布 taxonomy
- 对 CoT 的语义保持 / 破坏扰动
- DivideMix 式自适应加权
- meta-reweighting
- QBC 式高分歧位置预算升级
- EMA teacher 消融
