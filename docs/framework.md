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

- **门 A（2×2 左上格）**：正确 rollout 上，腐蚀敏感位置**之外**是否存在分散于
  推理主体的师生分歧？→ 决定正确分支方案 **(a)/(b)/(c)**。
- **门 B（`V(t)` 定位精度）**：标注 100–200 条错误 rollout 的首个出错步，
  `t*` 命中率是否 **≥~70%**？→ **硬定位** vs **`ΔV` 软加权**。
- **门 C（2×2 右下格）**：修正信号集中于答案位置这一现象的**普遍性** →
  决定 **motivation 强度**。

---

## v1 推迟项（明确不做，防止范围蔓延）

- CoT teacher（`COT_Reason` 字段）
- 特权阶梯与三分布 taxonomy
- 对 CoT 的语义保持 / 破坏扰动
- DivideMix 式自适应加权
- meta-reweighting
- QBC 式高分歧位置预算升级
- EMA teacher 消融
