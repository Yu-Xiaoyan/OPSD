# 三路分诊 rationale 验证（预注册，先落盘后跑；纯 probe，不训练）

> 承 v2 架构裁决（2026-07-15）：v2 主案 = **按 rollout 对错三路分诊 + 错支目标手术**
> （correct 支降权/跳过；wrong 支完整蒸馏 + C′ 促修正 + 漂移扣除；truncated 支 V(t) 分派）。
> 本文件验证分诊的三条 rationale。**写定即锁死判据，跑后不动。** 输出
> `probes/analysis/triage_rationale.json`。

## 数据

- **(a)/(b)**：现有 **3-seed 漂移分解数据**（`drift_shard*` / `drift_s1_shard*` / `drift_s2_shard*`，
  各 187 rollout = correct 107 + wrong 80），按 verifier-v2 桶（rollout 自带 `bucket`）重聚合。
  两桶 n（correct 107 / wrong 80）均 ≥20 → (a)(b) 不标 provisional。
- **(c)**：`diag_truncated.jsonl`（138 truncated，含 `V_values`[32]/`V_positions`，末位=截断点读数）。

## 预言（跑前锁死）

### (a) correct 桶漂移份额更高
150 步处 **correct 桶 overall drift share 高于 wrong 桶 ≥ 10pp（3-seed 均值）**。
- drift share 口径同主曲线：`100·Σdrift_k/(Σdrift_k+Σteach)`，按桶分别聚合。
- rationale：correct 桶"学生已会"，监督信号以漂移（拽回起点）为主 → 降权/跳过有据。

### (b) wrong 桶腐蚀敏感（修正需求）更强
**wrong 桶的腐蚀差分 δ 正向质量（P_T 加权口径，与 R-b 一致）≥ 1.5× correct 桶**。
- 每 rollout 标量 = `Σ_t Σ_v P_T·max(δ,0)` / T（冲突位置口径同 direction_axis：p2/p1≥0.3；
  与 R-b 一致取冲突位置上的 P_T 加权 δ 正部），按桶取均值。
- teacher=base 固定；δ=log π_T−log π_T̃（腐蚀=特权答案本体，复用 corrupt pipeline）。
- rationale：wrong 桶需"目标手术/促修正"，其腐蚀敏感成分应显著强于 correct 桶。

### (c) truncated 桶 V(t) 截断点读数区分"后续正轨/歪轨"
**AUC ≥ 0.65**。
- **操作化困境（已落盘）**：138 truncated **无一含 `\boxed`（可直接判对错子集 = 0 < 30）**
  → 直接口径不可行，启用**替代操作化（预注册）**：
  - **续写判对错**：对每条 truncated rollout，以 `student_prompt(problem)+truncated_completion`
    为前缀，用 **ckpt150 student** 续写（锁定协议 temp 1.0/top_p 1.0/top_k −1/min_p 0），
    **续写预算 cap = +4096 token**；含 `\boxed` 且 verifier-v2 判对 → **label=正轨(1)**；
    判错 → **歪轨(0)**；**仍截断（未出答案）→ 归 0（歪轨）**（保守：未收敛视为 off-track）。
  - **读数** = 截断点 V(t) = `V_values[-1]`（gt 答案在截断位的 log-likelihood）。
  - **AUC**(读数, label)；报正轨/歪轨各 n。
  - 若续写后 label 单一类（全 0 或全 1）致 AUC 未定义 → 记"未定义"，(c) 判**不成立**。

## 判定（锁死）

- **(a) 且 (b) 成立** → 分诊 rationale 获支撑，**训练解冻条件满足一半**（另一半 = 用户明示发令）。
- **(a) 不成立** → correct 支降权**失据**，**架构重审**。
- **(c) 不成立** → truncated 支改**保守默认（按 wrong 处理，完整蒸馏）**，方法保留但削弱。
- 各桶报 n；**correct 桶 n<20 → (a)(b) 标 provisional**（本数据 correct=107，不触发）。

## 实现与预算

- **(a)**：`probes/triage_drift.py`（CPU）——3-seed drift 数据按桶重聚合 → per-bucket/per-seed
  drift share 曲线 + 3-seed 均值 + (a) 判读。零 GPU。
- **(b)**：`probes/triage_delta.py`（GPU，1 卡）——correct+wrong 各 rollout 补腐蚀 forward
  （π_T/π_T̃，base），算冲突位 P_T 加权 δ 正部均值，按桶聚合 + (b) 判读。~187×2 forward，估 <30min。
- **(c)**：`probes/triage_truncated_vt.py`（GPU，1 卡，续写）——138 truncated 续写（cap +4096）
  + verifier-v2 + AUC。续写 138 条估 20–60min（vLLM 批量）；> 2h 则分片。
- 存储纪律 §4：只落派生标量/AUC，不存 [T,V]。所有作业先 py_compile + 逻辑自检再投。

---

## 结果

### (a) correct vs wrong 漂移份额 —— **FAIL**（`triage_a_drift.json`）

3-seed 均值 overall drift share（%），correct 与 wrong 桶**几乎完全重合**：

| step | correct | wrong | Δ(c−w) |
|--:|--:|--:|--:|
| 25 | 36.3 | 35.7 | +0.5 |
| 50 | 49.1 | 48.5 | +0.6 |
| 100 | 58.0 | 58.1 | −0.1 |
| 150 | 60.2 | 60.5 | **−0.3** |

- 150 步 Δ(correct−wrong) = **−0.3pp**（判据 ≥10）→ **FAIL**；per-seed 一致（−0.3/−0.2/−0.3）。
- 桶内对照 → off-policy 偏置两桶同量抵消，null 稳健。correct n=107 / wrong n=80，不 provisional。
- **判决**：**correct 桶漂移并不比 wrong 桶多** → "correct 支信号以漂移为主"**不成立** →
  **correct-支大幅降权/跳过失据 → 架构重审**（替代降权依据待另立，如 teach 绝对量 / 可学空间）。

### (b) wrong vs correct 腐蚀敏感（δ 正向质量，P_T 加权）—— **PASS（贴线）**（`triage_b_delta.json`）

冲突位 P_T 加权 δ 正部均值：correct **0.00773** / wrong **0.01198** → **wrong/correct = 1.55×**
（判据 ≥1.5）→ **PASS**（贴线，n=107/80，不 provisional）。
- **判决**：wrong 桶修正需求确高于 correct（1.55×，但仅略过阈）→ **wrong-支目标手术/C′ 前提成立（边际）**。

### (c) truncated V(t) AUC —— **FAIL（贴线）**（`triage_c_truncated.json`，job 33409）

138 truncated 以 ckpt150 续写（cap +4096，锁定协议）+ verifier-v2：
- 续写后 **正轨 68 / 歪轨 70**（歪轨 = wrong 61 + 仍 truncated 9）——**~49% 的 truncated 续写后达对**
  （只是原长度不够）。类别均衡，AUC 有定义。
- **AUC(截断点 V(t), 正轨label) = 0.648**（判据 ≥0.65）→ **FAIL（以 0.002 之差贴线）**。
- **判决（按预注册）**：**(c) 不成立 → truncated 支改保守默认（按 wrong 处理，完整蒸馏）**。
  V(t) 截断点读数对"后续正轨/歪轨"有**弱预测力**（0.648 > 0.5）但**未达分派门槛** → 不用 V(t) 分派。

### 综合判决

- **(a)(b) 半解冻条件 = 未满足**（(a) FAIL）。**wrong-支 C′ 有据、correct-支降权无据**。
- → **三路分诊架构需重审 correct 支**（见 `v2_design.md` 线 2 的架构重审标注）。训练维持冻结。

---

## 裁决二：correct 支重审 —— 教学含量探针（预注册，纯重聚合，不训练）

(a) 证伪了"correct 支漂移更多"的降权依据。改测**教学含量**（correct 支若教学含量低，则降权
重新获据，且与漂移无关）。**现有数据重聚合，跑前锁死判据。** 输出 `probes/analysis/triage_teaching.json`。

- **(i) teacher-student 分歧绝对量（决定性）**：3-seed drift 数据的 `per_k[k]["total"] =
  JSD(T_S, S_k)` 逐 token 绝对量，按桶聚合，报 **wrong/correct 比值**。主判 **step=50**（rollout
  由 ckpt50 生成，on-policy 匹配）；附全步轨迹。
- **(ii) δ 正向质量（辅助，不单独触发）**：P_T 加权 δ 正部的桶间比值 = triage (b) 的换口径复测。
  当前取 (b) 已测值 **wrong/correct = 1.55×**（P_T 加权 δ+ 均值，冲突位口径）。报各桶 n。

### 判据（锁死；由 (i) 触发）
- **(i) ≥ 1.5×** → correct 支降权**重新获据**，依据登记为 **"教学含量低"（与漂移无关）**。
- **(i) < 1.2×** → **取消 correct 支降权改等权**；方法差异化收缩为 **"错支手术 + 全局漂移扣除"**，
  v2 架构相应简化。
- **1.2 ~ 1.5×** → **轻降权**，报人判。
- (ii) 作辅助证据，不单独触发判定。各桶报 n（correct 107 / wrong 80，不 provisional）。

### 裁决二结果（`triage_teaching.json`）

- **(i) 教学含量 JSD(T_S,S_k) 逐 rollout 均值，池化 3-seed**（n: correct 321 / wrong 240）：

| step | correct | wrong | wrong/correct |
|--:|--:|--:|--:|
| 50 (主判) | 0.0279 | 0.0353 | **1.27×** |
| 150 | 0.0379 | 0.0470 | 1.24× |

  全步 1.24–1.27，**3-seed 完全一致**（@50 = 1.27/1.27/1.27）。
- **(ii) 辅助**：P_T 加权 δ+ 桶间比值 = **1.55×**（triage b 换口径复测，同向更强）。
- **判决（战略层终审 2026-07-15）**：**"correct 桶教学含量略低于 wrong（−21%），不支持跳过或
  大幅降权。"** 1.27× ∈ (1.2, 1.5) 不足以支撑预设降权 → **correct 支权重 ω 改为消融轴**
  （不由探针拍死），**主案默认 ω=1.0（等权）**——分诊的差异化**全部由错支手术承担，correct 支不动**。
  与 (b) 1.55×、(ii) 复测 **并列引用**。
- **探针阶段收官**：**不新增"可学空间"探针**——该参数将由**训练实验直接裁决**，间接代理探针无必要；
  另引 F4（(a)）教训——**机制想象驱动的探针需克制**。correct-支权重下沉为训练矩阵的 ω 消融轴。
