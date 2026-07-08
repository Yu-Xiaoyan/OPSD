# 机制验证包 M1–M4：门控在信号层面是否按设计工作

> 论文映射：分支二机制件（草稿在作者处，见 `docs/paper_map.md`）。

## 背景与目的

T2 与难度分层均显示 **v0 性能中性**（相对 OPSD 无稳健优势，预注册 P 已完全证伪）。
TRD 论断：**reweighting 在结构上不治 prefix failure**——降权只能减少冲突项的贡献，
不能在冲突位置提供正确信号（bimodal mixture 结构仍在）。本包验证两件事，各自预注册、如实报告：
- **(a) 机制正确**：门控确实正确识别并降权了"坏信号"（冲突/低价值位置）。
- **(b) 为何正确机制没兑现成分数**：质量占比核算 + 无新信号——正确的处置只作用于极小
  一块监督质量，分数天花板由算术而非机制失败决定。

## 预注册（结果产生前锁定）

- **M1 门控 vs 独立冲突度量**（防循环：门控用 ΔV；冲突用 teacher 分布**形状**——
  wrong 桶逐 token 的双峰性 top-2 质量比 / entropy 尖峰，与 ΔV 无共享输入）。
  检验 v0 低权重位置与高双峰位置的秩相关 / AUC。**预期：显著正相关**（门控从独立信号恢复了冲突结构）。
- **M2 梯度碎裂度**：wrong 桶内 logit 层逐 token 梯度方向一致性（两两余弦分布 / 合成梯度范数比），
  uniform vs v0 加权对比。**预期：v0 加权下碎裂度下降**；同时如实测下降幅度——TRD 预言"混合结构仍在"
  （降权只减冲突项、不在冲突位提供正确信号）。
- **M3 单步更新对齐**：同批数据分别按 OPSD loss 与 v0 loss 做一次小 lr 假想更新，测 held-out 锚集
  p(y*|x) 变化。**预期：v0 更新对答案似然的提升 ≥ OPSD**（轴 2 在更新层的直接检验）。
- **M4 质量占比核算**（零训练，最关键解释件）。**预期：prefix-failure 靶向触及的监督质量占比 X 很小
  （个位数%）→ 性能中性是算术必然而非机制失败**。

## 可行性纠正（数据现实 vs 指令）

保存的诊断数据（`probes/data/diag2x2_*`、`drift_*`、`diag_truncated`）含 `jsd_teacher_student`、
`clipped_kl`、截断桶 `V_values` 等**派生量**，但**不含 teacher 分布本身的形状**（top-2/entropy），
wrong 桶也未存 ΔV/V(t)。因此 **M1 无法零 GPU**（teacher 双峰性与 wrong 桶 ΔV 都需重跑 forward）。
→ **M1 并入 M2/M3 的 1 卡短作业**（同批 forward 中顺带抽 teacher entropy/top-2 与 V(t)）。
**仅 M4 为真正零 GPU**，已完成如下。

## M4 结果（零训练，已完成）

桶占比（gated 训练日志 225 步平均 + ckpt50 rollout dump 200 条 / 184,791 tokens）：

| bucket | example% (train log) | example% (dump) | **token%** (dump) | avg_len |
|---|---|---|---|---|
| correct | 30.0% | 21.0% | 15.7% | 691 |
| **wrong** | **7.8%** | 10.0% | **7.8%** | 722 |
| truncated | 62.1% | 69.0% | 76.5% | 1024（顶格） |

`gate/weight_mean=0.441`（uniform=1.0）。

**核算**：prefix-failure 的 ΔV 软加权**只作用于 wrong 桶**（bimodal-conflict 的病灶所在），
token 级仅 **~7.8%**；其中还需再乘"V-drop(ΔV<0)且 w<0.5"的子集比例（待 GPU 作业出 ΔV 精确化，
预计把靶向进一步缩到个位数低端）。truncated 桶虽占 76.5% token，但其处置是**基于 V(end) 的钝降权**
（恶化半桶 w=0.3），非 prefix-failure 的精准靶向。

**判定（支持 M4 预注册预期）**：**严格 prefix-failure 靶向 ≤ ~7.8% 的监督 token，且强降权子集更小。**
即便门控完美识别并中和这些冲突位置，可影响的监督质量也被限制在这一小块 →
**性能中性是算术必然（分数天花板由质量占比决定），而非"病灶不真"或"处置错误"。**
这与 (a) 待验（M1–M3）互补：病灶真实 + 处置正确 + 占比极小 = 中性。

## 待办
- **M1+M2+M3**：1 卡短作业（wrong 桶 rollout 上重跑 teacher/student forward，抽 per-token
  teacher entropy/top-2、logit 层梯度、单步更新前后锚集 p(y*|x)）。预算与脚本见下一步，等发令。
- M4 的 wrong 桶内 w<0.5 精确子集比例，随 M1 的 ΔV 一并回填。

## M1 / M2 结果 + M4 剂量终值（probe job `31291`，1 卡）

| 项 | 预注册预期 | 实测 | 判定 |
|---|---|---|---|
| **M1** | 门控低权重 ↔ teacher 高双峰，显著正相关 | spearman(w,top2)=**0.019**；spearman(w,entropy)=**0.022**；AUC(低w→高top2)=**0.489**；n=14,449 token | **不成立（NULL）** |
| **M2** | v0 加权下碎裂度下降 | 相干比 ‖Σw·g‖/Σw‖g‖：uniform **0.1429** → v0 **0.1460**（+2.2% 相对）；75% rollout 改善 | 方向支持，**幅度可忽略** |
| **M4 剂量** | X 小（3–5%） | wrong 桶内有效移除 **51.7%**；`Σ_wrong(1−w_t)/总质量` = **4.0%** | **命中** |
| M3 | v0 更新 ΔV ≥ OPSD | 旧结果四项全 0 = **probe bug**（measure 时 `disable_adapter` → 量的是 base，ΔV 构造性为 0） | 无效，已修（adapter-on），job `31490` 重跑 |

### 判读（定稿）

**门控作用于 outcome 轴（ΔV），与 conflict 轴（teacher 分布双峰性）实测正交**——AUC 0.49，等同抛硬币。
连同 **M2 微效**（降权只压掉约 2% 的冲突项，相干比本身仅 0.14，高度碎裂依旧）与 **M4 剂量 4%**，
三者构成 **"reweighting 不打 conflict 病灶" 的三重确认**，与 TRD 的结构论断
（*reweighting 在结构上不治 prefix failure*）**交叉验证**。

即：v0 的门控**病灶定位在另一根轴上**——它按结果似然（ΔV）降权，而 TRD 所指的病灶是 teacher 分布的
冲突（bimodal mixture）。两轴正交，故门控既不识别冲突位置（M1），也只能微弱削减冲突项（M2），
且可及的监督质量本就只有 4%（M4）。**性能中性由此获得完整解释：靶点错轴 + 剂量极小。**

**caveat（保留，不用于软化结论）**：双峰性以 top-2 质量比 / entropy 度量，teacher 用直接特权代理。
若认为该度量未捕捉 TRD 的 bimodal mixture，需换度量重测。**但按预注册的度量，结论就是 NULL。**
