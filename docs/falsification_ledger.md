# 证伪台账（falsification ledger）

> 统一登记**预注册后被实测证伪**的预言/机制想象。诚实台账：想象错了就如实记，附证据与新事实。
> 各条详情见来源 doc。

| # | 预言 / 机制想象 | 出处 | 实测 | 判决 |
|--:|---|---|---|---|
| F1 | **P-a：clip 是梯度死区，去 clip 释放性能**（B>A） | `v1_design.md` | B≤A 全 6 格，AIME25 跌破 base | **证伪**；clip 本 regime 净正 |
| F2 | **v1 P1：answer-only×gated 四格最优** | `v1_design.md` | 6 格无一最优落此格，常垫底 | **证伪** |
| F3 | **v1 P2：两种特权下 gated ≥ ungated** | `v1_design.md` | 12 项 10 项门控净负 | **证伪** |
| F4 | **分诊预言 (a)：correct 桶漂移份额 > wrong ≥10pp** | `triage_rationale_recon.md` | 桶间几乎不变（150 步 Δ=**−0.3pp**，3-seed 一致） | **证伪** |

## F4 详注（分诊预言 a，2026-07-15）

- **机制想象（战略层）**："correct 桶学生已会 → 监督信号以漂移为主 → 漂移份额应更高" →
  据此给 correct 支大幅降权/跳过。
- **实测**：3-seed 漂移分解按桶重聚合，correct 与 wrong 桶 overall drift share **每一步都几乎重合**
  （150 步 correct 60.2% vs wrong 60.5%，Δ=−0.3pp；per-seed −0.3/−0.2/−0.3）。**想象错误**。
- **新事实入档**：**漂移份额与 rollout 对错无关** → **漂移是全局权重位移现象（global weight
  displacement），非数据条件性（not data-conditional）**。
- **连带架构修正**（见 `v2_design.md` 线 2）：**漂移扣除**从"wrong 错支候选"**升为"全局手术候选，
  作用于所有支"**；**分诊 与 漂移扣除** 定位为**正交部件**——分诊 = 资源分配（对错走不同 loss），
  漂移扣除 = 去毒（从所有支的目标中减去回起点分量）。correct-支降权依据改由裁决二另立探针重审。

## 相关阴性（非预言证伪，但入台账参照）
- **M1 NULL**：门控与 conflict 轴正交（AUC 0.49），见 `mechanism_validation.md`。
- **Q-b 原全词表 1.00×**：尾部支配伪影，判"无信息"（非"假设被否"）；质量加权后 33× 反获支持，
  见 `direction_axis_recon.md`。
