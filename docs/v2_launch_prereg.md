# v2 训练放行预注册（跑前锁死；解冻发令 2026-07-15）

> 承 `v2_design.md` 线 2 架构定案 + 战略层解冻发令。**本文件 commit+push 后方可 qsub。**
> 判读线跑前写死，结果后不动。实现 = `v2_trainer.py`（扩展 `OPSDDeclipTrainer`），
> 冻结口径核验 + 20 步 smoke 绿后再投正式。

## 方法（loss 数学，pinned）

每 rollout 经 verifier-v2 分桶（correct / wrong / truncated；**truncated → 按 wrong 处理**，
(c) 收案）。teacher=base 固定。per-rollout 前向：
- `S` = student（带 grad）；`π_T` = base 特权 teacher；`π_T̃` = base 腐蚀 teacher（corrupt_solution）；
- `π_S0` = base **student prompt**（adapter off = 初始学生快照，供漂移扣除）。

**目标 logits 重构**（renorm 后取 `forward-KL(target‖S)`）：
- 基础：`log_tgt = log π_T`。
- **C′-1（δ 加权重构，仅 wrong 支）**：`log_tgt += λ · max(δ, 0)`，其中 `δ = log π_T − log π_T̃`（逐 vocab）。
- **B（全局漂移扣除，所有支，若开）**：`log_tgt −= γ · log π_S0`（PMI 式扣除"回起点"分量）。
- `target = log_softmax(log_tgt)`；`loss_i = Σ_v clamp(target·(log target − log S), max=0.05)`
  （forward KL，**保留 element-wise clip 0.05** = 冻结口径，P-a 净正）。
- **ω（correct 支权重）**：correct 支 loss × ω，wrong 支 × 1。`loss = Σ w·loss_i·nt / Σ w·nt`。

**pinned 超参**（默认，可调；记于此以透明）：
- **λ = 1.0**（C′-1 δ 权重）；**γ = 0.5**（漂移扣除强度）；**ρ = 0.0007**（不调参，仅 C 系用）。
- 其余全沿 repo 冻结口径：clip 0.05、TM-off、solution、seed42、gb30（2×5×3）、150 步、fixed teacher。
- max_len：主案 1024；max_len 臂 2048。

## 实验矩阵（各臂配置）

| # | run 名 | B | C′-1 | ω | max_len | 训练 |
|--:|---|:--:|:--:|:--:|:--:|:--:|
| 1 | `v2_main` 主案 | ✓ | ✓ | 1.0 | 1024 | ✓ |
| 2 | `v2_noB` 去漂移扣除 | ✗ | ✓ | 1.0 | 1024 | ✓ |
| 3 | `v2_noCp` 去 C′ | ✓ | ✗ | 1.0 | 1024 | ✓ |
| 4 | `v2_w05` ω=0.5 | ✓ | ✓ | 0.5 | 1024 | ✓ |
| 5 | `v2_len2048` max_len 轴 | ✓ | ✓ | 1.0 | 2048 | ✓（必跑） |
| 6 | `v2_w0` ω=0〔算力允许〕 | ✓ | ✓ | 0.0 | 1024 | ✓（选） |
| B1 | base（未训练） | — | — | — | — | ✗ eval 基线 |
| B2 | 官方 OPSD（复用 repro） | — | — | — | — | ✗ eval 基线，不重训 |

## 评测协议（锁定）
- temp 1.0 / top_p 1.0 / top_k −1 / min_p 0 / presence_penalty 0 / max_new_tokens 38912 / thinking ON。
- **AIME24/25 avg@12 + MATH500 avg@4 @ ckpt100/150**（base 单点）。阶段内对照。

## 判读线（跑前写死，结果后不动）
- **部件"有效"判据**：某方法臂相对**官方 OPSD 基线（B2）**，**AIME24 ≥ +1.5 且 MATH500 不降** →
  该部件（B / C′-1 / ω 档 / max_len）记 **"有效"**。
- **方法线转档判据**：**全臂 AIME24 增益 < +1.5** → 方法线（v2 主案）**转入台账**，论文按
  **A 路线（纯诊断）** 推进。
- **8/15 闸门规则不变**。

## 放行纪律（§7/§9）
- 冻结口径核验（λ/γ/ρ/gb30/seed/clip/step 落地一致）+ **20 步 smoke 绿**（qstat+log 双证）后，
  方投正式 150 步。smoke 未绿不投。
- 各臂正式作业号 + ETA 报回，用户终端 qstat 核对。

## 启动记录（2026-07-15）
- **main 臂 20 步 smoke 绿**（job 33437，exit 0，20/20，train_loss=3.62，v2 分桶 metrics 正常
  correct~0.51/wrong~0.21/truncated~0.27，~14.6s/step）→ 新 loss 全路径验证通过。
- **正式矩阵 qsub**（2 卡/臂，150 步）：main=33443 / noB=33444 / noCp=33445 / w05=33446 /
  len2048=33447 / w0=33448。ETA ~37min/run（2048 ~60min）+ 依赖 eval。
