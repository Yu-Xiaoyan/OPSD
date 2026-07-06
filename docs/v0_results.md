# v0 main run — 结果与判读（按 v0_predictions.md 预注册口径）

**Eval 完成 2026-07-07**（job `30605.gaas`，exit 0）。锁定协议：AIME24/25
avg@12，temp 1.0 / top_p 1.0 / top_k -1 / min_p 0 / max_new 38912 / thinking on。
30 题满量，单 seed。base = 未训练 Qwen3-1.7B。判读口径在结果产生前已 pre-register
（`docs/v0_predictions.md`），此处如实报告，无论支持与否。

## 结果表（avg@12, %）

### AIME24  (base = 49.2)
| step | OPSD | v0 | Δ(v0−OPSD) | v0 gain/base |
|---|---|---|---|---|
| 50 | 52.5 | 55.0 | **+2.5** | +5.8 |
| 100 | 55.0 | 56.9 | **+1.9** | **+7.7** (peak) |
| 150 | — | 56.1 | — | +6.9 |

### AIME25  (base = 35.0)
| step | OPSD | v0 | Δ(v0−OPSD) | v0 gain/base |
|---|---|---|---|---|
| 50 | 40.6 | 42.2 | **+1.7** | **+7.2** (peak) |
| 100 | 43.1 | 40.8 | **−2.2** | +5.8 |
| 150 | — | 41.7 | — | +6.7 |

OPSD peak gain（参考，= ckpt100 − base）：**AIME24 +5.8 / AIME25 +8.1**。

## 主判据（primary accept/reject）— v0 best gain/base vs OPSD

- **AIME24：v0 +7.7 (ckpt100) > OPSD +5.8 → ACCEPT。** 且逐 checkpoint 同步对比
  v0 全程领先（+2.5、+1.9），ckpt150 (56.1) 仍高于 OPSD 的 ckpt100 峰值。
- **AIME25：v0 +7.2 (ckpt50) < OPSD +8.1 → REJECT（略逊 0.9）。** OPSD 在 ckpt100
  冲到 43.1，v0 同步反而回落到 40.8（−2.2）。

**综合：分裂结果。** 两 benchmark 平均峰值增益 v0 7.45 vs OPSD 6.95（v0 略优），
但优势完全来自 AIME24；AIME25 上 OPSD 的 ckpt100 峰值 v0 未能追平。

## Prediction P（drift-erosion：v0 峰值更晚 且/或 post-peak 更稳）

- **峰值步：不支持"更晚"。** AIME24 v0 峰 = ckpt100 = OPSD 峰步（相同，不更晚）；
  AIME25 v0 峰 = ckpt50，比 OPSD（ckpt100）**更早**。预注册证伪条件的前半
  （"v0 peaks no later than OPSD"）成立。
- **post-peak 稳定性：无法对照。** OPSD baseline 只 eval 了 ckpt50/100，**缺
  ckpt150**，无法与 v0 的 post-peak 回落直接比较。v0 自身 post-peak 平缓
  （AIME24 ckpt100→150 仅 −0.8；AIME25 ckpt50→150 在 40.8–42.2 小幅波动），
  但没有 OPSD ckpt150 做对照，证伪条件的后半无法判定。
- **结论：Prediction P 在峰值步维度被证伪，稳定性维度数据不足。** drift-erosion
  的"延后饱和"版本不被本 run 支持。

## 注意事项 / 数据缺口

- **小样本 + 单 seed：** AIME 各 30 题，1 题 = 3.3pt。表中 1.7–2.5pt 的 Δ 落在
  单题量级，应读作趋势而非定论。预注册即规定单 seed 先行，multi-seed only after pass。
- **OPSD 缺 ckpt150：** 要完成 Prediction P 的 post-peak 对照，需补 OPSD baseline
  (`qwen31b_repro_3xh200_gb30`) 的 ckpt150 eval（单 checkpoint × 2 dataset，<2h）。
- **AIME24/AIME25 方向相反：** v0 在 AIME24 稳定占优、AIME25 峰值略逊，值得下一步
  排查是数据集特性还是 gating 在某类题上的副作用。

## 建议下一步（待决策，未擅自执行）

1. 补 OPSD ckpt150 eval → 完成 P 的 post-peak 对照。
2. 若继续，multi-seed（≥3）复跑 v0 + OPSD，看 AIME24 优势与 AIME25 劣势是否稳健。
