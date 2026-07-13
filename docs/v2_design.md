# v2 设计：EMA teacher（新鲜度轴）—— 只写设计与预算，不跑

> 论文映射：v2 主线设计载体（草稿在作者处，见 `docs/paper_map.md`）。**先写定，后开跑。**

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
