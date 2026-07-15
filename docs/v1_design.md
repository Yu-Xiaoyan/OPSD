# v1 设计：范式内特权配置探索（answer-only × 门控）

> 论文映射：v1 主实验的设计与预注册载体（草稿在作者处，见 `docs/paper_map.md`）；替代此前主实验草案。
> 逐字 prompt 见 `docs/version_genealogy.md`。**本文件写定后再开跑（预注册）。**

## 背景与动机

v0（solution 特权）性能中性，三线确认：**T2**（3-seed 无稳健优势）、**难度分层**（无难度结构化增益）、
**M4**（prefix-failure 靶向仅触及 ~7.8% 监督 token，剂量小 → 中性是算术必然而非机制失败）。

**新假设：问题在特权的形态。** solution 是一段 **off-policy 示范文本**；teacher 条件化其上会被拽向
示范的**文体分布**，而非纯粹的目标条件化。旁证：(1) 师生分歧大头在 structural/style token（competence
份额仅 34%）；(2) 官方 `jsd_token_clip` 的存在理由（压 teacher 高置信 copying/style 位的梯度）；
(3) 承诺点发现——特权答案的作用位点在**收尾决策**，非内容复读。

**answer-only 特权**：teacher 只见目标答案（`\boxed`），移除 off-policy 示范；置信须来自**朝已知答案的
自主推理** → 更贴合轴 1 grounded 定义。

**诚实反向风险（预注册）**：右下格诊断显示 teacher 修正力即便有完整 solution 也集中在答案位置；
answer-only 下修正信号可能**更薄**。**hard 层（base=0%）是试金石。**

## 阶段 1（侦察，1.7B，单 seed=42，4 run）

**2×2 矩阵**：{solution, answer-only} × {gated, ungated}。`max_completion_length=2048`（paper-OPSD 原生
长度，抬 wrong 桶质量占比 / 降截断率）。其余沿冻结：seed 42、global batch 30、150 步、clip 0.05、
fixed teacher。gated 格用**冻结的 v0 门控配置，一字不动**。

**teacher 特权实现（⚠️ 设计岔口，待确认后再跑）**：两格拟统一用**直接特权 teacher 模板**（非
reason-first 两阶段），仅改 Reference 段内容以**隔离"特权内容"这一唯一变量**。
- solution 格：Reference 段 = 完整 solution；answer-only 格：Reference 段 = 仅 `\boxed{Answer}` + 微调 transition。
- **岔口**：v0 原用 reason-first（两阶段）。若两格统一直接模板，则 solution×gated ≠ v0（属 2048 新
  regime，本就不与 1024 冻结 run 直接比）。**替代方案**：solution 格保持 reason-first、只 answer-only 用
  直接模板——但这会混淆"结构"与"内容"两个变量。**默认取直接-for-both（隔离干净）；若你要保 reason-first 请示下。**

**探针/泄露**：腐蚀探针腐蚀对象 = 特权答案本体（复用 `probes/corrupt_answers.py`）；V(t)/verifier 不变。
泄露检测器全程挂载、每 5 步 dump（answer-only 的 early-emission 风险重点盯）。

**评测**：AIME24/25 avg@12 + MATH500 avg@4 @ ckpt100/150，锁定协议。**比较仅限阶段内四格**
（2048 新 regime，不与 1024 冻结 run 直接比）。

### 预注册预言（结果前锁定）
- **P1（主）**：answer-only × gated 为四格最优。
- **P2（门控）**：两种特权下 gated ≥ ungated。
- **P3（风险）**：answer-only 在 hard 层（base=0%）增益**不劣于** solution；若显著更差 → 修正信号变薄
  成立，方向存疑。

### 预算
1024 训练 2 卡 ~54min/run；2048 生成约倍增（截断率降、avg 长度升）→ 估 **~75–95min/run**，4 run 串行
**~5–6.5h（≤10h）**→ 按预算自主权可直接提交训练（报 job id），eval 随后。

## 阶段 2 / 3（待发令）
- **阶段 2**：阶段 1 胜出配置 3-seed 确认。
- **阶段 3**：胜出配置迁移 4B（同模型 self-distill，仅换尺寸），主表候选。

## 纪律
冻结纪律在 gated 格内继续（门控超参一字不动）。M1–M3 probe（`31291`）与 ckpt50 eval 不受影响。

## 四格判读预期（据 M1–M4 结论更新；结果前锁定）

- **主效应看特权轴**（answer-only vs solution）。v1 的假设是特权**形态**问题，主效应应出现在这一轴。
- **门控轴预期为小效应**。M1（门控与 conflict 轴正交，AUC 0.49）、M2（微效 +2.2%）、M4（剂量 4.0%）
  三重确认 reweighting 不打 conflict 病灶 → 预期 gated 相对 ungated 的增量很小。
- **P1 若命中，归因必须分解为两部分报告，不得整体归给门控**：
  1. **特权主效应** = mean(answer 两格) − mean(solution 两格)；
  2. **门控增量** = (answer: gated − ungated)，并与 (solution: gated − ungated) 对照以检验交互项。
- P2（两种特权下 gated ≥ ungated）与 P3（answer-only 在 hard 层不劣于 solution）判读不变。

## v1 阶段1 结果（single seed=42，2048 regime；已完成，如实记录）

四格 eval（AIME24/25 avg@12 + MATH500 avg@4 @ ckpt100/150，完整性无 warn，数据 `results/v1_eval/`）：
- **P1（answer×gated 四格最优）不成立**：6 个 (bench,ckpt) 无一次最优落在 answer×gated；`ans_gated` 常垫底。
- **P2（gated ≥ ungated）不成立**：门控增量 12 项中 10 项为负（2048 下门控净负；AIME25 sol −3.9 / ans −3.1）。
- **特权主效应（answer−solution）不显著且方向混杂**：AIME24 微正、AIME25 明显负（−2.5/−2.5）、MATH500 ≈0。
- **M3（adapter-on 重跑）**：held-out ΔV OPSD +0.155 ≥ v0 +0.068；in-batch 亦 OPSD ≥ v0 → 不支持"v0 更新更对齐轴2"，与 M1/M2/M4 一致。
- 判读：v1"换 answer-only 特权更好"的核心假设**阶段1未获支持**；阶段2/3（3-seed、4B）暂缓。

---

# 【新主线】de-clip：用精准机制取代 clip 钝器

## 机制假设（先入档）
`jsd_token_clip` 是逐元素 `clamp(max=0.05)`，饱和区梯度为零 → **teacher-student 分歧最大处（教学信号最强处）梯度被掐死**，训练只吃浅层修正。这为五项阴性（T2/分层/M1/M2/M4；门控在无动态范围的 loss 上重分配故中性；教学信号集中高 KL 被顶、漂移弥散低 KL 畅通故漂移份额爬升）提供统一解释。同时 clip 是 2×2 表确认的**最强泄露抑制器**——拿掉它需替代性泄露控制。替代方案：**腐蚀探针驱动的精准干预**。

## 第 0 步（零训练）：Tier 1 性能评测
`qwen31b_paper_opsd_v1` 的 ckpt100/150（无 clip、TM-on、2048 regime）评 AIME24/25 avg@12 + MATH500 avg@4，锁定协议。
**对照口径**：2048/TM-on regime，不与 repo 1024 直接比；看相对 base 的增益量级 + "泄露涌现前(ckpt100) vs 后(ckpt150)"性能走势。目的：无 clip regime 性能上限初探。

## 第 1 步（4 训练 run，seed 42；基底=repo 冻结口径但 `jsd_token_clip=0`，其余全不动）
- **A) clipped 基线** = 已有主复现 run（`qwen31b_repro_3xh200_gb30`，复用不重跑）。
- **B) unclipped 裸跑**：clip=0，无替代机制。
- **C) unclipped + 腐蚀硬门**：clip=0；每 rollout 一次腐蚀 forward（T_S̃，无关腐蚀答案，复用 `probes/corrupt_answers.py`）；逐 token `c_t = JSD(T_S, T_S̃)`；**硬门 `w_t = 1[c_t ≤ ρ]`**（腐蚀敏感位直接扔掉，不参与蒸馏）。**ρ=0.0007**（诊断集 c_t 90 分位；出处 `probes/data/diag2x2_shard0.jsonl`，187 rollout/171,388 token；非零 37%、max 0.66；不调参）。
  - **修正说明**：原 `exp(-c_t/ρ)` 在 c_t 极偏分布（37% 非零、ρ 极小）下实为硬阈值，故改为显式硬门 `1[c_t ≤ ρ]`——与 D 的触发条件 `c_t > ρ` **完全对齐**，两臂差异单变量化（C=扔掉敏感位，D=敏感位学边缘化目标），不引入额外超参。
- **D) unclipped + 特权边缘化**：clip=0；腐蚀敏感位置（`c_t > ρ`）蒸馏目标替换为 `P̄_T = 0.5·[P_T(·|r)+P_T(·|r̃)]`（renormalize 后 forward KL），其余位置不变——泄露成分被**边缘化积掉**而非降权回避。
- **实现**：C/D 共享同一次腐蚀 forward（每步 +1 teacher forward，成本按 96.6ms 基准）；D 的目标混合在 logits 层做 log-mean-exp，**禁止物化全词表中间量**（流式纪律 §4）。走 v1 分支，`py_compile` + 20 步 smoke 后再投正式。
- **监测**：泄露检测器（keyword + early-emission，每 5 步 dump）+ 每 25 步 ckpt 的腐蚀质量（config 级）。
- **评测**：四臂 AIME24/25 + MATH500 @ ckpt100/150。

## 预注册（判读锁死）
- **P-a 性能释放**：B(unclipped) 相对 A(clipped) 增益为正——clip 梯度死区假设直接检验；**若 B≤A，假设死，战役止损**。
- **P-b 泄露代价**：B 的泄露指标（early-emission + 腐蚀质量）高于 A——repo 剩余抑制器（TM-off/1024/guard）是否兜底，两方向都有信息。
- **P-c 精准替代（主预言）**：C 和/或 D 满足「性能 ≥ B−0.5pt 且 泄露 ≤ A 水平」——精准机制拿到 clip 的安全性而不付性能税；**D 优于 C 则边缘化 > 降权**。
- **P-d 机制连带**：B/C/D 漂移份额曲线相对 A 下移（clip 是漂移主导化的机械原因；复用漂移扫描管线）。

## 排程
第0步 eval 立即挂；第1步 B/C/D 三训练（A 复用）+ 依赖 eval 过夜串行。预算自估：3 训练+eval ≤ 一夜半则直接提交，超出报预算等裁决。GPU 现 8/8 满，排队等卡。

## 第0步 Tier1 结果（如实入档；⚠️ regime caveat，不作 P-a 判据）

`paper_opsd_v1`（无 clip + TM-on + 2048 + 温和 guard，多旋钮混杂）相对 base：

| ckpt | AIME24 | AIME25 | MATH500 |
|---|---|---|---|
| 100 | 38.3 (**−10.8**) | 28.3 (**−6.7**) | 86.2 (**−4.6**) |
| 150 | 42.5 (**−6.7**) | 28.6 (**−6.4**) | 86.6 (**−4.3**) |

**三 benchmark 全部低于 base**，ckpt100→150 仅微升。**不作 P-a 判据**（多旋钮混杂，非单变量）——P-a 的干净检验是第1步 B(unclipped/1024/TM-off) vs A(clipped)，B 的 eval 已挂（job `32629/32630`）。

## 第1步 B 臂：泄露扫描（P-b 行为侧，零 GPU）

B(unclipped) 训练 rollout 泄露 vs A(repo, `leakage_over_training.md`)：

| 通道 | A (clipped) | B (unclipped) |
|---|---|---|
| keyword | 0.27% | 0.27%（持平） |
| early-emission clean | 1.28%, strong=0 | **1.52%, strong=5** |

- **B 的 early-emission 略升 + strong 从 0→5**：方向与"clip 是泄露抑制器"一致，但幅度温和——repo 剩余抑制器（TM-off/1024/guard）基本兜住行为泄露。
- 形态注记：B 的 early 反而 ≤100 步更高（1.65% vs >100步 1.26%），非"100 步后涌现"。
- **P-b 另一半（腐蚀质量/分布侧泄露）**需 teacher forward，随 C/D 批一起（共享腐蚀 forward）。

## P-a 判定：证伪 —— 机制假设撤回

A(clipped) vs B(unclipped)，单变量，均 1024/TM-off/solution/seed42：

| benchmark | A c100 | B c100(Δ) | A c150 | B c150(Δ) | base |
|---|---|---|---|---|---|
| AIME24 | 55.0 | 54.4(−0.6) | 54.2 | 48.1(−6.1) | 49.2 |
| AIME25 | 43.1 | 34.7(−8.3) | 42.8 | 33.3(−9.4) | 35.0 |
| MATH500 | 92.4 | 89.9(−2.5) | 91.3 | 89.8(−1.5) | 90.8 |

**B ≤ A 全部 6 格成立，且越训越差（AIME25 的 B 跌破 base）→ 按预注册 P-a 证伪。**

**撤回**：机制假设「`jsd_token_clip` 是梯度死区 → 拿掉会释放性能」**不成立，撤回**；据此提出的
「五阴性统一解释（clip 掐死高 KD 教学信号）」**随之撤回**。相反：本 regime 下 **clip 是把模型
托住的关键**——去 clip → teacher 高 KD 项（style/copying/泄露位）梯度全放开 → 训练被拽向
off-policy 文体 → 性能崩。与 B 泄露扫描（strong 0→5）、Tier1（无 clip paper_opsd 低于 base）互证。

**regime 条件性（关键限定）**：
- **TRD 报告 unclipped 胜**：4B/8B + synced teacher + 38k 长度；
- **我方 unclipped 崩**：1.7B + frozen teacher + 1024。
- → **clip 的净值是 regime 条件性的**，不是普适结论。B≤A 只在"我方小模型/frozen/短长度"成立。

**C/D 去向**：P-a 证伪 B 路线 ≠ C/D 必失败（C/D 是"去 clip 但控泄露/文体"的精准替代，正为救 B 的崩塌而设）。
但是否投 C/D 正式 run，改由**一次分布级腐蚀质量探针裁决**（见下节，预注册判读）。

## 裁决探针：分布级腐蚀质量（B vs A，config 级，预注册判读）

**测量**：`probes/corruption_quality.py`——对每个 checkpoint 在**其自身 rollout** 上算分布级腐蚀
质量 `mean token JSD(T_S, T_S̃)`（teacher=base 固定；`T_S̃` = 特权答案被无关腐蚀）。A=repro
（clipped），B=declipB（unclipped），step 100 与 145（无 step-150 dump，145 代理 ckpt150）。

**预注册判读规则（结果前锁死）**：
- **B ≥ 3× A** → 崩塌 copying 驱动 → C/D 靶点正确 → smoke 绿后投正式 run，P-c 存活；
- **B ≤ 1.5× A** → 崩塌由冲突/style 轴驱动（M1 已证与腐蚀门正交）→ C/D 够不到 → 取消，代码归档；
- **1.5× < 比值 < 3×** → 中间带，需人判。

**结果**（job `33402`，exit 0）：

| step | A (clipped) | B (unclipped) | B/A |
|---|---|---|---|
| 100 (≈ckpt100) | 0.00054 | 0.00154 | **2.84×** |
| 145 (≈ckpt150) | 0.00062 | 0.00163 | **2.63×** |

两点稳定落在 **~2.6–2.8×**（中间带），双 checkpoint 一致（非噪声抖动），**未达 3× 线**。

## 裁决：归档，不投 C/D 正式（用户 2026-07-14 裁定）

1. **严格预注册**：2.84 < 3×，未越 copying-driven 阈值，规则不支持自动投正式。
2. **战役层面**：P-a 证伪后 C/D 已无胜利条件——B 崩塌，C/D 最好结局 = 救回 A（clipped 基线）
   水平 = **零增益**；无正向上行空间，投正式不划算。
3. **裁决探针本身即产出**：B/A = 2.6–2.8× 双 checkpoint 稳定 → **clip 的第三重身份
   —— 分布级特权依赖抑制器 —— 确立**（见下"B 臂完整尸检"）。

**C/D 代码维持资产入库**（`declip_trainer.py` / `DeclipDataCollator` / `pbs/declip_smoke.pbs`，
smoke 全绿、实现正确性已验证）。**de-clip 战役正式关账。**

## B 臂完整尸检（clip 的三重身份，归档为资产；论文对应 §3 clip 小节，见 `paper_map.md`）

去 clip 后 B 的崩塌，三侧证据合成一具完整尸检：

| 侧 | 证据 | 出处 |
|---|---|---|
| **行为侧** | early-emission strong 0→5（clean 通道 1.28%→1.52%） | 上文"B 臂泄露扫描" |
| **性能侧** | 6 格 B≤A 全成立，AIME25 c150 跌破 base（−9.4pt vs A、−1.7 vs base） | 上文"P-a 判定" |
| **分布侧** | 腐蚀质量 B/A ≈ 2.7×（2.63–2.84；特权依赖显著高于 clipped） | 本节裁决探针 |

**clip 的三重身份**：(1) 梯度稳定器（原始 GOLD 设计意图，压 teacher 高置信位梯度）；
(2) 泄露抑制器（2×2 表已确认）；(3) **分布级特权依赖抑制器**（本尸检新确立——clip 系统性
压低 student 对特权答案的分布级依赖）。

**诚实标注：崩塌幅度与腐蚀量级不匹配。** 性能崩 ~9pt（AIME25）远大于 2.7× 的腐蚀信号所能
单独解释——**copying/特权依赖是显著伴随因素，但未必是主因**；**style/OOD 为主因候选**（去 clip
放开 teacher 高 KD 项的 style/结构位梯度，把 student 拽向 off-policy 文体 / 分布外）。主因归属
留待方向轴侦察（`docs/direction_axis_recon.md`）——**中间带的存在本身即"证据不足以断定 copying
为主因"的信号**。
