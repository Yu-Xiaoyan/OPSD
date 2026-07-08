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
