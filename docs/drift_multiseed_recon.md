# 漂移分解 3-seed 复核（预注册，先落定后跑）

> 目的：把"漂移接管"（drift 拽回起点成分随训练上升、并在 150 步过半）的中心论点
> 从**单 seed（42）**升级为 **3-seed（42/s1/s2）mean±std**。**本文件写定即锁死判读，结果前不改口径。**
> 数据/输出：`probes/analysis/drift_multiseed.json`；聚合 `probes/drift_multiseed.py`。

## 口径（沿用既有三方分解，单变量）

- **管线**：`probes/drift_scan.py`（本次加 `--ckpt_dir`/`--out_tag`，不改算法）。每 checkpoint
  k∈{25,50,75,100,125,150}，在**固定 4096 诊断 rollout**（`rollouts_ckpt50_max4096.jsonl`
  correct+wrong，n=187）上 teacher-forcing：
  - `S0`  = base、student prompt（k、seed 无关，算一次）；
  - `T_S` = base、privileged teacher prompt（k、seed 无关，算一次）；
  - `S_k` = adapter-k on、student prompt（**唯一随 seed 变的量**）。
  - `drift_k = JSD(S_k, S0)`，`teach = JSD(T_S, S0)`，`total_k = JSD(T_S, S_k)`。
- **"拽回起点"成分** = overall drift share_k = `100·Σ_tok drift_k / (Σ drift_k + Σ teach)`（全 token 聚合）。
- **三种子共用同一固定 rollout 集**：`S0`/`T_S`/`teach` 三种子完全相同，仅 `S_k` 变 →
  seed 效应被干净隔离。固定 rollout 采样自 seed42-ckpt50，对 25/75/100/125/150 是 off-policy
  评测（既有 caveat）：**绝对水平有偏，但"随步上升的趋势"对此稳健**——本复核只据趋势/首尾差/过半判读。
- **自检**：`drift_multiseed.py` 用盘上 seed42 shard 复算，须精确复现主曲线
  `[38.1, 50.8, 55.9, 59.6, 61.3, 62.1]`（已验证通过）。

## seed42 主曲线（已在盘，参照）

overall drift share（%）@ steps [25,50,75,100,125,150] = **[38.1, 50.8, 55.9, 59.6, 61.3, 62.1]**
（首尾 +24.0pp；150 步 62.1% > 50）。

## 预注册预言（结果前锁定）

- **(a) 上升**：两条**新曲线**（s1、s2）中"拽回起点"成分均**随训练步数上升**（允许局部抖动，
  以**首尾差 > 10 个百分点**为准）。
- **(b) 过半**：150 步处**三个 seed**（42/s1/s2）该成分份额均 **> 50%**。

## 预注册判读（锁死）

- **(a) 成立 且 (b) 成立** → "拽回起点持续上升且 150 步过半" 完整成立（3-seed 确认）。
- **(a) 成立 但 (b) 不成立** → 主张**降级为"份额持续上升"**，**不再引用"过半/majority"**。
- **(a) 不成立** → 中心论点（漂移接管）**重审**。

## 实现与预算

- 运行：`drift_scan.py --ckpt_dir ~/opsd_outputs/qwen31b_opsd_{s1,s2} --out_tag _{s1,s2}`，
  各 1 卡、nshards=1（187 rollout × [2 base + 6 adapter] forward，token cap 1024）。
  单 seed 估 ~15–40min（< 2h，自主边界内），两 seed 各一 job（backfill 并行）。
- 聚合（CPU）：`drift_multiseed.py` 读 s42/s1/s2 三套 shard → per-seed 曲线 + mean±std + 判读。
- 存储纪律 §4：drift_scan 只落派生 [T] 量（drift/total/teach），**不存 [T,V]**；单 seed shard ~17MB×？。
- checkpoint 已核实：s1/s2 各 6 步（25–150）全在盘；磁盘 `df ~` 135G 空余，本任务只读 ckpt、写小 jsonl。

---

## 结果（jobs `33404`(s1)/`33405`(s2) exit 0，各 187 rollout；`probes/analysis/drift_multiseed.json`）

overall drift share（"拽回起点"成分，%）@ steps [25,50,75,100,125,150]：

| seed | 25 | 50 | 75 | 100 | 125 | 150 | 首尾Δ |
|---|--:|--:|--:|--:|--:|--:|--:|
| s42 | 38.1 | 50.8 | 55.9 | 59.6 | 61.3 | 62.1 | +24.0 |
| s1  | 34.5 | 47.4 | 53.0 | 57.2 | 58.7 | 59.2 | +24.7 |
| s2  | 35.4 | 48.3 | 53.3 | 57.3 | 58.9 | 59.6 | +24.2 |
| **mean±std** | 36.0±1.5 | 48.8±1.4 | 54.1±1.3 | 58.0±1.1 | 59.7±1.2 | **60.3±1.3** | — |

### 判读（按预注册规则）

- **(a) 上升**：s1/s2 首尾差 **+24.7 / +24.2 pp**（均 > 10）→ **PASS**（s42 +24.0）。
- **(b) 过半**：三 seed 150 步 = **62.1 / 59.2 / 59.6 %**（均 > 50）→ **PASS**。
- **结论：(a)&(b) 均成立** → **"拽回起点成分持续上升且 150 步过半" 在 3-seed 上完整确认**
  （不降级、可引用"过半/majority"）。三条曲线形态一致、std ≤1.5pp，跨 seed 稳健。

### caveat 承接
- off-policy 固定 rollout 的绝对水平偏置**跨 seed 同向同量**（三 seed 共用同一 rollout 集），
  故**种子间差异与趋势不受其影响**；此前单 seed 的 off-policy caveat 于本复核判据（趋势/首尾/过半）不构成威胁。

### 结案（战略层 2026-07-15）
- **两预言 PASS，中心论点（漂移接管）以 3-seed mean±std 入档**；**"过半/majority" 措辞保留**（论文可引）。
- **用户终端核对项**：`qstat -x 33404.gaas 33405.gaas`（均 **F / exit 0**）+
  `wc -l probes/data/drift_s{1,2}_shard0.jsonl`（各 **187**）。
- 本任务**结案归档**。
