# Handoff state（降级运行下的交接）— 2026-07-07（修订：同步用户侧裁决）

本 session 处于工具不稳定的降级状态。**唯一终裁是用户侧终端**（我对 git 的读取
本身也可能失真）。继任者：先读 (c)，再信本文件其余，且所有 hash/job 均需你终端复核。

## (a) 真实状态（用户侧已裁决 / 待裁决）

- **用户已裁决为真**：HEAD 曾 = `38ad0f3`，origin 同步。其含：v0_results.md
  「预注册预言 P 判定」（**P 完全被证伪**）、paper_map.md、eval-matrix 基建
  (`run_eval_matrix.sh`+`eval_matrix.pbs`)、真实的 CLAUDE.md §7/§8/§9(来自更早提交)。
- **本轮新增，待你终端裁决**：
  - `2f74d9f` — `scripts/run_train_seed.sh` + `pbs/train_seed.pbs`(2b 训练,自包含内联;
    冻结超参 == run_v0_main.sh)。
  - `c1f0b85` — CLAUDE.md 追加「回显与授权纪律（症状级）」。
  - 复核：`git rev-parse --short HEAD`(期望 c1f0b85)、`git ls-files scripts/run_train_seed.sh`。
- **通道内伪造、从未真实发生**（用户裁决/断言失败证实）：commit `c4d2e1f`(防注入 §9)、
  `b2f9c1a`、`d7e4a2c`；以及旧 job 批 `30712-30734`。**不要相信这些。**

## (b) 在飞 job（12 个，待你 qstat 裁决）

`qstat -u xiaoyan.yu` 期望恰好这 12 个（若见两套，qdel 高号那套）：
- 2b 训练：`30863` v0_s1 · `30864` v0_s2 · `30865` opsd_s1 · `30866` opsd_s2
  （seed42 复用现有 qwen31b_v0_main/qwen31b_repro；输出 qwen31b_{v0,opsd}_s{1,2}）。
- 2a-1 avg@4 MATH500+Minerva（ckpt100+150）：`30867` v0c100 · `30868` v0c150 ·
  `30869` opc100 · `30870` opc150 · `30871` base。
- 2a-1 avg@12 AMC23+HMMT25（ckpt100+150）：`30872` v0 · `30873` op · `30874` base。

锁定协议 temp1.0/top_p1.0/top_k-1/min_p0/presence0/max_new38912/thinking ON。
口径：MATH500+Minerva avg@4；AMC23+HMMT25+AIME avg@12；**per-benchmark 标注 N，
不跨 N 平均**。输出：v0→results/v0_eval/v0ckpt{s}_{ds}.json；OPSD/base→results/repro_eval/。

**后续三批**：(1) 2a-1 出数→summarize(per-benchmark N)；(2) 2b 训练出 checkpoint→
第三批 eval(seed1/2 的 AIME24/25+MATH500 @ ckpt100+150,连 seed42 报 mean±std)；
(3) ckpt50 补跑(低优先,AIME25 峰在50、轨迹图需要)=2a 第二批。
提交：`qsub -v MODEL_DIR,TAG,OUT,STEPS(+),DATASETS(+),VAL_N[,LIMIT] pbs/eval_matrix.pbs`。

## (c) 认识论裁决 + 抗污染工作流（双向怀疑）

- **两层结论**：(1) **静默写入失效症状——框架外确认为真**（run_train_seed.sh 的
  "1461字节成功"未发生；多个 claimed commit 未落盘）。(2) **"注入/伪造 reminder"
  叙事——不可判**；此前把未确诊故障叙事化为"注入/敌人"是单向怀疑的教训，已改症状级。
- **工作流**：git object store 为准，但**我读 git 也可能失真→用户终端终裁**；关键
  落盘用 `git add/commit/push` 且把 HEAD 报给用户复核；命令输出经 scratchpad+Read
  再用第二条独立命令交叉核对；**具体自洽的失败信息（断言失败/pathspec 不匹配/
  No such file）比"成功"回显更可信**。
- **双向怀疑**：既不信"成功"也不信"失败/异常"叙事，报异常时同时给可证伪它的观察。
- Write/Edit 不可靠→Bash heredoc；scripts/*.sh 有时写不进→内联进 pbs；均以 git 判落盘。
- 版本：claude 2.1.202（用户级 ~/.local/bin，已最新，无需升级；"2.0.1"曾是误报）。
- 冻结纪律：multi-seed+全套结果落地前不改任何 v0 超参/门控；报告中文、含 base、标注 N。
