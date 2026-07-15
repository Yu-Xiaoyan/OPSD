# paper_map —— 论文章节 ↔ 仓库载体映射

> 论文 .tex 草稿**在作者处（不在 repo）**。凡"论文章节§X"，仓库对应载体一律为 docs/ 分析文档。
> 外部材料**标注不臆造**：repo 无该 .tex 时，修订内容落 docs 载体，由作者同步回外部 .tex。

| 论文章节 / .tex | 仓库载体（docs/） | 数据/脚本 | 备注 |
|---|---|---|---|
| §3 localized failure / `sec_localized_failure_v2.tex` | `docs/framework.md` + `docs/sec_localized_failure.md`（本次修订载体） | `probes/data/tstar_annotations.jsonl`、`annotate_tstar.py`、`rebucket_audit.py`、`answer_likelihood.py` | **.tex 在作者处**；修订见 `sec_localized_failure.md` |
| §clip / clip 三重身份 | `docs/v1_design.md`（B 臂尸检） | `corruption_quality.py` | de-clip 关账 |
| v2 三路分诊 + 错支手术 | `docs/v2_design.md`、`docs/v2_launch_prereg.md` | `v2_trainer.py`、`triage_*` | 训练矩阵在跑 |
| 机制验证 M1–M4 | `docs/mechanism_validation.md` | `mechanism_probe.py` | M1 NULL |
| 多 seed / 饱和口径 | `docs/v0_results.md`（§T2） | `summarize_multiseed.py` | 基准口径统一 2026-07-15 |
| 方向轴 / Purified 正交 | `docs/direction_axis_recon.md` | `direction_axis.py` | Q-a..Q-d + R-a/b/c |
| 数字账本 | `docs/number_ledger.md` | 全 `probes/analysis/*.json` | Section 3 总核对 |
| 证伪台账 | `docs/falsification_ledger.md` | — | F1–F4 |
