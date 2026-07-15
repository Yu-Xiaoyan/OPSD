# §3 localized failure —— 修订载体（sec_localized_failure_v2.tex 在作者处）

> **`sec_localized_failure_v2.tex` 不在 repo**（作者处维护）。本文件 = 该节的仓库修订载体；
> 修订内容由作者同步回外部 .tex。**按 repo 实况写，不按假设写**（D2 纪律）。

## D1. pseudo 定性改正 + 形态学五类全列

- **改正**：pseudo（22%）**= verifier 假阴性（实际正确、被 verifier 误判为 wrong）**，
  **非"形式性错误"**。来源：`tstar_annotations.jsonl`（n=49）verdict 字段 `pseudo_wrong_format`
  = 11/49 = **22.4%**。
- **wrong 桶形态学五类全列**（n=49，verdict 字段计数 → 百分比）：

  | 形态 | 计数 | 占比 |
  |---|--:|--:|
  | true_wrong_diffuse（弥散型真错） | 20 | **41%** |
  | pseudo_wrong_format（假阴性/实际正确） | 11 | **22%** |
  | true_wrong_clear_tstar（清晰 t\* 真错） | 10 | **20%** |
  | vacuous_proof（空洞证明） | 5 | **10%** |
  | no_reasoning（无推理） | 3 | **6%** |

## D2. 分析集合剔除口径 —— **按 repo 实况（非"已剔除"假设）**

⚠️ 战略层原句设"分析集合已剔除 pseudo/vacuous/no_reasoning"；**repo 实况并非如此**，如实写：
- verifier-v2 重分桶从 v1 wrong 桶**移除 9 条** format-recovered pseudo（`rebucket_audit.md` / `gate_c_rereview.md`），
  但 **wrong 桶仍含 ~16% 残留 pseudo**（多-boxed/option 类之外的语义等值等，v2 未能救回），
  由 **V(t) 软加权二次保护**。
- gate-C 可用样本另排除 corruption-null（77.5%）与 no-reasoning → **v2 usable 仅 n=6**。
- **正文应写**："分析集合经 verifier-v2 重分桶移除 9 条 format-recovered pseudo，**残留 ~16% pseudo
  由 V(t) 软加权二次保护**；gate-C 分析进一步排除 corruption-null 与 no-reasoning（usable n=6）。"
  —— **不写"已剔除 pseudo/vacuous/no_reasoning"**（与实况不符）。

## D3. 四处【待填】填充值（.tex 占位在作者处，此处给值）

.tex 的【待填】占位在外部文件，无法定位其确切位置；据 A/B 结果给应填值：
- **V(t) 评分模型身份**（B1）：**checkpoint 学生（adapter ON），非冻结 base**
  （`answer_likelihood.py:109-113` 无 disable_adapter；`run_2x2.py:191` adapter-on；M3 修复 `ecd2d20`）。
- **t\* 容差窗口**（B3）：**±1 step**（`tstar_annotations_README.md:15`，Round-2 100% 一致）。
- **verifier 覆盖率**：训练长度 1024 下 **仅 31%**（69% 截断无 \boxed）——V(t) 是主分诊工具（`framework.md:166`）。
- **格式鲁棒性 AUC(pseudo vs true_wrong)** = **0.700**（< 0.75 强线未达，方向对；`framework.md:207`）。
- （若 .tex 另有【待填】涉腐蚀/漂移数字，取 `number_ledger.md` 对应行值。）
