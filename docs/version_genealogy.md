# Version genealogy — 逐字 prompt / 配置记录

阶段性、逐字记录实验用的 teacher prompt 与关键配置改动，供论文方法节与复现引用。

## v1 阶段 1：teacher 特权 prompt（直接模板，2048）

**solution 格**（Reference = 完整 solution）：
```
Problem: {problem}

Here is a reference solution to this problem:
=== Reference Solution Begin ===
{solution}
=== Reference Solution End ===

After reading the reference solution above, make sure you truly understand the reasoning behind each step — do not copy or paraphrase it. Now, using your own words and independent reasoning, derive the same final answer to the problem above.
Please reason step by step, and put your final answer within \boxed{}.
```

**answer-only 格**（Reference = 仅 \boxed{Answer}，transition 微调）：
```
Problem: {problem}

Here is the correct final answer to this problem:
=== Reference Answer Begin ===
\boxed{Answer}
=== Reference Answer End ===

The correct final answer is given above. Do not simply restate it. Using your own independent step-by-step reasoning, derive this answer from the problem above.
Please reason step by step, and put your final answer within \boxed{}.
```

- 均经 `apply_chat_template(add_generation_prompt=True, enable_thinking=<teacher_thinking>)`。
- student prompt 不变（`Problem: {problem}\n\nPlease reason step by step, and put your final answer within \boxed{}.`）。
- gated 格门控配置 = 冻结 v0（τ=7.63 等，一字不动）。
- **待确认岔口**：solution 格是否改回 reason-first 两阶段（见 `docs/v1_design.md`）。

## paper-OPSD v1 配置在本环境的结果侧注脚（2026-07-10）

`qwen31b_paper_opsd_v1`（无 clip + TM-on + 2048 + 温和 guard）在本环境实测：
- **性能**：ckpt100/150 三 benchmark（AIME24/25、MATH500）**全部低于未训练 base**（−4-11pt）；
- **泄露**：非零、且随训练涌现（此配置的行为/分布泄露高于 repo 演化版）。

→ **两个维度都劣于 repo 演化版**（repo 主复现相对 base 为正、行为泄露近零）。这为
"**仓库为何从 paper-OPSD 静默演化到当前 repo 口径**（clip 0.05 + TM-off + 1024 + guard）"
提供了**结果侧注脚**：那些演化不是随意的，而是把一个"低于 base + 泄露"的配置调成了"高于 base + 泄露near-零"。

## clip 净值的 regime 条件性（2026-07-10，P-a 证伪后）

`jsd_token_clip` 的价值**取决于 regime，非普适**：
- **unclipped 胜**（TRD 报告）：4B/8B + synced teacher + 38k 长度；
- **unclipped 崩**（本环境实测 B vs A）：1.7B + frozen teacher + 1024，B 全格 ≤ A、AIME25 跌破 base。

→ 为"仓库为何保留 clip 0.05"再添一条结果侧注脚：在 repo 的小模型/frozen/短长度 regime 下，
clip 是净正（托住性能 + 抑泄露）；这不与 TRD"unclipped 更好"矛盾，两者只是不同 regime。
