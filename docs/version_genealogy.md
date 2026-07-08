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
