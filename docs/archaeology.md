# OPSD Archaeology Report

A read-only walkthrough of the OPSD (On-Policy Self-Distillation) codebase,
built on `trl`'s experimental **GOLD** trainer. All line numbers refer to the
state of the `repro` branch at the time of writing.

**One-paragraph summary.** A single Qwen3 model plays both roles. The
**student** sees only the problem and rolls out a full solution on-policy
(via vLLM). The **teacher** is the *same network* but conditioned on extra
**privileged context** — the ground-truth reference solution pasted into the
prompt. Both are then teacher-forced over the student's rollout tokens, and the
student is trained to match the teacher's next-token distribution via a
(clipped) KL/JSD loss. In the shipped configs the teacher is the **frozen base
model** (`--fixed_teacher`, LoRA adapters disabled) and the student is base +
LoRA.

---

## 1. Dataset

**Conclusion.** Training data is `siyanzhao/Openthoughts_math_30k_opsd`, a
~30k-example math dataset with `problem` and `solution` columns. The `solution`
field (a full worked solution/reasoning trace, not just the final answer) is
used verbatim as the teacher's privileged context. The dataset name is
**hard-coded and not parameterizable**, and it is loaded by **HF hub name**, so
offline runs require it to be pre-populated in the HF cache.

**Code location.** `opsd_train.py:266`

```python
dataset = load_dataset("siyanzhao/Openthoughts_math_30k_opsd")
train_dataset = dataset["train"]
```

Required columns are declared in the trainer:

`opsd_trainer.py:368`
```python
def _set_signature_columns_if_needed(self):
    super()._set_signature_columns_if_needed()
    required_columns = ["problem", "solution"]
```

And consumed in the collator (`data_collator.py:63`):
```python
problem = feature["problem"]
solution = feature["solution"]
```

**Answer field / `\boxed{}`.** The student user-message *instruction* asks for
`\boxed{}` (`data_collator.py:67`), and evaluation grading extracts the last
`\boxed{...}` (`eval/evaluate_math.py:15` `extract_boxed_answer`). The dataset's
own `solution` text is injected wholesale as the "reference solution"; whether
each `solution` string terminates in `\boxed{}` cannot be verified offline (the
dataset is not in the local cache here), but the code never parses the answer
out of `solution` — it passes the entire string to the teacher, so the exact
format is not load-bearing for training.

**Offline loadability / parameterization.**
- Not parameterizable: the hub name is a string literal at `opsd_train.py:266`
  (same literal in `sft_train.py:131` and `grpo_train.py:274`). To point at a
  local copy you must edit the source or set `HF_DATASETS_CACHE` and rely on
  `HF_HUB_OFFLINE=1` resolving the cached dataset.
- The eval datasets are likewise loaded by hub name
  (`eval/evaluate_math.py:222-242`), and **aime25 / hmmt25 use
  `trust_remote_code=True`** (`:238`, `:241`) — their loader scripts must also
  be cached. This is a real offline hazard (see §"Reproduction risks").

---

## 2. Teacher context construction

**Conclusion.** Student and teacher prompts are built by the collator using the
**Qwen3 chat template**. The teacher's privileged segment is the **full
reference solution** wrapped in explicit delimiters plus a "now re-derive it
yourself" transition instruction. Thinking mode is controlled per-role via the
Qwen3 `enable_thinking` flag on `apply_chat_template`.

**Code location — student prompt.** `data_collator.py:66-74`
```python
student_user_message = f"Problem: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
student_messages = [{"role": "user", "content": student_user_message}]
student_prompt = self.tokenizer.apply_chat_template(
    student_messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=self.student_thinking,   # default False
)
```

**Code location — teacher prompt (normal mode).** `data_collator.py:96-109`
```python
teacher_user_message = (
    f"Problem: {problem}\n\n"
    f"Here is a reference solution to this problem:\n"
    f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
    f"{self.transition_prompt}\n"
    f"Please reason step by step, and put your final answer within \\boxed{{}}."
)
teacher_prompt = self.tokenizer.apply_chat_template(
    teacher_messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=self.teacher_thinking,   # default True
)
```
`transition_prompt` (`data_collator.py:37-43`) tells the teacher *not* to copy
the reference but to independently re-derive the same answer. So the privileged
info = **the complete solution text**, not merely the answer.

**Thinking mode.** Defaults (`opsd_train.py:95-108`): `student_thinking=False`,
`teacher_thinking=True`. The non-thinking ablation sets both False
(`scripts/run_opsd_8b_nonthink.sh:36-37`). These flags flow
`opsd_train.py → OPSDTrainer → SelfDistillationDataCollator`
(`data_collator.py:26-27`, `71-73`, `107-109`).

**`reason_first` variant (off by default).** When `--reason_first` is set, the
teacher first *generates* an analysis of the solution, which is then spliced in
front of the transition prompt at runtime (`data_collator.py:76-94`,
reassembled in `opsd_trainer.py:1345-1362`). Not used in the shipped configs.

---

## 3. Alignment mechanism (student vs teacher logits over rollout tokens)

**Conclusion.** The teacher prompt is longer than the student prompt, so the two
sequences are offset. Alignment is done by slicing each side's logits with a
**per-side prompt-length offset** so that both sides' logit rows line up with the
**shared rollout tokens**. The offset is a single batch-level scalar per side
(`student_prompt_length`, `teacher_prompt_length`). The empty `<think></think>`
block that Qwen3 injects when `enable_thinking=False` lives **inside the student
prompt**, so it is consumed by the student prompt-length offset and never enters
the aligned rollout region.

**Code location — the offset slices.** `opsd_trainer.py:633-645, 689`
```python
student_prompt_len = inputs["student_prompt_length"]
teacher_prompt_len = inputs["teacher_prompt_length"]
sampled_token_ids = inputs["student_input_ids"][:, student_prompt_len:]
...
student_logits = outputs_student.logits[:, student_prompt_len - 1 : -1, :]
...
teacher_logits = outputs_teacher.logits[:, teacher_prompt_len - 1 : -1, :]
```
`[:, P-1 : -1, :]` are exactly the logits that *predict* the tokens at positions
`P … end-1`, i.e. the generated (rollout) tokens. Doing this with `P_s` for the
student and `P_t` for the teacher makes both `[B, G, V]` tensors correspond
row-for-row to the same `G` rollout tokens. `sampled_token_ids` (the rollout
ids) come from the student side and index both distributions.

**How the shared sequences are assembled.** In `training_step`, one rollout is
generated from the student prompt and then appended to *both* prompts:

`opsd_trainer.py:1391-1407`
```python
generation_ids = generated_ids[:, student_prompt_len:]
inputs["student_input_ids"] = generated_ids                      # [student_prompt | gen]
...
teacher_full_ids = torch.cat([teacher_prompts, generation_ids], dim=1)  # [teacher_prompt | gen]
inputs["teacher_input_ids"] = teacher_full_ids
```

**The prompt-length scalars** come from the collator, which pads all student
prompts to the batch max and all teacher prompts to the (larger) batch max, and
records those maxima (`data_collator.py:122, 136, 187, 201`). Per-example true
lengths are also kept (`:138`, `:202`) and used only for label masking
(`opsd_trainer.py:1411-1414`).

**Empty `<think></think>` handling (TM-off).** With `student_thinking=False`,
Qwen3's template appends an empty thinking block to the *prompt* (after the
generation prompt marker). Because it is part of the templated student prompt
string (`data_collator.py:71-73`), it is included in `student_prompt_length` and
is therefore *before* the `student_prompt_len` offset — it is skipped by the
logit slice and masked out of the labels. The teacher (thinking=True) has no
such block; the per-side offset absorbs the length difference. The two sides
only ever share the appended rollout tokens.

> Caveat worth flagging: the collator sets `padding_side="right"`
> (`data_collator.py:47`) and the vLLM path re-tokenizes prompts with
> `padding="longest"` before concatenating the completion
> (`opsd_trainer.py:984-1026`). For examples shorter than the batch-max prompt,
> right-padding places pad tokens *between* the real prompt and the rollout, and
> the offset uses the batch-max scalar. This is exactly the fragile area the
> README's "chat template / zero2 bug" fix (trl#5241) touches; it is correct
> only as long as tokenization is consistent across the collator and the vLLM
> re-tokenization. See risks §.

---

## 4. Student rollout generation (vLLM colocate)

**Conclusion.** In the shipped configs rollouts are generated by **vLLM in
`colocate` mode**, `n=1` per prompt, from the **student prompt only**. TM-off is
achieved purely by the chat template (`enable_thinking=False` → the student
emits an answer directly rather than a `<think>` trace). Rollouts **are already
dumped to disk** as prompt/completion text every 5 steps, plus logged to WandB —
but only *text*, not token ids or per-token teacher/student stats.

**Code location — generation dispatch.** `opsd_trainer.py:1364-1385`
```python
if self.use_vllm:
    self._wake_vllm_if_needed()
    result = self._generate_on_policy_outputs_vllm(
        inputs, self.generation_config, self.processing_class.pad_token_id)
    generated_ids, generated_attention_mask, _, prompt_texts, completion_texts = result
else:
    with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
        result = self.generate_on_policy_outputs(...)
```

**vLLM colocate core.** `opsd_trainer.py:855-1041` (`_generate_on_policy_outputs_vllm`):
decodes `inputs["student_prompts"]` to text, strips pad tokens
(`:861-869`), builds `SamplingParams(n=1, temperature, top_p, top_k, min_p,
max_tokens, presence_penalty)` (`:925-935`), calls
`self.vllm_engine.generate(...)` (`:949`), then re-tokenizes prompts and
concatenates padded completions (`:984-1026`). Engine is created in `__init__`
(`opsd_trainer.py:339-351`) with `distributed_executor_backend="external_launcher"`.
Weights are synced student→vLLM after optimizer steps via
`GOLDVLLMSyncCallback` (`:96-116`) → `_move_model_to_vllm` (`:1175`), which
merges/unmerges the LoRA adapter under ZeRO-3/FSDP gather.

**TM-off implementation.** No special generation flag — it is entirely the
template: `student_thinking=False` at `data_collator.py:71-73` produces the
non-thinking student prompt, so the rollout has no self-generated `<think>`.

**Rollout dumping (already present).** Buffered per step
(`opsd_trainer.py:1425-1429`), flushed every
`_generation_save_frequency = 5` steps (`:239`) by `_save_generation_outputs`
(`:1252-1285`) to `output_dir/generations/generations_step_{step}.json`, and
sampled into a WandB table in `log()` (`:1524-1537`).

**Where to add a richer dump hook (pointer only, no change made).** The existing
JSON dump stores only `{step, prompt, completion}` text. If you need the aligned
**token ids** or **per-token teacher/student log-probs / KL**, the cleanest
single place is inside `compute_loss` (`opsd_trainer.py:626-753`), right after
the teacher/student logits are sliced and aligned (`:645`, `:689`) and where
`generalized_jsd_loss` already has both distributions and the label mask — that
is the only point where the aligned per-token quantities exist together. A
secondary option is to extend the buffer append at `:1425-1429` (but only text
is available there).

---

## 5. Loss semantics

**Conclusion.**
- `--beta 0` → **forward KL** `KL(teacher ‖ student)` (mass-covering; the
  script naming `forwardbeta0` matches). `--beta 1` → reverse KL
  `KL(student ‖ teacher)`. `0<beta<1` → generalized JSD interpolation via the
  log-mixture.
- `--lmbda 1` is **inert** in this trainer — it is stored but never read; the
  overridden `training_step` is hard-wired to on-policy generation. (In stock
  GKD/GOLD, `lmbda` is the on-policy student-data fraction; here it only
  documents "100% on-policy".)
- `--jsd_token_clip 0.05` clamps the **per-element** (per token × per vocab)
  divergence contribution to a maximum, *before* the vocab sum and token
  masking — a stabilizer against high-KL style tokens.
- Gradients flow **only through the student logits**; the teacher forward runs
  under `torch.no_grad()`.

**Code location — divergence branches.** `opsd_trainer.py:441-464`
```python
if beta == 0:
    jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
elif beta == 1:
    jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
else:
    ... mixture_log_probs = logsumexp(student+log(1-beta), teacher+log(beta)) ...
    jsd = beta * kl_teacher + (1 - beta) * kl_student
if token_clip is not None:
    jsd = jsd.clamp(max=token_clip)
```
With `log_target=True`, `F.kl_div(A_logp, B_logp)` computes
`Σ exp(B)·(B − A)`. So `beta==0` → `Σ p_teacher·(log p_teacher − log p_student)
= KL(teacher ‖ student)` (**forward** in distillation convention);
`beta==1` → `KL(student ‖ teacher)` (**reverse**). Default `beta=0.5` gives
symmetric JSD.

**What the clip acts on.** `F.kl_div(..., reduction="none")` returns the full
`[B, S, V]` per-element tensor, so `jsd.clamp(max=token_clip)`
(`opsd_trainer.py:463-464`) caps each **(token, vocab)** contribution — not the
already-summed token KL. Masking then selects valid token rows and the reduction
sums over vocab and averages over tokens (`:467-473`):
```python
if labels is not None:
    mask = labels != -100
    jsd = jsd[mask]                     # [N_valid_tokens, V]
...
return jsd.sum() / mask.sum()          # mean over valid tokens of (Σ_vocab clamped)
```
This element-wise granularity explains why the tuned clip values are so
small and model-specific (`0.05`/`0.06` thinking; `1e-6`/`1e-7` non-thinking —
`scripts/*`): they cap individual vocab contributions, not the aggregate.
`--top_k_loss` (default 0/off) would instead restrict + renormalize to the
teacher's top-k tokens before the divergence (`:430-435`).

**Gradient path.** Teacher forward is wrapped in `torch.no_grad()`
(`opsd_trainer.py:683-687`), so `teacher_log_probs` is a constant target; only
`student_log_probs` (the `input` arg to `F.kl_div`) carries gradient. The
alternative `--use_tinker_loss` path is an explicit policy-gradient form with a
**detached** advantage (`:714`, `:726`), again keeping gradient only on the
student. The shipped OPSD configs use the JSD path (tinker loss off).

---

## 6. `--fixed_teacher`

**Conclusion.** Yes — with `fixed_teacher=True` the teacher forward is the
**base model with LoRA adapters disabled** (= the step-0 initial policy). It
requires `use_peft=True`. It is switched on per teacher forward pass via PEFT's
`disable_adapter()` context manager.

**Code location — the switch.** `opsd_trainer.py:676-687`
```python
if self.use_ema_teacher:
    adapter_context = self._ema_teacher_context(model)
elif self.fixed_teacher and is_peft_model(model):
    adapter_context = self.accelerator.unwrap_model(model).disable_adapter()
else:
    adapter_context = nullcontext()

with torch.no_grad(), adapter_context:
    outputs_teacher = model(input_ids=inputs["teacher_input_ids"], ...)
```
So inside the teacher pass the adapters are turned off → the network computes
with pure base weights, reading the privileged (solution-augmented) context;
outside, the student pass uses base + LoRA on the problem-only context. The
requirement is enforced at `opsd_train.py:165-168` and again at
`opsd_trainer.py:196-200`. The `reason_first` teacher generation path applies
the same `disable_adapter()` (`:773-777`). Three teacher strategies are mutually
exclusive: fixed (base), EMA (`_ema_teacher_context`, `:556-624`), or dynamic
(current student, `nullcontext`). All shipped OPSD scripts use `--fixed_teacher`.

---

## 7. `--num_train_epochs 30` → optimizer steps, and `--max_steps`

**Conclusion.** 30 epochs is a **nominal upper bound that is never reached in
practice** — the README states OPSD "peaks within 100 steps," checkpoints land
every 25 steps, and eval only looks at steps 25/50/75/100(/150). With ~30k
examples and an effective batch of 32, one epoch is already ~940 optimizer
steps, so runs are effectively stopped in the first epoch. `--max_steps` **is
supported** (it is a stock HF `TrainingArguments`/`SFTConfig`/`GOLDConfig`
field and, when > 0, overrides `num_train_epochs`) and would be the cleaner way
to cap a run.

**Effective batch size** is computed at `opsd_train.py:124-127`:
```python
effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps * num_processes
```
For every shipped config this is **32**:

| script | procs | per-device bs | grad-accum | eff. batch | steps/epoch (~30k) |
|---|---|---|---|---|---|
| `run_opsd_1b.sh` | 4 | 4 | 2 | 32 | ~940 |
| `run_opsd_4b.sh` | 8 | 4 | 1 | 32 | ~940 |
| `run_opsd_4b_nonthink.sh` | 4 | 4 | 2 | 32 | ~940 |
| `run_opsd_8b.sh` | 8 | 2 | 2 | 32 | ~940 |
| `run_opsd_8b_nonthink.sh` | 8 | 2 | 2 | 32 | ~940 |

So `30 × ~940 ≈ 28k` optimizer steps *nominally*, but the meaningful training
budget is **~100 steps**. Note none of the scripts pass `--max_steps`; they rely
on `--save_steps 25` + early-checkpoint evaluation instead.

---

## 8. Per-token statistics logging

**Conclusion.** **No.** There is no built-in per-token KL / entropy logging. The
only per-token *operation* is the `jsd_token_clip` clamp
(`opsd_trainer.py:463-464`); its intermediate per-token values are never
reduced-and-logged. The trainer's custom logging is limited to (a) aggregate
`on_policy_loss` / `off_policy_loss` scalars and (b) prompt/completion text
tables.

**Evidence.**
- `generalized_jsd_loss` returns a **scalar** only (`opsd_trainer.py:472-479`);
  no histogram or per-token tensor escapes.
- The `_metrics` dict is initialized (`:274`) and averaged/cleared in `log()`
  (`:1465-1466`, `:1516`) but **nothing ever appends to it** — so it always
  logs empty. The only populated custom metrics are `on_policy_loss` /
  `off_policy_loss` (`:1491-1503`), derived from the scalar loss.
- Text logging: `_textual_logs` / WandB completions table (`:283-288`,
  `:1524-1537`) and JSON generation dumps (`:1252-1285`).

The README's observation that style tokens ("wait", "think") show 6–15× higher
KL than math tokens is the *motivation* for `jsd_token_clip`, but the per-token
KL measurement that produced it is **not** wired into this trainer. If you want
that logging (a feature raised in community discussion), the natural home is
inside `generalized_jsd_loss` / `compute_loss` where the pre-reduction per-token
divergence and the teacher/student distributions are still in scope
(`opsd_trainer.py:441-473`, alongside the `:274`/`_metrics` plumbing that is
already half-built for it).

---

## Reproduction risk checklist (3×H200, offline)

The most likely failure points when reproducing official numbers on this setup:

1. **Offline dataset resolution (train + eval).** Both the training set
   (`opsd_train.py:266`) and all eval sets (`eval/evaluate_math.py:222-242`) are
   loaded by HF hub name, and **aime25/hmmt25 need `trust_remote_code=True`**
   loader scripts. With `HF_HUB_OFFLINE=1` and no network on the compute nodes,
   any of these that isn't pre-cached will hard-fail. Mitigation: pre-download
   `siyanzhao/Openthoughts_math_30k_opsd` and every eval dataset (with their
   remote-code scripts) into the HF cache from a login node before submitting.

2. **GPU-count mismatch (4/8 → 3).** Every script hard-codes `--num_processes`
   4 or 8 and is tuned for an **effective batch of 32**
   (`opsd_train.py:124-127`). On 3×H200 you must re-derive per-device batch ×
   grad-accum × 3 to hit 32, or the "~100 step" learning curve shifts. Also
   `vllm_tensor_parallel_size` must divide world size
   (`opsd_trainer.py:313-317`) — keep it at 1 for 3 GPUs; TP>1 is impossible
   with 3.

3. **Logit-alignment / chat-template fragility.** Alignment relies on batch-max
   prompt-length scalars plus `padding_side="right"` in the collator
   (`data_collator.py:47`) and a *separate* vLLM re-tokenization
   (`opsd_trainer.py:984-1026`). This is precisely the area of the README's
   trl#5241 "chat template / zero2" fix. Any drift in the Qwen3 chat template
   (empty-`<think>` handling), tokenizer, or `transformers`/vLLM version changes
   prompt lengths and **silently** misaligns teacher vs student → the KL signal
   becomes garbage with no error. Mitigation: pin `environment.yml` versions
   exactly; sanity-check a few aligned samples.

4. **`jsd_token_clip` is model- and granularity-specific.** The clip acts
   per (token,vocab) element before the vocab sum (`opsd_trainer.py:463-473`),
   and the tuned value differs sharply by config (`0.05`/`0.06` thinking vs
   `1e-6`/`1e-7` non-thinking). Carrying the wrong value across a model/mode
   silently over- or under-regularizes and won't reproduce the curve.

5. **Env/toolchain + colocate weight-sync on Hopper.** Reproduction depends on a
   finicky stack: torch 2.8 / vLLM 0.11 / flash-attn 2.8.3 / deepspeed 0.18.2 /
   CUDA build all matching on H200 (sm_90), plus the LoRA merge/unmerge vLLM
   sync under ZeRO
   (`opsd_trainer.py:1175-1245`). `pbs/smoke_test.pbs` exists to catch the
   flash-attn/GPU-visibility part; extend it to also exercise a 1–2 step OPSD
   train + vLLM sync before trusting a full run.

6. **(Bonus) Small-sample eval variance.** AIME24/25 are ~30 problems scored
   avg@12 at `temperature=1.0` (`eval/run_eval.sh`), and the teacher quality is
   just the frozen base model, so eval noise of a couple points is expected —
   don't treat a small gap vs the paper as a reproduction failure without
   multiple seeds.
