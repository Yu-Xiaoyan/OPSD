#!/bin/bash
# Tier 1 — paper-OPSD v1 FAITHFUL reproduction (leakage bridge MAIN experiment).
# Reproduces the config RLSD attacked, on our model/data. Relative to repo-OPSD
# (scripts/run_opsd_1b_3gpu.sh) it aligns FOUR knobs back to paper v1 (arXiv
# 2601.18734) while KEEPING the paper-original fixed teacher:
#   1. --student_thinking True         : paper Table 5 (thinking ON)
#   2. --max_completion_length 2048    : paper Table 6 (2k, NOT repo's 1024, NOT 4096)
#   3. --jsd_token_clip 0              : paper has no divergence clip
#   4. paper v1 Fig-2 MILD guard verbatim (via --transition_prompt_override):
#        "After understanding the reference solution, please try to solve this
#         problem using your own approach below:"
#      (NOT repo's hardened "do not copy or paraphrase" guard; NOT experiment-A's
#       neutral "Now, derive …" — that is a third text.)
#   --fixed_teacher KEPT: paper says "we fix the teacher policy to be the initial
#   policy … acts as regularization". LoRA retained (possible paper delta; noted
#   in docs/version_genealogy.md).
# Science question: does the RLSD-attacked (paper) version leak on our stack?
# Memory: batch 2 @2048 keeps the full-vocab JSD forward small (the 4096/batch-4
# bare run OOM'd at 136/140 GiB); expandable_segments set in the PBS wrapper.
accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 3 \
    --gradient_accumulation_steps 5 \
    --main_process_port 12955 \
    opsd_train.py \
    --model_name_or_path /home/xiaoyan.yu/models/Qwen3-1.7B \
    --student_thinking True \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 2 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 5 \
    --output_dir /home/xiaoyan.yu/opsd_outputs/ \
    --run_config qwen31b_paper_opsd_v1 \
    --num_train_epochs 30 \
    --max_steps 300 \
    --max_completion_length 2048 \
    --save_steps 50 \
    --logging_steps 2 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --max_length 20000 \
    --beta 0 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_tensor_parallel_size 1 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature 1.1 \
    --top_p 0.95 \
    --top_k 20 \
    --lmbda 1 \
    --fixed_teacher \
    --jsd_token_clip 0 \
    --transition_prompt_override 'After understanding the reference solution, please try to solve this problem using your own approach below:' \
    --wandb_project opsd-probes
