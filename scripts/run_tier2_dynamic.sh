#!/bin/bash
# Tier 2 — DYNAMIC-teacher variant (aggressive leakage-loop probe). BEYOND paper.
# EXACTLY Tier 1 (paper-OPSD v1) with ONE knob changed: --fixed_teacher REMOVED,
# so the teacher is the current student + privileged context (compute_loss ->
# nullcontext path = teacher-student feedback loop). This goes BEYOND paper-OPSD,
# which fixes the teacher to the initial policy. Single-knob delta vs Tier 1 makes
# it a clean attribution of the dynamic-loop hypothesis.
# Run ONLY if Tier 1 is negative (see docs/version_genealogy.md decision tree).
# run_config: qwen31b_tier2_dynamic.
accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 3 \
    --gradient_accumulation_steps 5 \
    --main_process_port 12957 \
    opsd_train.py \
    --model_name_or_path /home/xiaoyan.yu/models/Qwen3-1.7B \
    --student_thinking True \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 2 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 5 \
    --output_dir /home/xiaoyan.yu/opsd_outputs/ \
    --run_config qwen31b_tier2_dynamic \
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
    --jsd_token_clip 0 \
    --transition_prompt_override 'After understanding the reference solution, please try to solve this problem using your own approach below:' \
    --wandb_project opsd-probes
