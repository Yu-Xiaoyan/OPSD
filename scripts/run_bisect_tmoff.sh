#!/bin/bash
# Suppressor bisection variant 4/4: STUDENT THINKING off (TM-off, repo value).
# = Tier 1 (paper-OPSD v1) with the ONE knob student_thinking True -> False.
# Everything else stays at Tier 1 (2048, fixed_teacher, clip 0, paper mild guard).
# READING CAVEAT (per instruction): TM-off is a constructive gag on the keyword
# channel — citation lives inside <think>, so keyword->0 is EXPECTED channel
# closure, NOT suppression of the pathology. Judge this variant by early-emission
# + last-checkpoint corruption sensitivity (config-level: variant vs Tier 1's
# 3.3-5.5 vs repo's 0.47), NOT by the keyword rate.
accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 3 \
    --gradient_accumulation_steps 5 \
    --main_process_port 12967 \
    opsd_train.py \
    --model_name_or_path /home/xiaoyan.yu/models/Qwen3-1.7B \
    --student_thinking False \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 2 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 5 \
    --output_dir /home/xiaoyan.yu/opsd_outputs/ \
    --run_config qwen31b_bisect_tmoff \
    --num_train_epochs 30 \
    --max_steps 200 \
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
