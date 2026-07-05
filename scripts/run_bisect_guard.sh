#!/bin/bash
# Suppressor bisection variant 2/4: GUARD restored to repo value (hardened).
# = Tier 1 (paper-OPSD v1) with the ONE knob guard flipped: the paper v1 mild
# transition is dropped (no --transition_prompt_override), so the collator uses
# repo-OPSD's hardened guard ("...do not copy or paraphrase it. Now, using your
# own words and independent reasoning..."). Everything else stays at Tier 1
# (TM-on, 2048, fixed_teacher, clip 0). Isolates the anti-copy guard's contribution.
accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 3 \
    --gradient_accumulation_steps 5 \
    --main_process_port 12963 \
    opsd_train.py \
    --model_name_or_path /home/xiaoyan.yu/models/Qwen3-1.7B \
    --student_thinking True \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 2 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 5 \
    --output_dir /home/xiaoyan.yu/opsd_outputs/ \
    --run_config qwen31b_bisect_guard \
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
    --wandb_project opsd-probes
