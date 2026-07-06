#!/bin/bash
# v0 MAIN training run (OPSDGatedTrainer). Repro-identical to
# scripts/run_opsd_1b_3gpu.sh (global batch 30, 150 steps, save 25, TM-off, 1024,
# fixed_teacher, clip 0.05 — kept identical to the OPSD baseline for a clean v0
# vs OPSD comparison) with ONE change: --gated (frozen v0 gating: verifier-v2
# triage + real V(t) ΔV soft weighting on wrong, structural-token downweight on
# correct, V(end) split on truncated; no unlikelihood; corruption gate OFF).
# run_config qwen31b_v0_main, wandb project opsd-v0.
accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 3 \
    --gradient_accumulation_steps 2 \
    --main_process_port 12971 \
    opsd_train.py \
    --model_name_or_path /home/xiaoyan.yu/models/Qwen3-1.7B \
    --gated \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 5 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 2 \
    --output_dir /home/xiaoyan.yu/opsd_outputs/ \
    --run_config qwen31b_v0_main \
    --num_train_epochs 30 \
    --max_steps 150 \
    --max_completion_length 1024 \
    --save_steps 25 \
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
    --jsd_token_clip 0.05 \
    --wandb_project opsd-v0
