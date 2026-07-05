#!/bin/bash
# "Bare" leaky OPSD config (1.7B) — POSITIVE CONTROL for the leakage bridge
# experiment. Relative to scripts/run_opsd_1b_3gpu.sh (repo-OPSD), five knobs are
# flipped toward maximal leakage. See docs/version_genealogy.md.
#   1. --student_thinking True   : TM-on long CoT rollouts (paper-OPSD default)
#   2. (no --fixed_teacher)      : DYNAMIC teacher = current student + privileged
#                                  context (compute_loss -> nullcontext path).
#                                  NOTE: this goes BEYOND paper-OPSD, which fixes
#                                  the teacher to the initial policy; the bare run
#                                  is a max-leakage probe, not a paper replica.
#   3. --jsd_token_clip 0        : no per-token divergence clip (repo added 0.05)
#   4. --transition_prompt_override : neutral connective, no "do not copy" guard
#   5. 4096 tokens / 300 steps   : long generation, extended schedule
accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 3 \
    --gradient_accumulation_steps 2 \
    --main_process_port 12953 \
    opsd_train.py \
    --model_name_or_path /home/xiaoyan.yu/models/Qwen3-1.7B \
    --student_thinking True \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 2 \
    --output_dir /home/xiaoyan.yu/opsd_outputs/ \
    --run_config qwen31b_bare_leaky \
    --num_train_epochs 30 \
    --max_steps 300 \
    --max_completion_length 4096 \
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
    --transition_prompt_override '\n\nNow, derive the final answer to the problem above.' \
    --wandb_project opsd-probes
