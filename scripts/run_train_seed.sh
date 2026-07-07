#!/bin/bash
# multi-seed 训练(2b) 独立启动脚本。冻结口径同 scripts/run_v0_main.sh,仅参数化:
#   SEED  (必填) -> --seed，run_config 带 _s{SEED} 后缀(输出目录隔离)
#   GATED yes|no -> yes:--gated/qwen31b_v0_s{SEED}/opsd-v0; no:qwen31b_opsd_s{SEED}/opsd-repro
# 注: pbs/train_seed.pbs 为自包含内联版(不依赖本脚本);本脚本供直接运行/参考。
set -euo pipefail
: "${SEED:?need SEED}"; GATED="${GATED:-yes}"
if [ "$GATED" = "yes" ]; then GATE_ARG="--gated"; RC="qwen31b_v0_s${SEED}"; WB="opsd-v0"
else GATE_ARG=""; RC="qwen31b_opsd_s${SEED}"; WB="opsd-repro"; fi
PORT=$((12980 + SEED))
accelerate launch \
    --config_file accelerate.yaml --num_processes 3 --gradient_accumulation_steps 2 \
    --main_process_port $PORT opsd_train.py \
    --model_name_or_path /home/xiaoyan.yu/models/Qwen3-1.7B $GATE_ARG --seed $SEED \
    --learning_rate 5e-6 --max_grad_norm 0.1 --per_device_train_batch_size 5 \
    --gradient_checkpointing --output_dir /home/xiaoyan.yu/opsd_outputs/ --run_config $RC \
    --num_train_epochs 30 --max_steps 150 --max_completion_length 1024 --save_steps 25 \
    --logging_steps 2 --attn_implementation flash_attention_2 --torch_dtype bfloat16 \
    --max_length 20000 --beta 0 --use_vllm --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.6 --vllm_tensor_parallel_size 1 \
    --use_peft --lora_r 64 --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature 1.1 --top_p 0.95 --top_k 20 --lmbda 1 --fixed_teacher \
    --jsd_token_clip 0.05 --wandb_project $WB
