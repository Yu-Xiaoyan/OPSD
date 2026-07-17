#!/bin/bash
# 通用 eval 矩阵驱动 — 锁定协议 (docs/archaeology.md "Standard eval setting"):
#   temperature 1.0 / top_p 1.0 / top_k -1 / min_p 0 / presence_penalty 0 /
#   max_new 38912 / thinking ON / val_n=12 (avg@12)。
#
# 评 MODEL_DIR 下 STEPS 各 checkpoint × DATASETS 的矩阵；MODEL_DIR="" 时评 base
# (未训练 Qwen3-1.7B)。一 checkpoint 一 GPU stream，串行跑其所有 DATASETS；
# base 情形按 dataset round-robin 到各 GPU。输出 OUT/{TAG}ckpt{step}_{ds}.json
# (base: OUT/{TAG}_{ds}.json，TAG 空时为 OUT/base_{ds}.json)。
#
# 参数 (env):
#   MODEL_DIR  checkpoint 父目录 (空 = base)
#   TAG        输出前缀 (如 v0 / opsd / v0s1)；base 时用作文件名本身 (空 -> base)。
#              同一 OUT 下跑多个不同底座的 base 时必须各给 TAG，否则同名互相覆盖。
#   OUT        输出目录 (默认 results/matrix_eval)
#   DATASETS   空格分隔 (默认 "aime24 aime25")
#   STEPS      空格分隔 (默认 "50 100 150")；base 时忽略
#   LIMIT      -> --num_samples (空 = 全部题)
#   VAL_N      -> --val_n (默认 12)
#   NGPU       (默认 3)
#   DRYRUN     非空则只打印任务分配、不真跑 (登录节点校验用)
set -uo pipefail

PY="$HOME/.conda/envs/opsd/bin/python"
BASE_MODEL="${BASE_MODEL:-$HOME/models/Qwen3-1.7B}"
MODEL_DIR="${MODEL_DIR:-}"
TAG="${TAG:-}"
OUT="${OUT:-results/matrix_eval}"
DATASETS="${DATASETS:-aime24 aime25}"; DATASETS="${DATASETS//[,+]/ }"  # , + 或空格分隔
STEPS="${STEPS:-50 100 150}"; STEPS="${STEPS//[,+]/ }"                 # (qsub -v 用 + 避开逗号)
LIMIT="${LIMIT:-}"
VAL_N="${VAL_N:-12}"
NGPU="${NGPU:-3}"
DRYRUN="${DRYRUN:-}"

mkdir -p "$OUT" pbs/logs
LIMIT_ARG=""; [ -n "$LIMIT" ] && LIMIT_ARG="--num_samples $LIMIT"

run_one() {
    local gpu="$1" ds="$2" ckpt="$3" outtag="$4"
    if [ -n "$DRYRUN" ]; then
        echo "[dry] gpu=$gpu ds=$ds ckpt=${ckpt:-BASE} -> $OUT/${outtag}_${ds}.json"
        return 0
    fi
    local ckpt_arg=""; [ -n "$ckpt" ] && ckpt_arg="--checkpoint_dir $ckpt"
    echo "### START $(date '+%F %T') gpu=$gpu ds=$ds tag=$outtag ckpt=${ckpt:-BASE}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" eval/evaluate_math.py \
        --base_model "$BASE_MODEL" $ckpt_arg --dataset "$ds" \
        --val_n "$VAL_N" --temperature 1.0 --top_p 1.0 --top_k -1 --min_p 0 \
        --presence_penalty 0 --max_new_tokens 38912 \
        --tensor_parallel_size 1 --gpu_memory_utilization 0.9 $LIMIT_ARG \
        --output_file "$OUT/${outtag}_${ds}.json" \
        || echo "!!! FAILED ds=$ds tag=$outtag ckpt=${ckpt:-BASE}"
    echo "### END   $(date '+%F %T') ds=$ds tag=$outtag"
}

# 构造任务队列，每行 "gpu ds ckpt outtag"（ckpt 用 '-' 占位表示 base）
tasks=()
if [ -z "$MODEL_DIR" ]; then
    g=0
    for ds in $DATASETS; do
        tasks+=("$g $ds - ${TAG:-base}"); g=$(( (g+1) % NGPU ))
    done
else
    g=0
    for step in $STEPS; do
        ck="$MODEL_DIR/checkpoint-$step"
        if [ ! -d "$ck" ]; then echo "SKIP missing $ck"; continue; fi
        for ds in $DATASETS; do
            tasks+=("$g $ds $ck ${TAG}ckpt${step}")
            g=$(( (g+1) % NGPU ))   # 按 (ckpt,ds) 任务 round-robin，均衡且不闲置 GPU
        done
    done
fi

# 分派：每个 GPU 一个后台 stream，顺序执行分给它的任务
pids=()
for gpu in $(seq 0 $((NGPU-1))); do
    (
        for t in "${tasks[@]}"; do
            set -- $t
            [ "$1" = "$gpu" ] || continue
            ck="$3"; [ "$ck" = "-" ] && ck=""
            run_one "$1" "$2" "$ck" "$4"
        done
    ) > "pbs/logs/eval_matrix_gpu${gpu}.$$.log" 2>&1 &
    pids+=($!)
done

rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "=== results in $OUT ==="; ls -la "$OUT"/ 2>/dev/null | tail -30
exit $rc
