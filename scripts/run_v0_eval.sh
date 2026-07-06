#!/bin/bash
# v0 eval driver — evaluate qwen31b_v0_main checkpoints on AIME24/AIME25 under
# the LOCKED official protocol (docs/archaeology.md "Standard eval setting"):
#   temperature=1.0, thinking ON, max_new_tokens=38912, top_p 1.0 (top-p none),
#   top_k -1, min_p 0, presence_penalty 0, val_n=12 (avg@12).
# NOTE: the handoff's temp 0.6 is WRONG — do not use it. This is the frozen lock.
#
# Writes results/v0_eval/{tag}_{ds}.json. The base (untrained Qwen3-1.7B) number
# is model-independent and already lives in results/repro_eval/base_*.json — it
# is the shared reference, not re-evaluated here (summarize_v0.py reads it there).
#
# Quick sanity mode:  LIMIT=4 VAL_N=2 bash scripts/run_v0_eval.sh
#   LIMIT -> --num_samples (problem cap);  VAL_N -> --val_n (samples/problem).
# Full run (default): all problems, avg@12.
#
# Checkpoints:  STEPS env (default "50 100 150"), one GPU stream each (tp=1;
# Qwen3-1.7B num_key_value_heads=8 is not divisible by 3, so tp>1 would fail).
set -uo pipefail

PY="$HOME/.conda/envs/opsd/bin/python"
BASE="$HOME/models/Qwen3-1.7B"
CKPT_DIR="$HOME/opsd_outputs/qwen31b_v0_main"
OUT="results/v0_eval"
STEPS="${STEPS:-50 100 150}"
LIMIT="${LIMIT:-}"          # -> --num_samples (empty = all problems)
VAL_N="${VAL_N:-12}"        # -> --val_n (avg@N; locked default 12)

mkdir -p "$OUT"
LIMIT_ARG=""
[ -n "$LIMIT" ] && LIMIT_ARG="--num_samples $LIMIT"

run_eval() {
    local ds="$1" ckpt="$2" tag="$3"
    local ckpt_arg=""
    [ -n "$ckpt" ] && ckpt_arg="--checkpoint_dir $ckpt"
    echo "### START $(date '+%F %T')  dataset=$ds  tag=$tag  ckpt=${ckpt:-BASE}"
    "$PY" eval/evaluate_math.py \
        --base_model "$BASE" $ckpt_arg \
        --dataset "$ds" \
        --val_n "$VAL_N" \
        --temperature 1.0 \
        --top_p 1.0 \
        --top_k -1 \
        --min_p 0 \
        --presence_penalty 0 \
        --max_new_tokens 38912 \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization 0.9 \
        $LIMIT_ARG \
        --output_file "$OUT/${tag}_${ds}.json" \
        || echo "!!! FAILED dataset=$ds tag=$tag ckpt=${ckpt:-BASE}"
    echo "### END   $(date '+%F %T')  dataset=$ds  tag=$tag"
}

# one checkpoint per GPU; each stream does aime24 then aime25 serially.
stream() {
    local gpu="$1" step="$2"
    local ckpt="$CKPT_DIR/checkpoint-$step"
    if [ ! -d "$ckpt" ]; then
        echo "!!! SKIP checkpoint-$step: $ckpt does not exist yet"
        return 0
    fi
    CUDA_VISIBLE_DEVICES="$gpu" run_eval aime24 "$ckpt" "v0ckpt$step"
    CUDA_VISIBLE_DEVICES="$gpu" run_eval aime25 "$ckpt" "v0ckpt$step"
}

gpu=0
pids=()
for step in $STEPS; do
    stream "$gpu" "$step" > "pbs/logs/v0_eval_ckpt${step}.$$.log" 2>&1 &
    pids+=($!)
    gpu=$(( (gpu + 1) % 3 ))
done

rc=0
for p in "${pids[@]}"; do
    wait "$p" || rc=1
done

echo "=== v0 eval results ==="
ls -la "$OUT"/ 2>/dev/null
exit $rc
