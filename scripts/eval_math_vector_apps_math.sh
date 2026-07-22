#!/usr/bin/env bash
set -euo pipefail

VECTOR_PATH=${1:?"usage: $0 PATH_TO_MATH_VECTOR [GPU_INDEX]"}
GPU_INDEX=${2:-0}

MODEL=${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
LAYER=${LAYER:-20}
COEF=${COEF:--1.0}
MAX_EXAMPLES=${MAX_EXAMPLES:-300}
SAMPLE_SEED=${SAMPLE_SEED:-42}
MAX_TOKENS=${MAX_TOKENS:-10000}
BATCH_SIZE=${BATCH_SIZE:-25}
APPS_BATCH_SIZE=${APPS_BATCH_SIZE:-1}
APPS_SPLIT=${APPS_SPLIT:-test}
EVAL_WORKERS=${EVAL_WORKERS:-12}
APPS_TIMEOUT=${APPS_TIMEOUT:-10}
RUN_BASELINE=${RUN_BASELINE:-1}
RESULT_ROOT=${RESULT_ROOT:-results/math_vector_transfer}

if [[ ! -f "$VECTOR_PATH" ]]; then
  echo "Steering vector not found: $VECTOR_PATH" >&2
  exit 2
fi

COMMON_MATH=(
  --model_name_or_path "$MODEL"
  --max_tokens "$MAX_TOKENS"
  --use_chat_format
  --batch_size "$BATCH_SIZE"
  --dataset MATH500
  --remove_bos
  --random_sample
  --sample_seed "$SAMPLE_SEED"
  --max_examples "$MAX_EXAMPLES"
)

COMMON_APPS=(
  --model_name_or_path "$MODEL"
  --max_tokens "$MAX_TOKENS"
  --use_chat_format
  --batch_size "$APPS_BATCH_SIZE"
  --benchmark apps
  --apps_split "$APPS_SPLIT"
  --remove_bos
  --random_sample
  --stratify_by_difficulty
  --sample_seed "$SAMPLE_SEED"
  --max_examples "$MAX_EXAMPLES"
  --eval_workers "$EVAL_WORKERS"
  --timeout "$APPS_TIMEOUT"
)

if [[ "$RUN_BASELINE" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" python eval_MATH_steering.py \
    "${COMMON_MATH[@]}" \
    --save_dir "$RESULT_ROOT/MATH500/baseline"

  CUDA_VISIBLE_DEVICES="$GPU_INDEX" python eval_code_steering.py \
    "${COMMON_APPS[@]}" \
    --save_dir "$RESULT_ROOT/APPS/baseline"
fi

CUDA_VISIBLE_DEVICES="$GPU_INDEX" python eval_MATH_steering.py \
  "${COMMON_MATH[@]}" \
  --save_dir "$RESULT_ROOT/MATH500/math_vector" \
  --steering \
  --steering_vector "$VECTOR_PATH" \
  --steering_layer "$LAYER" \
  --steering_coef "$COEF"

CUDA_VISIBLE_DEVICES="$GPU_INDEX" python eval_code_steering.py \
  "${COMMON_APPS[@]}" \
  --save_dir "$RESULT_ROOT/APPS/math_vector" \
  --steering \
  --steering_vector "$VECTOR_PATH" \
  --steering_layer "$LAYER" \
  --steering_coef "$COEF"

echo "Finished. Results are under $RESULT_ROOT"
