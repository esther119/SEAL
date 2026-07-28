#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: DATASETS=math,apps,logiqa,livecodebench $0 VECTOR_PATH VECTOR_NAME [GPU_INDEX]" >&2
  exit 2
fi

VECTOR_PATH=$1
VECTOR_NAME=$2
GPU_INDEX=${3:-0}

MODEL=${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
LAYER=${LAYER:-20}
COEF=${COEF:--1.0}
MATH_MAX_EXAMPLES=${MATH_MAX_EXAMPLES:-500}
APPS_MAX_EXAMPLES=${APPS_MAX_EXAMPLES:-500}
LOGIQA_MAX_EXAMPLES=${LOGIQA_MAX_EXAMPLES:-500}
LCB_MAX_EXAMPLES=${LCB_MAX_EXAMPLES:-400}
SAMPLE_SEED=${SAMPLE_SEED:-42}
MAX_TOKENS=${MAX_TOKENS:-10000}
BATCH_SIZE=${BATCH_SIZE:-25}
APPS_BATCH_SIZE=${APPS_BATCH_SIZE:-1}
LOGIQA_BATCH_SIZE=${LOGIQA_BATCH_SIZE:-25}
LCB_BATCH_SIZE=${LCB_BATCH_SIZE:-25}
APPS_SPLIT=${APPS_SPLIT:-test}
LCB_RELEASE=${LCB_RELEASE:-release_v1}
EVAL_WORKERS=${EVAL_WORKERS:-12}
APPS_TIMEOUT=${APPS_TIMEOUT:-10}
RUN_BASELINE=${RUN_BASELINE:-1}
RESULT_ROOT=${RESULT_ROOT:-results/results_for_math_vectors}
DATASETS=${DATASETS:-math,apps,livecodebench}
# LogiQA 2.0 is ~half untranslated Chinese; keep the eval set English-only so it
# matches the English build set gen_logiqa_vllm.py filters for.
LOGIQA_ENGLISH_ONLY=${LOGIQA_ENGLISH_ONLY:-1}

if [[ ! "$VECTOR_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid VECTOR_NAME '$VECTOR_NAME'; use letters, numbers, dots, dashes, or underscores" >&2
  exit 2
fi

dataset_enabled() {
  case ",$DATASETS," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

IFS=',' read -r -a SELECTED_DATASETS <<< "$DATASETS"
for dataset in "${SELECTED_DATASETS[@]}"; do
  case "$dataset" in
    math|apps|logiqa|livecodebench) ;;
    *)
      echo "Unknown dataset '$dataset'. Choose from: math,apps,logiqa,livecodebench" >&2
      exit 2
      ;;
  esac
done

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
  --max_examples "$MATH_MAX_EXAMPLES"
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
  --sample_seed "$SAMPLE_SEED"
  --max_examples "$APPS_MAX_EXAMPLES"
  --eval_workers "$EVAL_WORKERS"
  --timeout "$APPS_TIMEOUT"
)

COMMON_LOGIQA=(
  --model_name_or_path "$MODEL"
  --max_tokens "$MAX_TOKENS"
  --use_chat_format
  --batch_size "$LOGIQA_BATCH_SIZE"
  --dataset LogiQA
  --remove_bos
  --random_sample
  --sample_seed "$SAMPLE_SEED"
  --max_examples "$LOGIQA_MAX_EXAMPLES"
)
if [[ "$LOGIQA_ENGLISH_ONLY" == "1" ]]; then
  COMMON_LOGIQA+=(--logiqa_english_only)
fi

COMMON_LCB=(
  --model_name_or_path "$MODEL"
  --max_tokens "$MAX_TOKENS"
  --use_chat_format
  --batch_size "$LCB_BATCH_SIZE"
  --benchmark livecodebench
  --release "$LCB_RELEASE"
  --remove_bos
  --max_examples "$LCB_MAX_EXAMPLES"
)

if [[ "$RUN_BASELINE" == "1" ]]; then
  if dataset_enabled math; then
    CUDA_VISIBLE_DEVICES="$GPU_INDEX" python eval_MATH_steering.py \
      "${COMMON_MATH[@]}" \
      --save_dir "$RESULT_ROOT/MATH500/baseline"
  fi

  if dataset_enabled apps; then
    CUDA_VISIBLE_DEVICES="$GPU_INDEX" python eval_code_steering.py \
      "${COMMON_APPS[@]}" \
      --save_dir "$RESULT_ROOT/APPS/baseline"
  fi

  if dataset_enabled logiqa; then
    CUDA_VISIBLE_DEVICES="$GPU_INDEX" python eval_MATH_steering.py \
      "${COMMON_LOGIQA[@]}" \
      --save_dir "$RESULT_ROOT/LogiQA/baseline"
  fi

  if dataset_enabled livecodebench; then
    CUDA_VISIBLE_DEVICES="$GPU_INDEX" python eval_code_steering.py \
      "${COMMON_LCB[@]}" \
      --save_dir "$RESULT_ROOT/LiveCodeBench/baseline"
  fi
fi

if dataset_enabled math; then
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" python eval_MATH_steering.py \
    "${COMMON_MATH[@]}" \
    --save_dir "$RESULT_ROOT/MATH500/$VECTOR_NAME" \
    --steering \
    --steering_vector "$VECTOR_PATH" \
    --steering_layer "$LAYER" \
    --steering_coef "$COEF"
fi

if dataset_enabled apps; then
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" python eval_code_steering.py \
    "${COMMON_APPS[@]}" \
    --save_dir "$RESULT_ROOT/APPS/$VECTOR_NAME" \
    --steering \
    --steering_vector "$VECTOR_PATH" \
    --steering_layer "$LAYER" \
    --steering_coef "$COEF"
fi

if dataset_enabled logiqa; then
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" python eval_MATH_steering.py \
    "${COMMON_LOGIQA[@]}" \
    --save_dir "$RESULT_ROOT/LogiQA/$VECTOR_NAME" \
    --steering \
    --steering_vector "$VECTOR_PATH" \
    --steering_layer "$LAYER" \
    --steering_coef "$COEF"
fi

if dataset_enabled livecodebench; then
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" python eval_code_steering.py \
    "${COMMON_LCB[@]}" \
    --save_dir "$RESULT_ROOT/LiveCodeBench/$VECTOR_NAME" \
    --steering \
    --steering_vector "$VECTOR_PATH" \
    --steering_layer "$LAYER" \
    --steering_coef "$COEF"
fi

echo "Finished. Results are under $RESULT_ROOT"
