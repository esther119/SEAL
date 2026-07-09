#!/usr/bin/env bash
# Transfer the MATH-derived steering vector to MMLU (philosophy subject).
# Runs baseline (no steering) then steered on the MMLU philosophy test set,
# reusing the shared MATH/LogiQA harness (answer-matching, choice accuracy).
#
# Usage: bash scripts/mmlu_philosophy.sh [gpu]
set -euo pipefail

# Reduce CUDA fragmentation OOMs during the steered (HF generate) phase.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

gpu=${1:-0}
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
TAG="DeepSeek-R1-Distill-Qwen-1.5B"
SUBJECT="philosophy"
VEC="results/results_for_math_vectors/MATH_train/${TAG}/baseline_10000/vector_500_500/layer_20_transition_reflection_steervec.pt"
OUT="results/results_for_math_vectors/MMLU/${TAG}"

echo "[1/2] baseline (no steering) — MMLU/${SUBJECT} ..."
CUDA_VISIBLE_DEVICES=$gpu python eval_MATH_vllm.py \
    --model_name_or_path "$MODEL" \
    --save_dir "${OUT}/transfer_baseline" \
    --dataset MMLU \
    --mmlu_subject "$SUBJECT" \
    --max_tokens 10000 \
    --use_chat_format \
    --remove_bos

echo "[2/2] steered (MATH vector, layer 20, coef -1.0) — MMLU/${SUBJECT} ..."
CUDA_VISIBLE_DEVICES=$gpu python eval_MATH_steering.py \
    --model_name_or_path "$MODEL" \
    --save_dir "${OUT}/transfer_steered" \
    --dataset MMLU \
    --mmlu_subject "$SUBJECT" \
    --max_tokens 10000 \
    --use_chat_format \
    --batch_size 40 \
    --remove_bos \
    --steering \
    --steering_vector "$VEC" \
    --steering_layer 20 \
    --steering_coef -1.0

echo "Done. metrics: ${OUT}/transfer_{baseline,steered}/metrics.json"
