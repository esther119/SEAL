#!/usr/bin/env bash
# Build the v_code MBPP steering vector using SEAL's own pipeline, fast (vLLM gen).
#   gen_mbpp_vllm.py (vLLM)  ->  hidden_analysis.py (HF forward)  ->  vector_generation.py
# Run in the SEAL env (vllm). ~40 min on A100.
set -euo pipefail
gpu=${1:-0}
: "${MODEL:=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
: "${MAX_TOKENS:=3000}"
: "${SAMPLE:=120}"                 # traces per class (50/50). ~120 -> ~11k activations (~1/10 SEAL)
: "${MAX_EXAMPLES:=374}"           # all of MBPP-full train (scored for correct/incorrect)
TAG=$(basename "$MODEL")
DIR="results/MBPP/${TAG}/baseline_${MAX_TOKENS}"

echo "[1/4] vLLM generate + score MBPP train ($MAX_EXAMPLES tasks) ..."
CUDA_VISIBLE_DEVICES=$gpu python -u gen_mbpp_vllm.py \
    --model_name_or_path "$MODEL" --save_dir "$DIR" \
    --max_tokens "$MAX_TOKENS" --max_examples "$MAX_EXAMPLES" --use_chat_format --remove_bos

echo "[2/4] hidden states — incorrect (layer 20) ..."
CUDA_VISIBLE_DEVICES=$gpu python -u hidden_analysis.py \
    --model_path "$MODEL" --data_path "$DIR/data.jsonl" --data_dir "$DIR" \
    --type incorrect --start 0 --sample "$SAMPLE" --keep_layers 20

echo "[3/4] hidden states — correct (layer 20) ..."
CUDA_VISIBLE_DEVICES=$gpu python -u hidden_analysis.py \
    --model_path "$MODEL" --data_path "$DIR/data.jsonl" --data_dir "$DIR" \
    --type correct --start 0 --sample "$SAMPLE" --keep_layers 20

echo "[4/4] build steering vector (layer 20) ..."
python -u vector_generation.py \
    --data_dir "$DIR" --prefixs "correct_0_${SAMPLE}" "incorrect_0_${SAMPLE}" \
    --layers 20 --save_prefix "mbpp_${SAMPLE}_${SAMPLE}"

echo "== done: $DIR/vector_mbpp_${SAMPLE}_${SAMPLE}/layer_20_transition_reflection_steervec.pt =="
echo "   (SEAL convention: vector = H_RT - H_E; apply with coef -1.0 in the eval)"
