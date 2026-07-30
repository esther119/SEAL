#!/usr/bin/env bash
# Phase 1: derive KEYWORD_SETS["logic"] from real LogiQA reasoning traces instead of
# guessing. Generates a probe batch of CJK-filtered LogiQA train traces, then measures
# how much of it the existing "math" keyword set leaves untagged ("other") and ranks
# candidate phrases by how many distinct untagged steps each one would recover.
#
# Usage:
#   bash scripts/discover_logic_keywords.sh [gpu_index]
#   PROBE_EXAMPLES=300 bash scripts/discover_logic_keywords.sh 0
#
# vLLM generation needs a GPU; the coverage/candidate-phrase pass only needs the
# tokenizer and can be re-run freely while iterating on KEYWORD_SETS["logic"].
set -euo pipefail

gpu=${1:-0}
: "${MODEL:=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
: "${MAX_TOKENS:=3000}"
: "${PROBE_EXAMPLES:=300}"          # enough for the phrase-frequency ranking to be
                                     # meaningful, not just a handful of anecdotes
: "${MIN_STEP_COUNT:=3}"
: "${TOP_K:=40}"
: "${SEED:=42}"

TAG=$(basename "$MODEL")
DIR="results/results_for_logic_vectors/LogiQA_train/${TAG}/keyword_probe"

echo "[1/2] generate ${PROBE_EXAMPLES} CJK-filtered LogiQA train traces ..."
CUDA_VISIBLE_DEVICES=$gpu python -u gen_logiqa_vllm.py \
    --model_name_or_path "$MODEL" \
    --save_dir "$DIR" \
    --split train \
    --filter_cjk \
    --max_examples "$PROBE_EXAMPLES" \
    --max_tokens "$MAX_TOKENS" \
    --use_chat_format \
    --remove_bos

echo "[2/2] measure 'math' keyword baseline coverage + rank candidate phrases ..."
python -u logic_keyword_coverage.py \
    --data_dir "$DIR" \
    --model_path "$MODEL" \
    --keywords math \
    --min_step_count "$MIN_STEP_COUNT" \
    --top_k "$TOP_K" \
    --seed "$SEED"

cat <<EOF

Keyword probe complete. Review, in order:
  $DIR/candidate_phrases_math.json   -- phrases ranked by how many 'other' steps they'd recover
  $DIR/review_samples_math.jsonl     -- actual step text (other / check / switch) to confirm
                                         candidates are genuine reflection/transition language,
                                         not coincidental substring matches

For each candidate you validate against real step text, add it to
KEYWORD_SETS["logic"] in hidden_analysis.py. Then re-run to see how much of the
'other' bucket it actually recovers:

  python logic_keyword_coverage.py \\
    --data_dir "$DIR" \\
    --model_path "$MODEL" \\
    --keywords logic \\
    --seed "$SEED"

Keep iterating until new candidates only pick off a small marginal fraction of the
remaining 'other' steps. Do not run scripts/generate_vector_logiqa.sh KEYWORDS=logic
until the logic list is frozen this way.
EOF
