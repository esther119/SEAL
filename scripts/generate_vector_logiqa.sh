#!/usr/bin/env bash
# Build a LogiQA steering vector using SEAL's own pipeline, fast (vLLM gen).
#   gen_logiqa_vllm.py (vLLM)  ->  hidden_analysis.py (HF forward)  ->  vector_generation.py
#
# KEYWORDS selects which check/switch keyword set tags the reasoning steps:
#   KEYWORDS=math  (default) -- control/ablation: reuse the existing math keyword set
#   KEYWORDS=logic            -- the LogiQA-derived set (see scripts/discover_logic_keywords.sh);
#                                refused below until KEYWORD_SETS["logic"] actually exists,
#                                so this can't silently fall back to an empty/undefined set.
# Run both to compare: the point is to test whether a dedicated keyword list actually
# changes the resulting vector's effect, not just to build "a" vector.
#
# Run in the SEAL env (vllm).
set -euo pipefail
gpu=${1:-0}
: "${MODEL:=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
: "${MAX_TOKENS:=3000}"
: "${SAMPLE:=500}"                 # traces per class (correct/incorrect) -- matches the
                                    # team's MATH/APPS scale target for LogiQA (see
                                    # docs/vector_build_plan.html), not MBPP's smaller 120+120.
: "${MAX_EXAMPLES:=}"              # empty = all streamed LogiQA2.0 train rows
: "${KEYWORDS:=math}"
TAG=$(basename "$MODEL")
DIR="results/results_for_logic_vectors/LogiQA_train/${TAG}/baseline_${MAX_TOKENS}"

if [[ "$KEYWORDS" == "logic" ]]; then
    python -c "
from hidden_analysis import KEYWORD_SETS
assert 'logic' in KEYWORD_SETS, (
    'KEYWORD_SETS[\"logic\"] does not exist yet in hidden_analysis.py. '
    'Run scripts/discover_logic_keywords.sh, review the candidate phrases, and add a '
    'reviewed KEYWORD_SETS[\"logic\"] entry before building the logic-keyword vector.'
)
"
fi

MAX_EX_ARGS=()
if [[ -n "$MAX_EXAMPLES" ]]; then
    MAX_EX_ARGS=(--max_examples "$MAX_EXAMPLES")
fi

if [[ -f "$DIR/math_eval.jsonl" && -f "$DIR/data.jsonl" ]]; then
    echo "[1/4] $DIR/math_eval.jsonl + data.jsonl already exist -- skipping generation "
    echo "      (trace generation doesn't depend on KEYWORDS; delete these files to force a re-run)"
    # Guard: reused traces may predate the grading fix. Under the legacy extractor a
    # generation with no </think> could be labelled CORRECT via a fabricated letter.
    # Refuse rather than silently build a vector from contaminated labels.
    python - "$DIR/math_eval.jsonl" <<'PY'
import json, sys
bad = tot = 0
for line in open(sys.argv[1]):
    r = json.loads(line)
    gen = r["model_generation"][0]
    tot += 1
    if "</think>" not in gen and r["all_eval"][0]:
        bad += 1          # no answer was stated, yet scored correct -> legacy grader
if bad:
    raise SystemExit(
        f"[guard] {bad}/{tot} reused traces are labelled CORRECT despite never closing "
        f"</think>. These were graded by the superseded extractor.\n"
        f"[guard] Re-grade them first:\n"
        f"[guard]   python scripts/relabel_logiqa_build.py --build_dir {sys.argv[1].rsplit('/',1)[0]} "
        f"--out_dir <dir>_regraded\n"
        f"[guard] then point this script at the regraded dir (or delete the traces to regenerate)."
    )
print(f"[guard] {tot} reused traces pass the grading check (no fabricated-correct labels).")
PY
else
    echo "[1/4] vLLM generate + score LogiQA train (CJK-filtered) ..."
    CUDA_VISIBLE_DEVICES=$gpu python -u gen_logiqa_vllm.py \
        --model_name_or_path "$MODEL" --save_dir "$DIR" \
        --max_tokens "$MAX_TOKENS" --split train --filter_cjk "${MAX_EX_ARGS[@]}" \
        --use_chat_format --remove_bos
fi

echo "[2/4] hidden states — incorrect (layer 20) ..."
CUDA_VISIBLE_DEVICES=$gpu python -u hidden_analysis.py \
    --model_path "$MODEL" --data_path "$DIR/data.jsonl" --data_dir "$DIR" \
    --type incorrect --start 0 --sample "$SAMPLE" --keep_layers 20 --keywords "$KEYWORDS"

echo "[3/4] hidden states — correct (layer 20) ..."
CUDA_VISIBLE_DEVICES=$gpu python -u hidden_analysis.py \
    --model_path "$MODEL" --data_path "$DIR/data.jsonl" --data_dir "$DIR" \
    --type correct --start 0 --sample "$SAMPLE" --keep_layers 20 --keywords "$KEYWORDS"

echo "[4/4] build steering vector (layer 20) ..."
python -u vector_generation.py \
    --data_dir "$DIR" --prefixs "correct_0_${SAMPLE}" "incorrect_0_${SAMPLE}" \
    --layers 20 --save_prefix "${KEYWORDS}_${SAMPLE}_${SAMPLE}" --overwrite

echo "== done: $DIR/vector_${KEYWORDS}_${SAMPLE}_${SAMPLE}/layer_20_transition_reflection_steervec.pt =="
echo "   (SEAL convention: vector = H_RT - H_E; apply with coef -1.0 in the eval)"
