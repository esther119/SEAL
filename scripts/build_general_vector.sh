#!/usr/bin/env bash
# Build Phase 1 S_general from MATH + APPS + LogiQA (unbalanced pool of all
# boundary vectors).
#
# In-repo layout (no external/v_code-SEAL required for the pooler itself):
#   data/APPS/baseline_10000/          shipped APPS traces
#   results/results_for_logic_vectors/LogiQA_train/.../baseline_3000_regraded/
#                                      canonical, strictly graded LogiQA traces
#   data/{MATH,APPS,LogiQA}/hidden_*_0_500/hidden.pt
#                                      DURABLE store (tracked; not under results/)
#   vectors/apps_v_code.pt             shipped code-domain home vector (cosine smoke)
#   apps/                              APPS generation + scoring
#   gen_logiqa_vllm.py + logic_utils.py
#                                      canonical LogiQA generation + scoring
#   hidden_analysis.py                 ONE extractor — --keywords math|code|logic
#
# Usage:
#   bash scripts/build_general_vector.sh            # full Phase 1 (GPU)
#   SKIP_HIDDEN=1 bash scripts/build_general_vector.sh   # pool only
#   SKIP_LOGIQA_GEN=1 bash scripts/build_general_vector.sh  # skip LogiQA gen if traces exist
#   BALANCE=1 bash scripts/build_general_vector.sh  # Phase 2 equal-domain subsample
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [[ -f .env ]]; then set -a; source .env; set +a; fi

: "${GPU:=0}"
: "${HF_HOME:=${HF_HOME:-/workspace/hf_cache}}"
: "${MODEL:=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
: "${STEER_LAYER:=20}"
: "${VEC_SAMPLES:=500}"
: "${SKIP_HIDDEN:=0}"
: "${SKIP_LOGIQA_GEN:=0}"
: "${BALANCE:=0}"
: "${BALANCE_SEED:=42}"
: "${MAX_CJK:=0.10}"
: "${LOGIQA_MAX_TOKENS:=3000}"
: "${LOGIQA_MAX_EXAMPLES:=2000}"

export HF_HOME CUDA_VISIBLE_DEVICES="$GPU"
[[ -n "${HF_TOKEN:-}" ]] && export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

MODEL_TAG="$(basename "$MODEL")"
MATH_DIR="results/results_for_math_vectors/MATH_train/${MODEL_TAG}/baseline_10000"
MATH_DATA="data/MATH/train.jsonl"
APPS_SHIPPED="data/APPS/baseline_10000"
APPS_DIR="results/APPS_train/${MODEL_TAG}/baseline_10000"
APPS_DATA="${APPS_DIR}/data.jsonl"
LOGIQA_SOURCE="data/LogiQA/train.jsonl"
LOGIQA_DIR="results/results_for_logic_vectors/LogiQA_train/${MODEL_TAG}/baseline_${LOGIQA_MAX_TOKENS}_regraded"
LOGIQA_DATA="${LOGIQA_DIR}/data.jsonl"
LOGIQA_EVAL="${LOGIQA_DIR}/math_eval.jsonl"

# Working copies (under gitignored results/)
MATH_HIDDEN_C="${MATH_DIR}/hidden_correct_0_${VEC_SAMPLES}/hidden.pt"
MATH_HIDDEN_I="${MATH_DIR}/hidden_incorrect_0_${VEC_SAMPLES}/hidden.pt"
APPS_HIDDEN_C="${APPS_DIR}/hidden_correct_0_${VEC_SAMPLES}/hidden.pt"
APPS_HIDDEN_I="${APPS_DIR}/hidden_incorrect_0_${VEC_SAMPLES}/hidden.pt"
LOGIQA_HIDDEN_C="${LOGIQA_DIR}/hidden_correct_0_${VEC_SAMPLES}/hidden.pt"
LOGIQA_HIDDEN_I="${LOGIQA_DIR}/hidden_incorrect_0_${VEC_SAMPLES}/hidden.pt"

# Durable copies (tracked under data/ — keep these from now on)
MATH_STORE_C="data/MATH/hidden_correct_0_${VEC_SAMPLES}/hidden.pt"
MATH_STORE_I="data/MATH/hidden_incorrect_0_${VEC_SAMPLES}/hidden.pt"
APPS_STORE_C="data/APPS/hidden_correct_0_${VEC_SAMPLES}/hidden.pt"
APPS_STORE_I="data/APPS/hidden_incorrect_0_${VEC_SAMPLES}/hidden.pt"
LOGIQA_STORE_C="data/LogiQA/hidden_correct_0_${VEC_SAMPLES}/hidden.pt"
LOGIQA_STORE_I="data/LogiQA/hidden_incorrect_0_${VEC_SAMPLES}/hidden.pt"

OUT_DIR="results/general"
if [[ "$BALANCE" == "1" ]]; then
    OUT_VEC="${OUT_DIR}/S_general_math_apps_logic_phase2.pt"
else
    OUT_VEC="${OUT_DIR}/S_general_math_apps_logic_phase1.pt"
fi
MATH_VEC="${MATH_DIR}/vector_${VEC_SAMPLES}_${VEC_SAMPLES}/layer_${STEER_LAYER}_transition_reflection_steervec.pt"
APPS_VEC="vectors/apps_v_code.pt"
LOGIQA_VEC="vectors/logiqa_v_logic.pt"

echo "=================================================================="
echo " S_general · MATH + APPS + LogiQA"
echo " MODEL=$MODEL  GPU=$GPU  LAYER=$STEER_LAYER  SAMPLES=$VEC_SAMPLES"
echo " BALANCE=$BALANCE  SKIP_HIDDEN=$SKIP_HIDDEN  SKIP_LOGIQA_GEN=$SKIP_LOGIQA_GEN"
echo " MATH_DIR=$MATH_DIR"
echo " APPS_DIR=$APPS_DIR"
echo " LOGIQA_DIR=$LOGIQA_DIR"
echo " durable hidden store: data/{MATH,APPS,LogiQA}/hidden_*_0_${VEC_SAMPLES}/"
echo " OUT=$OUT_VEC"
echo "=================================================================="

# ------------------------------------------------------------------ #
# Sanity: required inputs
# ------------------------------------------------------------------ #
[[ -f "$MATH_DIR/math_eval.jsonl" ]] || { echo "ERROR: missing $MATH_DIR/math_eval.jsonl"; exit 1; }
[[ -f "$MATH_DATA" ]] || { echo "ERROR: missing $MATH_DATA"; exit 1; }
[[ -f "$APPS_SHIPPED/math_eval.jsonl.gz" && -f "$APPS_SHIPPED/data.jsonl" ]] || {
    echo "ERROR: missing shipped APPS traces under $APPS_SHIPPED"
    echo "  Expected data.jsonl + math_eval.jsonl.gz (from v_code-SEAL baseline_10000)."
    exit 1
}
[[ -f "$LOGIQA_SOURCE" ]] || {
    echo "ERROR: missing canonical English LogiQA source $LOGIQA_SOURCE"
    exit 1
}
[[ -f "$APPS_VEC" ]] || echo "WARN: missing $APPS_VEC — cosine compare vs apps will be skipped."

# Restore durable hidden.pt into results/ working dirs when present.
restore_hidden() {
    local src="$1" dst="$2"
    if [[ -f "$src" && ! -f "$dst" ]]; then
        mkdir -p "$(dirname "$dst")"
        echo "[store] restore $(basename "$(dirname "$src")")/hidden.pt → $dst"
        cp "$src" "$dst"
    fi
}
restore_hidden "$MATH_STORE_C" "$MATH_HIDDEN_C"
restore_hidden "$MATH_STORE_I" "$MATH_HIDDEN_I"
restore_hidden "$APPS_STORE_C" "$APPS_HIDDEN_C"
restore_hidden "$APPS_STORE_I" "$APPS_HIDDEN_I"
restore_hidden "$LOGIQA_STORE_C" "$LOGIQA_HIDDEN_C"
restore_hidden "$LOGIQA_STORE_I" "$LOGIQA_HIDDEN_I"

# ------------------------------------------------------------------ #
# Stage APPS traces into results/ (decompress once)
# ------------------------------------------------------------------ #
mkdir -p "$APPS_DIR"
if [[ ! -f "$APPS_DIR/math_eval.jsonl" ]]; then
    echo "[apps] Staging shipped traces → $APPS_DIR"
    cp "$APPS_SHIPPED/data.jsonl" "$APPS_DATA"
    gunzip -c "$APPS_SHIPPED/math_eval.jsonl.gz" > "$APPS_DIR/math_eval.jsonl"
    cp "$APPS_SHIPPED/gen_config.json" "$APPS_DIR/gen_config.json" 2>/dev/null || true
    cp "$APPS_SHIPPED"/selection_*.json "$APPS_DIR/" 2>/dev/null || true
fi
[[ -f "$APPS_DATA" && -f "$APPS_DIR/math_eval.jsonl" ]] || {
    echo "ERROR: failed to stage APPS traces into $APPS_DIR"; exit 1;
}

# ------------------------------------------------------------------ #
# 0) LogiQA traces (canonical strict grader; English-only pool filter)
# ------------------------------------------------------------------ #
mkdir -p "$LOGIQA_DIR"
if [[ "$SKIP_HIDDEN" == "1" ]]; then
    echo "[logiqa] SKIP_HIDDEN=1 — not generating traces (need durable hidden.pt)."
elif [[ -f "$LOGIQA_HIDDEN_C" && -f "$LOGIQA_HIDDEN_I" ]]; then
    echo "[logiqa] working hidden.pt already present — skipping generation."
elif [[ "$SKIP_LOGIQA_GEN" == "1" ]]; then
    [[ -f "$LOGIQA_EVAL" && -f "$LOGIQA_DATA" ]] || {
        echo "ERROR: SKIP_LOGIQA_GEN=1 but missing $LOGIQA_EVAL"
        exit 1
    }
    echo "[logiqa] SKIP_LOGIQA_GEN=1 — using existing traces."
elif [[ -f "$LOGIQA_EVAL" && -f "$LOGIQA_DATA" ]]; then
    echo "[logiqa] canonical traces already exist — reusing them."
else
    echo "[logiqa] Generating $LOGIQA_MAX_EXAMPLES canonical English traces..."
    python -u gen_logiqa_vllm.py \
        --model_name_or_path "$MODEL" \
        --save_dir "$LOGIQA_DIR" \
        --max_tokens "$LOGIQA_MAX_TOKENS" \
        --max_examples "$LOGIQA_MAX_EXAMPLES" \
        --split train \
        --filter_cjk \
        --cjk_threshold "$MAX_CJK" \
        --use_chat_format \
        --remove_bos
fi

if [[ "$SKIP_HIDDEN" != "1" && ! ( -f "$LOGIQA_HIDDEN_C" && -f "$LOGIQA_HIDDEN_I" ) ]]; then
    python - "$LOGIQA_EVAL" "$VEC_SAMPLES" <<'PY'
import json
import sys

path, target = sys.argv[1], int(sys.argv[2])
rows = [json.loads(line) for line in open(path)]
fabricated = sum(
    "</think>" not in row["model_generation"][0] and bool(row["all_eval"][0])
    for row in rows
)
correct = sum(bool(row["all_eval"][0]) for row in rows)
incorrect = len(rows) - correct
if fabricated:
    raise SystemExit(
        f"ERROR: {fabricated} unfinished traces are labelled correct in {path}; "
        "regrade with scripts/relabel_logiqa_build.py before extraction."
    )
if correct < target or incorrect < target:
    raise SystemExit(
        f"ERROR: LogiQA pools under-filled: correct={correct}, incorrect={incorrect}, "
        f"target={target}. Increase LOGIQA_MAX_EXAMPLES and regenerate."
    )
print(
    f"[logiqa] strict-label guard passed: correct={correct}, "
    f"incorrect={incorrect}, target={target}"
)
PY
fi

# ------------------------------------------------------------------ #
# 1) MATH hidden states (layer STEER_LAYER, keywords=math)
# ------------------------------------------------------------------ #
if [[ "$SKIP_HIDDEN" == "1" ]]; then
    echo "[math] SKIP_HIDDEN=1 — not regenerating."
else
    if [[ -f "$MATH_HIDDEN_I" ]]; then
        echo "[math] incorrect hidden.pt exists, skipping."
    else
        echo "[math] Extracting incorrect hidden states (layer $STEER_LAYER, keywords=math)..."
        python hidden_analysis.py \
            --model_path "$MODEL" \
            --data_path "$MATH_DATA" \
            --data_dir "$MATH_DIR" \
            --type incorrect --start 0 --sample "$VEC_SAMPLES" \
            --keep_layers "$STEER_LAYER" \
            --keywords math
    fi
    if [[ -f "$MATH_HIDDEN_C" ]]; then
        echo "[math] correct hidden.pt exists, skipping."
    else
        echo "[math] Extracting correct hidden states (layer $STEER_LAYER, keywords=math)..."
        python hidden_analysis.py \
            --model_path "$MODEL" \
            --data_path "$MATH_DATA" \
            --data_dir "$MATH_DIR" \
            --type correct --start 0 --sample "$VEC_SAMPLES" \
            --keep_layers "$STEER_LAYER" \
            --keywords math
    fi
fi

# ------------------------------------------------------------------ #
# 2) APPS hidden states (SAME hidden_analysis.py, keywords=code)
# ------------------------------------------------------------------ #
if [[ "$SKIP_HIDDEN" == "1" ]]; then
    echo "[apps] SKIP_HIDDEN=1 — not regenerating."
else
    if [[ -f "$APPS_HIDDEN_I" ]]; then
        echo "[apps] incorrect hidden.pt exists, skipping."
    else
        echo "[apps] Extracting incorrect hidden states (layer $STEER_LAYER, keywords=code)..."
        python hidden_analysis.py \
            --model_path "$MODEL" \
            --data_path "$APPS_DATA" \
            --data_dir "$APPS_DIR" \
            --type incorrect --start 0 --sample "$VEC_SAMPLES" \
            --keep_layers "$STEER_LAYER" \
            --keywords code
    fi
    if [[ -f "$APPS_HIDDEN_C" ]]; then
        echo "[apps] correct hidden.pt exists, skipping."
    else
        echo "[apps] Extracting correct hidden states (layer $STEER_LAYER, keywords=code)..."
        python hidden_analysis.py \
            --model_path "$MODEL" \
            --data_path "$APPS_DATA" \
            --data_dir "$APPS_DIR" \
            --type correct --start 0 --sample "$VEC_SAMPLES" \
            --keep_layers "$STEER_LAYER" \
            --keywords code
    fi
fi

# ------------------------------------------------------------------ #
# 3) LogiQA hidden states (SAME hidden_analysis.py, keywords=logic)
# ------------------------------------------------------------------ #
if [[ "$SKIP_HIDDEN" == "1" ]]; then
    echo "[logiqa] SKIP_HIDDEN=1 — not regenerating."
else
    [[ -f "$LOGIQA_EVAL" && -f "$LOGIQA_DATA" ]] || {
        echo "ERROR: missing LogiQA traces under $LOGIQA_DIR"
        exit 1
    }
    if [[ -f "$LOGIQA_HIDDEN_I" ]]; then
        echo "[logiqa] incorrect hidden.pt exists, skipping."
    else
        echo "[logiqa] Extracting incorrect hidden states (layer $STEER_LAYER, keywords=logic)..."
        python hidden_analysis.py \
            --model_path "$MODEL" \
            --data_path "$LOGIQA_DATA" \
            --eval_path "$LOGIQA_EVAL" \
            --data_dir "$LOGIQA_DIR" \
            --type incorrect --start 0 --sample "$VEC_SAMPLES" \
            --keep_layers "$STEER_LAYER" \
            --keywords logic
    fi
    if [[ -f "$LOGIQA_HIDDEN_C" ]]; then
        echo "[logiqa] correct hidden.pt exists, skipping."
    else
        echo "[logiqa] Extracting correct hidden states (layer $STEER_LAYER, keywords=logic)..."
        python hidden_analysis.py \
            --model_path "$MODEL" \
            --data_path "$LOGIQA_DATA" \
            --eval_path "$LOGIQA_EVAL" \
            --data_dir "$LOGIQA_DIR" \
            --type correct --start 0 --sample "$VEC_SAMPLES" \
            --keep_layers "$STEER_LAYER" \
            --keywords logic
    fi
fi

for f in \
    "$MATH_HIDDEN_C" "$MATH_HIDDEN_I" \
    "$APPS_HIDDEN_C" "$APPS_HIDDEN_I" \
    "$LOGIQA_HIDDEN_C" "$LOGIQA_HIDDEN_I"
do
    [[ -f "$f" ]] || { echo "ERROR: missing $f — run without SKIP_HIDDEN on a GPU pod."; exit 1; }
done

# ------------------------------------------------------------------ #
# Persist hidden.pt under data/ (tracked; survives results/ cleanups)
# ------------------------------------------------------------------ #
store_hidden() {
    local src="$1" dst="$2"
    mkdir -p "$(dirname "$dst")"
    if [[ -f "$dst" ]] && cmp -s "$src" "$dst"; then
        echo "[store] up to date: $dst"
    else
        cp "$src" "$dst"
        echo "[store] saved $dst"
    fi
}
echo "[store] Persisting hidden.pt → data/{MATH,APPS,LogiQA}/ (durable, commit these)"
store_hidden "$MATH_HIDDEN_C" "$MATH_STORE_C"
store_hidden "$MATH_HIDDEN_I" "$MATH_STORE_I"
store_hidden "$APPS_HIDDEN_C" "$APPS_STORE_C"
store_hidden "$APPS_HIDDEN_I" "$APPS_STORE_I"
store_hidden "$LOGIQA_HIDDEN_C" "$LOGIQA_STORE_C"
store_hidden "$LOGIQA_HIDDEN_I" "$LOGIQA_STORE_I"

# ------------------------------------------------------------------ #
# 4) Pool ALL boundary vectors → S_general Phase 1 (or Phase 2 if BALANCE=1)
# Prefer durable data/ copies so pooling does not depend on results/.
# ------------------------------------------------------------------ #
mkdir -p "$OUT_DIR"
COMPARE_FLAGS=()
[[ -f "$MATH_VEC" ]] && COMPARE_FLAGS+=(--compare "math=${MATH_VEC}")
[[ -f "$APPS_VEC" ]] && COMPARE_FLAGS+=(--compare "apps=${APPS_VEC}")
[[ -f "$LOGIQA_VEC" ]] && COMPARE_FLAGS+=(--compare "logic=${LOGIQA_VEC}")

BALANCE_FLAGS=()
if [[ "$BALANCE" == "1" ]]; then
    BALANCE_FLAGS=(--balance --seed "$BALANCE_SEED")
    echo "[pool] Building Phase 2 S_general (equal-domain balance, seed=$BALANCE_SEED)..."
else
    echo "[pool] Building Phase 1 S_general (no balancing)..."
fi

python build_general_vector.py \
    --domain "math=${MATH_STORE_C},${MATH_STORE_I}" \
    --domain "apps=${APPS_STORE_C},${APPS_STORE_I}" \
    --domain "logic=${LOGIQA_STORE_C},${LOGIQA_STORE_I}" \
    --layer "$STEER_LAYER" \
    --out "$OUT_VEC" \
    "${BALANCE_FLAGS[@]}" \
    "${COMPARE_FLAGS[@]}"

echo "=================================================================="
echo " DONE."
echo "   vector : $OUT_VEC"
echo "   meta   : ${OUT_VEC%.pt}.meta.json"
echo "   hidden : $MATH_STORE_C"
echo "            $MATH_STORE_I"
echo "            $APPS_STORE_C"
echo "            $APPS_STORE_I"
echo "            $LOGIQA_STORE_C"
echo "            $LOGIQA_STORE_I"
echo "   apply  : coef -1.0 at layer $STEER_LAYER"
echo "   next   : git add data/MATH/hidden_* data/APPS/hidden_* data/LogiQA/hidden_* && commit"
echo "=================================================================="
