#!/usr/bin/env bash
# Build Phase 1 S_general from MATH + APPS (unbalanced pool of all boundary vectors).
#
# In-repo layout (no external/v_code-SEAL required):
#   data/APPS/baseline_10000/   shipped APPS traces (math_eval.jsonl.gz + data.jsonl)
#   vectors/apps_v_code.pt      shipped code-domain home vector (cosine smoke only)
#   apps/                       APPS generation + scoring (for future re-gen)
#   hidden_analysis.py          ONE extractor — --keywords math|code
#
# Usage:
#   bash scripts/build_general_vector.sh            # full Phase 1 (GPU)
#   SKIP_HIDDEN=1 bash scripts/build_general_vector.sh   # pool only
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

export HF_HOME CUDA_VISIBLE_DEVICES="$GPU"
[[ -n "${HF_TOKEN:-}" ]] && export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

MODEL_TAG="$(basename "$MODEL")"
MATH_DIR="results/results_for_math_vectors/MATH_train/${MODEL_TAG}/baseline_10000"
MATH_DATA="data/MATH/train.jsonl"
APPS_SHIPPED="data/APPS/baseline_10000"
APPS_DIR="results/APPS_train/${MODEL_TAG}/baseline_10000"
APPS_DATA="${APPS_DIR}/data.jsonl"

OUT_DIR="results/general"
OUT_VEC="${OUT_DIR}/S_general_math_apps_phase1.pt"
MATH_VEC="${MATH_DIR}/vector_${VEC_SAMPLES}_${VEC_SAMPLES}/layer_${STEER_LAYER}_transition_reflection_steervec.pt"
APPS_VEC="vectors/apps_v_code.pt"

MATH_HIDDEN_C="${MATH_DIR}/hidden_correct_0_${VEC_SAMPLES}/hidden.pt"
MATH_HIDDEN_I="${MATH_DIR}/hidden_incorrect_0_${VEC_SAMPLES}/hidden.pt"
APPS_HIDDEN_C="${APPS_DIR}/hidden_correct_0_${VEC_SAMPLES}/hidden.pt"
APPS_HIDDEN_I="${APPS_DIR}/hidden_incorrect_0_${VEC_SAMPLES}/hidden.pt"

echo "=================================================================="
echo " S_general Phase 1 · MATH + APPS (unbalanced pool)"
echo " MODEL=$MODEL  GPU=$GPU  LAYER=$STEER_LAYER  SAMPLES=$VEC_SAMPLES"
echo " MATH_DIR=$MATH_DIR"
echo " APPS_DIR=$APPS_DIR"
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
[[ -f "$APPS_VEC" ]] || echo "WARN: missing $APPS_VEC — cosine compare vs apps will be skipped."

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

for f in "$MATH_HIDDEN_C" "$MATH_HIDDEN_I" "$APPS_HIDDEN_C" "$APPS_HIDDEN_I"; do
    [[ -f "$f" ]] || { echo "ERROR: missing $f — run without SKIP_HIDDEN on a GPU pod."; exit 1; }
done

# ------------------------------------------------------------------ #
# 3) Pool ALL boundary vectors → S_general Phase 1
# ------------------------------------------------------------------ #
mkdir -p "$OUT_DIR"
COMPARE_FLAGS=()
[[ -f "$MATH_VEC" ]] && COMPARE_FLAGS+=(--compare "math=${MATH_VEC}")
[[ -f "$APPS_VEC" ]] && COMPARE_FLAGS+=(--compare "apps=${APPS_VEC}")

echo "[pool] Building Phase 1 S_general (no balancing)..."
python build_general_vector.py \
    --domain "math=${MATH_HIDDEN_C},${MATH_HIDDEN_I}" \
    --domain "apps=${APPS_HIDDEN_C},${APPS_HIDDEN_I}" \
    --layer "$STEER_LAYER" \
    --out "$OUT_VEC" \
    "${COMPARE_FLAGS[@]}"

echo "=================================================================="
echo " DONE."
echo "   vector : $OUT_VEC"
echo "   meta   : ${OUT_VEC%.pt}.meta.json"
echo "   apply  : coef -1.0 at layer $STEER_LAYER"
echo "=================================================================="
