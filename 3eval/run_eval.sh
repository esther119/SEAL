#!/usr/bin/env bash
# Run the 3-benchmark eval (MBPP / GSM8K / LogiQA 2.0) of a steering vector on RunPod.
# The vector is an INPUT — point VECTOR at your .pt:
#   VECTOR=/workspace/vectors/mbpp_v_code_steervec.pt bash run_eval.sh
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$DIR"
[[ -f .env ]] && { set -a; source .env; set +a; }
: "${HF_HOME:=/workspace/hf_cache}"; export HF_HOME
[[ -n "${HF_TOKEN:-}" ]] && export HF_TOKEN

: "${VECTOR:?set VECTOR=/path/to/steervec.pt (the input steering vector, SEAL: S = H_RT - H_E)}"
: "${N:=100}"
: "${COEF:=-1.0}"         # SEAL vector = H_RT - H_E; coef -1.0 pushes toward execution
: "${LAYER:=20}"
: "${MAX_TOKENS:=3072}"
: "${BATCH_SIZE:=16}"     # batched decode (the real speedup); A100-80GB handles 16 easily
: "${BENCHMARKS:=mbpp gsm8k logiqa}"

echo "=================================================================="
echo " 3-eval | vector=$VECTOR | n=$N coef=$COEF layer=$LAYER max_tokens=$MAX_TOKENS batch=$BATCH_SIZE"
echo " benchmarks=$BENCHMARKS"
echo "=================================================================="

python -u run_eval.py --vector "$VECTOR" --n "$N" --coef "$COEF" --layer "$LAYER" \
    --max_tokens "$MAX_TOKENS" --batch_size "$BATCH_SIZE" --benchmarks $BENCHMARKS 2>&1 | tee eval_run.log

echo "== done: results/summary.json (+ per-benchmark json) =="
