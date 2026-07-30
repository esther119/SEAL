#!/usr/bin/env bash
# Rebuild S_logic from FINISHED traces only, using the relabelled build set.
#
# The shipped vector was extracted from pools labelled by extract_choice_legacy,
# whose fallback assigns a letter to generations that never closed </think>.
# 152/500 of its "correct" pool and 234/500 of its "incorrect" pool had never
# stated an answer -- non-terminating spirals counted as reasoning outcomes.
#
# Prereq (CPU, already run):
#   python scripts/relabel_logiqa_build.py \
#       --build_dir <...>/baseline_3000 --out_dir <...>/baseline_3000_finished
#
# Generation is NOT repeated -- the same 2,000 traces are reused, only relabelled.
set -euo pipefail
gpu=${1:-0}
: "${MODEL:=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
: "${SAMPLE:=500}"
: "${KEYWORDS:=logic}"
TAG=$(basename "$MODEL")
DIR="results/results_for_logic_vectors/LogiQA_train/${TAG}/baseline_3000_finished"

if [[ ! -f "$DIR/math_eval.jsonl" ]]; then
    echo "Missing $DIR — run scripts/relabel_logiqa_build.py first." >&2
    exit 2
fi

python - "$DIR" "$SAMPLE" <<'PY'
import json, sys
d, need = sys.argv[1], int(sys.argv[2])
rows = [json.loads(l) for l in open(f"{d}/math_eval.jsonl")]
c = sum(1 for r in rows if r["all_eval"][0]); i = len(rows) - c
print(f"[check] finished traces {len(rows)} | correct {c} | incorrect {i}")
if c < need or i < need:
    raise SystemExit(f"[check] cannot fill {need}+{need}: correct {c}, incorrect {i}")
print(f"[check] OK to build {need}+{need}")
PY

echo "[1/3] hidden states — incorrect (layer 20, finished only) ..."
CUDA_VISIBLE_DEVICES=$gpu python -u hidden_analysis.py \
    --model_path "$MODEL" --data_path "$DIR/data.jsonl" --data_dir "$DIR" \
    --type incorrect --start 0 --sample "$SAMPLE" --keep_layers 20 --keywords "$KEYWORDS"

echo "[2/3] hidden states — correct (layer 20, finished only) ..."
CUDA_VISIBLE_DEVICES=$gpu python -u hidden_analysis.py \
    --model_path "$MODEL" --data_path "$DIR/data.jsonl" --data_dir "$DIR" \
    --type correct --start 0 --sample "$SAMPLE" --keep_layers 20 --keywords "$KEYWORDS"

echo "[3/3] build steering vector (layer 20) ..."
python -u vector_generation.py \
    --data_dir "$DIR" --prefixs "correct_0_${SAMPLE}" "incorrect_0_${SAMPLE}" \
    --layers 20 --save_prefix "${KEYWORDS}_finished_${SAMPLE}_${SAMPLE}" --overwrite

NEW="$DIR/vector_${KEYWORDS}_finished_${SAMPLE}_${SAMPLE}/layer_20_transition_reflection_steervec.pt"
OLD="results/results_for_logic_vectors/LogiQA_train/${TAG}/baseline_3000/vector_logic_500_500/layer_20_transition_reflection_steervec.pt"
echo "== done: $NEW =="

python - "$NEW" "$OLD" <<'PY'
import sys, torch, torch.nn.functional as F
new = torch.load(sys.argv[1], weights_only=True).float()
print(f"[new] dim {tuple(new.shape)} norm {new.norm():.3f}")
try:
    old = torch.load(sys.argv[2], weights_only=True).float()
    cos = F.cosine_similarity(new.flatten(), old.flatten(), dim=0)
    print(f"[old] norm {old.norm():.3f}")
    print(f"cos(S_logic_finished, S_logic_original) = {cos:.4f}")
    print("  -> contamination was cosmetic" if cos > 0.99 else
          "  -> vector materially changed; the shipped one should be replaced")
except FileNotFoundError:
    print("(original vector not present for comparison)")
PY
