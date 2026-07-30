#!/usr/bin/env bash
# Rebuild S_logic from the re-graded extraction set.
#
# The shipped-then-superseded vector was built from pools labelled by
# extract_choice_legacy, whose fallback fabricates an answer for generations that
# never closed </think>. The fix is to the GRADER, not to pool membership:
#
#   * unfinished (no </think>) -> scored INCORRECT, and KEPT in the pool
#   * no trace is excluded, matching v_math / v_code (which exclude nothing)
#   * 500 correct + 500 incorrect, first-in first-fill in file order (greedy)
#
# vector_generation pools boundaries from BOTH files into one contrast
# S = mean(check u switch) - mean(other); the correct/incorrect label is only a
# sampling control and never enters the formula.
#
# Prereq (CPU):
#   python scripts/relabel_logiqa_build.py \
#       --build_dir <...>/baseline_3000 --out_dir <...>/baseline_3000_regraded
#
# Generation is NOT repeated -- the same 2,000 traces are reused, only re-graded.
set -euo pipefail
gpu=${1:-0}
: "${MODEL:=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
: "${SAMPLE:=500}"
: "${KEYWORDS:=logic}"
: "${LAYER:=20}"
TAG=$(basename "$MODEL")
DIR="results/results_for_logic_vectors/LogiQA_train/${TAG}/baseline_3000_regraded"

if [[ ! -f "$DIR/math_eval.jsonl" ]]; then
    echo "Missing $DIR — run scripts/relabel_logiqa_build.py first." >&2
    exit 2
fi

python - "$DIR" "$SAMPLE" <<'PY'
import json, sys
d, need = sys.argv[1], int(sys.argv[2])
rows = [json.loads(l) for l in open(f"{d}/math_eval.jsonl")]
c = sum(1 for r in rows if r["all_eval"][0]); i = len(rows) - c
unf = sum(1 for r in rows if r.get("unfinished"))
first_i = [r for r in rows if not r["all_eval"][0]][:need]
print(f"[check] traces {len(rows)} (none dropped) | correct {c} | incorrect {i} (unfinished {unf})")
print(f"[check] first-{need} incorrect pool: {sum(1 for r in first_i if r.get('unfinished'))} unfinished, "
      f"{sum(1 for r in first_i if not r.get('unfinished'))} answered-wrong")
if c < need or i < need:
    raise SystemExit(f"[check] cannot fill {need}+{need}: correct {c}, incorrect {i}")
print(f"[check] OK to build {need}+{need}")
PY

echo "[1/3] hidden states — incorrect (layer $LAYER, includes unfinished) ..."
CUDA_VISIBLE_DEVICES=$gpu python -u hidden_analysis.py \
    --model_path "$MODEL" --data_path "$DIR/data.jsonl" --data_dir "$DIR" \
    --type incorrect --start 0 --sample "$SAMPLE" --keep_layers "$LAYER" --keywords "$KEYWORDS"

echo "[2/3] hidden states — correct (layer $LAYER) ..."
CUDA_VISIBLE_DEVICES=$gpu python -u hidden_analysis.py \
    --model_path "$MODEL" --data_path "$DIR/data.jsonl" --data_dir "$DIR" \
    --type correct --start 0 --sample "$SAMPLE" --keep_layers "$LAYER" --keywords "$KEYWORDS"

echo "[3/3] build steering vector (layer $LAYER) ..."
python -u vector_generation.py \
    --data_dir "$DIR" --prefixs "correct_0_${SAMPLE}" "incorrect_0_${SAMPLE}" \
    --layers "$LAYER" --save_prefix "${KEYWORDS}_${SAMPLE}_${SAMPLE}" --overwrite

NEW="$DIR/vector_${KEYWORDS}_${SAMPLE}_${SAMPLE}/layer_${LAYER}_transition_reflection_steervec.pt"
echo "== done: $NEW =="

python - "$NEW" vectors/logiqa_v_logic.pt "$DIR" "$LAYER" <<'PY'
import sys, torch, torch.nn.functional as F
new = torch.load(sys.argv[1], weights_only=True).float()
print(f"[new] dim {tuple(new.shape)} norm {new.norm():.3f} finite {bool(torch.isfinite(new).all())}")
try:
    prev = torch.load(sys.argv[2], weights_only=True).float()
    print(f"[prev shipped] norm {prev.norm():.3f}  cos {F.cosine_similarity(new.flatten(), prev.flatten(), dim=0):.4f}")
except Exception as e:
    print(f"(no comparison: {e})")
d, layer = sys.argv[3], int(sys.argv[4])
tot = {"check": 0, "switch": 0, "other": 0}
for kind in ("correct", "incorrect"):
    data = torch.load(f"{d}/hidden_{kind}_0_500/hidden.pt", weights_only=False)[layer]
    c = s = o = 0
    for k in data:
        h = data[k]; ci = len(h["check_index"]); si = len(h["switch_index"])
        c += ci; s += si; o += h["step"].shape[0] - ci - si
    print(f"[pool] {kind:<9} traces {len(data)} | boundaries {c+s+o} | check {c} switch {s} other {o}")
    tot["check"] += c; tot["switch"] += s; tot["other"] += o
T = sum(tot.values())
print(f"[pool] TOTAL {T} | check {tot['check']} ({tot['check']/T*100:.1f}%) switch {tot['switch']} other {tot['other']}")
PY
