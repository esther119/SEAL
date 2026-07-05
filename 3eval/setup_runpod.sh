#!/usr/bin/env bash
# One-time env setup for the 3-benchmark eval on a RunPod GPU pod (PyTorch CUDA image).
#   cp .env.example .env   # optional HF_TOKEN
#   bash setup_runpod.sh
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$DIR"
[[ -f .env ]] && { set -a; source .env; set +a; echo "[setup] loaded .env"; }
: "${HF_HOME:=/workspace/hf_cache}"; export HF_HOME; mkdir -p "$HF_HOME"
echo "[setup] HF_HOME=$HF_HOME"

pip install -q --upgrade pip
pip install -q -r requirements.txt
[[ -n "${HF_TOKEN:-}" ]] && python - <<'PY'
import os; from huggingface_hub import login; login(token=os.environ["HF_TOKEN"], add_to_git_credential=False); print("[setup] HF login OK")
PY

python - <<'PY'
import torch, transformers, datasets
assert torch.cuda.is_available(), "no CUDA — pick a GPU pod / PyTorch base image"
print(f"[setup] torch {torch.__version__} CUDA ({torch.cuda.get_device_name(0)})")
print(f"[setup] transformers {transformers.__version__} datasets {datasets.__version__}")
import steering, benchmarks
print(f"[setup] benchmarks: {list(benchmarks.BENCHMARKS)} | steering OK")
PY
echo "[setup] done -> VECTOR=/path/to/steervec.pt bash run_eval.sh"
