#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: DATASETS=math,apps,livecodebench $0 VECTOR_PATH VECTOR_NAME [GPU_INDEX]" >&2
  exit 2
fi

VECTOR_PATH=$1
VECTOR_NAME=$2
GPU_INDEX=${3:-0}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

VECTOR_NAME="$VECTOR_NAME" exec \
  "$SCRIPT_DIR/eval_math_vector_apps_math.sh" "$VECTOR_PATH" "$GPU_INDEX"
