#!/usr/bin/env bash
# Unattended MMLU run for Vast.ai: run baseline + steered, commit + push the
# results, then STOP the instance (disk persists, GPU billing halts). Designed
# so you can launch it and go to bed.
#
# Launch inside tmux so it survives disconnects:
#
#   tmux new -s seal
#   VAST_API_KEY=xxxx VAST_INSTANCE_ID=1234567 bash scripts/run_mmlu_unattended.sh 0
#   # Ctrl+B then D to detach; close your laptop.
#
# Safety / notes:
#   - `git push` must already be authenticated on this box (you've pushed before).
#   - Your data is safe even if the push fails: this uses `stop`, never `destroy`,
#     so the disk persists and you can push manually in the morning.
#   - Omit VAST_API_KEY / VAST_INSTANCE_ID to skip auto-stop (it will still run +
#     commit + push, then leave the box up for you to stop from the dashboard).
#   - Get VAST_INSTANCE_ID from the Vast dashboard (the numeric instance id).
#
# NOT using `set -e`: we deliberately continue to the commit + stop steps even if
# a generation phase errors, so partial results are saved and the box still stops.
set -uo pipefail

gpu=${1:-0}
cd /workspace/SEAL
LOG="$PWD/mmlu_run.log"
source /venv/main/bin/activate 2>/dev/null || true

echo "[$(date)] START MMLU baseline+steered (gpu $gpu)" | tee -a "$LOG"
bash scripts/mmlu_philosophy.sh "$gpu" 2>&1 | tee -a "$LOG"
echo "[$(date)] generation finished (status ${PIPESTATUS[0]})" | tee -a "$LOG"

echo "[$(date)] committing + pushing results" | tee -a "$LOG"
git add -f results/results_for_math_vectors/MMLU mmlu_run.log 2>/dev/null
git commit -m "MMLU philosophy transfer results (unattended run)" 2>&1 | tee -a "$LOG" \
  || echo "[$(date)] nothing new to commit" | tee -a "$LOG"
git push origin HEAD 2>&1 | tee -a "$LOG" \
  || echo "[$(date)] PUSH FAILED — results are still on disk; push manually later" | tee -a "$LOG"

if [[ -n "${VAST_API_KEY:-}" && -n "${VAST_INSTANCE_ID:-}" ]]; then
  echo "[$(date)] stopping instance $VAST_INSTANCE_ID (disk persists, billing stops)" | tee -a "$LOG"
  pip install -q vastai 2>/dev/null || true
  vastai set api-key "$VAST_API_KEY" >/dev/null 2>&1
  vastai stop instance "$VAST_INSTANCE_ID" 2>&1 | tee -a "$LOG"
else
  echo "[$(date)] VAST_API_KEY/VAST_INSTANCE_ID not set — leaving box up; stop it from the dashboard." | tee -a "$LOG"
fi

echo "[$(date)] DONE" | tee -a "$LOG"
