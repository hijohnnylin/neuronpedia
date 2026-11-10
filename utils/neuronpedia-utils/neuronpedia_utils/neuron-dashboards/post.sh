#!/bin/bash
# After dash.py ends: finalize, upload, check R2, write .idx files, then stop this pod.
# Usage: post.sh MODEL RUN_LOG
MODEL=$1
RUN_LOG=$2
OUT=/workspace/nd/$MODEL
LOG=/workspace/nd/${MODEL}_post.log
. /root/env.sh
cd /root
echo "$(date) waiting for dash.py" >> "$LOG"
while pgrep -f "dash.py --model $MODEL" > /dev/null; do sleep 60; done
if ! grep -q "DONE" "$RUN_LOG"; then
  echo "$(date) dash.py did not finish; pod stays up" >> "$LOG"
  exit 1
fi
echo "$(date) finalize" >> "$LOG"
python finalize.py --run "$OUT" --corpus /workspace/corpus --workers 24 >> "$LOG" 2>&1 || { echo "$(date) finalize failed" >> "$LOG"; exit 1; }
python verify_r2.py "$MODEL" "$OUT" /workspace/corpus >> "$LOG" 2>&1 || { echo "$(date) verify failed" >> "$LOG"; exit 1; }
python build_idx.py "$MODEL" >> "$LOG" 2>&1 || { echo "$(date) build_idx failed" >> "$LOG"; exit 1; }
echo "$(date) ALL DONE; stopping pod" >> "$LOG"
touch /workspace/nd/ALL_DONE
export RUNPOD_API_KEY=$(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD_API_KEY=' | cut -d= -f2-)
POD=$(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD_POD_ID=' | cut -d= -f2-)
runpodctl stop pod "$POD" >> "$LOG" 2>&1
