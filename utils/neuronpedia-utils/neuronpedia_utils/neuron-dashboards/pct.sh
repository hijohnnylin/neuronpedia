#!/bin/bash
# After dash.py ends: build the neuron percentile table and upload it to R2.
# Runs beside post.sh; done long before finalize ends and the pod stops.
# Usage: pct.sh MODEL RUN_LOG
MODEL=$1
RUN_LOG=$2
OUT=/workspace/nd/$MODEL
LOG=/workspace/nd/${MODEL}_pct.log
. /root/env.sh
cd /root
echo "$(date) waiting for dash.py" >> "$LOG"
while pgrep -f "dash.py --model $MODEL" > /dev/null; do sleep 60; done
grep -q "DONE" "$RUN_LOG" || { echo "$(date) dash.py did not finish" >> "$LOG"; exit 1; }
TABLE=$OUT/neuron_percentiles.safetensors
python build_percentiles.py "$OUT/raw" "$TABLE" >> "$LOG" 2>&1 || { echo "$(date) build failed" >> "$LOG"; exit 1; }
python -c "
import finalize
finalize.upload(finalize.r2(), '$TABLE', '$MODEL/neuron_percentiles.safetensors')
" >> "$LOG" 2>&1 || { echo "$(date) upload failed" >> "$LOG"; exit 1; }
echo "$(date) PCT DONE" >> "$LOG"
