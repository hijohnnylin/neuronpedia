# used to benchmark how graph generation speed based on number of tokens and batch size

#!/bin/bash

URL="http://localhost:5004/generate-graph"
MODEL_ID="google/gemma-2-2b"
MAX_N_LOGITS=10
DESIRED_LOGIT_PROB=0.95
NODE_THRESHOLD=0.8
EDGE_THRESHOLD=0.98

DIGIT_COUNTS=(24 256)
BATCH_SIZES=(48 96)

echo "digits,batch_size,seconds" > output.csv
echo "Results will be printed to the console and saved to output.csv"

for batch in "${BATCH_SIZES[@]}"; do
  for digits in "${DIGIT_COUNTS[@]}"; do
    prompt=$(yes 123456789 | tr -d '\n' | head -c $((digits - 1)))
    start=$(date +%s.%N)
    curl -s -X POST "$URL" \
      -H "Content-Type: application/json" \
      -H "x-secret-key: $SECRET" \
      -d '{
        "prompt": "'"$prompt"'",
        "model_id": "'"$MODEL_ID"'",
        "batch_size": '"$batch"',
        "max_n_logits": '"$MAX_N_LOGITS"',
        "desired_logit_prob": '"$DESIRED_LOGIT_PROB"',
        "node_threshold": '"$NODE_THRESHOLD"',
        "edge_threshold": '"$EDGE_THRESHOLD"',
        "slug_identifier": "count-'"$digits"'-'"$batch"'"
      }' > /dev/null
    end=$(date +%s.%N)
    elapsed=$(echo "$end - $start" | bc)
    echo "$digits,$batch,$elapsed" >> benchmark.csv
    echo "Digits: $digits, Batch Size: $batch, Time: $elapsed seconds"
  done
done