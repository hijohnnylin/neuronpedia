#!/usr/bin/env bash
#
# Example: Compute tsne reduction and launch interactive viewer
# for WildChat-4.8M and FineWeb-10BT embeddings
#

set -euo pipefail

MODEL="Qwen__Qwen3-Embedding-0.6B"
WILDCHAT_DIR="/workspace/embeddings/${MODEL}/allenai__WildChat-4.8M/train"
FINEWEB_DIR="/workspace/embeddings/${MODEL}/HuggingFaceFW__fineweb__sample-10BT/train"

OUTPUT_DIR="/workspace/visualizations/${MODEL}"
OUTPUT_FILE="${OUTPUT_DIR}/wildchat-fineweb-tsne-2d.parquet"

mkdir -p "${OUTPUT_DIR}"

echo "==> Computing tsne reduction (this may take a while for large datasets)..."
uv run chatspace reduce-embeddings \
  --embedding-dirs "${WILDCHAT_DIR}" "${FINEWEB_DIR}" \
  --output "${OUTPUT_FILE}" \
  --method tsne \
  --n-components 2 \
  --max-samples 50000 \
  --n-neighbors 15 \
  --min-dist 0.1 \
  --metric cosine \
  --seed 42

echo ""
echo "==> Reduction complete! Saved to ${OUTPUT_FILE}"
echo ""
echo "==> Launching interactive viewer..."
echo "    Open your browser to http://127.0.0.1:8050"
echo ""

uv run chatspace visualize-embeddings \
  --input "${OUTPUT_FILE}" \
  --host 0.0.0.0 \
  --port 8050
