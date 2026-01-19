#!/usr/bin/env bash
#
# Launch viewer for existing UMAP reduction
#

set -euo pipefail

UMAP_FILE="/workspace/visualizations/Qwen__Qwen3-Embedding-0.6B/wildchat-fineweb-tsne-2d.parquet"

echo "==> Launching interactive viewer..."
echo "    Open your browser to http://127.0.0.1:8050"
echo ""

uv run chatspace visualize-embeddings \
  --input "${UMAP_FILE}" \
  --host 0.0.0.0 \
  --port 8050
