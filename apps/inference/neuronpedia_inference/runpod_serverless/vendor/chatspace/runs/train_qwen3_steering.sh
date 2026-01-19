#!/usr/bin/env bash
# Train steering vectors across all Qwen-3-32B persona datasets.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

export HF_HOME="${HF_HOME:-/workspace/hf-cache}"

uv run python "${ROOT_DIR}/scripts/train_all_steering.py" \
  --model Qwen/Qwen3-32B \
  --trait-prefix qwen-3-32b__trait__ \
  --role-prefix qwen-3-32b__role__ \
  --output-root /workspace/steering_runs_qwen3_layer_31 \
  --learning-rate 0.5 \
  --target-layer 31 \
  --skip-existing \
  "$@"
