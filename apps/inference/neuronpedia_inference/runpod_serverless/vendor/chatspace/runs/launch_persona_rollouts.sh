#!/usr/bin/env bash
# Launch the persona rollout scheduler with the vector-backed dataset lists.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
LOG_DIR="/workspace/logs"
TRAITS_FILE="/workspace/steering_runs_scheduler_prod_acc1/traits_with_vectors.txt"
ROLES_FILE="/workspace/steering_runs_scheduler_prod_acc1/roles_with_vectors.txt"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_path="${LOG_DIR}/rollout_scheduler_prod_acc1_${timestamp}.log"

mkdir -p "${LOG_DIR}"
{
  printf '[%s] rollout scheduler starting\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "${ROOT_DIR}/scripts/rollout_scheduler_run.sh" \
    --traits-file "${TRAITS_FILE}" \
    --roles-file "${ROLES_FILE}" \
    --include-roles \
    --avoid-gpu0 \
    --num-gpus 7 \
    --run-root /workspace/steering_runs_scheduler_prod_acc1 \
    --output-root /workspace/steering_rollouts_prod_acc1 \
    --model Qwen/Qwen3-32B \
    --target-layer 31 \
    --steering-no-system \
    --skip-existing \
    --rollouts 4 \
    --steering-scales 200 400 800 \
    --trained-scales 200 400 800 \
    --activation-scales 200 400 800 \
    --activation-match-learned \
    --minilm-eval
} |& tee -a "${log_path}"

echo "Logs written to ${log_path}"
