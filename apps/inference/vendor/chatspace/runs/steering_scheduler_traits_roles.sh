#!/usr/bin/env bash
# Recorded invocation for the steering scheduler sweep over traits and roles.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

"${ROOT_DIR}/scripts/steering_scheduler_run.sh" \
  --include-roles \
  --skip-existing \
  --reuse-base-model \
  --run-root /workspace/steering_runs_scheduler \
  --trait-prefix "qwen-3-32b__trait__" \
  --role-prefix "qwen-3-32b__role__" \
  --traits-file /workspace/persona_traits_over_100k.txt \
  --roles-file /workspace/persona_roles_over_100k.txt \
  --model Qwen/Qwen3-32B \
  --target-layer 31 \
  -- \
  --device-map cuda \
  --learning-rate 0.5 \
  --lr-scheduler cosine \
  --target-tokens 100000 \
  --val-target-tokens 10000 \
  --num-epochs 5 \
  "$@" \
  # avoid gpu0 to allow concurrent interactive exploration
  # --avoid-gpu0 
  # smoke test
  # --run-root /workspace/steering_runs_scheduler_test \
  # --datasets qwen-3-32b__role__leviathan \

