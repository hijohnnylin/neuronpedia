"""Shared path and configuration constants for workspace tooling."""

from __future__ import annotations

from pathlib import Path

PERSONA_ROOT = Path("/workspace/persona-data")
PROCESSED_PERSONA_ROOT = Path("/workspace/datasets/processed/persona")
STEERING_RUN_ROOT = Path("/workspace/steering_runs")
STEERING_SCHEDULER_RUN_ROOT = Path("/workspace/steering_runs_scheduler")
STEERING_ROLLOUT_ROOT = Path("/workspace/steering_rollouts")
PERSONA_TRAITS_FILE = Path("/workspace/persona_traits_over_100k.txt")
PERSONA_ROLES_FILE = Path("/workspace/persona_roles_over_100k.txt")
