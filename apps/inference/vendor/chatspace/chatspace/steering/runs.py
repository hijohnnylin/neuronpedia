"""Helpers for discovering completed steering runs and artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

from .constants import SUMMARY_FILENAME
from ..utils import sanitize_component


@dataclass(frozen=True)
class RunAttempt:
    """Represents a single steering run attempt written by the scheduler."""

    dataset: str
    path: Path
    run_id: Optional[str]
    attempt: Optional[int]
    ended_at: Optional[str]


def iter_run_attempts(run_root: Path) -> Iterator[RunAttempt]:
    """Yield run attempts discovered via summary.json files."""
    run_root = run_root.expanduser()
    for summary_path in sorted(run_root.rglob(SUMMARY_FILENAME)):
        try:
            payload = json.loads(summary_path.read_text())
        except Exception:
            continue
        dataset = payload.get("dataset")
        if not dataset:
            continue
        yield RunAttempt(
            dataset=dataset,
            path=summary_path.parent,
            run_id=payload.get("run_id"),
            attempt=payload.get("attempt"),
            ended_at=payload.get("ended_at"),
        )


def _dataset_from_path(path: Path) -> Optional[str]:
    for part in path.parts:
        if "__role__" in part or "__trait__" in part:
            return part
    # Legacy layouts may use the directory name directly.
    candidate = path.name
    if "__role__" in candidate or "__trait__" in candidate:
        return candidate
    return None


def collect_run_dirs(run_root: Path) -> Dict[str, Path]:
    """Return the most recent artifact directory for each dataset."""
    run_root = run_root.expanduser()
    best: dict[str, tuple[tuple[str, float], Path]] = {}

    def _update(dataset: str, path: Path, ended_at: Optional[str]) -> None:
        key = (ended_at or "", path.stat().st_mtime)
        current = best.get(dataset)
        if current is None or key > current[0]:
            best[dataset] = (key, path)

    # First prefer explicit summaries written by chatspace.steering.job.
    for attempt in iter_run_attempts(run_root):
        _update(attempt.dataset, attempt.path, attempt.ended_at)

    # Fallback: direct artifact directories without summaries.
    for vec_path in run_root.rglob("steering_vector.pt"):
        dataset = _dataset_from_path(vec_path.parent)
        if not dataset:
            continue
        _update(dataset, vec_path.parent, None)

    return {dataset: path for dataset, (_, path) in best.items()}


def latest_run_dir(run_root: Path, dataset: str) -> Optional[Path]:
    """Return the latest artifact directory for a dataset, if any."""
    dataset = dataset.strip()
    if not dataset:
        return None
    mapping = collect_run_dirs(run_root)
    if dataset in mapping:
        return mapping[dataset]

    # Final fallback: direct directory named after dataset.
    direct = run_root / dataset
    if (direct / "steering_vector.pt").exists():
        return direct

    slug = sanitize_component(dataset)
    slug_dir = run_root / slug
    if (slug_dir / "steering_vector.pt").exists():
        return slug_dir
    return None


def has_successful_run(run_root: Path, dataset: str) -> bool:
    """Return True if a dataset already has a completed run."""
    dataset = dataset.strip()
    if not dataset:
        return False
    mapping = collect_run_dirs(run_root)
    if dataset in mapping:
        # Ensure the summary exists alongside the directory.
        summary_path = mapping[dataset] / SUMMARY_FILENAME
        if summary_path.exists():
            return True
    # For legacy directories, treat presence of steering_vector as success.
    direct = run_root / dataset
    if (direct / "steering_vector.pt").exists():
        return True
    slug = sanitize_component(dataset)
    if (run_root / slug / "steering_vector.pt").exists():
        return True
    return False


def list_trained_datasets(run_root: Path) -> list[str]:
    """List datasets with completed steering vectors under the run root."""
    mapping = collect_run_dirs(run_root)
    return sorted(mapping.keys())
