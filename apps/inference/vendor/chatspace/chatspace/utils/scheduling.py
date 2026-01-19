"""Shared helpers for planning strided work across GPUs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

try:
    import torch
except Exception:
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class WorkerPlan:
    """Dataset allocation for a single worker."""

    index: int
    datasets: List[str]


def assign_by_stride(datasets: Sequence[str], worker_count: int) -> list[WorkerPlan]:
    """Assign datasets to workers using modulo stride selection."""
    if worker_count < 1:
        raise ValueError("worker_count must be >= 1")
    plans: list[WorkerPlan] = []
    for worker_idx in range(worker_count):
        subset = [dataset for idx, dataset in enumerate(datasets) if idx % worker_count == worker_idx]
        if subset:
            plans.append(WorkerPlan(index=worker_idx, datasets=subset))
    return plans


def detect_available_gpus(exclude: Iterable[int] | None = None, limit: int | None = None) -> list[int]:
    """Query torch for visible CUDA devices, applying exclusion/limit filters."""
    exclude_set = set(int(x) for x in (exclude or []))
    if torch is None or not torch.cuda.is_available():
        return []
    device_count = torch.cuda.device_count()
    visible = [idx for idx in range(device_count) if idx not in exclude_set]
    if limit is not None and limit >= 0:
        visible = visible[:limit]
    return visible
