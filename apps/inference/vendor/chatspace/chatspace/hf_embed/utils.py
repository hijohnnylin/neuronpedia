"""Utility functions for path management, checksums, and data processing."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import torch

from ..utils import (
    ensure_dir as _ensure_dir_shared,
    iso_now as _iso_now_shared,
    sanitize_component as _sanitize_component_shared,
)


def _iso_now() -> str:
    """Return current UTC timestamp in ISO format."""
    return _iso_now_shared()


def _ensure_dir(path: Path) -> None:
    """Create directory and parents if they don't exist."""
    _ensure_dir_shared(path)


def _sanitize_component(value: str) -> str:
    """Sanitize a string for use in file paths."""
    return _sanitize_component_shared(value, lowercase=False)


def _compute_sha256(path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _prepare_paths(
    output_root: Path,
    model_name: str,
    dataset: str,
    subset: Optional[str],
    split: str,
    manifest_relpath: Optional[Path] = None,
) -> dict[str, Path]:
    """Prepare output directory structure and return key paths."""
    dataset_component = _sanitize_component(dataset)
    if subset:
        dataset_component = f"{dataset_component}__{_sanitize_component(subset)}"
    split_component = _sanitize_component(split)
    model_component = _sanitize_component(model_name)

    embeddings_dir = output_root / "embeddings" / model_component / dataset_component / split_component
    indexes_dir = output_root / "indexes" / model_component / dataset_component
    cache_dir = output_root / "cache"
    logs_dir = output_root / "logs"

    for path in [embeddings_dir, indexes_dir, cache_dir, logs_dir]:
        _ensure_dir(path)

    manifest_path = manifest_relpath
    if manifest_path is None:
        manifest_path = indexes_dir / f"manifest-{split_component}.json"

    run_path = indexes_dir / f"run-{split_component}.json"

    return {
        "embeddings_dir": embeddings_dir,
        "indexes_dir": indexes_dir,
        "manifest_path": manifest_path,
        "run_path": run_path,
        "logs_dir": logs_dir,
    }


def _observe_git_sha() -> Optional[str]:
    """Get current git commit SHA, if available."""
    try:
        import subprocess

        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def _compute_norms(embedding_batch: torch.Tensor) -> torch.Tensor:
    """Compute L2 norms for a batch of embeddings."""
    return torch.linalg.vector_norm(embedding_batch, dim=1)


def _next_power_of_two(value: int) -> int:
    """Round up to next power of two."""
    if value <= 0:
        return 1
    return 1 << (value - 1).bit_length()


def _enumerate_bucket_sizes(bucket_min_tokens: int, bucket_max_tokens: int) -> list[int]:
    """Enumerate all power-of-2 bucket sizes within the given range."""
    sizes: list[int] = []
    size = _next_power_of_two(bucket_min_tokens)
    while size <= bucket_max_tokens:
        if size not in sizes:
            sizes.append(size)
        if size == bucket_max_tokens:
            break
        size *= 2

    if bucket_max_tokens not in sizes:
        sizes.append(bucket_max_tokens)

    return sorted(set(sizes))
