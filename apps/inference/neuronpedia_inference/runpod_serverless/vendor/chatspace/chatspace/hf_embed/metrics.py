"""Statistics tracking for the embedding pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class StatsUpdate:
    """Statistics update message from worker processes."""

    rows_processed: int = 0
    rows_skipped: int = 0
    embedding_dim: Optional[int] = None
    min_norm: Optional[float] = None
    max_norm: Optional[float] = None


@dataclass
class PipelineStats:
    """Statistics about processed rows and embeddings."""

    total_rows: int = 0
    skipped_rows: int = 0
    embedding_dim: Optional[int] = None
    min_norm: Optional[float] = None
    max_norm: Optional[float] = None

    def register_rows(self, count: int) -> None:
        """Register successfully processed rows."""
        self.total_rows += count

    def register_skipped(self, count: int = 1) -> None:
        """Register skipped rows."""
        self.skipped_rows += count

    def update_embedding_dim(self, dim: int) -> None:
        """Update embedding dimension, ensuring consistency."""
        if self.embedding_dim is None:
            self.embedding_dim = dim
        elif dim is not None and self.embedding_dim != dim:
            raise ValueError(f"Inconsistent embedding dimension: expected {self.embedding_dim}, received {dim}")

    def update_norm_bounds(self, min_norm: Optional[float], max_norm: Optional[float]) -> None:
        """Update min/max norm bounds."""
        if min_norm is None or max_norm is None:
            return
        self.min_norm = min_norm if self.min_norm is None else min(self.min_norm, min_norm)
        self.max_norm = max_norm if self.max_norm is None else max(self.max_norm, max_norm)