"""Parquet shard writer and manifest generation."""

from __future__ import annotations

import json
import multiprocessing as mp
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from chatspace import __version__ as CHATSPACE_VERSION

from .config import SentenceTransformerConfig
from .utils import _compute_sha256, _iso_now


@dataclass
class EmbeddedBatch:
    """A batch of rows with computed embeddings."""

    rows: list[dict[str, Any]]
    embeddings: torch.Tensor


class _ShardWriter:
    """Accumulates embedded rows and writes Parquet shards."""

    def __init__(self, cfg: SentenceTransformerConfig, paths: dict[str, Path], git_sha: Optional[str]) -> None:
        self._cfg = cfg
        self._paths = paths
        self._git_sha = git_sha
        self._current_rows: list[dict[str, Any]] = []
        self._current_norms: list[float] = []
        self._shards: list[dict[str, Any]] = []
        self._shard_index = 0

    @property
    def shards(self) -> list[dict[str, Any]]:
        """Return list of shard metadata."""
        return self._shards

    def run(
        self,
        batch_queue: mp.Queue[Any],
        stop_token: Any,
        shard_metadata_queue: Optional[mp.Queue[Any]] = None,
        error_queue: Optional[mp.Queue[tuple[str, Exception]]] = None,
    ) -> None:
        """Run writer loop in background process."""
        try:
            while True:
                item = batch_queue.get()

                if item is stop_token:
                    break

                self._append_batch(item)

            self._flush_remaining()

            # Send shard metadata back to main process
            if shard_metadata_queue is not None:
                shard_metadata_queue.put(self._shards)

        except Exception as exc:
            if error_queue is not None:
                error_queue.put(("writer", exc))

    def _append_batch(self, batch: EmbeddedBatch) -> None:
        """Append a batch to the current accumulator."""
        if not batch.rows:
            return

        embeddings_cpu = batch.embeddings.detach().to(device="cpu", dtype=torch.float32)
        if embeddings_cpu.shape[0] != len(batch.rows):
            raise ValueError(
                f"Mismatch between rows ({len(batch.rows)}) and embeddings ({embeddings_cpu.shape[0]})"
            )

        embeddings_list = embeddings_cpu.tolist()
        norms_tensor = torch.linalg.vector_norm(embeddings_cpu, dim=1)
        norms_list = norms_tensor.tolist()

        for row, embedding in zip(batch.rows, embeddings_list):
            row["embedding"] = embedding

        self._current_rows.extend(batch.rows)
        self._current_norms.extend(norms_list)
        while len(self._current_rows) >= self._cfg.rows_per_shard:
            self._write_shard(self._cfg.rows_per_shard)

    def _flush_remaining(self) -> None:
        """Write any remaining rows as a final shard."""
        if self._current_rows:
            self._write_shard(len(self._current_rows))

    def _write_shard(self, rows_to_write: int) -> None:
        """Write a shard to disk and record metadata."""
        shard_rows = self._current_rows[:rows_to_write]
        shard_norms = self._current_norms[:rows_to_write]
        self._current_rows = self._current_rows[rows_to_write:]
        self._current_norms = self._current_norms[rows_to_write:]

        shard_created_at = _iso_now()
        shard_path = self._paths["embeddings_dir"] / f"shard-{self._shard_index:05d}.parquet"
        table = pa.Table.from_pylist(shard_rows)
        pq.write_table(table, shard_path)
        file_size = shard_path.stat().st_size
        checksum = _compute_sha256(shard_path)

        if not shard_norms:
            raise ValueError("Cannot write shard with empty norms list")
        if not shard_rows or shard_rows[0].get("embedding") is None:
            raise ValueError("Cannot write shard with missing embeddings")
        shard_min_norm = min(shard_norms)
        shard_max_norm = max(shard_norms)
        embedding_dim = len(shard_rows[0]["embedding"])

        self._shards.append(
            {
                "path": str(shard_path),
                "rows": len(shard_rows),
                "bytes": file_size,
                "sha256": checksum,
                "embedding_dim": embedding_dim,
                "min_norm": float(shard_min_norm),
                "max_norm": float(shard_max_norm),
                "created_at": shard_created_at,
                "shard_index": self._shard_index,
                "tool": {
                    "package": "chatspace",
                    "version": CHATSPACE_VERSION,
                    "git_sha": self._git_sha,
                },
            }
        )
        self._shard_index += 1


def _config_to_dict(cfg: SentenceTransformerConfig) -> dict[str, Any]:
    """Convert config to JSON-serializable dict."""
    data = asdict(cfg)
    data["output_root"] = str(cfg.output_root)
    if cfg.manifest_relpath is not None:
        data["manifest_relpath"] = str(cfg.manifest_relpath)
    return data


def write_manifest(
    cfg: SentenceTransformerConfig,
    paths: dict[str, Path],
    shards: list[dict[str, Any]],
    stats: Any,
    created_at: str,
    run_id: str,
    duration: float,
    git_sha: Optional[str],
) -> None:
    """Write manifest and run summary files."""
    preprocessing_note = None
    if cfg.extract_first_assistant:
        preprocessing_note = f"Extracted first assistant response from '{cfg.text_field}' field; skipped rows without assistant responses"

    manifest = {
        "dataset": cfg.dataset,
        "subset": cfg.subset,
        "split": cfg.split,
        "model": cfg.model_name,
        "source": cfg.source_label,
        "rows_total": stats.total_rows,
        "rows_skipped": stats.skipped_rows,
        "embedding_dim": stats.embedding_dim,
        "rows_per_shard": cfg.rows_per_shard,
        "shards": shards,
        "created_at": created_at,
        "run_id": run_id,
        "min_norm": stats.min_norm,
        "max_norm": stats.max_norm,
        "preprocessing": preprocessing_note,
        "run_config": _config_to_dict(cfg),
    }

    manifest_path = paths["manifest_path"]
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    run_summary = {
        "dataset": cfg.dataset,
        "subset": cfg.subset,
        "split": cfg.split,
        "model": cfg.model_name,
        "created_at": created_at,
        "run_id": run_id,
        "duration_seconds": duration,
        "rows_total": stats.total_rows,
        "rows_skipped": stats.skipped_rows,
        "num_shards": len(shards),
        "embedding_dim": stats.embedding_dim,
        "git_sha": git_sha,
        "manifest_path": str(manifest_path),
        "min_norm": stats.min_norm,
        "max_norm": stats.max_norm,
        "tool_version": CHATSPACE_VERSION,
    }

    run_path = paths["run_path"]
    with run_path.open("w", encoding="utf-8") as fh:
        json.dump(run_summary, fh, indent=2)