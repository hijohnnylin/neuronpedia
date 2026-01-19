"""Configuration dataclass for SentenceTransformer embedding pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class SentenceTransformerConfig:
    """Configuration for embedding datasets with SentenceTransformer models."""

    dataset: str
    subset: Optional[str] = None
    split: str = "train"
    text_field: str = "text"
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    batch_size: int = 32
    rows_per_shard: int = 8192
    max_rows: Optional[int] = None
    output_root: Path = Path("/workspace")
    manifest_relpath: Optional[Path] = None
    seed: Optional[int] = None
    dtype: Optional[str] = "bfloat16"
    device: Optional[str] = None
    attention_impl: Optional[str] = "flash_attention_2"
    tokenizer_padding: str = "left"
    trust_remote_code: bool = True
    num_workers: int = 1
    prefetch_batches: int = 4
    bucket_min_tokens: int = 128
    bucket_max_tokens: int = 32768
    tokens_per_batch: Optional[int] = None
    compile_model: bool = False
    compile_mode: Optional[str] = "default"
    progress: bool = True
    source_label: str = "huggingface"
    run_id: Optional[str] = None
    max_rows_per_file: Optional[int] = None
    resume: bool = False
    extract_first_assistant: bool = False
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    tokenizer_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.rows_per_shard <= 0:
            raise ValueError("rows_per_shard must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.prefetch_batches <= 0:
            raise ValueError("prefetch_batches must be positive")
        if self.bucket_min_tokens <= 0:
            raise ValueError("bucket_min_tokens must be positive")
        if self.bucket_max_tokens <= 0:
            raise ValueError("bucket_max_tokens must be positive")
        if self.bucket_min_tokens > self.bucket_max_tokens:
            raise ValueError("bucket_min_tokens must be less than or equal to bucket_max_tokens")
        if self.tokens_per_batch is not None and self.tokens_per_batch <= 0:
            raise ValueError("tokens_per_batch must be positive if provided")
        if self.max_rows is not None and self.max_rows <= 0:
            raise ValueError("max_rows must be positive if provided")
