"""HuggingFace dataset embedding pipeline using SentenceTransformer models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import SentenceTransformerConfig

if TYPE_CHECKING:
    from .pipeline import run_sentence_transformer

__all__ = ["SentenceTransformerConfig", "run_sentence_transformer"]


def __getattr__(name: str):
    """Lazy import to avoid loading heavy dependencies at import time."""
    if name == "run_sentence_transformer":
        from .pipeline import run_sentence_transformer
        return run_sentence_transformer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")