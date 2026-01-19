"""Dataset loading and conversation extraction utilities."""

from __future__ import annotations

import logging
from typing import Any, Iterator, Optional

from datasets import IterableDataset, load_dataset

from .config import SentenceTransformerConfig


def _load_dataset(cfg: SentenceTransformerConfig) -> IterableDataset:
    """Load dataset from HuggingFace in streaming mode."""
    dataset_kwargs: dict[str, Any] = {}
    if cfg.subset:
        dataset_kwargs["name"] = cfg.subset
    logging.info("Loading dataset %s subset=%s split=%s (streaming)", cfg.dataset, cfg.subset, cfg.split)
    return load_dataset(cfg.dataset, split=cfg.split, streaming=True, **dataset_kwargs)


def _extract_first_assistant_response(conversation: list[dict[str, Any]]) -> Optional[str]:
    """Extract the first assistant response from a conversation list.

    Args:
        conversation: List of conversation turns with 'role' and 'content' fields

    Returns:
        The content of the first assistant response, or None if not found

    Raises:
        TypeError: If conversation is not a list
    """
    if not isinstance(conversation, list):
        raise TypeError(f"Expected conversation to be a list, got {type(conversation).__name__}")

    for turn in conversation:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role", "").lower()
        if role in ("assistant", "model"):
            content = turn.get("content")
            if content:
                return str(content).strip()

    return None


def _rows_from_dataset(ds: IterableDataset, cfg: SentenceTransformerConfig) -> Iterator[dict[str, Any]]:
    """Stream rows from dataset, optionally extracting assistant responses.

    Args:
        ds: Iterable dataset to process
        cfg: Configuration with text field and extraction settings

    Yields:
        Dictionary rows, potentially with extracted text
    """
    for idx, row in enumerate(ds):
        if cfg.max_rows is not None and idx >= cfg.max_rows:
            break
        if cfg.seed is not None:
            # Deterministic hashing to decide keep/drop could be added here; for now no-op.
            pass

        row_dict = dict(row)

        # If extract_first_assistant is enabled, extract from conversation field
        if cfg.extract_first_assistant and cfg.text_field in row_dict:
            conversation = row_dict.get(cfg.text_field)
            assistant_text = _extract_first_assistant_response(conversation)
            if assistant_text:
                # Store original conversation in metadata, replace text_field with extracted text
                row_dict[f"_original_{cfg.text_field}"] = conversation
                row_dict[cfg.text_field] = assistant_text
            else:
                # Skip rows without assistant responses
                continue

        yield row_dict