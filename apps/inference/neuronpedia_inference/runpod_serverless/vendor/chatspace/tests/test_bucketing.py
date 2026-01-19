"""Tests for chatspace.hf_embed.bucketing module."""

from pathlib import Path

import pytest
import torch

from chatspace.hf_embed.bucketing import (
    TokenBatch,
    _BucketBuffer,
    _select_bucket_size,
    _token_sequence_length,
    _pad_and_stack_tokens,
    _effective_batch_size,
)
from chatspace.hf_embed.config import SentenceTransformerConfig


def test_token_batch_creation():
    """Test TokenBatch dataclass."""
    rows = [{"text": "hello"}, {"text": "world"}]
    features = {"input_ids": torch.tensor([[1, 2, 3]])}
    batch = TokenBatch(rows=rows, features=features, bucket_size=128)

    assert batch.rows == rows
    assert "input_ids" in batch.features
    assert batch.bucket_size == 128


def test_bucket_buffer_initialization():
    """Test BucketBuffer initialization."""
    buffer = _BucketBuffer(bucket_size=256)
    assert buffer.bucket_size == 256
    assert len(buffer) == 0
    assert len(buffer.rows) == 0


def test_bucket_buffer_add():
    """Test adding items to buffer."""
    buffer = _BucketBuffer(bucket_size=256)
    row = {"text": "test"}
    tokens = {"input_ids": torch.tensor([1, 2, 3]), "attention_mask": torch.tensor([1, 1, 1])}

    buffer.add(row, tokens)
    assert len(buffer) == 1
    assert buffer.rows[0] == row


def test_bucket_buffer_pop():
    """Test popping items from buffer."""
    buffer = _BucketBuffer(bucket_size=4)
    pad_values = {"input_ids": 0, "attention_mask": 0}

    # Add 3 items
    for i in range(3):
        buffer.add(
            {"text": f"item{i}"},
            {"input_ids": torch.tensor([1, 2]), "attention_mask": torch.tensor([1, 1])},
        )

    # Pop 2 items
    batch = buffer.pop(2, pad_values)
    assert batch is not None
    assert len(batch.rows) == 2
    assert len(buffer) == 1

    # Pop remaining
    batch = buffer.pop(1, pad_values)
    assert batch is not None
    assert len(batch.rows) == 1
    assert len(buffer) == 0


def test_bucket_buffer_flush():
    """Test flushing buffer."""
    buffer = _BucketBuffer(bucket_size=4)
    pad_values = {"input_ids": 0, "attention_mask": 0}

    # Add items
    for i in range(3):
        buffer.add(
            {"text": f"item{i}"},
            {"input_ids": torch.tensor([1, 2]), "attention_mask": torch.tensor([1, 1])},
        )

    batch = buffer.flush(pad_values)
    assert batch is not None
    assert len(batch.rows) == 3
    assert len(buffer) == 0


def test_select_bucket_size():
    """Test bucket size selection."""
    cfg = SentenceTransformerConfig(
        dataset="test",
        bucket_min_tokens=128,
        bucket_max_tokens=1024,
    )

    # Below min - should round up to min
    assert _select_bucket_size(64, cfg) == 128
    assert _select_bucket_size(100, cfg) == 128

    # Within range - should round to next power of 2
    assert _select_bucket_size(128, cfg) == 128
    assert _select_bucket_size(200, cfg) == 256
    assert _select_bucket_size(256, cfg) == 256
    assert _select_bucket_size(500, cfg) == 512

    # Above max - should cap at max
    assert _select_bucket_size(1024, cfg) == 1024
    assert _select_bucket_size(2000, cfg) == 1024


def test_token_sequence_length():
    """Test sequence length extraction from tokens."""
    # With attention mask
    tokens = {
        "input_ids": torch.tensor([1, 2, 3, 0, 0]),
        "attention_mask": torch.tensor([1, 1, 1, 0, 0]),
    }
    assert _token_sequence_length(tokens) == 3

    # Without attention mask, use input_ids length
    tokens = {"input_ids": torch.tensor([1, 2, 3, 4])}
    assert _token_sequence_length(tokens) == 4

    # Other tensors
    tokens = {"embeddings": torch.tensor([1.0, 2.0, 3.0])}
    assert _token_sequence_length(tokens) == 3


def test_pad_and_stack_tokens():
    """Test padding and stacking."""
    bucket_size = 8
    pad_values = {"input_ids": 0, "attention_mask": 0}

    token_slices = {
        "input_ids": [torch.tensor([1, 2, 3]), torch.tensor([4, 5])],
        "attention_mask": [torch.tensor([1, 1, 1]), torch.tensor([1, 1])],
    }

    features = _pad_and_stack_tokens(token_slices, bucket_size, pad_values)

    assert "input_ids" in features
    assert "attention_mask" in features
    assert features["input_ids"].shape == (2, 8)
    assert features["attention_mask"].shape == (2, 8)

    # Check padding
    assert features["input_ids"][0].tolist() == [1, 2, 3, 0, 0, 0, 0, 0]
    assert features["input_ids"][1].tolist() == [4, 5, 0, 0, 0, 0, 0, 0]


def test_pad_and_stack_tokens_truncation():
    """Test that tokens are truncated if longer than bucket size."""
    bucket_size = 4
    pad_values = {"input_ids": 0}

    token_slices = {
        "input_ids": [torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])],
    }

    features = _pad_and_stack_tokens(token_slices, bucket_size, pad_values)

    assert features["input_ids"].shape == (1, 4)
    assert features["input_ids"][0].tolist() == [1, 2, 3, 4]


def test_effective_batch_size_fixed():
    """Test effective batch size with fixed batch_size."""
    cfg = SentenceTransformerConfig(
        dataset="test",
        batch_size=32,
        tokens_per_batch=None,
    )

    # Without tokens_per_batch, should return batch_size
    assert _effective_batch_size(128, cfg) == 32
    assert _effective_batch_size(512, cfg) == 32


def test_effective_batch_size_token_budget():
    """Test effective batch size with tokens_per_batch."""
    cfg = SentenceTransformerConfig(
        dataset="test",
        batch_size=32,
        tokens_per_batch=4096,
    )

    # With tokens_per_batch, should calculate based on bucket size
    assert _effective_batch_size(128, cfg) == 32  # 4096 / 128 = 32
    assert _effective_batch_size(256, cfg) == 16  # 4096 / 256 = 16
    assert _effective_batch_size(512, cfg) == 8  # 4096 / 512 = 8
    assert _effective_batch_size(1024, cfg) == 4  # 4096 / 1024 = 4

    # Should be at least 1
    assert _effective_batch_size(8192, cfg) == 1  # 4096 / 8192 < 1, but min is 1
