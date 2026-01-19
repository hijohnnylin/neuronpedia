"""Tests for chatspace.hf_embed.config module."""

from pathlib import Path

import pytest

from chatspace.hf_embed.config import SentenceTransformerConfig


def test_config_defaults():
    """Test default configuration values."""
    cfg = SentenceTransformerConfig(dataset="test/dataset")

    assert cfg.dataset == "test/dataset"
    assert cfg.subset is None
    assert cfg.split == "train"
    assert cfg.text_field == "text"
    assert cfg.model_name == "Qwen/Qwen3-Embedding-0.6B"
    assert cfg.batch_size == 32
    assert cfg.rows_per_shard == 8192
    assert cfg.max_rows is None
    assert cfg.output_root == Path("/workspace")
    assert cfg.bucket_min_tokens == 128
    assert cfg.bucket_max_tokens == 32768
    assert cfg.compile_model is False
    assert cfg.extract_first_assistant is False


def test_config_validation_positive_values():
    """Test validation for positive value requirements."""
    # rows_per_shard must be positive
    with pytest.raises(ValueError, match="rows_per_shard must be positive"):
        SentenceTransformerConfig(dataset="test", rows_per_shard=0)

    with pytest.raises(ValueError, match="rows_per_shard must be positive"):
        SentenceTransformerConfig(dataset="test", rows_per_shard=-1)

    # batch_size must be positive
    with pytest.raises(ValueError, match="batch_size must be positive"):
        SentenceTransformerConfig(dataset="test", batch_size=0)

    # prefetch_batches must be positive
    with pytest.raises(ValueError, match="prefetch_batches must be positive"):
        SentenceTransformerConfig(dataset="test", prefetch_batches=0)


def test_config_validation_bucket_tokens():
    """Test validation for bucket token ranges."""
    # bucket_min_tokens must be positive
    with pytest.raises(ValueError, match="bucket_min_tokens must be positive"):
        SentenceTransformerConfig(dataset="test", bucket_min_tokens=0)

    # bucket_max_tokens must be positive
    with pytest.raises(ValueError, match="bucket_max_tokens must be positive"):
        SentenceTransformerConfig(dataset="test", bucket_max_tokens=0)

    # min must be <= max
    with pytest.raises(ValueError, match="bucket_min_tokens must be less than or equal to bucket_max_tokens"):
        SentenceTransformerConfig(dataset="test", bucket_min_tokens=1024, bucket_max_tokens=512)


def test_config_validation_optional_positive():
    """Test validation for optional positive values."""
    # tokens_per_batch (optional) must be positive if provided
    with pytest.raises(ValueError, match="tokens_per_batch must be positive"):
        SentenceTransformerConfig(dataset="test", tokens_per_batch=0)

    # max_rows (optional) must be positive if provided
    with pytest.raises(ValueError, match="max_rows must be positive"):
        SentenceTransformerConfig(dataset="test", max_rows=0)

    # Should work when None
    cfg = SentenceTransformerConfig(dataset="test", tokens_per_batch=None, max_rows=None)
    assert cfg.tokens_per_batch is None
    assert cfg.max_rows is None


def test_config_custom_values():
    """Test configuration with custom values."""
    cfg = SentenceTransformerConfig(
        dataset="my/dataset",
        subset="config",
        split="test",
        text_field="content",
        model_name="custom/model",
        batch_size=64,
        rows_per_shard=4096,
        max_rows=1000,
        bucket_min_tokens=256,
        bucket_max_tokens=16384,
        tokens_per_batch=65536,
        compile_model=True,
        extract_first_assistant=True,
    )

    assert cfg.dataset == "my/dataset"
    assert cfg.subset == "config"
    assert cfg.split == "test"
    assert cfg.text_field == "content"
    assert cfg.model_name == "custom/model"
    assert cfg.batch_size == 64
    assert cfg.rows_per_shard == 4096
    assert cfg.max_rows == 1000
    assert cfg.bucket_min_tokens == 256
    assert cfg.bucket_max_tokens == 16384
    assert cfg.tokens_per_batch == 65536
    assert cfg.compile_model is True
    assert cfg.extract_first_assistant is True
