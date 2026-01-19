"""Tests for chatspace.hf_embed.utils module."""

from pathlib import Path
from unittest.mock import patch
import tempfile

import pytest
import torch

from chatspace.hf_embed.utils import (
    _iso_now,
    _ensure_dir,
    _sanitize_component,
    _compute_sha256,
    _prepare_paths,
    _observe_git_sha,
    _compute_norms,
    _next_power_of_two,
    _enumerate_bucket_sizes,
)


def test_iso_now():
    """Test ISO timestamp generation."""
    result = _iso_now()
    assert isinstance(result, str)
    assert "T" in result  # ISO format includes T separator
    assert result.endswith("+00:00") or result.endswith("Z")  # UTC timezone


def test_ensure_dir():
    """Test directory creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "nested" / "dirs"
        _ensure_dir(test_path)
        assert test_path.exists()
        assert test_path.is_dir()

        # Should not raise if called again
        _ensure_dir(test_path)


def test_sanitize_component():
    """Test path component sanitization."""
    assert _sanitize_component("org/model") == "org__model"
    assert _sanitize_component("simple") == "simple"
    assert _sanitize_component("a/b/c") == "a__b__c"


def test_compute_sha256():
    """Test SHA256 checksum computation."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"test content")
        f.flush()
        path = Path(f.name)

    try:
        checksum = _compute_sha256(path)
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA256 produces 64 hex chars
        # Verify determinism
        assert _compute_sha256(path) == checksum
    finally:
        path.unlink()


def test_prepare_paths():
    """Test path preparation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = _prepare_paths(
            output_root=Path(tmpdir),
            model_name="org/model",
            dataset="dataset/name",
            subset="config",
            split="train",
        )

        assert "embeddings_dir" in paths
        assert "indexes_dir" in paths
        assert "manifest_path" in paths
        assert "run_path" in paths
        assert "logs_dir" in paths

        # Check that directories were created
        assert paths["embeddings_dir"].exists()
        assert paths["indexes_dir"].exists()
        assert paths["logs_dir"].exists()

        # Check sanitization
        assert "org__model" in str(paths["embeddings_dir"])
        assert "dataset__name__config" in str(paths["embeddings_dir"])


def test_observe_git_sha():
    """Test git SHA observation."""
    # This may or may not return a SHA depending on environment
    result = _observe_git_sha()
    if result is not None:
        assert isinstance(result, str)
        assert len(result) == 40  # Git SHA is 40 hex chars


def test_compute_norms():
    """Test L2 norm computation."""
    embeddings = torch.tensor([[3.0, 4.0], [5.0, 12.0]])
    norms = _compute_norms(embeddings)

    assert norms.shape == (2,)
    assert torch.allclose(norms, torch.tensor([5.0, 13.0]))


def test_next_power_of_two():
    """Test power-of-2 rounding."""
    assert _next_power_of_two(0) == 1
    assert _next_power_of_two(1) == 1
    assert _next_power_of_two(2) == 2
    assert _next_power_of_two(3) == 4
    assert _next_power_of_two(7) == 8
    assert _next_power_of_two(8) == 8
    assert _next_power_of_two(9) == 16
    assert _next_power_of_two(100) == 128
    assert _next_power_of_two(1024) == 1024


def test_enumerate_bucket_sizes():
    """Test bucket size enumeration."""
    sizes = _enumerate_bucket_sizes(128, 1024)
    assert sizes == [128, 256, 512, 1024]

    sizes = _enumerate_bucket_sizes(64, 256)
    assert sizes == [64, 128, 256]

    # Edge case: min equals max
    sizes = _enumerate_bucket_sizes(512, 512)
    assert sizes == [512]
