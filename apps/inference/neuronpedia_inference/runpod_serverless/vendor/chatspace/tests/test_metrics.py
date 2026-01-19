"""Tests for chatspace.hf_embed.metrics module."""

import pytest

from chatspace.hf_embed.metrics import PipelineStats, StatsUpdate


def test_pipeline_stats_initialization():
    """Test PipelineStats initial state."""
    stats = PipelineStats()
    assert stats.total_rows == 0
    assert stats.skipped_rows == 0
    assert stats.embedding_dim is None
    assert stats.min_norm is None
    assert stats.max_norm is None


def test_pipeline_stats_register_rows():
    """Test row registration."""
    stats = PipelineStats()
    stats.register_rows(10)
    assert stats.total_rows == 10
    stats.register_rows(5)
    assert stats.total_rows == 15


def test_pipeline_stats_register_skipped():
    """Test skipped row registration."""
    stats = PipelineStats()
    stats.register_skipped()
    assert stats.skipped_rows == 1
    stats.register_skipped(5)
    assert stats.skipped_rows == 6


def test_pipeline_stats_update_embedding_dim():
    """Test embedding dimension updates."""
    stats = PipelineStats()

    # First update sets the dimension
    stats.update_embedding_dim(768)
    assert stats.embedding_dim == 768

    # Same dimension is ok
    stats.update_embedding_dim(768)
    assert stats.embedding_dim == 768

    # Different dimension raises error
    with pytest.raises(ValueError, match="Inconsistent embedding dimension"):
        stats.update_embedding_dim(1024)


def test_pipeline_stats_update_norm_bounds():
    """Test norm bound updates."""
    stats = PipelineStats()

    # First update
    stats.update_norm_bounds(0.5, 2.0)
    assert stats.min_norm == 0.5
    assert stats.max_norm == 2.0

    # Updates should track min/max
    stats.update_norm_bounds(0.3, 1.5)
    assert stats.min_norm == 0.3
    assert stats.max_norm == 2.0

    stats.update_norm_bounds(0.8, 3.0)
    assert stats.min_norm == 0.3
    assert stats.max_norm == 3.0

    # None values should be ignored
    stats.update_norm_bounds(None, None)
    assert stats.min_norm == 0.3
    assert stats.max_norm == 3.0


def test_stats_update_initialization():
    """Test StatsUpdate dataclass initialization."""
    update = StatsUpdate()
    assert update.rows_processed == 0
    assert update.rows_skipped == 0
    assert update.embedding_dim is None
    assert update.min_norm is None
    assert update.max_norm is None


def test_stats_update_with_values():
    """Test StatsUpdate with specific values."""
    update = StatsUpdate(
        rows_processed=100,
        rows_skipped=5,
        embedding_dim=768,
        min_norm=0.5,
        max_norm=2.0,
    )
    assert update.rows_processed == 100
    assert update.rows_skipped == 5
    assert update.embedding_dim == 768
    assert update.min_norm == 0.5
    assert update.max_norm == 2.0
