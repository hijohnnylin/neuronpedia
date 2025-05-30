# ABOUTME: Unit tests for the LayerActivationCache implementation
# ABOUTME: Tests LRU eviction, cache hits/misses, and proper recency tracking

import time
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformer_lens import ActivationCache

from neuronpedia_inference.layer_activation_cache import (
    CacheEntry,
    LayerActivationCache,
)


class TestLayerActivationCache:
    """Test suite for LayerActivationCache."""

    @pytest.fixture
    def cache(self):
        """Create a fresh cache instance for each test."""
        # Reset singleton
        LayerActivationCache._instance = None
        return LayerActivationCache(max_entries=3)

    @pytest.fixture
    def mock_activation_cache(self):
        """Create a mock ActivationCache."""
        return MagicMock(spec=ActivationCache)

    def test_singleton_pattern(self):
        """Test that get_instance returns the same instance."""
        cache1 = LayerActivationCache.get_instance()
        cache2 = LayerActivationCache.get_instance()
        assert cache1 is cache2

    def test_compute_token_hash(self, cache):
        """Test token hashing is consistent."""
        tokens = torch.tensor([1, 2, 3, 4, 5])
        hash1 = cache._compute_token_hash(tokens)
        hash2 = cache._compute_token_hash(tokens)
        assert hash1 == hash2
        assert len(hash1) == 16  # Should be truncated to 16 chars

    def test_cache_miss(self, cache):
        """Test cache miss behavior."""
        tokens = torch.tensor([1, 2, 3])
        result = cache.get(tokens, layer_num=5)
        assert result is None
        assert cache.misses == 1
        assert cache.hits == 0

    def test_cache_hit(self, cache, mock_activation_cache):
        """Test cache hit behavior."""
        tokens = torch.tensor([1, 2, 3])
        layer_num = 5

        # Store in cache
        cache.put(tokens, layer_num, mock_activation_cache)

        # Retrieve from cache
        result = cache.get(tokens, layer_num)
        assert result is not None
        assert result.activation_cache == mock_activation_cache
        assert cache.hits == 1
        assert cache.misses == 0

    def test_lru_eviction(self, cache, mock_activation_cache):
        """Test LRU eviction when cache is full."""
        # Fill cache to capacity (3 entries)
        tokens1 = torch.tensor([1, 1])
        tokens2 = torch.tensor([2, 2])
        tokens3 = torch.tensor([3, 3])
        tokens4 = torch.tensor([4, 4])

        cache.put(tokens1, 0, mock_activation_cache)
        cache.put(tokens2, 0, mock_activation_cache)
        cache.put(tokens3, 0, mock_activation_cache)

        assert len(cache.cache) == 3
        assert cache.evictions == 0

        # Add one more - should evict the first
        cache.put(tokens4, 0, mock_activation_cache)

        assert len(cache.cache) == 3
        assert cache.evictions == 1

        # First entry should be evicted
        assert cache.get(tokens1, 0) is None
        # Others should still be there
        assert cache.get(tokens2, 0) is not None
        assert cache.get(tokens3, 0) is not None
        assert cache.get(tokens4, 0) is not None

    def test_access_order_update(self, cache, mock_activation_cache):
        """Test that accessing an entry updates its position."""
        tokens1 = torch.tensor([1, 1])
        tokens2 = torch.tensor([2, 2])
        tokens3 = torch.tensor([3, 3])
        tokens4 = torch.tensor([4, 4])

        # Fill cache
        cache.put(tokens1, 0, mock_activation_cache)
        cache.put(tokens2, 0, mock_activation_cache)
        cache.put(tokens3, 0, mock_activation_cache)

        # Access the first entry to move it to end
        entry1 = cache.get(tokens1, 0)
        assert entry1 is not None
        assert entry1.access_count == 1

        # Add new entry - should evict tokens2 (now oldest)
        cache.put(tokens4, 0, mock_activation_cache)

        assert cache.get(tokens2, 0) is None  # Evicted
        assert cache.get(tokens1, 0) is not None  # Still there
        assert cache.get(tokens3, 0) is not None  # Still there
        assert cache.get(tokens4, 0) is not None  # New entry

    def test_stop_at_layer_caching(self, cache, mock_activation_cache):
        """Test that stop_at_layer is part of cache key."""
        tokens = torch.tensor([1, 2, 3])

        # Same tokens but different stop_at_layer should be different entries
        cache.put(tokens, 0, mock_activation_cache, stop_at_layer=5)
        cache.put(tokens, 0, mock_activation_cache, stop_at_layer=10)

        assert len(cache.cache) == 2  # Two different entries

    def test_add_sae_features(self, cache, mock_activation_cache):
        """Test adding SAE features to cache entry."""
        tokens = torch.tensor([1, 2, 3])
        features = torch.randn(10, 768)

        cache.put(tokens, 0, mock_activation_cache)
        cache.add_sae_features(tokens, 0, "sae_1", features)

        retrieved = cache.get_sae_features(tokens, 0, "sae_1")
        assert retrieved is not None
        assert torch.equal(retrieved, features)

    def test_cache_stats(self, cache, mock_activation_cache):
        """Test cache statistics reporting."""
        tokens1 = torch.tensor([1, 1])
        tokens2 = torch.tensor([2, 2])

        # Generate some activity
        cache.get(tokens1, 0)  # Miss
        cache.put(tokens1, 0, mock_activation_cache)
        cache.get(tokens1, 0)  # Hit
        cache.get(tokens1, 0)  # Hit
        cache.get(tokens2, 0)  # Miss

        stats = cache.get_stats()
        assert stats["size"] == 1
        assert stats["max_size"] == 3
        assert stats["hits"] == 2
        assert stats["misses"] == 2
        assert stats["hit_rate"] == 0.5
        assert stats["evictions"] == 0

    def test_clear_cache(self, cache, mock_activation_cache):
        """Test clearing the cache."""
        tokens = torch.tensor([1, 2, 3])
        cache.put(tokens, 0, mock_activation_cache)
        cache.get(tokens, 0)  # Generate a hit

        cache.clear()

        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0
        assert cache.evictions == 0

    def test_access_time_tracking(self, cache, mock_activation_cache):
        """Test that access times are tracked correctly."""
        tokens = torch.tensor([1, 2, 3])

        # Store entry
        cache.put(tokens, 0, mock_activation_cache)
        time.sleep(0.01)  # Small delay

        # Access entry
        entry = cache.get(tokens, 0)
        assert entry is not None
        assert entry.last_access > entry.timestamp

    def test_concurrent_layer_caching(self, cache, mock_activation_cache):
        """Test caching multiple layers for same tokens."""
        tokens = torch.tensor([1, 2, 3])

        # Cache different layers for same tokens
        for layer in range(5):
            cache.put(tokens, layer, mock_activation_cache)

        # Should have 3 entries (limited by max_entries)
        assert len(cache.cache) == 3
        assert cache.evictions == 2  # Two entries were evicted

