# ABOUTME: Provides an LRU cache for layer activations to avoid redundant forward passes
# ABOUTME: Caches raw activations and SAE-encoded features for the 5 most recently used layers

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from transformer_lens import ActivationCache

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached activation entry."""

    activation_cache: ActivationCache
    raw_activations: dict[str, torch.Tensor]  # hook_name -> tensor
    sae_features: dict[str, torch.Tensor]  # sae_id -> encoded features
    token_hash: str
    timestamp: float
    access_count: int = 0
    last_access: float = 0.0


class LayerActivationCache:
    """
    LRU cache for layer activations with configurable size.
    Caches both raw activations and SAE-encoded features.
    """

    _instance = None

    @classmethod
    def get_instance(cls):
        """Get the global LayerActivationCache instance, creating it if it doesn't exist"""
        if cls._instance is None:
            cls._instance = LayerActivationCache()
        return cls._instance

    def __init__(self, max_entries: int = 5):
        self.max_entries = max_entries
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _compute_token_hash(self, tokens: torch.Tensor) -> str:
        """Compute a hash of the input tokens for cache key."""
        # Convert tensor to bytes and hash
        token_bytes = tokens.cpu().numpy().tobytes()
        return hashlib.sha256(token_bytes).hexdigest()[:16]

    def _make_cache_key(
        self, token_hash: str, layer_num: int, stop_at_layer: Optional[int]
    ) -> str:
        """Create a cache key from token hash and layer info."""
        return f"{token_hash}_L{layer_num}_stop{stop_at_layer}"

    def get(
        self, tokens: torch.Tensor, layer_num: int, stop_at_layer: Optional[int] = None
    ) -> Optional[CacheEntry]:
        """
        Retrieve cached activations for given tokens and layer.
        Updates access order and statistics.
        """
        token_hash = self._compute_token_hash(tokens)
        cache_key = self._make_cache_key(token_hash, layer_num, stop_at_layer)

        if cache_key in self.cache:
            # Update access order (move to end)
            entry = self.cache.pop(cache_key)
            entry.access_count += 1
            entry.last_access = time.time()
            self.cache[cache_key] = entry

            self.hits += 1
            logger.debug(f"Cache hit for layer {layer_num} (key: {cache_key})")
            return entry

        self.misses += 1
        logger.debug(f"Cache miss for layer {layer_num} (key: {cache_key})")
        return None

    def put(
        self,
        tokens: torch.Tensor,
        layer_num: int,
        activation_cache: ActivationCache,
        stop_at_layer: Optional[int] = None,
    ) -> None:
        """
        Store activations in cache, evicting oldest entry if needed.
        """
        token_hash = self._compute_token_hash(tokens)
        cache_key = self._make_cache_key(token_hash, layer_num, stop_at_layer)

        # Check if we need to evict
        if len(self.cache) >= self.max_entries and cache_key not in self.cache:
            # Evict least recently used (first item)
            evicted_key, evicted_entry = self.cache.popitem(last=False)
            self.evictions += 1
            logger.debug(
                f"Evicted cache entry {evicted_key} "
                f"(accessed {evicted_entry.access_count} times)"
            )

        # Create new entry
        entry = CacheEntry(
            activation_cache=activation_cache,
            raw_activations={},
            sae_features={},
            token_hash=token_hash,
            timestamp=time.time(),
            last_access=time.time(),
        )

        self.cache[cache_key] = entry
        logger.debug(f"Cached activations for layer {layer_num} (key: {cache_key})")

    def add_raw_activation(
        self,
        tokens: torch.Tensor,
        layer_num: int,
        hook_name: str,
        activation: torch.Tensor,
        stop_at_layer: Optional[int] = None,
    ) -> None:
        """Add raw activation tensor to existing cache entry."""
        token_hash = self._compute_token_hash(tokens)
        cache_key = self._make_cache_key(token_hash, layer_num, stop_at_layer)

        if cache_key in self.cache:
            self.cache[cache_key].raw_activations[hook_name] = activation

    def add_sae_features(
        self,
        tokens: torch.Tensor,
        layer_num: int,
        sae_id: str,
        features: torch.Tensor,
        stop_at_layer: Optional[int] = None,
    ) -> None:
        """Add SAE-encoded features to existing cache entry."""
        token_hash = self._compute_token_hash(tokens)
        cache_key = self._make_cache_key(token_hash, layer_num, stop_at_layer)

        if cache_key in self.cache:
            self.cache[cache_key].sae_features[sae_id] = features

    def get_sae_features(
        self,
        tokens: torch.Tensor,
        layer_num: int,
        sae_id: str,
        stop_at_layer: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """Retrieve cached SAE features if available."""
        entry = self.get(tokens, layer_num, stop_at_layer)
        if entry and sae_id in entry.sae_features:
            return entry.sae_features[sae_id]
        return None

    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        logger.info("Layer activation cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0

        return {
            "size": len(self.cache),
            "max_size": self.max_entries,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions,
            "entries": {
                key: {
                    "access_count": entry.access_count,
                    "age": time.time() - entry.timestamp,
                    "last_access": time.time() - entry.last_access,
                }
                for key, entry in self.cache.items()
            },
        }

    def log_stats(self) -> None:
        """Log cache statistics."""
        stats = self.get_stats()
        logger.info(
            f"LayerActivationCache stats: "
            f"size={stats['size']}/{stats['max_size']}, "
            f"hits={stats['hits']}, misses={stats['misses']}, "
            f"hit_rate={stats['hit_rate']:.2%}, "
            f"evictions={stats['evictions']}"
        )

