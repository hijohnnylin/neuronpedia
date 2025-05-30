# ABOUTME: Integration tests demonstrating real-world cache performance improvements
# ABOUTME: Uses actual endpoints to measure end-to-end timing improvements

import asyncio
import time
from typing import Dict, List

import pytest
from fastapi.testclient import TestClient

from neuronpedia_inference.layer_activation_cache import LayerActivationCache
from neuronpedia_inference.server import app


@pytest.mark.integration
class TestCachePerformanceIntegration:
    """Integration tests measuring real performance improvements."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear cache before each test."""
        cache = LayerActivationCache.get_instance()
        cache.clear()
        yield
        # Print cache stats after test
        cache.log_stats()

    def test_activation_all_endpoint_performance(self, client):
        """Test performance improvement for activation/all endpoint."""
        # Test payload
        payload = {
            "prompt": "The quick brown fox jumps over the lazy dog",
            "model": "gpt2-small",
            "source_set": "res-jb",
            "selected_sources": ["0-res-jb", "2-res-jb", "4-res-jb", "6-res-jb"],
            "num_results": 10,
            "sort_by_token_indexes": [],
            "ignore_bos": False,
        }
        
        print("\n=== Activation/All Endpoint Performance ===")
        
        # First request - cold cache
        start = time.time()
        response1 = client.post("/v1/activation/all", json=payload)
        first_time = time.time() - start
        assert response1.status_code == 200
        print(f"First request (cold cache): {first_time*1000:.2f}ms")
        
        # Subsequent requests - warm cache
        warm_times = []
        for i in range(3):
            start = time.time()
            response = client.post("/v1/activation/all", json=payload)
            elapsed = time.time() - start
            warm_times.append(elapsed)
            assert response.status_code == 200
            print(f"Request {i+2} (warm cache): {elapsed*1000:.2f}ms")
        
        # Calculate improvement
        avg_warm_time = sum(warm_times) / len(warm_times)
        improvement = (first_time - avg_warm_time) / first_time * 100
        speedup = first_time / avg_warm_time
        
        print(f"\nImprovement: {improvement:.1f}% faster")
        print(f"Speedup: {speedup:.1f}x")
        
        # Get cache stats
        cache = LayerActivationCache.get_instance()
        stats = cache.get_stats()
        print(f"Cache hit rate: {stats['hit_rate']:.2%}")
        
        assert improvement > 20  # At least 20% improvement

    def test_activation_single_performance(self, client):
        """Test performance for single activation endpoint."""
        prompt = "Artificial intelligence is transforming the world"
        
        print("\n=== Activation/Single Endpoint Performance ===")
        
        # Test different layers
        layers = ["0-res-jb", "3-res-jb", "6-res-jb", "9-res-jb"]
        timings = {"cold": [], "warm": []}
        
        # First pass - cold cache
        for layer in layers:
            payload = {
                "prompt": prompt,
                "source": layer,
                "index": 100,
            }
            
            start = time.time()
            response = client.post("/v1/activation/single", json=payload)
            elapsed = time.time() - start
            timings["cold"].append(elapsed)
            assert response.status_code == 200
            print(f"Layer {layer} (cold): {elapsed*1000:.2f}ms")
        
        # Second pass - warm cache
        for layer in layers:
            payload = {
                "prompt": prompt,
                "source": layer,
                "index": 100,
            }
            
            start = time.time()
            response = client.post("/v1/activation/single", json=payload)
            elapsed = time.time() - start
            timings["warm"].append(elapsed)
            assert response.status_code == 200
            print(f"Layer {layer} (warm): {elapsed*1000:.2f}ms")
        
        # Calculate aggregate improvement
        total_cold = sum(timings["cold"])
        total_warm = sum(timings["warm"])
        improvement = (total_cold - total_warm) / total_cold * 100
        
        print(f"\nTotal cold: {total_cold*1000:.2f}ms")
        print(f"Total warm: {total_warm*1000:.2f}ms")
        print(f"Improvement: {improvement:.1f}%")
        
        assert improvement > 30  # At least 30% improvement

    def test_mixed_endpoint_usage(self, client):
        """Test cache effectiveness across different endpoint types."""
        prompt = "Machine learning models are becoming increasingly sophisticated"
        
        print("\n=== Mixed Endpoint Usage Pattern ===")
        
        results = []
        
        # 1. First hit activation/all
        start = time.time()
        response = client.post("/v1/activation/all", json={
            "prompt": prompt,
            "model": "gpt2-small",
            "source_set": "res-jb",
            "selected_sources": ["0-res-jb", "1-res-jb", "2-res-jb"],
            "num_results": 5,
        })
        elapsed = time.time() - start
        assert response.status_code == 200
        results.append(("activation/all", elapsed, "COLD"))
        
        # 2. Then hit activation/single for layer already cached
        start = time.time()
        response = client.post("/v1/activation/single", json={
            "prompt": prompt,
            "source": "1-res-jb",
            "index": 50,
        })
        elapsed = time.time() - start
        assert response.status_code == 200
        results.append(("activation/single", elapsed, "WARM"))
        
        # 3. Hit topk for another cached layer
        start = time.time()
        response = client.post("/v1/activation/topk-by-token", json={
            "prompt": prompt,
            "source": "2-res-jb",
            "top_k": 5,
        })
        elapsed = time.time() - start
        assert response.status_code == 200
        results.append(("activation/topk", elapsed, "WARM"))
        
        # Print results
        for endpoint, timing, cache_state in results:
            print(f"{endpoint:<20} {timing*1000:>8.2f}ms ({cache_state})")
        
        # Verify warm requests are faster
        cold_time = results[0][1]
        warm_times = [r[1] for r in results[1:]]
        assert all(warm < cold_time * 0.5 for warm in warm_times)

    @pytest.mark.asyncio
    async def test_concurrent_cache_benefits(self, client):
        """Test cache performance under concurrent-like access."""
        prompts = [
            "The future of AI",
            "The future of AI",  # Duplicate
            "Climate change impacts",
            "The future of AI",  # Another duplicate
        ]
        
        print("\n=== Concurrent Access Pattern ===")
        
        async def make_request(prompt: str, index: int):
            start = time.time()
            response = client.post("/v1/activation/single", json={
                "prompt": prompt,
                "source": "5-res-jb",
                "index": 100,
            })
            elapsed = time.time() - start
            cache_status = "HIT" if index > 0 and prompt == prompts[0] else "MISS"
            return (index, prompt, elapsed, cache_status)
        
        # Simulate concurrent requests
        tasks = [make_request(prompt, i) for i, prompt in enumerate(prompts)]
        results = await asyncio.gather(*tasks)
        
        # Print results
        for idx, prompt, timing, status in sorted(results):
            print(f"Request {idx}: '{prompt[:20]}...' - {timing*1000:.2f}ms ({status})")
        
        # Verify cache hits are faster
        hit_times = [r[2] for r in results if r[3] == "HIT"]
        miss_times = [r[2] for r in results if r[3] == "MISS"]
        
        if hit_times and miss_times:
            avg_hit = sum(hit_times) / len(hit_times)
            avg_miss = sum(miss_times) / len(miss_times)
            print(f"\nAverage hit time: {avg_hit*1000:.2f}ms")
            print(f"Average miss time: {avg_miss*1000:.2f}ms")
            assert avg_hit < avg_miss * 0.7  # Hits should be at least 30% faster

    def test_cache_stats_endpoint(self, client):
        """Test that cache stats are properly reported in health endpoint."""
        # Generate some cache activity
        test_prompt = "Testing cache statistics"
        
        # Make a few requests
        for i in range(3):
            client.post("/v1/activation/single", json={
                "prompt": test_prompt if i < 2 else "Different prompt",
                "source": "0-res-jb",
                "index": 10,
            })
        
        # Check health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "cache_stats" in data
        
        stats = data["cache_stats"]
        print("\n=== Cache Statistics from /health ===")
        print(f"Cache size: {stats['size']}/{stats['max_size']}")
        print(f"Hit rate: {stats['hit_rate']:.2%}")
        print(f"Hits: {stats['hits']}, Misses: {stats['misses']}")
        print(f"Evictions: {stats['evictions']}")
        
        assert stats["hits"] >= 1  # Should have at least one hit
        assert stats["misses"] >= 2  # Should have at least two misses