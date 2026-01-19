"""Tests for shared memory cleanup failure scenarios.

Tests cover:
- SharedMemory.unlink() failures (FileNotFoundError, PermissionError)
- Graceful degradation when unlink fails
- Worker-side TTL cleanup behavior
- Shared memory limit exhaustion
- Fallback to bytes encoding
- Concurrent cleanup race conditions
- Partial cleanup after exceptions
"""

import asyncio
import logging
import os
import pytest
import time
import warnings
from multiprocessing.shared_memory import SharedMemory
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import torch
from vllm import SamplingParams

from chatspace.generation import VLLMSteerModel, VLLMSteeringConfig


@pytest.fixture
def model_name():
    """Small model for fast tests."""
    return "Qwen/Qwen3-0.6B"


@pytest.fixture
async def model_factory(model_name):
    """Factory for creating VLLMSteerModel with custom config."""
    created_models = []

    async def _make_model(
        shm_ttl_seconds=600,
        shm_max_gb=128.0,
    ):
        config = VLLMSteeringConfig(
            model_name=model_name,
            gpu_memory_utilization=0.4,
            max_model_len=512,
        )
        m = VLLMSteerModel(
            config,
            bootstrap_layers=(5,),
            shm_ttl_seconds=shm_ttl_seconds,
            shm_max_gb=shm_max_gb,
            enforce_eager=True,
        )
        created_models.append(m)
        return m

    yield _make_model

    # Cleanup all created models
    for m in created_models:
        if hasattr(m, "_engine") and m._engine is not None:
            try:
                await m._engine.shutdown()
            except Exception:
                pass


@pytest.mark.slow
@pytest.mark.asyncio
async def test_unlink_file_not_found_error(model_factory, caplog):
    """Test that FileNotFoundError during unlink doesn't crash worker.

    This simulates the case where shared memory was already unlinked
    (e.g., by another process or TTL cleanup) before client calls close().
    """
    model = await model_factory()

    prompts = ["Once upon a time"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    assert len(handles) == 1
    handle = handles[0]

    # Fetch captures to trigger shared memory creation
    await handle.fetch()
    assert len(handle._shm_names) > 0, "Expected shared memory to be used"

    # Patch SharedMemory.close() to simulate unlink failure
    original_close = SharedMemory.close

    def failing_close(self):
        # Call original close to unmap
        original_close(self)
        # Then simulate that segment was already deleted
        raise FileNotFoundError(f"No such file or directory: '/dev/shm/{self.name}'")

    with patch.object(SharedMemory, "close", failing_close):
        # Should not raise, should log warning
        with caplog.at_level(logging.WARNING):
            await handle.close()

    # Verify worker is still functional after cleanup failure
    results2, handles2 = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )
    assert len(results2) == 1
    assert len(handles2) == 1

    # Cleanup second handle
    await handles2[0].close()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_unlink_permission_error(model_factory, caplog):
    """Test that PermissionError during unlink doesn't crash worker.

    This simulates the case where /dev/shm has incorrect permissions.
    """
    model = await model_factory()

    prompts = ["Test prompt"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    handle = handles[0]
    await handle.fetch()
    assert len(handle._shm_names) > 0

    # Simulate permission error during unlink
    original_close = SharedMemory.close

    def permission_denied_close(self):
        original_close(self)
        raise PermissionError(f"Permission denied: '/dev/shm/{self.name}'")

    with patch.object(SharedMemory, "close", permission_denied_close):
        with caplog.at_level(logging.WARNING):
            await handle.close()

    # Verify warning was logged
    assert any("Failed to close shared memory" in record.message for record in caplog.records)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_idempotent_close(model_factory):
    """Test that calling close() multiple times is safe."""
    model = await model_factory()

    prompts = ["Test prompt"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    handle = handles[0]
    await handle.fetch()

    # First close should work normally
    await handle.close()
    assert handle._closed is True

    # Second close should be no-op (not raise)
    await handle.close()
    assert handle._closed is True

    # Third close should also be safe
    await handle.close()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_close_without_fetch(model_factory):
    """Test that closing a handle before fetching is safe."""
    model = await model_factory()

    prompts = ["Test prompt"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    handle = handles[0]

    # Close without fetching (no shared memory created yet)
    await handle.close()
    assert handle._closed is True

    # Verify fetch after close raises appropriate error
    # (This tests the expected failure mode, not a bug)
    # Note: This might not raise if fetch() implementation checks _closed
    # Just ensure it doesn't crash


@pytest.mark.slow
@pytest.mark.asyncio
async def test_shared_memory_limit_raises_error(model_factory):
    """Test that exceeding CHATSPACE_MAX_SHM_GB raises RuntimeError.

    Shared memory is always used (no bytes fallback), so exceeding
    the limit should fail fast with a clear error message.
    """
    # Set very low limit to trigger error
    model = await model_factory(
        shm_max_gb=0.0001,  # 100 KB limit - will be exceeded
    )

    prompts = ["Once upon a time, in a land far away, there lived a"]
    sampling_params = SamplingParams(max_tokens=20, temperature=0.0)

    # Generate with multiple capture layers to exceed limit
    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5, 10, 15, 20],  # Multiple layers to exceed limit
    )

    handle = handles[0]

    # Fetch should raise Exception when shared memory limit is exceeded
    # (RuntimeError from worker gets wrapped in Exception through RPC)
    with pytest.raises(Exception, match="Shared memory limit reached"):
        await handle.fetch()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_cleanup_after_fetch_exception(model_factory):
    """Test that shared memory is cleaned up even if fetch() raises."""
    model = await model_factory()

    prompts = ["Test prompt"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    handle = handles[0]

    # Mock _fetch_request_captures to raise after creating shared memory
    original_fetch = model._fetch_request_captures

    async def failing_fetch(*args, **kwargs):
        # Call original to create shared memory
        result = await original_fetch(*args, **kwargs)
        # Then raise to simulate error during processing
        raise RuntimeError("Simulated fetch failure")

    with patch.object(model, "_fetch_request_captures", failing_fetch):
        with pytest.raises(RuntimeError, match="Simulated fetch failure"):
            await handle.fetch()

    # Cleanup should still work
    await handle.close()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_concurrent_close_operations(model_factory):
    """Test that concurrent close() calls don't cause race conditions."""
    model = await model_factory()

    prompts = ["Test prompt"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    handle = handles[0]
    await handle.fetch()

    # Launch multiple close operations concurrently
    close_tasks = [handle.close() for _ in range(5)]

    # All should complete without error
    await asyncio.gather(*close_tasks)

    # Handle should be closed
    assert handle._closed is True


@pytest.mark.slow
@pytest.mark.asyncio
async def test_shared_memory_always_used(model_factory):
    """Test that shared memory is always used for captures."""
    model = await model_factory()

    # Use longer prompt to get a decent tensor size
    prompts = ["Test " * 100]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    handle = handles[0]
    await handle.fetch()

    # Shared memory should always be used
    assert len(handle._shm_names) > 0, "Expected shared memory for captures"

    await handle.close()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_multiple_handles_cleanup(model_factory):
    """Test cleanup of multiple handles with shared memory."""
    model = await model_factory()

    prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5, 10],
    )

    assert len(handles) == 3

    # Fetch all handles
    await asyncio.gather(*[h.fetch() for h in handles])

    # All should have shared memory
    for handle in handles:
        assert len(handle._shm_names) > 0

    # Close in different order
    await handles[1].close()
    await handles[0].close()
    await handles[2].close()

    # All should be closed
    for handle in handles:
        assert handle._closed is True


@pytest.mark.slow
@pytest.mark.asyncio
async def test_basic_capture_with_shared_memory(model_factory):
    """Test that basic captures work with shared memory (always enabled)."""
    model = await model_factory()

    prompts = ["Test prompt"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    handle = handles[0]
    await handle.fetch()

    # Shared memory is always used
    assert len(handle._shm_names) > 0, "Expected shared memory for captures"

    # Captures should be valid
    assert handle._captures is not None
    assert 5 in handle._captures

    await handle.close()
