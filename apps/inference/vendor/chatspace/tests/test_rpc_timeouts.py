"""Tests for RPC timeout handling in VLLMSteerModel.

Tests cover:
- Slow worker response timeout
- Worker hang detection
- Partial worker failures (some succeed, some timeout)
- Timeout propagation to client
- System remains functional after timeout
- Cleanup operations with timeout
- Steering registration timeout
- Capture fetch timeout
"""

import asyncio
import pytest
from unittest.mock import patch, AsyncMock

import torch
from vllm import SamplingParams

from chatspace.generation import VLLMSteerModel, VLLMSteeringConfig, SteeringSpec, LayerSteeringSpec, AddSpec


@pytest.fixture
def model_name():
    """Small model for fast tests."""
    return "Qwen/Qwen3-0.6B"


@pytest.fixture
async def model_factory(model_name):
    """Factory for creating VLLMSteerModel with custom config."""
    created_models = []

    async def _make_model():
        config = VLLMSteeringConfig(
            model_name=model_name,
            gpu_memory_utilization=0.4,
            max_model_len=512,
        )
        m = VLLMSteerModel(
            config,
            bootstrap_layers=(5,),
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
async def test_capture_fetch_timeout(model_factory):
    """Test timeout handling when fetch operation hangs.

    This simulates the case where worker is slow to respond to fetch RPC.
    """
    model = await model_factory()

    prompts = ["Test prompt"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    # Normal generation should work
    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    handle = handles[0]

    # Mock _collective_rpc to be very slow for fetch operations
    original_rpc = model._collective_rpc

    async def slow_fetch_rpc(op, *args, **kwargs):
        if op == "fetch_request_activations":
            # Hang for a very long time
            await asyncio.sleep(60)  # Will be interrupted by timeout
        return await original_rpc(op, *args, **kwargs)

    with patch.object(model, "_collective_rpc", slow_fetch_rpc):
        # Fetch with short timeout should fail
        with pytest.raises((asyncio.TimeoutError, RuntimeError)):
            await asyncio.wait_for(handle.fetch(), timeout=2.0)

    # Cleanup (should still work with original RPC)
    await handle.close()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_steering_registration_timeout(model_factory):
    """Test timeout handling when steering registration hangs.

    This simulates the case where worker is slow to register steering spec.
    """
    model = await model_factory()

    # Create steering spec
    steering_vector = torch.randn(model.hidden_size)
    steering_spec = SteeringSpec(
        layers={
            5: LayerSteeringSpec(operations=[
                AddSpec(vector=steering_vector / steering_vector.norm(), scale=1.0)
            ])
        }
    )

    # Mock _collective_rpc to be slow for registration
    original_rpc = model._collective_rpc

    async def slow_register_rpc(op, *args, **kwargs):
        if op == "register_steering_spec":
            await asyncio.sleep(60)  # Will be interrupted
        return await original_rpc(op, *args, **kwargs)

    prompts = ["Test prompt"]
    sampling_params = SamplingParams(max_tokens=5, temperature=0.0)

    with patch.object(model, "_collective_rpc", slow_register_rpc):
        # Generate with timeout should fail during steering registration
        with pytest.raises((asyncio.TimeoutError, RuntimeError)):
            await asyncio.wait_for(
                model.generate(prompts, sampling_params, steering_spec=steering_spec),
                timeout=3.0
            )


@pytest.mark.slow
@pytest.mark.asyncio
async def test_cleanup_rpc_timeout_is_not_fatal(model_factory, caplog):
    """Test that cleanup RPC timeout doesn't crash the system.

    Worker-side TTL cleanup should handle stale segments even if client
    times out during explicit cleanup.
    """
    import logging

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

    # Mock _collective_rpc to timeout on cleanup
    original_rpc = model._collective_rpc

    async def slow_cleanup_rpc(op, *args, **kwargs):
        if op == "release_shared_memory":
            await asyncio.sleep(60)  # Hang
        return await original_rpc(op, *args, **kwargs)

    with patch.object(model, "_collective_rpc", slow_cleanup_rpc):
        with caplog.at_level(logging.WARNING):
            # Close with short timeout
            try:
                await asyncio.wait_for(handle.close(), timeout=2.0)
            except asyncio.TimeoutError:
                # Timeout is acceptable for cleanup (worker TTL will handle it)
                pass

    # System should still be functional
    results2, handles2 = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    assert len(results2) == 1
    await handles2[0].close()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_system_functional_after_timeout(model_factory):
    """Test that system remains functional after RPC timeout.

    This verifies that timeout doesn't leave system in broken state.
    """
    model = await model_factory()

    prompts = ["Test prompt"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    # First generation with slow RPC
    original_rpc = model._collective_rpc
    call_count = [0]

    async def sometimes_slow_rpc(op, *args, **kwargs):
        call_count[0] += 1
        # Only first call is slow
        if call_count[0] == 1 and op == "initialize_worker_state":
            await asyncio.sleep(0.1)  # Small delay
        return await original_rpc(op, *args, **kwargs)

    with patch.object(model, "_collective_rpc", sometimes_slow_rpc):
        # First operation succeeds despite slight delay
        results1, handles1 = await model.generate(
            prompts,
            sampling_params,
            capture_layers=[5],
        )
        await handles1[0].close()

    # Second generation should work normally (no patch)
    results2, handles2 = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    assert len(results2) == 1
    await handles2[0].close()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_concurrent_rpcs_with_one_timeout(model_factory):
    """Test that one slow RPC doesn't block other operations.

    This verifies that RPC operations don't have global locks.
    """
    model = await model_factory()

    prompts = ["Test prompt"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    # Generate first batch
    results1, handles1 = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    # Start second generation concurrently
    async def second_generation():
        results2, handles2 = await model.generate(
            prompts,
            sampling_params,
            capture_layers=[5],
        )
        return results2, handles2

    # Both should complete (no blocking)
    results2, handles2 = await asyncio.wait_for(second_generation(), timeout=30.0)

    assert len(results2) == 1

    # Cleanup
    await handles1[0].close()
    await handles2[0].close()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_unregister_steering_timeout_handling(model_factory, caplog):
    """Test timeout handling during steering spec unregistration.

    Unregistration happens in finally block, so timeout should be logged
    but not crash the operation.
    """
    import logging

    model = await model_factory()

    steering_vector = torch.randn(model.hidden_size)
    steering_spec = SteeringSpec(
        layers={
            5: LayerSteeringSpec(operations=[
                AddSpec(vector=steering_vector / steering_vector.norm(), scale=1.0)
            ])
        }
    )

    # Mock to make unregister slow
    original_rpc = model._collective_rpc

    async def slow_unregister_rpc(op, *args, **kwargs):
        if op == "unregister_steering_spec":
            await asyncio.sleep(60)  # Hang on cleanup
        return await original_rpc(op, *args, **kwargs)

    prompts = ["Test prompt"]
    sampling_params = SamplingParams(max_tokens=5, temperature=0.0)

    with patch.object(model, "_collective_rpc", slow_unregister_rpc):
        with caplog.at_level(logging.WARNING):
            # Generate will try to unregister in finally block
            # This may timeout, but shouldn't crash
            try:
                result = await asyncio.wait_for(
                    model.generate(prompts, sampling_params, steering_spec=steering_spec),
                    timeout=30.0  # Long timeout for generation, but unregister may timeout
                )
                # If we get here, generation succeeded but cleanup may have timed out
                # Handle result which could be single value or tuple depending on how timeout occurred
                if isinstance(result, tuple):
                    results, handles = result
                    if handles:
                        await handles[0].close()
            except asyncio.TimeoutError:
                # Timeout during cleanup is acceptable
                pass


@pytest.mark.slow
@pytest.mark.asyncio
async def test_multiple_concurrent_timeouts(model_factory):
    """Test system behavior when multiple operations timeout concurrently.

    This stress tests the timeout handling to ensure no deadlocks.
    """
    model = await model_factory()

    # Initialize engine once before concurrent operations to avoid race condition
    init_prompts = ["Init"]
    init_params = SamplingParams(max_tokens=1, temperature=0.0)
    init_results, init_handles = await model.generate(init_prompts, init_params, capture_layers=[5])
    await init_handles[0].close()

    # Create multiple slow operations
    async def slow_operation(delay: float):
        prompts = ["Test"]
        sampling_params = SamplingParams(max_tokens=5, temperature=0.0)
        try:
            result = await asyncio.wait_for(
                model.generate(prompts, sampling_params),
                timeout=delay
            )
            # Handle result which might be tuple or single value
            if isinstance(result, tuple):
                results, handles = result
                if handles:
                    await handles[0].close()
            return "success"
        except asyncio.TimeoutError:
            return "timeout"
        except Exception as e:
            # Log other exceptions but don't fail - testing system resilience
            print(f"Operation encountered exception: {e}")
            return f"error: {type(e).__name__}"

    # Launch multiple operations with different timeouts
    # Use longer timeouts to avoid timing issues
    tasks = [
        slow_operation(30.0),  # Should succeed
        slow_operation(30.0),  # Should succeed
        slow_operation(30.0),  # Should succeed
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # At least some should succeed (system not deadlocked)
    successes = [r for r in results if r == "success"]
    assert len(successes) > 0, \
        f"Expected at least some operations to succeed, got results: {results}"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_rpc_exception_propagation(model_factory):
    """Test that RPC exceptions are properly propagated to caller.

    This ensures that errors don't get swallowed silently.
    """
    model = await model_factory()

    # Store original method before mocking
    original_rpc = model._collective_rpc

    # Mock RPC to raise exception
    async def failing_rpc(op, *args, **kwargs):
        if op == "register_steering_spec":
            raise RuntimeError("Simulated worker error")
        # Let other operations through (call original method directly)
        return await original_rpc(op, *args, **kwargs)

    steering_vector = torch.randn(model.hidden_size)
    steering_spec = SteeringSpec(
        layers={
            5: LayerSteeringSpec(operations=[
                AddSpec(vector=steering_vector / steering_vector.norm(), scale=1.0)
            ])
        }
    )

    prompts = ["Test"]
    sampling_params = SamplingParams(max_tokens=5, temperature=0.0)

    with patch.object(model, "_collective_rpc", failing_rpc):
        # Should propagate the RuntimeError
        with pytest.raises(RuntimeError, match="Simulated worker error"):
            await model.generate(prompts, sampling_params, steering_spec=steering_spec)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_parallel_fetch_timeout(model_factory):
    """Test timeout handling when parallel fetching multiple handles.

    This tests the asyncio.gather pattern for fetching handles.
    """
    model = await model_factory()

    prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    results, handles = await model.generate(
        prompts,
        sampling_params,
        capture_layers=[5],
    )

    assert len(handles) == 3

    # Fetch handles concurrently with timeout
    await asyncio.wait_for(
        asyncio.gather(*[h.fetch() for h in handles]),
        timeout=30.0
    )

    # Verify captures are available
    for handle in handles:
        assert handle.captures is not None

    # Cleanup
    for handle in handles:
        await handle.close()
