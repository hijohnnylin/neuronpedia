"""Pytest configuration for steerllm test suite."""

from __future__ import annotations

import gc
import time
import warnings

import pytest

# vLLM brings in SWIG-backed helper types that currently emit DeprecationWarnings
# under Python 3.11. Filter them here so GPU-enabled steering tests run cleanly.
warnings.filterwarnings(
    "ignore",
    message=r"builtin type SwigPy(?:Packed|Object) has no __module__ attribute",
    category=DeprecationWarning,
)


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        # --run-slow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(autouse=True, scope="function")
def aggressive_cleanup(request):
    """Ensure proper cleanup after each test to prevent state pollution.

    This fixture addresses test isolation issues where vLLM engine state
    from one test can interfere with subsequent tests. It:
    1. Allows time for async cleanup to complete
    2. Forces garbage collection to release resources
    3. Clears CUDA cache if available
    4. Synchronizes CUDA operations
    """
    yield

    # Only perform aggressive cleanup for slow integration tests that use the GPU/engine
    if not request.node.get_closest_marker("slow"):
        return

    # Give a moment for async cleanup to complete
    # Use time.sleep instead of await to avoid event loop issues
    time.sleep(0.2)

    # Force garbage collection to release any lingering references
    gc.collect()
    gc.collect()  # Second pass to catch objects freed by first pass

    # Clear CUDA cache if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass

    # Brief pause to ensure cleanup completes before next test
    time.sleep(0.2)
