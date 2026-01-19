"""Pytest configuration for chatspace test suite."""

from __future__ import annotations

import asyncio
import gc
import time
import warnings

import pytest

# Register custom markers to avoid pytest warnings
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "veryslow: marks tests as very slow (requires --run-veryslow)"
    )


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
    parser.addoption(
        "--run-veryslow", action="store_true", default=False, help="run very slow tests (implies --run-slow)"
    )


def pytest_collection_modifyitems(config, items):
    run_slow = config.getoption("--run-slow")
    run_veryslow = config.getoption("--run-veryslow")

    # --run-veryslow implies --run-slow
    if run_veryslow:
        run_slow = True

    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    skip_veryslow = pytest.mark.skip(reason="need --run-veryslow option to run")

    for item in items:
        if "veryslow" in item.keywords:
            if not run_veryslow:
                item.add_marker(skip_veryslow)
        elif "slow" in item.keywords:
            if not run_slow:
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
