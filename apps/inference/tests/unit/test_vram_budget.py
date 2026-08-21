"""The VRAM admission controller: does it actually ration, and does it always give back?

A leak here is worse than no budget at all -- the server would wedge with every request
waiting on bytes that nothing will ever release -- so most of these tests are about the
release path (success, exception, cancellation) rather than the happy path.

No GPU and no model: the budget is plain accounting over asyncio primitives.
"""

from __future__ import annotations

import asyncio
from functools import wraps

import pytest

from neuronpedia_inference.shared import RequestTooLarge, VramBudget

GIB = 1024**3


def async_test(func):
    """Run an async test body. pytest-asyncio is not a dependency of this project."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper


@pytest.fixture
def one_gib_budget() -> VramBudget:
    budget = VramBudget()
    budget.configure(GIB)
    return budget


def test_a_budget_of_zero_is_disabled():
    """The default, and what a non-CUDA device measures: rationing off, reserve a no-op."""
    budget = VramBudget()
    assert not budget.enabled
    assert budget.total_bytes == 0


@async_test
async def test_reserving_and_releasing_returns_the_bytes(one_gib_budget: VramBudget):
    async with one_gib_budget.reserve(GIB // 4):
        assert one_gib_budget.available_bytes == GIB - GIB // 4
    assert one_gib_budget.available_bytes == GIB


@async_test
async def test_bytes_are_returned_even_when_the_request_raises(
    one_gib_budget: VramBudget,
):
    """The leak that would wedge the server: an endpoint that fails mid-request."""
    with pytest.raises(RuntimeError):
        async with one_gib_budget.reserve(GIB // 2):
            raise RuntimeError("endpoint blew up")

    assert one_gib_budget.available_bytes == GIB


@async_test
async def test_bytes_are_returned_when_the_request_is_cancelled(
    one_gib_budget: VramBudget,
):
    """Clients disconnect mid-request (the lens stream in particular), which cancels the task."""
    started = asyncio.Event()

    async def hold():
        async with one_gib_budget.reserve(GIB // 2):
            started.set()
            await asyncio.sleep(3600)

    task = asyncio.create_task(hold())
    await started.wait()
    assert one_gib_budget.available_bytes == GIB // 2

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert one_gib_budget.available_bytes == GIB


@async_test
async def test_a_request_that_cannot_ever_fit_is_rejected_immediately(
    one_gib_budget: VramBudget,
):
    """Waiting would never help, so this must not block for the lock timeout."""
    with pytest.raises(RequestTooLarge) as excinfo:
        async with one_gib_budget.reserve(4 * GIB):
            pass

    assert excinfo.value.needed_bytes == 4 * GIB
    assert excinfo.value.budget_bytes == GIB
    # The message has to name both numbers: it is what the caller sees.
    assert "4.00 GiB" in str(excinfo.value)
    assert "1.00 GiB" in str(excinfo.value)
    assert one_gib_budget.available_bytes == GIB


@async_test
async def test_an_oversized_request_waits_for_room_rather_than_racing(
    one_gib_budget: VramBudget,
):
    """The whole point: two requests that each fit, but do not fit TOGETHER, serialize."""
    order: list[str] = []
    first_holding = asyncio.Event()
    release_first = asyncio.Event()

    async def first():
        async with one_gib_budget.reserve(3 * GIB // 4):
            order.append("first-in")
            first_holding.set()
            await release_first.wait()
        order.append("first-out")

    async def second():
        await first_holding.wait()
        async with one_gib_budget.reserve(3 * GIB // 4):
            order.append("second-in")

    first_task = asyncio.create_task(first())
    second_task = asyncio.create_task(second())

    await first_holding.wait()
    # Let `second` reach the reserve and block there.
    await asyncio.sleep(0.05)
    assert order == ["first-in"], "second must not have been admitted alongside first"

    release_first.set()
    await asyncio.gather(first_task, second_task)

    assert order == ["first-in", "first-out", "second-in"]
    assert one_gib_budget.available_bytes == GIB


@async_test
async def test_requests_that_fit_together_run_concurrently(one_gib_budget: VramBudget):
    """The other half of the point: cheap requests must not be serialized needlessly."""
    both_in = asyncio.Event()
    count = 0

    async def cheap():
        nonlocal count
        async with one_gib_budget.reserve(GIB // 4):
            count += 1
            if count == 2:
                both_in.set()
            await asyncio.wait_for(both_in.wait(), timeout=1)

    await asyncio.gather(cheap(), cheap())
    assert one_gib_budget.available_bytes == GIB


@async_test
async def test_waiting_for_memory_times_out_rather_than_hanging(
    one_gib_budget: VramBudget,
):
    """A held budget must not block a new request forever; the caller turns this into a 503."""
    async with one_gib_budget.reserve(GIB):
        with pytest.raises(asyncio.TimeoutError):
            async with one_gib_budget.reserve(GIB, timeout=0.05):
                pass

    assert one_gib_budget.available_bytes == GIB


@async_test
async def test_reserving_nothing_is_free(one_gib_budget: VramBudget):
    """Endpoints with no estimator (and estimators that failed) pass 0."""
    async with one_gib_budget.reserve(0):
        assert one_gib_budget.available_bytes == GIB


@async_test
async def test_a_disabled_budget_admits_everything():
    budget = VramBudget()
    async with budget.reserve(9999 * GIB):
        assert budget.available_bytes == 0
