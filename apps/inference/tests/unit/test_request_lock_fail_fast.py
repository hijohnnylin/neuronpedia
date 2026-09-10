"""``with_request_lock`` must refuse a fail-fast request at every gate it guards.

The steer endpoints are admitted by the decorator rather than inline, so ``fail_if_busy`` only
reaches them if the decorator reads it off the request and passes it to all three gates. One gate
left blocking holds the connection for the full timeout, which is the failure this guards: the
client is waiting to be refused so it can try a different pod, and a slow refusal is no better
than none.
"""

import asyncio
from functools import wraps

import pytest
from pydantic import BaseModel

from neuronpedia_inference.sae_cache import sae_cache
from neuronpedia_inference.shared import (
    ConcurrencyLimiter,
    RequestBusy,
    VramBudget,
    budget,
    limiter,
    with_request_lock,
)

GIB = 1024**3


def async_test(func):  # type: ignore[no-untyped-def]
    """Run an async test body. pytest-asyncio is not a dependency of this project.

    ``wraps`` is what lets pytest still see the fixture parameters through the wrapper.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        return asyncio.run(func(*args, **kwargs))

    return wrapper


class FailFastRequest(BaseModel):
    """Stands in for a steer request: what matters is that the field exists."""

    fail_if_busy: bool = False


class PlainRequest(BaseModel):
    """An endpoint that never declared the field, which must keep queueing."""


@pytest.fixture
def isolated_limits(monkeypatch: pytest.MonkeyPatch):
    """A one-slot limiter and a 1 GiB budget, swapped in for the process-wide globals."""
    fresh_limiter = ConcurrencyLimiter()
    fresh_limiter.configure(concurrent=True, max_concurrent=1)
    fresh_budget = VramBudget()
    fresh_budget.configure(GIB)

    for module in ("neuronpedia_inference.shared",):
        monkeypatch.setattr(f"{module}.limiter", fresh_limiter)
        monkeypatch.setattr(f"{module}.budget", fresh_budget)
    return fresh_limiter, fresh_budget


@async_test
async def test_a_taken_slot_refuses_a_fail_fast_request(isolated_limits):
    fresh_limiter, _ = isolated_limits

    @with_request_lock(exclusive=False)
    async def handler(request: FailFastRequest) -> str:  # noqa: ARG001
        return "served"

    async with fresh_limiter.slot(exclusive=False):
        with pytest.raises(RequestBusy):
            await handler(FailFastRequest(fail_if_busy=True))


@async_test
async def test_a_full_budget_refuses_a_fail_fast_request(isolated_limits):
    """The gate after the slot. The slot check is instant, so this is the one that used to hang."""
    _, fresh_budget = isolated_limits

    @with_request_lock(exclusive=False, cost=lambda _request: GIB // 2)
    async def handler(request: FailFastRequest) -> str:  # noqa: ARG001
        return "served"

    async with fresh_budget.reserve(GIB):
        with pytest.raises(RequestBusy):
            await handler(FailFastRequest(fail_if_busy=True))

    assert fresh_budget.available_bytes == GIB


@async_test
async def test_a_free_server_still_serves_a_fail_fast_request(isolated_limits):
    @with_request_lock(exclusive=False, cost=lambda _request: GIB // 4)
    async def handler(request: FailFastRequest) -> str:  # noqa: ARG001
        return "served"

    assert await handler(FailFastRequest(fail_if_busy=True)) == "served"


@async_test
async def test_without_the_flag_a_busy_server_still_queues(isolated_limits):
    """The default has to stay "wait": every caller that never sets the flag depends on it."""
    fresh_limiter, _ = isolated_limits

    @with_request_lock(exclusive=False)
    async def handler(request: FailFastRequest) -> str:  # noqa: ARG001
        return "served"

    held = await fresh_limiter.acquire(exclusive=False)
    queued = asyncio.create_task(handler(FailFastRequest(fail_if_busy=False)))
    await asyncio.sleep(0)
    assert not queued.done()

    held.release()
    assert await queued == "served"


@async_test
async def test_an_endpoint_without_the_field_is_unaffected(isolated_limits):
    """Read with getattr, so an endpoint that never declared the field keeps the old behaviour."""
    fresh_limiter, _ = isolated_limits

    @with_request_lock(exclusive=False)
    async def handler(request: PlainRequest) -> str:  # noqa: ARG001
        return "served"

    held = await fresh_limiter.acquire(exclusive=False)
    queued = asyncio.create_task(handler(PlainRequest()))
    await asyncio.sleep(0)
    assert not queued.done()

    held.release()
    assert await queued == "served"


@async_test
async def test_a_full_sae_cache_refuses_a_fail_fast_request(monkeypatch: pytest.MonkeyPatch):
    """The third gate. Only reachable with SAE paging on, and it can block for an eviction."""
    monkeypatch.setattr(sae_cache, "_enabled", True, raising=False)
    monkeypatch.setattr(sae_cache, "_budget_bytes", GIB, raising=False)
    if not sae_cache.enabled:
        pytest.skip("sae_cache exposes no way to enable it in a unit test")

    async with sae_cache.reserve(GIB, timeout=1.0):
        with pytest.raises(RequestBusy):
            async with sae_cache.reserve(GIB, fail_if_busy=True):
                pass


def test_the_steer_requests_carry_the_flag():
    """Without the field the decorator has nothing to read, so the wire contract is the test."""
    from neuronpedia_inference.schemas.steer import (
        SteerCompletionChatRequest,
        SteerCompletionRequest,
    )

    for model in (SteerCompletionRequest, SteerCompletionChatRequest):
        assert "fail_if_busy" in model.model_fields
        assert model.model_fields["fail_if_busy"].default is False
        assert model.model_fields["fail_if_busy"].alias == "failIfBusy"


def test_the_globals_are_what_the_decorator_uses():
    """Guards the fixture above: if these stop being module attributes it patches nothing."""
    assert isinstance(limiter, ConcurrencyLimiter)
    assert isinstance(budget, VramBudget)
