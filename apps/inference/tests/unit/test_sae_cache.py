"""SAE paging: does the GPU cache stay inside its budget, and never evict live weights?

The dangerous failure is not a cache miss, it is evicting an SAE a request is mid-encode
on -- the tensors would silently move back to the host and the next kernel would fault, or
worse, the request would read the wrong device. So most of what is tested here is the
interaction between reservations (which bound how much can be in use) and eviction (which
must skip whatever is in use).

Everything runs on the CPU: `stage_in("cpu")` is a no-op move, which leaves exactly the
residency bookkeeping under test and none of CUDA's behaviour.
"""

from __future__ import annotations

import asyncio
from functools import wraps
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from neuronpedia_inference.sae_cache import (
    HostPinner,
    SAEGpuCache,
    _collect_host_state,
    state_nbytes,
)
from neuronpedia_inference.startup_memory import resolve_sae_gpu_budget_bytes

MIB = 1024**2


def async_test(func):
    """Run an async test body. pytest-asyncio is not a dependency of this project."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper


class FakeSAE(torch.nn.Module):
    """Stand-in for a SAELens SAE: weights, a buffer, and the placement attributes.

    SAELens keeps ``sae.device`` / ``sae.cfg.device`` alongside the parameters and endpoints
    read them to place their own tensors, so they are part of what paging has to maintain.
    """

    def __init__(self, d_sae: int = 64, d_in: int = 16):
        super().__init__()
        self.W_enc = torch.nn.Parameter(torch.zeros(d_in, d_sae))
        self.W_dec = torch.nn.Parameter(torch.zeros(d_sae, d_in))
        self.register_buffer("threshold", torch.zeros(d_sae))
        self.device = torch.device("cpu")
        self.cfg = SimpleNamespace(device="cpu")


def make_cache(budget_bytes: int) -> SAEGpuCache:
    cache = SAEGpuCache()
    cache.configure(budget_bytes=budget_bytes, device="cpu", pinned_host_bytes=0)
    return cache


def sae_bytes(d_sae: int = 64, d_in: int = 16) -> int:
    return state_nbytes(_collect_host_state(FakeSAE(d_sae, d_in)))


def test_disabled_by_default():
    cache = SAEGpuCache()
    assert cache.enabled is False
    assert cache.acquire("anything") is None


def test_register_reports_size_and_keeps_weights_off_the_gpu():
    cache = make_cache(10 * MIB)
    module = FakeSAE()
    nbytes = cache.register("0-res", module)

    assert nbytes == sae_bytes()
    assert cache.resident_bytes == 0
    assert cache.host_bytes == nbytes


def test_acquire_stages_in_and_hits_thereafter():
    cache = make_cache(10 * MIB)
    module = FakeSAE()
    cache.register("0-res", module)

    assert cache.acquire("0-res") is module
    assert cache.resident_bytes == sae_bytes()
    assert (cache.hits, cache.misses) == (0, 1)

    cache.acquire("0-res")
    assert (cache.hits, cache.misses) == (1, 1)
    assert cache.resident_bytes == sae_bytes()


def test_stage_out_restores_the_host_tensors_in_place():
    """Eviction must not swap the Parameter objects an endpoint may be holding."""
    cache = make_cache(10 * MIB)
    module = FakeSAE()
    cache.register("0-res", module)
    original_param = module.W_dec

    cache.acquire("0-res")
    cache._evict_locked("0-res")

    assert module.W_dec is original_param
    assert module.W_dec.device.type == "cpu"
    assert cache.resident_bytes == 0


def test_stage_out_reports_the_weights_as_being_back_on_the_host():
    """Stale placement attributes would send an endpoint's tensors to the wrong device."""
    cache = make_cache(10 * MIB)
    module = FakeSAE()
    cache.register("0-res", module)

    cache.acquire("0-res")
    cache._evict_locked("0-res")

    assert module.device == torch.device("cpu")
    assert module.cfg.device == "cpu"


def test_an_evicted_sae_can_be_staged_back_in():
    cache = make_cache(sae_bytes())
    module = FakeSAE()
    cache.register("0-res", module)
    weights = module.W_dec

    cache.acquire("0-res")
    cache._evict_locked("0-res")
    assert cache.acquire("0-res") is module

    assert cache.resident_bytes == sae_bytes()
    assert module.W_dec is weights


def test_evicts_least_recently_used_to_stay_within_budget():
    one = sae_bytes()
    cache = make_cache(2 * one)
    for sae_id in ("a", "b", "c"):
        cache.register(sae_id, FakeSAE())

    cache.acquire("a")
    cache.acquire("b")
    assert cache.resident_bytes == 2 * one

    cache.acquire("c")  # evicts "a", the least recently used
    assert cache.resident_bytes == 2 * one
    assert cache.evictions == 1
    assert cache._resident.keys() == {"b", "c"}


def test_reuse_refreshes_recency():
    one = sae_bytes()
    cache = make_cache(2 * one)
    for sae_id in ("a", "b", "c"):
        cache.register(sae_id, FakeSAE())

    cache.acquire("a")
    cache.acquire("b")
    cache.acquire("a")  # "b" is now the least recently used
    cache.acquire("c")

    assert cache._resident.keys() == {"a", "c"}


def test_budget_grows_to_fit_a_single_oversized_sae():
    """A source that cannot fit is unservable forever, which is worse than overshooting."""
    cache = make_cache(1)
    cache.register("huge", FakeSAE())
    assert cache.budget_bytes == sae_bytes()


@async_test
async def test_reservation_bounds_concurrent_residency():
    one = sae_bytes()
    cache = make_cache(2 * one)
    started = []

    async def request(sae_id: str):
        async with cache.reserve(one, timeout=5):
            started.append(sae_id)
            await asyncio.sleep(0.05)

    tasks = [asyncio.create_task(request(s)) for s in ("a", "b", "c")]
    await asyncio.sleep(0.01)
    assert len(started) == 2, "the third request must wait for residency to free up"

    await asyncio.gather(*tasks)
    assert len(started) == 3


@async_test
async def test_eviction_skips_a_source_another_request_is_using():
    one = sae_bytes()
    cache = make_cache(2 * one)
    for sae_id in ("a", "b"):
        cache.register(sae_id, FakeSAE())

    holding = asyncio.Event()
    release = asyncio.Event()

    async def holder():
        async with cache.reserve(one, timeout=5):
            cache.acquire("a")
            holding.set()
            await release.wait()

    async def contender():
        await holding.wait()
        async with cache.reserve(one, timeout=5):
            cache.acquire("b")
            # "a" is the LRU entry and would normally be evicted to make room for "b".
            assert cache._records["a"].on_gpu, "evicted an SAE that was in use"
        release.set()

    await asyncio.gather(asyncio.create_task(holder()), asyncio.create_task(contender()))


@async_test
async def test_a_request_holds_only_its_most_recent_source():
    """The one-at-a-time contract: acquiring b releases the claim on a."""
    one = sae_bytes()
    cache = make_cache(2 * one)
    for sae_id in ("a", "b", "c"):
        cache.register(sae_id, FakeSAE())

    async with cache.reserve(one, timeout=5):
        cache.acquire("a")
        cache.acquire("b")
        cache.acquire("c")  # must be able to evict "a", which is no longer held

    assert cache.resident_bytes <= 2 * one
    assert cache._records["c"].on_gpu


@async_test
async def test_reservation_is_released_when_the_request_raises():
    one = sae_bytes()
    cache = make_cache(one)

    with pytest.raises(RuntimeError):
        async with cache.reserve(one, timeout=5):
            raise RuntimeError("boom")

    # A second request must not hang waiting on the first one's bytes.
    async with cache.reserve(one, timeout=1):
        pass


@async_test
async def test_reservation_is_released_when_the_request_is_cancelled():
    """Clients disconnect mid-request; a leaked reservation would wedge the cache."""
    one = sae_bytes()
    cache = make_cache(one)
    started = asyncio.Event()

    async def hold():
        async with cache.reserve(one, timeout=5):
            started.set()
            await asyncio.sleep(3600)

    task = asyncio.create_task(hold())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert cache._reserved_bytes == 0
    assert cache._active == set()


@async_test
async def test_reserve_is_a_noop_when_paging_is_off():
    cache = SAEGpuCache()
    async with cache.reserve(10 * MIB, timeout=0) as residency:
        assert residency is None


@pytest.fixture
def paged_manager():
    """A SAEManager wired to a paging cache, with the Hub and SAELens stubbed out."""
    from neuronpedia_inference import sae_manager as manager_module

    modules = {sae_id: FakeSAE() for sae_id in ("0-res-test", "1-res-test")}

    def fake_load(release, sae_id, device, dtype):  # noqa: ARG001
        module = modules[sae_id]
        module.cfg = SimpleNamespace(
            device=device,
            d_sae=64,
            d_in=16,
            metadata=SimpleNamespace(neuronpedia_id=sae_id),
        )
        return module, f"blocks.{sae_id[0]}.hook_resid_pre"

    config = SimpleNamespace(
        model_id="test-model",
        custom_hf_model_id=None,
        sae_dtype="float32",
        max_loaded_saes=500,
        sae_config=[{"set": "res-test", "saes": list(modules)}],
    )
    cache = SAEGpuCache()

    with (
        patch.object(manager_module, "sae_cache", cache),
        patch.object(manager_module.Config, "get_instance", return_value=config),
        patch.object(manager_module, "get_saelens_neuronpedia_directory_df"),
        patch.object(manager_module, "resolve_saelens_model_id", return_value="m"),
        patch.object(
            manager_module,
            "get_sae_lens_ids_from_neuronpedia_id",
            side_effect=lambda model_id, neuronpedia_id, df_exploded: (  # noqa: ARG005
                "release",
                neuronpedia_id,
            ),
        ),
        patch.object(manager_module.SaeLensSAE, "load", side_effect=fake_load),
    ):
        manager = manager_module.SAEManager(
            num_layers=0,
            device="cpu",
            sae_gpu_budget_bytes=sae_bytes(),
            sae_pinned_host_bytes=0,
        )
        manager.load_saes()
        yield manager, cache


def test_manager_loads_saes_to_the_host_and_records_their_size(paged_manager):
    manager, cache = paged_manager

    assert manager.paging_enabled is True
    assert cache.host_bytes == 2 * sae_bytes()
    assert manager.get_sae_nbytes("0-res-test") == sae_bytes()
    # sae_data["sae"] is empty under paging: reading it would hand out host weights.
    assert manager.sae_data["0-res-test"]["sae"] is None


def test_manager_warms_the_cache_up_to_the_budget(paged_manager):
    """The warm fill is what makes the post-startup free-VRAM measurement honest."""
    _manager, cache = paged_manager
    assert cache.resident_bytes == sae_bytes()
    assert cache.stats()["resident_count"] == 1


def test_ensure_source_does_not_stage_anything_in(paged_manager):
    """Priming metadata for 26 selected sources must not drag all 26 onto the GPU."""
    manager, cache = paged_manager
    before = cache.resident_bytes

    manager.ensure_source("1-res-test")

    assert cache.resident_bytes == before
    assert cache.misses == 0


def test_get_sae_stages_in_through_the_cache(paged_manager):
    manager, cache = paged_manager

    sae = manager.get_sae("1-res-test")

    assert sae is cache.peek("1-res-test")
    assert cache._records["1-res-test"].on_gpu
    assert cache.resident_bytes == sae_bytes()  # the other one was evicted


def test_pinner_falls_back_when_over_budget():
    pinner = HostPinner()
    pinner.configure(0)
    state = _collect_host_state(FakeSAE())
    pinned_state, pinned = pinner.pin(state)

    assert pinned is False
    assert pinned_state is state
    assert pinner.used_bytes == 0
    assert pinner.refused_count == 1


@pytest.mark.parametrize(
    "setting,expected",
    [
        (None, 0),
        ("", 0),
        ("0", 0),
        ("off", 0),
        ("8", 8 * 1024**3),
        ("1.5", int(1.5 * 1024**3)),
        ("nonsense", 0),
    ],
)
def test_resolve_gpu_budget_settings(setting, expected):
    assert resolve_sae_gpu_budget_bytes(setting, device="cuda:0", is_vllm=True, vllm_gpu_utilization=0.5) == expected


def test_resolve_gpu_budget_is_off_on_cpu():
    assert resolve_sae_gpu_budget_bytes("auto", device="cpu", is_vllm=False, vllm_gpu_utilization=0.0) == 0
