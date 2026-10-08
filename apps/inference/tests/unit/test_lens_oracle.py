"""``/v1/lens/oracle``'s startup settings and read cache. No weights and no GPU."""

from __future__ import annotations

import asyncio

import pytest
from interp_engine.oracle import ALL_LAYERS, FAST_LAYERS, OraclePartial, OracleRead

from neuronpedia_inference.endpoints.lens.oracle import (
    ORACLE_REGISTRY,
    _each_position,
    _ReadCache,
    parse_oracle_layers,
    resolve_oracle_ref,
)


def test_auto_loads_the_registered_adapter_and_nothing_elsewhere():
    assert resolve_oracle_ref("auto", "Qwen/Qwen3.6-27B") == tuple(ORACLE_REGISTRY["Qwen/Qwen3.6-27B"].split(":"))
    assert resolve_oracle_ref(None, "google/gemma-3-1b-it") is None
    assert resolve_oracle_ref("off", "Qwen/Qwen3.6-27B") is None
    assert resolve_oracle_ref("me/ao:run1", "google/gemma-3-1b-it") == ("me/ao", "run1")
    with pytest.raises(ValueError, match="ORACLE_LENS"):
        resolve_oracle_ref("me/ao", "Qwen/Qwen3.6-27B")


def test_the_layer_setting_switches_between_all_fast_and_a_list():
    assert parse_oracle_layers("all", ALL_LAYERS) == ALL_LAYERS
    assert parse_oracle_layers("fast", ALL_LAYERS) == FAST_LAYERS
    assert parse_oracle_layers("20, 60", ALL_LAYERS) == (20, 60)
    with pytest.raises(ValueError, match="outside"):
        parse_oracle_layers("20,63", ALL_LAYERS)


def test_the_cache_keys_on_the_prefix_and_evicts_the_oldest():
    cache = _ReadCache(size=2)
    read = OracleRead(layer=20, text="- a\n", bullets=["a"], token_ids=[1], finish="bullets")
    k1 = cache.key("ao", [1, 2, 3], 20, 2, 96)
    assert k1 != cache.key("ao", [1, 2, 4], 20, 2, 96)
    assert k1 != cache.key("ao", [1, 2, 3], 20, 3, 96)
    cache.put(k1, read)
    cache.put(cache.key("ao", [1], 20, 2, 96), read)
    assert cache.get(k1) is read
    cache.put(cache.key("ao", [2], 20, 2, 96), read)
    assert cache.get(k1) is read
    assert cache.get(cache.key("ao", [1], 20, 2, 96)) is None


def _read(layer: int, text: str) -> OracleRead:
    return OracleRead(layer=layer, text=text, bullets=[text], token_ids=[1], finish="eos")


async def _stream(layer: int, delay: float, log: list[str], name: str):
    log.append(f"{name} start")
    yield OraclePartial(layer, name[:1])
    await asyncio.sleep(delay)
    yield _read(layer, name)
    log.append(f"{name} end")


def _run(streams, together):
    async def go():
        return [(p, item) async for p, item in _each_position(streams, together=together)]

    return asyncio.run(go())


def test_positions_run_together_and_each_item_carries_its_position():
    log: list[str] = []
    got = _run({3: _stream(20, 0.05, log, "slow"), 7: _stream(24, 0.0, log, "fast")}, together=True)
    reads = [(p, item.text) for p, item in got if isinstance(item, OracleRead)]
    assert reads == [(7, "fast"), (3, "slow")]
    assert log[:2] == ["slow start", "fast start"]
    assert {(p, item.text) for p, item in got if isinstance(item, OraclePartial)} == {(3, "s"), (7, "f")}


def test_positions_run_one_after_another_when_not_together():
    log: list[str] = []
    got = _run({3: _stream(20, 0.01, log, "a"), 7: _stream(24, 0.0, log, "b")}, together=False)
    assert [p for p, _ in got] == [3, 3, 7, 7]
    assert log == ["a start", "a end", "b start", "b end"]


def test_an_error_in_one_position_ends_the_others():
    log: list[str] = []

    async def boom():
        yield OraclePartial(20, "x")
        raise RuntimeError("worker died")

    async def go():
        async for _ in _each_position({1: boom(), 2: _stream(24, 5.0, log, "long")}, together=True):
            pass

    with pytest.raises(RuntimeError, match="worker died"):
        asyncio.run(asyncio.wait_for(go(), timeout=2.0))
    assert "long end" not in log
