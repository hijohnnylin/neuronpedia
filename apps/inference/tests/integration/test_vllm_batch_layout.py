"""Regression guard for the vLLM-internal batch-layout assumptions the per-request
demux relies on (``interp_engine/vllm_capture/_demux.py``, in the engine repo).

The demux attributes each row of the batched forward's flattened ``[num_tokens, hidden]``
tensor to its originating request. That mapping reads vLLM internals that changed in
0.25.1 and are NOT part of vLLM's public API, so a future vLLM bump can silently break
it. These tests fail LOUDLY (with a pointer to the code to fix) if any assumption moves:

1. The input-preparation seam exists on BOTH GPU runners, which is not academic: vLLM
   sends MoE architectures (gpt-oss and friends) to the V1 runner and everything else
   to V2, and the two expose different methods.
   - V2: ``model_runner.prepare_inputs(scheduler_output, ...)`` returns an
     ``InputBatch`` exposing ``req_ids`` + ``num_scheduled_tokens``. (Older vLLM used
     ``model_runner.input_batch`` / a ``[cached, new]`` scheduler ordering -- both gone
     in 0.25.1, which instead sorts the batch by token count. We therefore read the
     InputBatch that ``prepare_inputs`` actually produced.)
   - V1: ``model_runner._prepare_inputs(scheduler_output, num_scheduled_tokens)``, whose
     return value is unrelated to the layout; the row counts are the argument and the
     order is ``model_runner.input_batch.req_ids``.
2. That ``(req_ids, seq_lens)`` ordering EXACTLY matches the flat tensor the hooks see --
   verified by CONTENT (distinct-length concurrent captures must each equal their
   single-request baseline), not by trusting the ordering.
3. vLLM appends a child suffix to the ``request_id`` we pass to ``generate()``
   (``"<request_id>-<hash>"``); the demux resolves a batch req_id back to our
   registration by prefix (``_resolve_rid``).

Every test runs against both runners: the dense test model would always take the V2 path,
so ``VLLM_USE_V2_MODEL_RUNNER`` pins each parametrization instead of requiring a MoE model
(a 20B one) to reach V1.

Requires CUDA + vLLM, so skipped on the CPU CI / macOS.

Run on a GPU box:
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
    .venv/bin/python -m pytest tests/integration/test_vllm_batch_layout.py -v

``spawn`` matters: importing this module touches CUDA (the skipif above), so a forked
EngineCore dies with "Cannot re-initialize CUDA in forked subprocess" and the fixture
turns that into a SKIP -- green, but guarding nothing. Check for PASSED, not just a
zero exit.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import importlib.util
import os
from typing import Any

import pytest
import torch
from interp_engine import Address

_VLLM = importlib.util.find_spec("vllm") is not None

# Markers so the GPU CI workflow selects this via `-m "cuda or vllm"`; the skipif keeps it
# out of the CPU suite (and off a GPU box without vLLM).
pytestmark = [
    pytest.mark.cuda,
    pytest.mark.vllm,
    pytest.mark.skipif(
        not (torch.cuda.is_available() and _VLLM),
        reason="vLLM per-request-demux layout guard requires CUDA + vLLM",
    ),
]

MODEL = "Qwen/Qwen3-0.6B"
LAYER = 8
POINT = Address("resid_post", LAYER)
POINTS = [POINT]
# What crosses ``collective_rpc``: the worker takes and returns canonical address strings, and keys
# its store with them. Derived from POINT rather than spelled out, so the tests that drive the
# worker directly cannot drift from the grammar in ``interp_engine.address``.
WIRE_KEY = str(POINT)


async def _cancel_pending_tasks() -> None:
    """Cancel and await every other task on the running loop.

    This is what ``asyncio.run`` does on the way out; a hand-made loop has to do it
    itself (see the teardown in ``demux``).
    """
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in pending:
        task.cancel()
    if pending:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=10)


def _restore_env(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


def _close_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Drain, then close, a hand-made loop.

    ``AsyncLLM.shutdown()`` only *requests* cancellation of its background tasks
    (``output_handler`` and ``EngineCoreOutputQueueTask``): they are cancelled the next
    time their loop runs. Closing the loop before that leaves them pending, which pytest
    reports at end of session as "Task was destroyed but it is pending!" followed by an
    "Event loop is closed" traceback out of ``output_handler`` when it is GC'd.
    """
    with contextlib.suppress(Exception):
        loop.run_until_complete(_cancel_pending_tasks())
        loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()
    asyncio.set_event_loop(None)


@pytest.fixture(scope="module", params=["v2", "v1"])
def demux(request: pytest.FixtureRequest):
    """(backend, run, runner_api) sharing ONE persistent event loop, once per runner.

    ``AsyncLLM`` starts a background engine loop bound to the event loop it is created
    in, so every async call must run on that SAME loop. Using ``asyncio.run`` per call
    would create (and immediately close) a fresh loop, leaving the engine's background
    loop dead -> ``generate()`` hangs forever. We therefore keep one module loop and run
    every coroutine on it via the returned ``run`` helper.

    Each parametrization builds (and tears down) its own engine, since the runner is
    fixed at engine-construction time.
    """
    runner_api = request.param
    # This file drives its own backend instead of the harness's cached server, so release
    # that server first -- otherwise its EngineCore still holds its VRAM reservation and
    # this engine fails to start with "Free memory ... less than desired". It must happen
    # *before* the env setup below, because tearing the server down also restores the
    # environment it captured at boot (which would undo these variables).
    from tests.harness import shutdown_running_server

    shutdown_running_server()

    # Same portable sampler path as tests/harness.py (FlashInfer JIT needs nvcc).
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    # Pin the runner for this parametrization. vLLM reads this lazily out of the
    # environment when it builds VllmConfig, so it must be set before the engine starts.
    prev_runner = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if runner_api == "v2" else "0"

    from interp_engine import VLLMModel

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    b = VLLMModel(MODEL, dtype="bfloat16", gpu_memory_utilization=0.4, max_model_len=1024)
    try:
        loop.run_until_complete(b._ensure_engine())
    except Exception as exc:  # noqa: BLE001
        # The batch-layout assumptions these tests guard only apply once the engine is
        # up. If the engine can't even start on this box (e.g. FlashInfer's JIT can't
        # find nvcc / a CUDA toolkit, or there isn't enough free VRAM), that's an
        # environment gap, not a layout regression -- skip rather than error so the
        # suite stays green off the provisioned GPU CI runner.
        _close_loop(loop)
        _restore_env("VLLM_USE_V2_MODEL_RUNNER", prev_runner)
        pytest.skip(f"vLLM engine could not initialize in this environment: {type(exc).__name__}: {str(exc)[:200]}")

    def run(coro: Any) -> Any:
        return loop.run_until_complete(coro)

    yield b, run, runner_api

    # Tear the engine down so the EngineCore subprocess doesn't orphan (holding VRAM):
    # the next parametrization needs that VRAM back to start its own engine.
    with contextlib.suppress(Exception):
        b.engine.shutdown()  # pyright: ignore[reportOptionalMemberAccess]
    _close_loop(loop)
    _restore_env("VLLM_USE_V2_MODEL_RUNNER", prev_runner)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _ids(backend: Any, text: str) -> list[int]:
    return backend.tokenizer(text, add_special_tokens=True)["input_ids"]


def _debug(backend: Any, run: Any) -> dict[str, Any]:
    dbg = run(backend.engine.collective_rpc("demux_debug"))
    return dbg[0] if isinstance(dbg, list | tuple) else dbg


def test_prepare_inputs_snapshot_populated(demux: Any):
    """Assumption 1: the layout snapshot works on whichever runner is in play.

    A successful capture proves the whole chain; the debug snapshot additionally asserts
    the patch is installed on the expected runner API and that no extraction error
    occurred (a renamed attribute would surface here as ``last_error``).
    """
    backend, run, runner_api = demux
    ids = _ids(backend, "The capital of France is")
    caps = run(backend.capture(ids, POINTS))
    assert POINT in caps, "capture returned nothing -> layout snapshot failed"

    d = _debug(backend, run)
    assert d["patched"] is True, "input-preparation patch not installed"
    assert d["runner_api"] == runner_api, (
        f"expected to exercise the {runner_api} runner seam but the demux patched "
        f"{d['runner_api']!r}; VLLM_USE_V2_MODEL_RUNNER may no longer select the runner, "
        "leaving one branch of _ensure_patched untested"
    )
    assert d["last_error"] is None, (
        "vLLM's batch-layout assumption broke -- update _meta_from_input_batch / "
        f"_meta_from_v1_inputs / _ensure_patched in vllm_capture/_demux.py. Error: {d['last_error']}"
    )
    assert d["last_meta"] is not None
    req_ids, seq_lens = d["last_meta"]
    assert len(req_ids) == len(seq_lens) >= 1
    # single in-flight request: its row count must equal the prompt length.
    assert sum(seq_lens) == len(ids), (
        f"snapshot rows {sum(seq_lens)} != prompt len {len(ids)} -> num_scheduled_tokens wrong"
    )


def test_concurrent_distinct_length_attribution(demux: Any):
    """Assumption 2: the snapshot ordering matches the real flat-tensor layout.

    Distinct prompt LENGTHS + CONTENT run concurrently (vLLM batches them). Each request's
    capture must have exactly its own row count and match its single-request baseline --
    so a reordered/mis-sliced batch (the thing that broke on the 0.25.1 sort change) is
    caught by content, not assumed away.
    """
    backend, run, _ = demux
    texts = [
        "Hi there",
        "The capital of France is Paris and",
        "Once upon a midnight dreary while I pondered weak and weary over",
    ]
    ids = [_ids(backend, t) for t in texts]
    assert len({len(x) for x in ids}) == len(ids), "prompts must have distinct lengths"

    async def _go():
        baselines = []
        for i in ids:
            cap = await backend.capture(i, POINTS)
            baselines.append(cap[POINT].float())
        results = await asyncio.gather(*[backend.capture(i, POINTS) for i in ids])
        return baselines, [r[POINT].float() for r in results]

    baselines, concurrent = run(_go())
    for i, (base, cur) in enumerate(zip(baselines, concurrent)):
        assert cur.shape[0] == len(ids[i]), (
            f"req {i}: got {cur.shape[0]} rows, expected {len(ids[i])} -> batch row layout wrong "
            "(update the layout readers in vllm_capture/_demux.py)"
        )
        cos = torch.nn.functional.cosine_similarity(cur.flatten(), base.flatten(), dim=0).item()
        assert cos > 0.999, (
            f"req {i}: concurrent capture disagrees with its baseline (cos={cos:.5f}) -> rows "
            "mis-attributed across requests (batch ordering assumption broke)"
        )


def test_request_id_child_suffix_prefix_resolution(demux: Any):
    """Assumption 3: vLLM's batch req_id starts with the request_id we passed to generate().

    Register capture under a known id, run that id, and confirm (a) the capture landed
    (so ``_resolve_rid``'s prefix match is doing its job) and (b) the observed batch id
    still starts with ours -- the invariant ``_resolve_rid`` depends on.
    """
    from vllm import SamplingParams

    backend, run, _ = demux
    rid = "np-layoutguard-0001"
    ids = _ids(backend, "Hello world, this is a test")

    async def _go():
        await backend.engine.collective_rpc("register_capture", args=(rid, [WIRE_KEY]))
        await backend._run_one(
            {"prompt_token_ids": ids},
            SamplingParams(max_tokens=1, temperature=0.0),
            request_id=rid,
        )
        payloads = await backend.engine.collective_rpc("collect_request", args=(rid,))
        return payloads[0] if isinstance(payloads, list | tuple) else payloads

    payload = run(_go())
    dbg = _debug(backend, run)

    # (a) capture landed -> registered id matched the (suffixed) batch id via prefix.
    assert WIRE_KEY in payload, (
        "capture empty for a known request_id -> _resolve_rid prefix match broke "
        "(vLLM may have changed how it derives the batch request_id)"
    )
    # (b) document/guard the exact invariant _resolve_rid relies on.
    observed = dbg["last_meta"][0][0]
    assert observed == rid or observed.startswith(rid), (
        f"vLLM batch req_id {observed!r} no longer starts with our request_id {rid!r}; "
        "update _resolve_rid in vllm_capture/_demux.py"
    )
