"""Shared test harness: engine/model matrix + server bring-up + capability gating.

Two engines are exercised across the suite:

- ``eager`` -> :class:`interp_engine.EagerModel` (CPU or CUDA).
- ``vllm``     -> :class:`interp_engine.VLLMModel` (CUDA-only).

and a small model matrix that mirrors the three archetypes the endpoints must serve:

- a pretrained/base model (``openai-community/gpt2``; completion/activation/lens, no chat),
- a non-thinking instruct model (``google/gemma-3-270m-it``; gated -> needs ``HF_TOKEN``),
- a thinking instruct model (``Qwen/Qwen3.5-0.8B``; hybrid Gated-DeltaNet + Gated-Attention,
  supports ``enable_thinking``).

The gating helpers here let a test declare what it needs (CUDA, vLLM, a gated HF repo, an
instruct/thinking model) and ``pytest.skip`` cleanly when the box can't provide it, so the
same suite is green on a CPU laptop, this CUDA dev box, and the managed GPU CI runner.
"""

from __future__ import annotations

import contextlib
import gc
import json
import os
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient
from interp_engine import VLLMModel

import neuronpedia_inference.server as server
from neuronpedia_inference.args import parse_env_and_args
from neuronpedia_inference.config import Config
from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.server import app, initialize
from neuronpedia_inference.shared import Model

X_SECRET_KEY = "cat"

# --- engines -----------------------------------------------------------------

EAGER = "eager"
VLLM = "vllm"


# --- model matrix ------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """One row of the model matrix.

    ``model_id`` is the Hugging Face repo id passed as ``MODEL_ID`` (weights load from
    it directly). SAELens short names (e.g. ``gpt2-small``) are resolved from
    ``np_model_to_hf.json`` at SAE load time. ``sae_sets``/``include_sae`` load SAEs for
    the activation endpoints; instruct models here ship without SAEs so their lists are
    empty (chat/tokenize/completion don't need an SAE).
    """

    key: str
    model_id: str
    dtype: str = "float32"
    is_chat: bool = False
    is_thinking: bool = False
    is_gated: bool = False
    sae_sets: list[str] = field(default_factory=list)
    include_sae: list[str] = field(default_factory=list)


GPT2 = ModelSpec(
    key="gpt2",
    model_id="openai-community/gpt2",
    dtype="float32",
    sae_sets=["res-jb"],
    # 7-res-jb is what the endpoint tests have always used. 10-res-jb carries the curated
    # "cats" feature (16899) that tests/model_expectations.yaml reads and steers with; it is
    # the layer prod calibrated against, and one res-jb SAE is small enough that loading a
    # second costs less than re-deriving the expectations on layer 7 would.
    include_sae=["7-res-jb", "10-res-jb"],
)

GEMMA_IT = ModelSpec(
    key="gemma-3-270m-it",
    model_id="google/gemma-3-270m-it",
    dtype="float32",
    is_chat=True,
    is_gated=True,
)

QWEN_THINKING = ModelSpec(
    key="qwen3.5-0.8b",
    model_id="Qwen/Qwen3.5-0.8B",
    dtype="bfloat16",
    is_chat=True,
    is_thinking=True,
)

# The one checkpoint that ships vector assets. The read path can be unit-tested with
# synthetic tensors, but whether capture, the pinned template kwargs and the projection agree
# can only be seen on the model they were fitted for. No SAEs -- a readout needs the forward
# pass and the capture hooks, nothing else.
#
# Deliberately not in MODELS: 16GB of gated weights is more than CI should pull, so its
# model_expectations.yaml row is `tier: manual` and `test_manual_rows_are_not_in_the_ci_matrix`
# holds it out of the matrix. Tests using this spec opt in the same way, via
# NP_RUN_MANUAL_TESTS.
LLAMA_8B_TRAITS = ModelSpec(
    key="llama3.1-8b-it",
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    dtype="bfloat16",
    is_chat=True,
    is_gated=True,
)

MODELS: dict[str, ModelSpec] = {m.key: m for m in (GPT2, GEMMA_IT, QWEN_THINKING)}


# --- capability probes -------------------------------------------------------


def cuda_available() -> bool:
    return torch.cuda.is_available()


def vllm_available() -> bool:
    """Whether the vLLM backend imported (independent of whether the engine can start)."""
    try:
        from neuronpedia_inference.server import VLLM_AVAILABLE

        return bool(VLLM_AVAILABLE)
    except Exception:  # noqa: BLE001
        return False


def hf_token_present() -> bool:
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


def require_cuda() -> None:
    if not cuda_available():
        pytest.skip("requires a CUDA GPU")


def require_vllm() -> None:
    require_cuda()
    if not vllm_available():
        pytest.skip("requires the vLLM backend (import failed)")


def require_hf_token(spec: ModelSpec) -> None:
    if spec.is_gated and not hf_token_present():
        pytest.skip(f"{spec.model_id} is gated and HF_TOKEN is not set")


# --- server bring-up ---------------------------------------------------------

_ENV_KEYS = (
    "MODEL_ID",
    "FORCE_BACKEND",
    "SAE_SETS",
    "MODEL_DTYPE",
    "SAE_DTYPE",
    "TOKEN_LIMIT",
    "DEVICE",
    "INCLUDE_SAE",
    "EXCLUDE_SAE",
    "MAX_LOADED_SAES",
    "SECRET",
    "VLLM_GPU_MEMORY_UTILIZATION",
    "VLLM_USE_FLASHINFER_SAMPLER",
)


def _model_instance() -> Any:
    """``Model._instance`` is only present after the first ``set_instance`` / assignment."""
    return getattr(Model, "_instance", None)


def _reset_singletons() -> None:
    # Drop a CUDA EagerModel off-device before losing the reference so the next vLLM
    # bring-up isn't fighting orphaned weights for free memory.
    model = _model_instance()
    if model is not None and not isinstance(model, VLLMModel):
        with contextlib.suppress(Exception):
            hf = getattr(model, "hf_model", None) or getattr(model, "model", None)
            if hf is not None and hasattr(hf, "to"):
                hf.to("cpu")
            del model
    Config._instance = None
    SAEManager._instance = None
    Model._instance = None  # type: ignore[assignment]
    # Allow a subsequent ``initialized_server`` / TestClient lifespan to re-init cleanly.
    server.initialized = False
    server.initialization_error = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _shutdown_vllm_engine() -> None:
    """Tear down a live vLLM EngineCore so it doesn't orphan VRAM across tests."""
    model = _model_instance()
    if not isinstance(model, VLLMModel) or model.engine is None:
        return
    with contextlib.suppress(Exception):
        model.engine.shutdown()
    model.engine = None


@dataclass
class _RunningServer:
    key: tuple[Any, ...]
    client: TestClient
    stack: ExitStack


# At most one server is live at a time. Booting a backend (especially a vLLM
# EngineCore) costs seconds and a few GiB of VRAM, so it is kept running and handed to
# every test that asks for the same configuration; a differing config, an unhealthy
# server, or the end of the session tears it down.
_running: _RunningServer | None = None


def shutdown_running_server() -> None:
    """Tear down the cached server, if any. Idempotent."""
    global _running
    if _running is None:
        return
    srv, _running = _running, None
    srv.stack.close()


def _restore_env(prev_env: dict[str, str | None]) -> None:
    for k, v in prev_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _server_healthy(srv: _RunningServer) -> bool:
    """Whether the cached server can serve another test.

    A test may have killed the backend (a crashed EngineCore, a swapped-out singleton),
    and reusing it would fail every later test in confusing ways -- so reuse is only
    allowed when the model is loaded, the vLLM engine is alive, and /health answers.
    """
    if not server.initialized:
        return False
    model = _model_instance()
    if model is None:
        return False
    if isinstance(model, VLLMModel):
        engine = model.engine
        if engine is None or getattr(engine, "errored", False):
            return False
    try:
        return srv.client.get("/health").status_code == 200
    except Exception:  # noqa: BLE001
        return False


def _boot_server(
    key: tuple[Any, ...],
    spec: ModelSpec,
    engine: str,
    device: str,
    max_loaded_saes: int,
) -> _RunningServer:
    env_update = {
        "MODEL_ID": spec.model_id,
        "FORCE_BACKEND": engine,
        "SAE_SETS": json.dumps(spec.sae_sets),
        "MODEL_DTYPE": spec.dtype,
        # bf16, because that is what local_scripts/pods.yaml serves (`--sae_dtype bfloat16`).
        # An activation magnitude or a steer band measured against fp32 SAEs is a number about
        # a configuration nothing runs, and it costs twice the VRAM to produce.
        "SAE_DTYPE": "bfloat16",
        "TOKEN_LIMIT": "500",
        "DEVICE": device,
        "INCLUDE_SAE": json.dumps(spec.include_sae),
        "EXCLUDE_SAE": json.dumps([]),
        "MAX_LOADED_SAES": str(max_loaded_saes),
        "SECRET": X_SECRET_KEY,
    }
    if engine == VLLM:
        vllm_defaults = {
            # vLLM's default 0.9 util refuses to start when anything else holds a few GiB
            # (desktop compositor, a prior EagerModel CUDA load, ...). Tests only need gpt2 /
            # small instruct models.
            "VLLM_GPU_MEMORY_UTILIZATION": "0.5",
            # FlashInfer's sampler JIT needs nvcc; many GPU boxes (and this one) only ship
            # the driver, so keep vLLM startable without a CUDA toolkit. Managed GPU CI has
            # CUDA preinstalled and can set this back to 1.
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            # Deliberately NOT setting VLLM_ALLOW_INSECURE_SERIALIZATION here, and nothing
            # else does either: the engine reaches its worker hooks by name through the
            # InterpWorkerExtension worker_extension_cls, so no callable is ever pickled.
            # Setting it here would hide a regression back to callable RPC, which fails
            # only on a real out-of-process engine core -- i.e. only on GPU CI or in prod.
        }
        # An explicit value from the environment wins, so CI can tune these.
        env_update.update({k: v for k, v in vllm_defaults.items() if os.environ.get(k) is None})

    stack = ExitStack()
    try:
        # Teardown order is the reverse of registration: shut the vLLM engine down while
        # the portal is still alive (shutdown after the loop closes hangs), then close
        # the client, then drop singletons, then put the environment back.
        stack.callback(_restore_env, {k: os.environ.get(k) for k in _ENV_KEYS})
        stack.callback(_reset_singletons)
        os.environ.update(env_update)
        _reset_singletons()
        server.args = parse_env_and_args()

        # Suppress the app's own startup handler, which fires initialization into a
        # background task we cannot await. We run ``initialize()`` ourselves below.
        server.initialized = True
        client = stack.enter_context(TestClient(app))
        stack.callback(_shutdown_vllm_engine)
        server.initialized = False

        # Initialize on the TestClient's portal -- the same event loop that will serve
        # every request. AsyncLLM binds to the loop that creates it, so initializing on
        # a throwaway ``asyncio.run`` loop would leave the engine bound to a closed loop.
        portal = client.portal
        assert portal is not None, "TestClient must be entered before initializing"
        portal.call(initialize)
    except BaseException:
        stack.close()
        raise
    return _RunningServer(key=key, client=client, stack=stack)


@contextmanager
def initialized_server(
    spec: ModelSpec,
    *,
    engine: str = EAGER,
    device: str | None = None,
    max_loaded_saes: int = 1,
) -> Generator[TestClient, None, None]:
    """Yield a TestClient against a real inference server for ``spec`` on ``engine``.

    The server is *cached across tests*: asking for a configuration that is already
    running reuses it instead of reloading the model, so a whole file of vLLM tests pays
    for one EngineCore. Exiting the context therefore does not shut the server down;
    that happens when a different configuration is requested, when the server stops
    looking healthy, or at end of session (see the ``_shutdown_servers`` fixture).

    Because only one server runs at a time, these contexts must not be *nested* with
    different configurations -- the inner one tears the outer one down.

    Raises (does not skip) on load failure; callers decide whether a failure is a skip
    (missing weights/arch) or a real error, since that distinction is test-specific.
    """
    global _running
    if engine == VLLM:
        require_vllm()
    require_hf_token(spec)

    resolved_device = device or ("cuda" if (engine == VLLM or cuda_available()) else "cpu")
    # The SAE lists are part of the key, not just the model key: a caller can hand in a
    # ``replace()``d spec that loads different SAEs (tests/model_expectations.yaml does, to
    # give an otherwise SAE-less matrix model activation coverage). Keying on ``spec.key``
    # alone would hand that caller the previous server, whose SAEManager knows nothing about
    # the sources it is about to ask for -- a 400 that looks like a bad request rather than a
    # reused server.
    key = (
        spec.key,
        engine,
        resolved_device,
        max_loaded_saes,
        tuple(spec.sae_sets),
        tuple(spec.include_sae),
    )

    if _running is not None and (_running.key != key or not _server_healthy(_running)):
        shutdown_running_server()
    if _running is None:
        _running = _boot_server(key, spec, engine, resolved_device, max_loaded_saes)

    try:
        yield _running.client
    except BaseException as exc:
        # A skip is environmental and leaves the server usable; anything else may have
        # left the backend wedged, so don't hand it to the next test.
        if not isinstance(exc, pytest.skip.Exception):
            shutdown_running_server()
        raise


@contextmanager
def try_initialized_server(spec: ModelSpec, *, engine: str = EAGER, **kwargs: Any) -> Generator[TestClient, None, None]:
    """Like :func:`initialized_server` but turns a *load* failure into a ``pytest.skip``.

    Use for the extra-model coverage (gemma/qwen) and vLLM paths that legitimately can't run
    off the provisioned GPU CI runner (weights not cached, arch not supported by this vLLM,
    no CUDA toolkit for FlashInfer's JIT, ...). Environment gaps skip; they are not failures.

    Only the bring-up is guarded -- assertion failures inside the ``with`` body propagate
    normally so genuine regressions still fail.
    """
    cm = initialized_server(spec, engine=engine, **kwargs)
    try:
        client = cm.__enter__()
    except pytest.skip.Exception:
        raise
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"could not initialize {spec.model_id} on {engine}: {type(exc).__name__}: {str(exc)[:200]}")
        return  # unreachable; keeps type-checkers happy
    try:
        yield client
    finally:
        cm.__exit__(None, None, None)
