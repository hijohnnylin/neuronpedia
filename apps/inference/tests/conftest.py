import asyncio
import gc
import json
import os
import warnings
from typing import Any

import pytest
import torch
from _pytest.terminal import TerminalReporter
from fastapi.testclient import TestClient

import neuronpedia_inference.server as server
from neuronpedia_inference.args import parse_env_and_args
from neuronpedia_inference.config import Config
from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.server import app, initialize
from neuronpedia_inference.shared import Model

BOS_TOKEN_STR = "<|endoftext|>"
TEST_PROMPT = "Hello, world!"
X_SECRET_KEY = "cat"

_HF_TOKEN_ABSENT_MSG = (
    "HF_TOKEN (and HUGGING_FACE_HUB_TOKEN) is not set: gated-model tests "
    "(@pytest.mark.gated, e.g. google/gemma-3-270m-it) will be SKIPPED. "
    "Set HF_TOKEN to exercise gated models."
)


def _hf_token_present() -> bool:
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


# Set during collection: whether this run actually gave up coverage for want of a token.
# A run that deselects the marker outright -- `-m "not gated"`, which is how the CPU CI job
# invokes pytest -- needs no token, so warning it about one is noise that trains people to
# ignore the warning in the runs where it means something.
_gated_tests_skipped = False


# ``tests/harness.py`` keeps at most ONE server alive, so a configuration that reappears
# after a different one ran pays for a second bring-up -- about a minute for a vLLM
# EngineCore. ``test_engine_parity`` measures eager-CPU and then vLLM gpt2, which tears
# down whatever ran before it either way; hoisting it ahead of the (vLLM gpt2) matrix
# module lets the matrix reuse parity's engine instead of booting its own.
_MODULES_FIRST = ("test_engine_parity.py",)


def _module_rank(item: pytest.Item) -> int:
    return 0 if os.path.basename(str(item.path)) in _MODULES_FIRST else 1


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    config: pytest.Config,  # noqa: ARG001
    items: list[pytest.Item],
) -> None:
    """Reorder for server reuse, and skip gated tests when there's no token to run them.

    ``trylast`` so ``items`` is what will actually run: pytest's own ``-m`` filtering happens
    in this same hook, and seeing the post-filter list is what lets the token warning below
    stay quiet on a run that never asked for a gated model.
    """
    # Stable, so everything else keeps its collection order.
    items.sort(key=_module_rank)

    # Deterministically skip anything marked ``gated`` when no HF token is available.
    if _hf_token_present():
        return
    gated = [item for item in items if "gated" in item.keywords]
    if not gated:
        return
    skip_gated = pytest.mark.skip(reason="HF_TOKEN not set; skipping gated-model test")
    for item in gated:
        item.add_marker(skip_gated)

    global _gated_tests_skipped
    _gated_tests_skipped = True
    # Warned here as well as in the summary: collection warnings surface early enough to
    # explain an otherwise-green run that quietly covered less than it looks like.
    warnings.warn(_HF_TOKEN_ABSENT_MSG, stacklevel=1)


def pytest_terminal_summary(
    terminalreporter: TerminalReporter,
    exitstatus: int,  # noqa: ARG001
    config: pytest.Config,  # noqa: ARG001
) -> None:
    # Warn again in the end-of-run summary so the skip isn't silently missed.
    if _gated_tests_skipped:
        terminalreporter.write_line("")
        terminalreporter.write_line(f"WARNING: {_HF_TOKEN_ABSENT_MSG}", yellow=True, bold=True)


MODEL_ID = "openai-community/gpt2"
# SAELens still keys gpt2 SAEs under the Neuronpedia short id; error messages and
# directory lookups use this after resolve_saelens_model_id.
SAELENS_MODEL_ID = "gpt2-small"
SAE_SOURCE_SET = "res-jb"
SAE_SELECTED_SOURCES = ["7-res-jb"]
ABS_TOLERANCE = 0.1
N_COMPLETION_TOKENS = 10
TEMPERATURE = 0
STRENGTH = 10.0  # Steering mechanism (feature or vector) specific strength
STRENGTH_MULTIPLIER = 10.0  # Multiplier across all steering mechanisms
FREQ_PENALTY = 0.0
SEED = 42
STEER_SPECIAL_TOKENS = False
STEER_FEATURE_INDEX = 5
INVALID_SAE_SOURCE = "fake-source"


# Canonical singletons captured after the (expensive) one-time session init. The
# per-test ``initialize_models`` fixture restores these so a test that mutates the
# global Model/Config/SAEManager (e.g. the FORCE_BACKEND engine suite, which nulls
# the singletons in its teardown) cannot leak broken state into later tests.
_CANONICAL: dict[str, Any] = {}


@pytest.fixture(scope="session")
def _session_init():  # pyright: ignore[reportUnusedFunction]  # referenced by name as a fixture dependency
    """Run the real /initialize logic once per session (model + SAE load).

    Loading openai-community/gpt2 + the res-jb SAE is expensive, so it happens exactly
    once; the resulting singletons are cached in ``_CANONICAL`` for cheap per-test
    restoration.
    """
    # The harness keeps its server running between tests on purpose, which means its
    # ``_restore_env`` has NOT run by the time this fixture is first requested -- and the
    # module reordering above puts a vLLM module first, so what is still in the environment
    # is that server's configuration. Retire it before reading the environment, so this
    # fixture initializes against a known-empty process rather than inheriting one.
    from tests.harness import shutdown_running_server

    shutdown_running_server()

    # Set environment variables for testing
    os.environ.update(
        {
            "MODEL_ID": MODEL_ID,
            # Pinned, not merely defaulted: ``FORCE_BACKEND`` is set by the harness for
            # whichever engine it last booted, so leaving this out let a leaked "vllm" turn
            # the CPU-eager gpt2 below into a vLLM engine -- built here under
            # ``asyncio.run``, whose loop closes on the way out, then driven from each
            # TestClient's own portal. That combination used to hang the whole session.
            "FORCE_BACKEND": "eager",
            "SAE_SETS": json.dumps(["res-jb"]),
            "MODEL_DTYPE": "float16",
            "SAE_DTYPE": "float32",
            "TOKEN_LIMIT": "500",
            "DEVICE": "cpu",
            "INCLUDE_SAE": json.dumps(["7-res-jb"]),  # Only load the specific SAE we want
            "EXCLUDE_SAE": json.dumps([]),
            "MAX_LOADED_SAES": "1",
            "SECRET": X_SECRET_KEY,
        }
    )

    # Re-parse args after setting environment variables
    # This is important to refresh the module-level args in the server module
    server.args = parse_env_and_args()

    # Initialize the model and SAEs
    asyncio.run(initialize())

    _CANONICAL["config"] = Config._instance
    _CANONICAL["sae_manager"] = SAEManager._instance
    _CANONICAL["model"] = Model._instance

    yield

    # Cleanup
    _CANONICAL.clear()
    Config._instance = None
    SAEManager._instance = None
    Model._instance = None  # type: ignore
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


@pytest.fixture
def initialize_models(_session_init: None):  # noqa: ARG001
    """Per-test fixture that (re)binds the canonical singletons.

    Function-scoped so it re-establishes the shared gpt2 / res-jb state before every
    test, defending against cross-test contamination from suites that swap out the
    global Model/Config/SAEManager. The heavy model load still happens only once
    (in ``_session_init``); this only reassigns the ``_instance`` pointers.
    """
    Config._instance = _CANONICAL["config"]
    SAEManager._instance = _CANONICAL["sae_manager"]
    Model._instance = _CANONICAL["model"]
    yield


@pytest.fixture
def client(initialize_models: None):  # noqa: ARG001
    return TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def _shutdown_harness_servers():
    """Tear down the harness's cached server (and its vLLM EngineCore) at session end.

    ``tests.harness.initialized_server`` keeps a backend running for reuse across tests,
    so something has to stop it; leaking an EngineCore holds VRAM and can keep the
    pytest process alive after the summary.
    """
    yield
    from tests.harness import shutdown_running_server

    shutdown_running_server()
