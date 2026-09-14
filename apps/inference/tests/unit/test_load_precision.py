"""``--quantization`` and ``--kv_cache_dtype``: from start.py flag to the ``load_model`` call.

Both are one name on ``load_model`` (interp-engine >= 1.8) and this server passes them through by
that name, so the tests here are about the edges: a pod that sets neither makes the same call it
always did, an engine too old to take them is refused with the pin to bump rather than dying in a
constructor, and the KV cache dtype reaches the serving-limit arithmetic as a byte width.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from neuronpedia_inference import server
from neuronpedia_inference.server import _kv_cache_dtype_for_sizing, _load_precision_kwargs

START_PY = Path(server.__file__).resolve().parents[1] / "start.py"


def _new_load_model(hf_model_id, *, backend="auto", dtype="auto", quantization="", kv_cache_dtype="auto", **kw):
    """The 1.8 signature, so these tests do not depend on which engine the venv holds."""


def _old_load_model(hf_model_id, *, backend="auto", dtype="auto", **backend_kwargs):
    """The 1.7 signature: the two names would fall into ``**backend_kwargs``."""


def test_a_pod_that_sets_neither_adds_nothing_to_the_call(monkeypatch):
    monkeypatch.setattr(server, "load_model", _old_load_model)
    assert _load_precision_kwargs("", "auto") == {}
    assert _load_precision_kwargs("", "") == {}


def test_both_names_reach_load_model_as_load_model_spells_them(monkeypatch):
    monkeypatch.setattr(server, "load_model", _new_load_model)
    assert _load_precision_kwargs("fp8", "fp8") == {"quantization": "fp8", "kv_cache_dtype": "fp8"}
    assert _load_precision_kwargs("bnb-4bit", "auto") == {"quantization": "bnb-4bit"}
    assert _load_precision_kwargs("", "fp8") == {"kv_cache_dtype": "fp8"}


def test_the_pinned_engine_takes_both_names():
    """Against the engine the venv really holds, so a pin rolled back below 1.8 fails here first."""
    assert _load_precision_kwargs("fp8", "fp8") == {"quantization": "fp8", "kv_cache_dtype": "fp8"}


def test_an_engine_too_old_for_the_flag_is_refused_with_the_pin_to_bump(monkeypatch):
    """On 1.7 the name would land in a constructor as an opaque TypeError, after the pod was paid for."""
    monkeypatch.setattr(server, "load_model", _old_load_model)
    with pytest.raises(ValueError, match="quantization needs interp-engine >= 1.8"):
        _load_precision_kwargs("fp8", "auto")
    with pytest.raises(ValueError, match="kv_cache_dtype needs interp-engine >= 1.8"):
        _load_precision_kwargs("", "fp8")


def test_the_kv_cache_is_priced_at_its_own_dtype_not_the_models():
    """An fp8 cache holds twice the tokens; sizing it at bf16 would admit half the concurrency."""
    assert _kv_cache_dtype_for_sizing("bfloat16", "auto") == "bfloat16"
    assert _kv_cache_dtype_for_sizing("bfloat16", "") == "bfloat16"
    assert _kv_cache_dtype_for_sizing("bfloat16", "fp8") == "float8"
    assert _kv_cache_dtype_for_sizing("bfloat16", "fp8_e4m3") == "float8"


def test_start_py_exports_the_flags_as_the_env_vars_args_py_reads():
    """The whole start.py -> env -> args.py handoff, run the way a pod runs it."""
    env = {k: v for k, v in os.environ.items() if k not in ("MODEL_QUANTIZATION", "KV_CACHE_DTYPE")}
    code = (
        "import os, sys; sys.argv = ['start.py', '--quantization', 'fp8', '--kv_cache_dtype', 'fp8', "
        "'--list_models']; import start; "
        "import neuronpedia_inference.args as a; a.list_available_options = lambda: None; "
        "start.main(); print(os.environ['MODEL_QUANTIZATION'], os.environ['KV_CACHE_DTYPE'])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=START_PY.parent, env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.split()[-2:] == ["fp8", "fp8"]


def test_args_py_defaults_mean_as_stored_and_the_model_dtype(monkeypatch):
    monkeypatch.delenv("MODEL_QUANTIZATION", raising=False)
    monkeypatch.delenv("KV_CACHE_DTYPE", raising=False)
    from neuronpedia_inference.args import parse_env_and_args

    args = parse_env_and_args()
    assert args.quantization == ""
    assert args.kv_cache_dtype == "auto"
    monkeypatch.setenv("MODEL_QUANTIZATION", "fp8")
    monkeypatch.setenv("KV_CACHE_DTYPE", "fp8")
    args = parse_env_and_args()
    assert (args.quantization, args.kv_cache_dtype) == ("fp8", "fp8")
