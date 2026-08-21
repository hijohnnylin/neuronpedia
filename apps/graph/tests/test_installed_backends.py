"""Checks graph's engine list against the circuit-tracer that is actually installed.

``MODEL_ENGINES`` hand-duplicates circuit-tracer's ``Backend`` literal, and the two live in
different repositories: nothing but this test says they still agree. ``server.py`` passes the
resolved value straight into ``ReplacementModel.from_pretrained``, so a fork rev that predates an
engine listed here fails at model load -- after ``load_transcoder_from_hub`` has already spent
minutes downloading weights, and only for whoever set that value.

A subset rather than an equality, on purpose: the fork growing a fourth backend is not this app's
problem, but this app offering one the installed fork cannot build is.
"""

import importlib.util
import os
from typing import get_args

import pytest

from neuronpedia_graph.runtime_env import MODEL_ENGINES

# Same split as tests/test_openapi.py, for the same reason: locally a missing attribution extra is
# a skip, but under CI it is a failure, since a job that stopped installing the extra would
# otherwise go green with this check silently absent.
if importlib.util.find_spec("circuit_tracer") is None:
    if os.environ.get("CI"):
        raise RuntimeError(
            "circuit_tracer is not installed but CI is set, so this test would skip instead of run. "
            "The graph job must sync with `--extra circuit-tracer`."
        )
    pytest.skip("needs an attribution extra: uv sync --extra circuit-tracer", allow_module_level=True)

from circuit_tracer.replacement_model.replacement_model import Backend  # noqa: E402


def test_every_engine_we_offer_exists_in_the_installed_circuit_tracer():
    missing = sorted(set(MODEL_ENGINES) - set(get_args(Backend)))
    assert not missing, (
        f"runtime_env.MODEL_ENGINES offers {missing}, which the installed circuit-tracer does not "
        "accept. Check the `circuit-tracer` rev pinned in pyproject.toml against uv.lock."
    )
