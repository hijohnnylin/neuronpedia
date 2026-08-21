"""The engine attributes this app reads through ``getattr`` defaults, checked against the real class.

``engine_adapter`` and ``/capabilities`` ask the model what it can do with calls shaped like
``getattr(model, "static_points", ())``. The default is deliberate -- an eager model has no such
attribute, and neither did any vLLM model before the graph backends existed -- but it also means a
rename upstream cannot fail. It returns the default, every pod reports that it captures nothing, and
a router reading ``/capabilities`` stops sending capture traffic to a fleet that would have served
it. No exception, no log line, and the tests here would all still pass, because they run against
``SimpleNamespace`` stubs that spell the names the same way the app does.

So spell them once against the class that really has to have them. This file is the only place in
the app's tests that touches the engine's vLLM class, and it should fail loudly the day one of these
is renamed -- which is exactly what interp-engine 1.3 did to ``frozen_points`` and
``writes_available``.
"""

from __future__ import annotations

import pytest

# The three the capability report and every refusal are derived from. Reading any of them wrong is
# silent, so each is named here rather than covered by a "the class looks about right" check.
CAPABILITY_ATTRS = ("static_points", "static_writes", "graph_replay", "hooks_available")


@pytest.fixture(scope="module")
def vllm_model_class() -> type:
    return pytest.importorskip("interp_engine.vllm_backend").VLLMModel


@pytest.mark.parametrize("name", CAPABILITY_ATTRS)
def test_the_capability_attributes_exist_on_the_real_class(vllm_model_class: type, name: str) -> None:
    assert hasattr(vllm_model_class, name), (
        f"VLLMModel has no {name!r}. Every getattr default in engine_adapter.py and "
        "endpoints/capabilities.py now reports this pod as incapable instead of raising. Find what "
        "the engine renamed it to and follow the rename here."
    )


def test_the_deferred_declaration_call_still_exists(vllm_model_class: type) -> None:
    """``STATIC_POINTS=sae`` calls this after the SAEs load, before engine warmup. Unlike the reads
    above it would raise on a rename, but it raises during startup on one deploy shape only, so it
    is cheaper to catch here."""
    assert callable(getattr(vllm_model_class, "configure_static", None))


def test_the_static_module_still_provides_what_sae_declaration_needs() -> None:
    """``sae_static_addresses`` imports these by name inside the function, so a rename surfaces as a
    500 on a ``STATIC_POINTS=sae`` pod's first request rather than at import time."""
    static = pytest.importorskip("interp_engine.vllm_capture.static")
    for name in ("ATTN_STATIC_POINT", "static_unsupported_reason", "steer_write_for_sae_point"):
        assert hasattr(static, name), f"interp_engine.vllm_capture.static has no {name!r}"


def test_the_backend_names_this_app_asks_for_are_the_ones_the_engine_offers() -> None:
    """``_vllm_engine_backend`` returns these three strings. ``load_model`` validates its ``backend``
    argument, so a typo here is caught at startup -- but only on the pod shape that hits it, and the
    generate and static shapes are the two nobody runs locally."""
    load = pytest.importorskip("interp_engine.load")
    assert set(load.VLLM_BACKENDS) >= {"vllm", "vllm-static", "vllm-generate"}
