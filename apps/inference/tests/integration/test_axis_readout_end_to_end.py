"""Readout axes against the model they were fitted for.

Everything else about axes is tested on synthetic tensors: the projection arithmetic, the
per-axis failure isolation, which layers get captured, how readouts are shaped. None of that
touches a model, and none of it would notice the failure that matters most here -- capture
reading a different point than the projection assumes, or the prompt being rendered differently
than the fit assumed. Both produce plausible numbers rather than an error.

So this brings up ``meta-llama/Llama-3.1-8B-Instruct`` and asks for real readouts over a real
conversation. The axes are sent with the request, which is the only way one reaches this server,
and their directions are synthetic: what is pinned here is the wiring, not the values. A fitted
direction's numbers depend on what the model happens to say, so asserting a particular reading
would pin the generation rather than the readout, while a basis vector makes the plumbing legible
-- the reading is one coordinate of the residual stream at one layer, so two axes that disagree
about which layer or which coordinate cannot come back equal by accident.

Run against both backends. The two capture activations by different mechanisms -- eager reads
hook output directly, vLLM reads back a recorded point -- and the vLLM path additionally builds
a steering spec that used to be built for a readout too, which made a readout carrying no
features a 500. Only the unit tests with a stub backend covered any of that.

Manual: 16GB of gated weights, so set ``NP_RUN_MANUAL_TESTS=1`` (and ``HF_TOKEN``) to run.
"""

from __future__ import annotations

import math
import os

import pytest

from neuronpedia_inference.shared import Model
from tests.harness import (
    EAGER,
    LLAMA_8B_TRAITS,
    VLLM,
    X_SECRET_KEY,
    shutdown_running_server,
    try_initialized_server,
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("NP_RUN_MANUAL_TESTS"),
    reason="manual test (downloads 16GB of gated weights); set NP_RUN_MANUAL_TESTS=1 to run",
)

ENDPOINT = "/v1/steer/completion-chat"

# vLLM is CUDA-only and its bring-up self-skips, so naming both here costs nothing off a GPU.
ENGINES = [pytest.param(EAGER, id="eager"), pytest.param(VLLM, id="vllm", marks=pytest.mark.cuda)]

# The harness defaults vLLM to half the card, which is sized for gpt2 and the small instruct
# models. This model's weights alone are ~16GiB, so on a 32GiB card that default leaves about
# 0.1GiB for the KV cache and the engine core refuses to start -- reported as a skip, which
# would quietly mean the vLLM half of this file never ran. The harness lets an explicit
# environment value win for exactly this. Still a fraction rather than all of it, so a desktop
# compositor or a leftover eager load does not tip it over; a card too small for the model
# skips, which is the honest outcome.
VLLM_UTILIZATION_FOR_8B = "0.85"


@pytest.fixture
def engine_env(engine, monkeypatch):
    if engine != VLLM:
        return engine
    # Shut any cached server down *before* setting the variable. The harness restores the
    # environment a server booted with when it tears that server down, and it tears the old one
    # down on the way to booting the new one -- so setting this first would have it wiped in
    # between, leaving the harness default and a bring-up failure reported as a skip.
    shutdown_running_server()
    monkeypatch.setenv("VLLM_GPU_MEMORY_UTILIZATION", VLLM_UTILIZATION_FOR_8B)
    return engine


# Two turns, so a readout has more than one point to report and an ordering to get wrong. The
# content is bland on purpose: this test is about the plumbing, not about moving a trait.
CONVERSATION = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
    {"role": "user", "content": "And of Japan?"},
]

# Read at 19 and 13 respectively, which is the point: one request, two layers, one forward.
DEEP = "mit_deep"
DEEP_LAYER = 19
SHALLOW = "mit_shallow"
SHALLOW_LAYER = 13


def _axis(axis_id: str, layer: int, component: int) -> dict:
    """One inline axis reading a single coordinate of the residual stream at ``layer``.

    Called inside a server context, because the width of the direction has to be this model's
    and the endpoint refuses any other.
    """
    direction = [0.0] * Model.get_instance().d_model
    direction[component] = 1.0
    return {"id": axis_id, "layer": layer, "direction": direction, "author": "mit"}


def _both_axes() -> list[dict]:
    return [_axis(DEEP, DEEP_LAYER, 0), _axis(SHALLOW, SHALLOW_LAYER, 1)]


def _request(**overrides) -> dict:
    return {
        "prompt": CONVERSATION,
        "model": LLAMA_8B_TRAITS.model_id,
        "steer_method": "SIMPLE_ADDITIVE",
        "normalize_steering": False,
        "types": ["DEFAULT"],
        "n_completion_tokens": 16,
        "temperature": 0,
        "strength_multiplier": 0.0,
        "freq_penalty": 0.0,
        "seed": 42,
        "steer_special_tokens": False,
        **overrides,
    }


def _post(client, **overrides):
    resp = client.post(ENDPOINT, json=_request(**overrides), headers={"X-SECRET-KEY": X_SECRET_KEY})
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.mark.parametrize("engine", ENGINES)
def test_two_traits_at_different_layers_come_back_from_one_request(engine_env):
    """The property the single-layer format could not express, end to end.

    Empathy is fitted at layer 19 and toxicity at 13. Both are read from one generation, so a
    mix-up here would report one trait's activations under the other's name -- which is exactly
    the kind of error that still looks like a plausible chart.
    """
    with try_initialized_server(LLAMA_8B_TRAITS, engine=engine_env) as client:
        body = _post(client, custom_axes=_both_axes())

        readouts = {readout["id"]: readout for readout in body["axes"]}
        assert sorted(readouts) == sorted([DEEP, SHALLOW])
        assert readouts[DEEP]["layer"] == DEEP_LAYER
        assert readouts[SHALLOW]["layer"] == SHALLOW_LAYER
        assert readouts[DEEP]["author"] == "mit"

        for readout in readouts.values():
            # Two assistant turns: the one in the prompt and the one just generated.
            turns = readout["turns"]
            assert len(turns) == 2, readout
            for turn in turns:
                assert math.isfinite(turn["value"]), readout
                # Calibrated, so a real reading lands near [-1, 1] without being clipped to it.
                assert abs(turn["value"]) < 10.0, readout
                assert turn["snippet"]
            # Unsteered, so there is nothing for a post-cap value to describe: the response
            # omits the key rather than sending null. Spelled camelCase because that is what
            # this endpoint aliases to -- the snake_case name is absent for the wrong reason,
            # so asserting on it would pass without checking anything. The steered test below
            # is what makes this meaningful, by pinning that the key does appear under steering.
            assert all(turn.get("valuePostCap") is None for turn in turns)


@pytest.mark.parametrize("engine", ENGINES)
def test_the_traits_are_measuring_different_things(engine_env):
    """Two axes over one generation must not report the same number.

    The failure this catches is a capture keyed by something other than the layer, where every
    axis reads whichever activation was stored last. Both readouts would look reasonable; they
    would just be the same.
    """
    with try_initialized_server(LLAMA_8B_TRAITS, engine=engine_env) as client:
        body = _post(client, custom_axes=_both_axes())
        by_id = {readout["id"]: readout for readout in body["axes"]}
        deep = [turn["value"] for turn in by_id[DEEP]["turns"]]
        shallow = [turn["value"] for turn in by_id[SHALLOW]["turns"]]
        assert deep != shallow


def test_a_readout_does_not_change_what_the_model_says():
    """Measuring is not steering: the same seed generates the same text with and without axes.

    The axis pins ``date_string``, which is what makes the readout path re-render the prompt. If
    that render reached generation, asking for a readout would quietly change the answer being
    read -- so the pinned date is the point of the test, not incidental to it.
    """
    with try_initialized_server(LLAMA_8B_TRAITS) as client:
        dated = _axis(DEEP, DEEP_LAYER, 0) | {"render": {"template_kwargs": {"date_string": "26 Jul 2024"}}}
        without = _post(client)
        with_axes = _post(client, custom_axes=[dated])

        def text(body: dict) -> str:
            return next(output["raw"] for output in body["outputs"] if output["type"] == "DEFAULT")

        assert text(with_axes) == text(without)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_readout_only_request_carries_no_features(engine_env):
    """The shape an unsteered readout page sends: types=[DEFAULT], axes=[...], nothing to steer with.

    Named separately from the tests above, which happen to send the same shape, because this is
    the request that used to fail. Steering used to be validated for every request and the vLLM
    steering spec built for every request, so measuring without steering meant either a 400 for
    the missing features or a 500 from building a spec out of none. A caller worked around it by
    faking a zero-strength vector, which is a steer being pretended at to take a measurement.
    """
    with try_initialized_server(LLAMA_8B_TRAITS, engine=engine_env) as client:
        request = _request(custom_axes=[_axis(DEEP, DEEP_LAYER, 0)])
        assert "features" not in request and "vectors" not in request
        resp = client.post(ENDPOINT, json=request, headers={"X-SECRET-KEY": X_SECRET_KEY})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert [readout["id"] for readout in body["axes"]] == [DEEP]
        assert all(math.isfinite(turn["value"]) for turn in body["axes"][0]["turns"])


@pytest.mark.parametrize("engine", ENGINES)
def test_a_steered_readout_reports_both_sides_of_the_cap(engine_env):
    """Steering and measuring in one request: the readout gets a pre-cap and a post-cap value.

    The other direction of the same boundary. Here the vLLM steering spec *is* built, and the
    readout needs a second capture under steering -- so this is the path where a spec shared
    between generation and readout would show up as the two values being equal.
    """
    with try_initialized_server(LLAMA_8B_TRAITS, engine=engine_env) as client:
        model = Model.get_instance()
        # A real direction, not the zero vector a caller would once have sent to satisfy the
        # old "exactly one of features or vectors" check: vLLM's spec builder rejects a
        # zero-norm vector outright, so that workaround never worked on this backend anyway.
        steering_vector = [0.0] * model.d_model
        steering_vector[0] = 1.0
        resp = client.post(
            ENDPOINT,
            json=_request(
                types=["STEERED", "DEFAULT"],
                custom_axes=[_axis(DEEP, DEEP_LAYER, 0)],
                strength_multiplier=1.0,
                vectors=[
                    {
                        "steering_vector": steering_vector,
                        "strength": 1.0,
                        "hook": "blocks.15.hook_resid_post",
                    }
                ],
            ),
            headers={"X-SECRET-KEY": X_SECRET_KEY},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()

        by_type = {readout["type"]: readout for readout in body["axes"] if readout["id"] == DEEP}
        assert sorted(by_type) == ["DEFAULT", "STEERED"]
        steered_turns = by_type["STEERED"]["turns"]
        assert steered_turns
        for turn in steered_turns:
            assert math.isfinite(turn["value"])
            # Measured under steering, so this side exists -- unlike the unsteered case above.
            assert turn["valuePostCap"] is not None
            assert math.isfinite(turn["valuePostCap"])


def test_an_axis_fitted_for_other_weights_is_refused():
    """A direction of the wrong width is a 400 naming the axis, not a 500 out of the matmul.

    The unit tests check this against a stub, which fixes the width by construction. Here it is
    the real model answering, which is the only thing that knows what its width is -- and the
    mistake is a realistic one, since a direction fitted on the 70B is simply longer.
    """
    with try_initialized_server(LLAMA_8B_TRAITS) as client:
        too_wide = {"id": "lu_assistant-axis", "layer": 40, "direction": [0.1] * 8192}
        resp = client.post(
            ENDPOINT,
            json=_request(custom_axes=[too_wide]),
            headers={"X-SECRET-KEY": X_SECRET_KEY},
        )
        assert resp.status_code == 400, resp.text
        assert "lu_assistant-axis" in resp.json()["error"]
