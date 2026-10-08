"""The sampling knobs of a steer request: decided by the engine, applied by it, reported once.

The stream is exercised through the real generator with a stub backend that records the
keywords ``generate_stream`` was handed, which each backend maps to its own sampler.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from interp_engine import RecommendedSampling, SamplingSettings, resolve_sampling

from neuronpedia_inference.endpoints.steer.completion import _run_batched_generate
from neuronpedia_inference.inference_utils.sampling import (
    resolve_request_sampling,
    sampling_report,
    state_settings_once,
)
from neuronpedia_inference.inference_utils.steering import SteeringSettings, format_sse_message, remove_sse_formatting
from neuronpedia_inference.schemas import (
    NPSamplingSettings,
    NPSteerCompletionOutput,
    NPSteerType,
    NPSteerVector,
    SteerCompletionRequest,
    SteerCompletionResponse,
)

GEMMA_LIKE = RecommendedSampling(temperature=1.0, top_k=64, top_p=0.95, do_sample=True, source="x")


class _Model:
    """A backend that resolves knobs the way every engine backend does."""

    def __init__(self, stated: RecommendedSampling | None = None) -> None:
        self.recommended_sampling = stated or RecommendedSampling()

    def sampling_settings(self, **knobs):
        return resolve_sampling(self.recommended_sampling, **knobs)


def _request(**knobs) -> SteerCompletionRequest:
    return SteerCompletionRequest(
        prompt="Hi",
        model="m",
        steer_method="SIMPLE_ADDITIVE",  # type: ignore[arg-type]
        normalize_steering=False,
        types=[NPSteerType.DEFAULT],
        vectors=[NPSteerVector(steering_vector=[0.0] * 8, strength=1.0, hook="blocks.0.hook_resid_post")],
        n_completion_tokens=8,
        strength_multiplier=1.0,
        seed=16,
        **knobs,
    )


# --- resolving the request ------------------------------------------------------


def test_an_unset_request_runs_with_the_checkpoints_recommendation():
    settings = resolve_request_sampling(_Model(GEMMA_LIKE), _request())
    assert settings == SamplingSettings(temperature=1.0, top_k=64, top_p=0.95, presence_penalty=0.0)


def test_the_requests_knobs_win_and_the_penalty_is_its_own():
    settings = resolve_request_sampling(_Model(GEMMA_LIKE), _request(temperature=0.3, top_k=0, presence_penalty=1.5))
    assert settings == SamplingSettings(temperature=0.3, top_k=None, top_p=0.95, presence_penalty=1.5)


def test_a_checkpoint_stating_nothing_is_neutral():
    assert resolve_request_sampling(_Model(), _request()) == SamplingSettings(1.0, None, None, 0.0)


def test_freq_penalty_is_accepted_and_ignored():
    """Older clients still send it. It changes nothing, and the report says so by omission."""
    settings = resolve_request_sampling(_Model(), _request(freq_penalty=1.0))
    assert settings.presence_penalty == 0.0
    assert "freq_penalty" not in sampling_report(settings, 16).model_dump()


def test_the_request_bounds_the_penalty():
    with pytest.raises(ValueError):
        _request(presence_penalty=3.0)
    with pytest.raises(ValueError):
        _request(top_p=0.0)


# --- reporting once ---------------------------------------------------------------


def _frame(text: str) -> str:
    return format_sse_message(
        SteerCompletionResponse(outputs=[NPSteerCompletionOutput(type=NPSteerType.DEFAULT, output=text)]).to_wire_json()
    )


def test_a_stream_states_its_settings_on_the_first_frame_only():
    report = sampling_report(SamplingSettings(0.6, 20, 0.95, 1.5), 16)

    async def frames():
        for text in ["a", "ab", "abc"]:
            yield _frame(text)

    async def run():
        return [
            json.loads(remove_sse_formatting(f))
            async for f in state_settings_once(frames(), report, SteerCompletionResponse)
        ]

    out = asyncio.run(run())
    assert [f["outputs"][0]["output"] for f in out] == ["a", "ab", "abc"], "the frames themselves are untouched"
    # The wire is camelCase, as every schema here is.
    assert out[0]["sampling"] == {"temperature": 0.6, "topK": 20, "topP": 0.95, "presencePenalty": 1.5, "seed": 16}
    assert all("sampling" not in f for f in out[1:])


def test_the_report_keeps_null_filters_and_drops_nothing_else():
    """``top_k: null`` is information (no filtering), so it survives the wire's exclude_none."""
    report = sampling_report(SamplingSettings(1.0, None, None, 0.0), None)
    wire = json.loads(SteerCompletionResponse(outputs=[], sampling=report).to_wire_json())
    assert wire["sampling"] == {"temperature": 1.0, "presencePenalty": 0.0}
    assert NPSamplingSettings.model_validate(wire["sampling"]).top_k is None


# --- what reaches the engine ------------------------------------------------------------


class _RecordingBackend:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def to_tokens(self, _prompt, **_kwargs):
        return [[1, 2]]

    async def generate_stream(self, _prompt_token_ids, **kwargs):
        self.calls.append(kwargs)
        yield "ok"


def test_the_resolved_settings_reach_the_engine_as_its_own_keywords():
    backend = _RecordingBackend()
    settings = SamplingSettings(temperature=0.6, top_k=None, top_p=0.95, presence_penalty=1.5)

    async def run():
        async for _ in _run_batched_generate(
            model=backend,  # type: ignore[arg-type]
            prompt="Hi",
            settings=SteeringSettings(features=[], strength_multiplier=1.0),
            steer_types=[NPSteerType.DEFAULT],
            seed=7,
            sampling=settings,
            max_new_tokens=4,
        ):
            pass

    asyncio.run(run())
    (call,) = backend.calls
    assert {k: call[k] for k in ("temperature", "top_k", "top_p", "presence_penalty", "seed")} == {
        "temperature": 0.6,
        "top_k": None,
        "top_p": 0.95,
        "presence_penalty": 1.5,
        "seed": 7,
    }


def test_the_wire_shape_of_the_eager_report_matches_the_vllm_one():
    """One schema for both backends; nothing about the report says which one ran."""
    model = SimpleNamespace(sampling_settings=_Model(GEMMA_LIKE).sampling_settings)
    report = sampling_report(resolve_request_sampling(model, _request(presence_penalty=0.5)), 1)  # type: ignore[arg-type]
    assert report == NPSamplingSettings(temperature=1.0, top_k=64, top_p=0.95, presence_penalty=0.5, seed=1)
