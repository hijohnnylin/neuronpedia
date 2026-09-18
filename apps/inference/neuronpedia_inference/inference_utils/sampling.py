"""The sampling knobs of a steer request, decided once and reported once.

A request may leave any knob unset. The engine decides it: the checkpoint's own
``generation_config.json`` recommendation where there is one, neutral where there is not
(``interp_engine.sampling.resolve_sampling``). Both generation backends then run with the same
:class:`SamplingSettings`, and the response carries them back so the reader sees what produced
the text -- once per response, on the first frame of a stream (:func:`state_settings_once`).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Protocol

from interp_engine import SamplingSettings

from neuronpedia_inference.inference_utils.steering import format_sse_message, remove_sse_formatting
from neuronpedia_inference.schemas import NPSamplingSettings, SteerCompletionChatResponse, SteerCompletionResponse

logger = logging.getLogger(__name__)


class _SamplingRequest(Protocol):
    temperature: float | None
    top_k: int | None
    top_p: float | None
    presence_penalty: float | None
    freq_penalty: float | None


class _Samples(Protocol):
    def sampling_settings(
        self,
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
    ) -> SamplingSettings: ...


def resolve_request_sampling(model: _Samples, request: _SamplingRequest) -> SamplingSettings:
    """What this request's generation runs with, on whichever backend holds ``model``."""
    if request.freq_penalty:
        # Accepted for older clients, applied by nothing: it scaled with a token's count and
        # punished function words in a long reply. The presence penalty is the repetition control.
        logger.info("freq_penalty=%s ignored; presence_penalty is the repetition control", request.freq_penalty)
    return model.sampling_settings(
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        presence_penalty=request.presence_penalty,
    )


def sampling_report(settings: SamplingSettings, seed: int | None) -> NPSamplingSettings:
    """The settings on the wire."""
    return NPSamplingSettings(
        temperature=settings.temperature,
        top_k=settings.top_k,
        top_p=settings.top_p,
        presence_penalty=settings.presence_penalty,
        seed=seed,
    )


async def state_settings_once(
    frames: AsyncIterator[str],
    report: NPSamplingSettings,
    schema: type[SteerCompletionResponse] | type[SteerCompletionChatResponse],
) -> AsyncIterator[str]:
    """The SSE ``frames`` with ``report`` attached to the first one and to no other.

    One parse and re-serialization, of the first frame only; the rest pass through untouched.
    """
    first = True
    async for frame in frames:
        if first:
            first = False
            parsed = schema.model_validate_json(remove_sse_formatting(frame))
            parsed.sampling = report
            yield format_sse_message(parsed.to_wire_json())
            continue
        yield frame
