"""Streaming-frame invariants for /steer/completion, with a stub backend.

The non-streaming handler reads the last frame as the answer, so a stream with no frame was a
500 ("Generator yielded no items"). A model that samples EOS first, or emits only special
tokens the detokenizer drops, streams no delta -- and an empty completion is a valid result.
"""

from __future__ import annotations

import asyncio
import json

from neuronpedia_inference.endpoints.steer.completion import _vllm_run_batched_generate
from neuronpedia_inference.inference_utils.steering import (
    SteeringSettings,
    remove_sse_formatting,
)
from neuronpedia_inference.schemas import NPSteerType, NPSteerVector


class StubBackend:
    """Replays canned text deltas, per call, in order."""

    def __init__(self, deltas_by_call: list[list[str]]):
        self.deltas_by_call = deltas_by_call
        self.calls = 0

    async def generate(self, _prompt, _sampling_params, **_kwargs):
        deltas = self.deltas_by_call[self.calls]
        self.calls += 1

        async def stream():
            for delta in deltas:
                yield delta

        return stream()


async def _frames(model: StubBackend, steer_types: list[NPSteerType]) -> list[dict]:
    return [
        json.loads(remove_sse_formatting(sse))
        async for sse in _vllm_run_batched_generate(
            model=model,  # type: ignore[arg-type]
            prompt="Hi",
            settings=SteeringSettings(
                features=[
                    NPSteerVector(
                        steering_vector=[0.0] * 7 + [1.0],
                        strength=0.0,
                        hook="blocks.0.hook_resid_post",
                    )
                ],
                strength_multiplier=1.0,
            ),
            steer_types=steer_types,
            seed=1,
            temperature=0.0,
            max_new_tokens=32,
        )
    ]


def _outputs(frame: dict) -> list[tuple[str, str]]:
    return [(o["type"], o["output"]) for o in frame["outputs"]]


def test_empty_generation_still_emits_one_frame_per_type():
    frames = asyncio.run(_frames(StubBackend([[], []]), [NPSteerType.STEERED, NPSteerType.DEFAULT]))
    assert len(frames) == 2
    assert _outputs(frames[-1]) == [("STEERED", ""), ("DEFAULT", "")]


def test_one_empty_type_does_not_add_a_frame_to_the_other():
    """The closing frame is only for a type that streamed nothing; a type with deltas is unchanged."""
    frames = asyncio.run(_frames(StubBackend([["Hel", "lo"], []]), [NPSteerType.STEERED, NPSteerType.DEFAULT]))
    assert len(frames) == 3
    assert _outputs(frames[1]) == [("STEERED", "Hello"), ("DEFAULT", "")]
    assert _outputs(frames[-1]) == [("STEERED", "Hello"), ("DEFAULT", "")]


def test_a_generation_with_deltas_emits_one_frame_per_delta():
    frames = asyncio.run(_frames(StubBackend([["a", "b", "c"]]), [NPSteerType.DEFAULT]))
    assert len(frames) == 3
    assert _outputs(frames[-1]) == [("DEFAULT", "abc")]
