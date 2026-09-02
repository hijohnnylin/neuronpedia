"""Streaming-frame invariants for /steer/completion-chat, with a stub backend.

The composed ``chat_template`` is only recoverable if the stream still carries the special
tokens that mark chat structure (harmony's ``<|channel|>``/``<|message|>``, turn-end markers).
vLLM's detokenizer drops them by default, which silently produced a response with no assistant
turn at all, so that setting is asserted here rather than left to review.
"""

from __future__ import annotations

import asyncio
import json

import torch

from neuronpedia_inference.endpoints.steer.completion_chat import (
    _vllm_chat_generate,
    messages_for_render,
)
from neuronpedia_inference.inference_utils.steering import (
    SteeringSettings,
    remove_sse_formatting,
)
from neuronpedia_inference.schemas import NPSteerChatMessage, NPSteerType, NPSteerVector

HARMONY_VOCAB = ["<|start|>", "<|channel|>", "<|message|>", "<|end|>", "<|return|>"]
PROMPT = "<|start|>user<|message|>Hi<|end|><|start|>assistant"
DELTAS = [
    "<|channel|>analysis",
    "<|message|>Greeting.",
    "<|end|><|start|>assistant<|channel|>final",
    "<|message|>Hello!",
    "<|return|>",
]


class StubTokenizer:
    def get_added_vocab(self) -> dict[str, int]:
        return dict.fromkeys(HARMONY_VOCAB, 0)

    def decode(self, _tokens) -> str:
        return PROMPT


class StubBackend:
    """Captures what the endpoint asks for and replays canned harmony deltas."""

    def __init__(self):
        self.tokenizer = StubTokenizer()
        self.sampling_params = None
        self.prompt_token_ids = None

    async def generate_steered(self, prompt_token_ids, sampling_params, **_kwargs):
        self.sampling_params = sampling_params
        self.prompt_token_ids = prompt_token_ids

        async def stream():
            for delta in DELTAS:
                yield delta

        return stream()


async def _frames(model: StubBackend) -> list[dict]:
    return [
        json.loads(remove_sse_formatting(sse))
        async for sse in _vllm_chat_generate(
            model=model,  # type: ignore[arg-type]
            promptTokenized=torch.tensor([1, 2, 3]),
            inputPrompt=[NPSteerChatMessage(role="user", content="Hi")],
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
            steer_types=[NPSteerType.DEFAULT],
            seed=1,
            temperature=0.0,
            max_new_tokens=32,
        )
    ]


def _messages(frame: dict) -> list[dict]:
    # camelCase because these are wire frames, not python attribute names.
    return frame["outputs"][0]["chatTemplate"]


def test_stream_keeps_special_tokens():
    """Dropping these makes the assistant turn unrecoverable — see module docstring."""
    model = StubBackend()
    asyncio.run(_frames(model))
    assert model.sampling_params is not None
    assert model.sampling_params.skip_special_tokens is False


def test_generation_uses_the_endpoints_own_token_ids():
    """Vector readouts index captured rows by prompt position.

    That only holds if the backend generates from the ids the endpoint tokenized, rather
    than re-tokenizing a decode of them, so the endpoint passes ids and not a string.
    """
    model = StubBackend()
    asyncio.run(_frames(model))
    assert model.prompt_token_ids == [1, 2, 3]


def test_frames_compose_prompt_messages_plus_assistant_turn():
    frames = asyncio.run(_frames(StubBackend()))
    assert len(frames) == len(DELTAS)

    # Thinking is visible before the answer arrives, then merges into one assistant message.
    assert _messages(frames[1])[-1] == {
        "role": "assistant",
        "content": "<think>Greeting.</think>",
    }
    final = _messages(frames[-1])
    assert final == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "<think>Greeting.</think>Hello!"},
    ]
    # The prompt scaffold is never re-parsed into messages, and raw stays prompt+generation.
    assert frames[-1]["outputs"][0]["raw"] == PROMPT + "".join(DELTAS)
    assert _messages(frames[-1])[0]["content"] == "Hi"


# --- what the template sees vs. what the client gets back --------------------
def _turn(role: str, content: str) -> NPSteerChatMessage:
    return NPSteerChatMessage(role=role, content=content)


def test_render_drops_prior_turn_reasoning():
    """Turn 2's prompt must not re-render turn 1's chain of thought."""
    convo = [
        _turn("user", "Hi"),
        _turn("assistant", "<think>Greeting.</think>Hello!"),
        _turn("user", "And Spain?"),
    ]
    assert messages_for_render(convo, blank_system_prompt=False) == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "And Spain?"},
    ]


def test_render_leaves_the_request_messages_untouched():
    """``promptChat`` is echoed back to the client, so stripping must not mutate it."""
    convo = [_turn("assistant", "<think>Greeting.</think>Hello!")]
    messages_for_render(convo, blank_system_prompt=False)
    assert convo[0].content == "<think>Greeting.</think>Hello!"


def test_render_drops_a_reasoning_only_turn_entirely():
    convo = [_turn("user", "Hi"), _turn("assistant", "<think>Still thinking.</think>")]
    assert messages_for_render(convo, blank_system_prompt=False) == [{"role": "user", "content": "Hi"}]


def test_render_does_not_strip_user_turns():
    """A user who types the tags means them literally."""
    convo = [_turn("user", "What does <think>this</think> mean?")]
    assert messages_for_render(convo, blank_system_prompt=False) == [
        {"role": "user", "content": "What does <think>this</think> mean?"}
    ]


def test_render_blanks_a_leading_system_prompt_when_asked():
    """The published-artifact case: the turn stays so the structure matches the fit."""
    convo = [_turn("system", "You are terse."), _turn("user", "Hi")]
    assert messages_for_render(convo, blank_system_prompt=True) == [
        {"role": "system", "content": ""},
        {"role": "user", "content": "Hi"},
    ]
