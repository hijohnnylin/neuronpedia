"""The verbalizer's generation rules, checked on a stub backend so no weights are needed.

The rules are the app's and hold on every engine backend: the stop tag ends the request and stays
in the text; EOS ends it as a stop; running out of tokens is a length; the text is decoded from
all ids so far, not joined from per-token strings; and leaving early closes the stream.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest
import torch
from interp_engine import SamplingSettings

from verbalizer import Verbalizer, resolve_backend

GREEDY = SamplingSettings(temperature=0.0, top_k=None, top_p=None, presence_penalty=0.0)


@dataclass
class _Step:
    token_id: int
    token_str: str


class _Tokenizer:
    """Decodes ids by a table; two ids join to one multibyte character to catch a per-token join."""

    eos_token_id = 9
    _table = {1: "<explanation>", 2: "caf", 3: "\u00e9", 4: "</explanation>", 5: " more", 9: "<eos>"}

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        assert skip_special_tokens is False
        # 2 then 3 is the pair the multibyte check leans on: decoded together they are one word.
        return "".join(self._table[i] for i in ids)


@dataclass
class _Backend:
    """Yields the ids it was given and records whether the consumer closed the stream."""

    ids: list[int]
    closed: bool = False
    calls: list[dict] = field(default_factory=list)

    async def generate_steps_from_embeds(self, prompt_embeds: torch.Tensor, **kwargs):
        self.calls.append({"shape": tuple(prompt_embeds.shape), **kwargs})
        try:
            for i in self.ids:
                yield _Step(i, "")
        finally:
            self.closed = True


def _verbalizer(ids: list[int]) -> tuple[Verbalizer, _Backend]:
    v = Verbalizer.__new__(Verbalizer)
    backend = _Backend(ids)
    v.model = backend  # pyright: ignore[reportAttributeAccessIssue]
    v.tokenizer = _Tokenizer()
    v._build_prompt_embeds = lambda activation, prompt: torch.zeros(5, 8)  # pyright: ignore[reportAttributeAccessIssue]
    return v, backend


def _collect(v: Verbalizer) -> list[dict]:
    async def run():
        return [f async for f in v.async_generate_stream([0.0], sampling=GREEDY, max_new_tokens=50)]

    return asyncio.run(run())


def test_the_stop_tag_ends_the_request_and_stays_in_the_text() -> None:
    v, backend = _verbalizer([1, 2, 3, 4, 5, 5])
    frames = _collect(v)
    assert [f["text"] for f in frames] == [
        "<explanation>",
        "<explanation>caf",
        "<explanation>caf\u00e9",
        "<explanation>caf\u00e9</explanation>",
    ]
    meta = frames[-1]["meta_info"]
    assert meta == {
        "finish_reason": {"type": "stop", "matched": "</explanation>"},
        "completion_tokens": 4,
        "prompt_tokens": 5,
    }
    assert backend.closed, "breaking on the stop tag must close the backend's stream"
    assert backend.calls[0] == {
        "shape": (5, 8),
        "max_tokens": 50,
        "temperature": 0.0,
        "top_k": None,
        "top_p": None,
        "presence_penalty": 0.0,
    }


def test_eos_is_a_stop_and_exhaustion_is_a_length() -> None:
    v, _ = _verbalizer([1, 2, 9])
    meta = _collect(v)[-1]["meta_info"]
    assert meta["finish_reason"] == {"type": "stop", "matched": None}
    assert meta["completion_tokens"] == 3

    v, _ = _verbalizer([1, 2, 3])
    meta = _collect(v)[-1]["meta_info"]
    assert meta["finish_reason"] == {"type": "length", "matched": None}


def test_async_generate_extracts_the_explanation_body() -> None:
    v, _ = _verbalizer([1, 2, 3, 4])
    text = asyncio.run(v.async_generate([0.0], sampling=GREEDY, max_new_tokens=8))
    assert text == "caf\u00e9"
    raw = asyncio.run(v.async_generate([0.0], extract_explanation=False, sampling=GREEDY, max_new_tokens=8))
    assert raw == "<explanation>caf\u00e9</explanation>"


def test_a_shut_down_verbalizer_refuses_to_generate() -> None:
    v, _ = _verbalizer([1])
    v.model = None
    with pytest.raises(RuntimeError, match="shut down"):
        _collect(v)


@pytest.mark.parametrize(
    ("backend", "device", "expected"),
    [
        ("auto", "cuda:0", "vllm-generate"),
        ("auto", "mps", "eager"),
        ("auto", "cpu", "eager"),
        ("vllm", "cuda:1", "vllm-generate"),
        ("eager", "cuda:0", "eager"),
        ("mlx", "mps", "mlx"),
    ],
)
def test_resolve_backend(backend: str, device: str, expected: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NLA_VERBALIZER_ENFORCE_EAGER", raising=False)
    assert resolve_backend(backend, device) == expected


def test_resolve_backend_refusals_and_the_enforce_eager_hatch(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="needs a CUDA device"):
        resolve_backend("vllm", "mps")
    with pytest.raises(ValueError, match="not one of"):
        resolve_backend("sglang", "cuda:0")
    monkeypatch.setenv("NLA_VERBALIZER_ENFORCE_EAGER", "1")
    assert resolve_backend("auto", "cuda:0") == "vllm"
