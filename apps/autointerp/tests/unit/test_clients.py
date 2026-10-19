"""Tests for the OpenRouter client the explain and score routes use."""

import asyncio
import json

import httpx

from neuronpedia_autointerp.clients import DEFAULT_MAX_TOKENS, REASONING_EFFORT, LowReasoningOpenRouter


def client_with_reply(content: str | None, sent: list[dict]) -> LowReasoningOpenRouter:
    """Make a client whose HTTP calls record the body and return `content`."""

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    client = LowReasoningOpenRouter(model="openai/gpt-5-nano", api_key="test")
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


def test_sends_low_reasoning_and_a_larger_token_limit():
    """The body carries the reasoning effort and the larger default limit."""
    sent: list[dict] = []
    reply = asyncio.run(client_with_reply("[1, 0]", sent).generate([{"role": "user", "content": "hi"}]))
    assert reply.text == "[1, 0]"
    assert sent[0]["reasoning"] == {"effort": REASONING_EFFORT}
    assert sent[0]["max_tokens"] == DEFAULT_MAX_TOKENS
    assert sent[0]["temperature"] == 1.0


def test_caller_limits_override_the_defaults():
    """A scorer that sets max_tokens or temperature still gets what it asked for."""
    sent: list[dict] = []
    asyncio.run(client_with_reply("ok", sent).generate("hi", max_tokens=50, temperature=0.0))
    assert sent[0]["max_tokens"] == 50
    assert sent[0]["temperature"] == 0.0
