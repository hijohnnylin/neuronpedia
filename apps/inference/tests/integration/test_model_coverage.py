"""Instruct/thinking-model coverage (beyond the gpt2-small base model).

Two archetypes the base model can't exercise:

- ``google/gemma-3-270m-it`` -- non-thinking instruct, **gated** (needs ``HF_TOKEN``; the
  suite auto-skips these when it is absent).
- ``Qwen/Qwen3.5-0.8B`` -- thinking instruct with a hybrid Gated-DeltaNet + Gated-Attention
  trunk and an ``enable_thinking`` chat-template switch.

These bring up a real server (no SAEs -- chat/template paths don't need them) and hit the
HTTP endpoints. Weights are downloaded on demand; if a model can't load in this environment
(uncached + offline, arch unsupported by this transformers/vLLM, ...) the test skips rather
than fails, so the same file is green locally and on the GPU CI runner where the weights,
token, and GPU are all present.

Arch support for the hybrid Qwen3.5 VL family is probed cheaply (config-only) in
``test_qwen35_arch_support_probe`` before the heavier load tests; unsupported engines are
skipped rather than failed.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from interp_engine import EagerModel
from transformers import AutoConfig

from neuronpedia_inference.shared import Model
from tests.harness import (
    GEMMA_IT,
    QWEN_THINKING,
    VLLM,
    X_SECRET_KEY,
    try_initialized_server,
    vllm_available,
)

CHAT_MESSAGES = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
]


def _qwen35_architectures() -> list[str]:
    """Config-only probe of Qwen3.5-0.8B's HF architecture class names."""
    try:
        cfg = AutoConfig.from_pretrained(QWEN_THINKING.model_id, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"could not load config for {QWEN_THINKING.model_id}: {exc}")
    return list(getattr(cfg, "architectures", None) or [])


def _vllm_supports_architectures(architectures: list[str]) -> bool:
    if not architectures:
        return False
    try:
        from vllm import ModelRegistry

        supported = set(ModelRegistry.get_supported_archs())
    except Exception:  # noqa: BLE001
        return False
    return any(arch in supported for arch in architectures)


def _apply_chat_template(
    client: TestClient,
    *,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = False,
    chat_template_kwargs: dict[str, bool] | None = None,
) -> Any:
    body: dict[str, Any] = {
        "messages": messages,
        "addGenerationPrompt": add_generation_prompt,
    }
    if chat_template_kwargs is not None:
        body["chatTemplateKwargs"] = chat_template_kwargs
    resp = client.post("/v1/apply-chat-template", json=body, headers={"X-SECRET-KEY": X_SECRET_KEY})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _assert_spans_cover_tokens(data: Any) -> None:
    spans = data["spans"]
    assert [s["position"] for s in spans] == list(range(len(spans)))
    assert len(data["tokens"]) == len(spans)
    roles = {s.get("role") for s in spans if s.get("messageIndex") is not None}
    assert "user" in roles and "assistant" in roles


# --- gemma-3-270m-it: non-thinking instruct (gated) --------------------------


@pytest.mark.gated
@pytest.mark.chat
def test_gemma_apply_chat_template_spans():
    with try_initialized_server(GEMMA_IT) as client:
        data = _apply_chat_template(client, messages=CHAT_MESSAGES)
        _assert_spans_cover_tokens(data)

        # Token ids match the tokenizer's own chat-template rendering.
        model = Model.get_instance()
        assert isinstance(model, EagerModel)
        tok = model.tokenizer
        assert tok is not None
        ref = tok.apply_chat_template(CHAT_MESSAGES, add_generation_prompt=False, tokenize=True)
        if isinstance(ref, dict) or hasattr(ref, "input_ids"):
            ref = ref["input_ids"]
        assert data["tokens"] == list(ref)


@pytest.mark.gated
@pytest.mark.chat
def test_gemma_chat_completion_default():
    """A DEFAULT (unsteered) chat completion should generate non-empty text end-to-end."""
    with try_initialized_server(GEMMA_IT) as client:
        # gemma loads on the EagerModel backend here (chat models ship without SAEs).
        model = Model.get_instance()
        assert isinstance(model, EagerModel)
        d_model = model.d_model
        hook = f"blocks.{max(0, model.n_layers // 2)}.hook_resid_post"
        req = {
            "prompt": [{"content": "Say hello.", "role": "user"}],
            "model": GEMMA_IT.model_id,
            "steer_method": "SIMPLE_ADDITIVE",
            "normalize_steering": False,
            "types": ["DEFAULT"],
            "vectors": [{"steering_vector": [0.0] * d_model, "strength": 0.0, "hook": hook}],
            "n_completion_tokens": 8,
            "temperature": 0,
            "strength_multiplier": 0.0,
            "freq_penalty": 0.0,
            "seed": 42,
            "steer_special_tokens": False,
        }
        resp = client.post(
            "/v1/steer/completion-chat",
            json=req,
            headers={"X-SECRET-KEY": X_SECRET_KEY},
        )
        assert resp.status_code == 200, resp.text
        # Chat results expose ``raw`` (full rendered string), not completion's ``output``.
        outputs = {o["type"]: o["raw"] for o in resp.json()["outputs"]}
        assert "DEFAULT" in outputs
        assert isinstance(outputs["DEFAULT"], str) and len(outputs["DEFAULT"]) > 0


# --- Qwen3.5-0.8B: thinking instruct -----------------------------------------


@pytest.mark.thinking
def test_qwen35_arch_support_probe():
    """Cheap config-only check: EagerModel resolves hybrid Qwen3.5; vLLM support is reported.

    EagerModel loads multimodal ``*ForConditionalGeneration`` via the concrete arch class and
    ``resolve_arch`` finds the text trunk under ``language_model``. vLLM may or may not list
    the hybrid VL arch yet — when it doesn't, vLLM-marked cases below skip rather than fail.
    """
    arches = _qwen35_architectures()
    assert arches, "Qwen3.5-0.8B config must declare architectures"
    # Hybrid Qwen3.5 ships as Qwen3_5* (text) or a conditional-generation multimodal wrapper.
    assert any("Qwen3" in a for a in arches), f"unexpected architectures: {arches}"

    if vllm_available() and not _vllm_supports_architectures(arches):
        pytest.skip(f"vLLM does not list {arches} yet; Qwen3.5 vLLM cases will skip on load")


@pytest.mark.thinking
@pytest.mark.chat
def test_qwen_thinking_toggle_changes_tokens():
    """The ``enable_thinking`` switch must flow through the chat template and change tokens."""
    with try_initialized_server(QWEN_THINKING) as client:
        msgs = [{"role": "user", "content": "Hello"}]
        think = _apply_chat_template(
            client,
            messages=msgs,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": True},
        )
        nothink = _apply_chat_template(
            client,
            messages=msgs,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )
        assert think["tokens"] != nothink["tokens"]


@pytest.mark.thinking
@pytest.mark.chat
def test_qwen_apply_chat_template_spans():
    with try_initialized_server(QWEN_THINKING) as client:
        data = _apply_chat_template(client, messages=CHAT_MESSAGES)
        _assert_spans_cover_tokens(data)


@pytest.mark.thinking
@pytest.mark.chat
def test_qwen_persona_flag_does_not_crash():
    """``is_assistant_axis`` must be accepted end-to-end even when PersonaData is unloaded.

    Without per-model PCA assets PersonaData stays uninitialized and the monitor no-ops
    (``assistant_axis`` absent/null); the chat completion itself must still succeed.
    """
    with try_initialized_server(QWEN_THINKING) as client:
        model = Model.get_instance()
        assert isinstance(model, EagerModel)
        d_model = model.d_model
        hook = f"blocks.{max(0, model.n_layers // 2)}.hook_resid_post"
        req = {
            "prompt": [{"content": "Say hi in one word.", "role": "user"}],
            "model": QWEN_THINKING.model_id,
            "steer_method": "SIMPLE_ADDITIVE",
            "normalize_steering": False,
            "types": ["DEFAULT"],
            "vectors": [{"steering_vector": [0.0] * d_model, "strength": 0.0, "hook": hook}],
            "n_completion_tokens": 8,
            "temperature": 0,
            "strength_multiplier": 0.0,
            "freq_penalty": 0.0,
            "seed": 42,
            "steer_special_tokens": False,
            "is_assistant_axis": True,
        }
        resp = client.post(
            "/v1/steer/completion-chat",
            json=req,
            headers={"X-SECRET-KEY": X_SECRET_KEY},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        outputs = {o["type"]: o["raw"] for o in body["outputs"]}
        assert "DEFAULT" in outputs
        assert isinstance(outputs["DEFAULT"], str) and len(outputs["DEFAULT"]) > 0
        # No PCA assets for this model => persona monitor no-ops; key may be absent.
        assert body.get("assistantAxis") in (None, [])


@pytest.mark.thinking
@pytest.mark.chat
@pytest.mark.cuda
@pytest.mark.vllm
def test_qwen_vllm_apply_chat_template_when_supported():
    """Exercise Qwen3.5 on vLLM when the arch is registered; skip otherwise."""
    arches = _qwen35_architectures()
    if not (vllm_available() and _vllm_supports_architectures(arches)):
        pytest.skip(f"vLLM does not support {arches}")
    with try_initialized_server(QWEN_THINKING, engine=VLLM) as client:
        data = _apply_chat_template(client, messages=CHAT_MESSAGES)
        _assert_spans_cover_tokens(data)
