"""CPU-only contract for resolving text-stack modules on the vLLM worker model.

Plain causal LMs expose ``model.layers`` / ``model.norm``. Multimodal
``*ForConditionalGeneration`` wrappers (Qwen3.5/3.6, Gemma 4) nest the whole text LM under
``language_model``, beside the vision tower, so every one of these lookups is a level deeper
and the wrapper's own ``compute_logits`` merely delegates to it. Each resolver below must find
the text stack in both layouts and must never return a vision-tower module.
"""

from __future__ import annotations

import pytest
import torch.nn as nn
from interp_engine.vllm_capture import (
    _get_layers,
    _worker_final_norm,
    _worker_logits_processor,
    _worker_unembed_weight,
)


class _DecoderLayer(nn.Module):
    def __init__(self, d: int = 4):
        super().__init__()
        self.self_attn = nn.Linear(d, d)
        self.mlp = nn.Linear(d, d)


class _VisionBlock(nn.Module):
    def __init__(self, d: int = 4):
        super().__init__()
        self.proj = nn.Linear(d, d)


class _VisionTower(nn.Module):
    """Stands in for ``visual``: has ``blocks`` and a ``norm``, must never be picked."""

    def __init__(self, n: int = 2, d: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList([_VisionBlock(d) for _ in range(n)])
        self.norm = nn.LayerNorm(d)


class _Trunk(nn.Module):
    def __init__(self, n: int, vocab: int, d: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([_DecoderLayer(d) for _ in range(n)])
        self.norm = nn.LayerNorm(d)


class _LogitsProcessor(nn.Module):
    def __init__(self, soft_cap: float | None, scale: float):
        super().__init__()
        self.soft_cap = soft_cap
        self.scale = scale


class _CausalStyle(nn.Module):
    """Minimal stand-in for vLLM ``Qwen3ForCausalLM`` / Llama-family."""

    def __init__(self, n: int = 3, vocab: int = 8, d: int = 4, scale: float = 1.0):
        super().__init__()
        self.model = _Trunk(n, vocab, d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.logits_processor = _LogitsProcessor(soft_cap=None, scale=scale)


class _MultimodalStyle(nn.Module):
    """Minimal stand-in for vLLM ``Qwen3_5ForConditionalGeneration`` (Qwen3.6)."""

    def __init__(self, n: int = 3, vocab: int = 8, d: int = 4, scale: float = 1.0):
        super().__init__()
        self.visual = _VisionTower(d=d)
        self.language_model = _CausalStyle(n, vocab, d, scale=scale)


def test_get_layers_on_causal_model():
    model = _CausalStyle(n=4)
    assert _get_layers(model) is model.model.layers


def test_get_layers_on_qwen36_multimodal_wrapper():
    model = _MultimodalStyle(n=5)
    layers = _get_layers(model)
    assert layers is model.language_model.model.layers
    assert len(layers) == 5


def test_final_norm_resolves_in_both_layouts():
    causal = _CausalStyle()
    assert _worker_final_norm(causal) is causal.model.norm

    multimodal = _MultimodalStyle()
    assert _worker_final_norm(multimodal) is multimodal.language_model.model.norm


def test_logits_processor_is_found_on_the_nested_text_lm():
    """A miss here is silent: it reads as no-softcap + unit-scale rather than raising."""
    model = _MultimodalStyle()
    assert _worker_logits_processor(model) is model.language_model.logits_processor


def test_unembed_weight_prefers_lm_head_over_the_nested_embedding():
    model = _MultimodalStyle()
    assert _worker_unembed_weight(model) is model.language_model.lm_head.weight


def test_unembed_weight_falls_back_to_tied_embed_tokens():
    model = _MultimodalStyle()
    del model.language_model.lm_head
    assert _worker_unembed_weight(model) is model.language_model.model.embed_tokens.weight


def test_resolvers_never_pick_the_vision_tower():
    model = _MultimodalStyle()
    layers = _get_layers(model)
    assert layers is not model.visual.blocks
    assert isinstance(layers[0], _DecoderLayer)
    assert _worker_final_norm(model) is not model.visual.norm


def test_missing_text_stack_raises():
    class _NoTrunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = _VisionTower()

    with pytest.raises(RuntimeError, match="decoder layers"):
        _get_layers(_NoTrunk())
