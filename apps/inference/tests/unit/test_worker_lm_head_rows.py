"""CPU-only contract for unembedding-row lookup on the vLLM worker model.

Gemma 1/2 in vLLM never create ``lm_head`` under tied embeddings — ``compute_logits``
uses ``model.embed_tokens`` directly. Gemma 4 multimodal wrappers nest the text LM under
``language_model``, so ``lm_head`` is not top-level. ``worker_lm_head_rows`` must resolve
both instead of raising.

Under tensor parallelism the head is vocab-sharded. Each rank returns only the rows it
owns; the client merges them. Indexing a global id outside a rank's shard would be an
out-of-bounds GPU assert that poisons the CUDA context, so the owned-mask path is load-
bearing, not cosmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from interp_engine.vllm_capture import (
    _local_lm_head_rows,
    _worker_unembed_weight,
    decode_tensor_payload,
    merge_lm_head_row_payloads,
    worker_lm_head_rows,
)
from interp_engine.vllm_capture.lens import unembed as unembed_module


class _EmbedTokens(nn.Module):
    def __init__(self, vocab: int, d: int):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(vocab * d, dtype=torch.float32).reshape(vocab, d))


class _Gemma2Inner(nn.Module):
    def __init__(self, vocab: int, d: int):
        super().__init__()
        self.embed_tokens = _EmbedTokens(vocab, d)


class _Gemma2Style(nn.Module):
    """Minimal stand-in for vLLM ``Gemma2ForCausalLM`` (no ``lm_head``)."""

    def __init__(self, vocab: int = 8, d: int = 4):
        super().__init__()
        self.model = _Gemma2Inner(vocab, d)


class _Gemma4LanguageModel(nn.Module):
    def __init__(self, vocab: int, d: int):
        super().__init__()
        self.lm_head = nn.Linear(d, vocab, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.arange(vocab * d, dtype=torch.float32).reshape(vocab, d))


class _Gemma4MultimodalStyle(nn.Module):
    """Minimal stand-in for vLLM ``Gemma4ForConditionalGeneration`` (nested text LM)."""

    def __init__(self, vocab: int = 8, d: int = 4):
        super().__init__()
        self.language_model = _Gemma4LanguageModel(vocab, d)


class _LlamaStyle(nn.Module):
    """Minimal stand-in for vLLM models that expose ``lm_head``."""

    def __init__(self, vocab: int = 8, d: int = 4):
        super().__init__()
        self.lm_head = nn.Linear(d, vocab, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.arange(vocab * d, dtype=torch.float32).reshape(vocab, d))


def _fake_worker(model: nn.Module) -> SimpleNamespace:
    return SimpleNamespace(model_runner=SimpleNamespace(model=model))


@dataclass
class _ShardIndices:
    """Stand-in for ``VocabParallelEmbeddingShardIndices`` (org vocab only, no LoRA)."""

    padded_org_vocab_start_index: int
    padded_org_vocab_end_index: int
    padded_added_vocab_start_index: int
    padded_added_vocab_end_index: int
    org_vocab_start_index: int
    org_vocab_end_index: int
    added_vocab_start_index: int
    added_vocab_end_index: int

    @property
    def num_org_vocab_padding(self) -> int:
        return (
            self.padded_org_vocab_end_index
            - self.padded_org_vocab_start_index
            - (self.org_vocab_end_index - self.org_vocab_start_index)
        )


class _ShardedLMHead(nn.Module):
    """One TP rank's slice of a vocab-parallel lm_head."""

    def __init__(self, full_weight: torch.Tensor, start: int, end: int, tp_size: int):
        super().__init__()
        self.tp_size = tp_size
        self.shard_indices = _ShardIndices(
            padded_org_vocab_start_index=start,
            padded_org_vocab_end_index=end,
            padded_added_vocab_start_index=full_weight.shape[0],
            padded_added_vocab_end_index=full_weight.shape[0],
            org_vocab_start_index=start,
            org_vocab_end_index=end,
            added_vocab_start_index=full_weight.shape[0],
            added_vocab_end_index=full_weight.shape[0],
        )
        self.weight = nn.Parameter(full_weight[start:end].clone())


class _ShardedLlamaRank(nn.Module):
    def __init__(self, lm_head: _ShardedLMHead):
        super().__init__()
        self.lm_head = lm_head


def test_unembed_weight_falls_back_to_tied_embed_tokens():
    model = _Gemma2Style()
    weight = _worker_unembed_weight(model)
    assert weight is model.model.embed_tokens.weight


def test_unembed_weight_falls_back_to_nested_language_model_lm_head():
    model = _Gemma4MultimodalStyle()
    weight = _worker_unembed_weight(model)
    assert weight is model.language_model.lm_head.weight


def test_unembed_weight_prefers_lm_head_when_present():
    model = _LlamaStyle()
    weight = _worker_unembed_weight(model)
    assert weight is model.lm_head.weight


def test_worker_lm_head_rows_on_gemma2_tied_embeddings():
    model = _Gemma2Style()
    payload = worker_lm_head_rows(_fake_worker(model), [1, 3, 5])
    rows = merge_lm_head_row_payloads([1, 3, 5], [payload])
    expected = model.model.embed_tokens.weight[[1, 3, 5]].detach()
    assert rows.shape == (3, 4)
    assert torch.equal(rows, expected)


def test_worker_lm_head_rows_on_gemma4_multimodal():
    model = _Gemma4MultimodalStyle()
    payload = worker_lm_head_rows(_fake_worker(model), [0, 2, 7])
    rows = merge_lm_head_row_payloads([0, 2, 7], [payload])
    expected = model.language_model.lm_head.weight[[0, 2, 7]].detach()
    assert torch.equal(rows, expected)


def test_worker_lm_head_rows_on_lm_head():
    model = _LlamaStyle()
    payload = worker_lm_head_rows(_fake_worker(model), [0, 2])
    rows = merge_lm_head_row_payloads([0, 2], [payload])
    expected = model.lm_head.weight[[0, 2]].detach()
    assert torch.equal(rows, expected)


def test_worker_lm_head_rows_answers_when_unsharded(monkeypatch):
    monkeypatch.setattr(unembed_module, "_worker_tp_world_size", lambda: 1)
    model = _LlamaStyle()
    payload = worker_lm_head_rows(_fake_worker(model), [0, 2])
    assert payload["owned"] == [True, True]
    rows = decode_tensor_payload(payload["rows"])
    assert torch.equal(rows, model.lm_head.weight[[0, 2]].detach())


class TestVocabParallelGather:
    """TP>1: each rank returns its owned rows; the client merges them."""

    @pytest.fixture
    def full_weight(self):
        # Distinct rows so a wrong-rank gather is obvious.
        return torch.arange(16 * 3, dtype=torch.float32).reshape(16, 3)

    def test_local_rows_only_select_owned_ids(self, full_weight, monkeypatch):
        # Without the owned-mask, indexing token 12 on rank 0 (which only holds 0..7)
        # would be an out-of-bounds index_select on a real sharded weight.
        monkeypatch.setattr(unembed_module, "_worker_tp_world_size", lambda: 2)
        rank0 = _ShardedLMHead(full_weight, 0, 8, tp_size=2)
        owned, rows = _local_lm_head_rows(rank0, [1, 12, 3])
        assert owned == [True, False, True]
        assert rows is not None
        assert torch.equal(rows, full_weight[[1, 3]])

    def test_merge_across_two_ranks_recovers_full_rows(self, full_weight, monkeypatch):
        monkeypatch.setattr(unembed_module, "_worker_tp_world_size", lambda: 2)
        token_ids = [1, 12, 3, 15]
        payloads = [
            worker_lm_head_rows(
                _fake_worker(_ShardedLlamaRank(_ShardedLMHead(full_weight, 0, 8, 2))),
                token_ids,
            ),
            worker_lm_head_rows(
                _fake_worker(_ShardedLlamaRank(_ShardedLMHead(full_weight, 8, 16, 2))),
                token_ids,
            ),
        ]
        assert payloads[0]["owned"] == [True, False, True, False]
        assert payloads[1]["owned"] == [False, True, False, True]
        merged = merge_lm_head_row_payloads(token_ids, payloads)
        assert torch.equal(merged, full_weight[token_ids])

    def test_merge_raises_when_an_id_is_unowned(self, full_weight, monkeypatch):
        monkeypatch.setattr(unembed_module, "_worker_tp_world_size", lambda: 2)
        # Both ranks only cover 0..15; ask for 99.
        payloads = [
            worker_lm_head_rows(
                _fake_worker(_ShardedLlamaRank(_ShardedLMHead(full_weight, 0, 8, 2))),
                [99],
            ),
            worker_lm_head_rows(
                _fake_worker(_ShardedLlamaRank(_ShardedLMHead(full_weight, 8, 16, 2))),
                [99],
            ),
        ]
        with pytest.raises(RuntimeError, match="not found on any TP rank"):
            merge_lm_head_row_payloads([99], payloads)

    def test_merge_raises_on_double_claim(self):
        from interp_engine.vllm_capture import encode_tensor_payload

        row = encode_tensor_payload(torch.ones(1, 2))
        payloads = [
            {"owned": [True], "rows": row},
            {"owned": [True], "rows": row},
        ]
        with pytest.raises(RuntimeError, match="more than one TP rank"):
            merge_lm_head_row_payloads([0], payloads)
