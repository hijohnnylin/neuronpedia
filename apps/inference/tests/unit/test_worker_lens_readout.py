"""CPU-only contract for worker-side top-k lens readout.

``worker_lens_readout`` must match the eager path's softcap / log_z / optional
non-word mask (with final-row top-1 preserved) without returning full logits.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
from interp_engine.vllm_capture import (
    decode_tensor_payload,
    encode_tensor_payload,
    worker_lens_readout,
)


class _FakeInner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.Identity()


class _FakeLM(nn.Module):
    """Minimal vLLM-shaped model: ``model.norm`` + ``compute_logits`` via ``lm_head``."""

    # vLLM attaches this at runtime; not an nn.Module/Parameter.
    logits_processor: Any = None

    def __init__(self, vocab: int = 16, d: int = 4):
        super().__init__()
        self.model = _FakeInner()
        self.lm_head = nn.Linear(d, vocab, bias=False)
        with torch.no_grad():
            # Distinct, deterministic rows so argmax/topk are stable.
            self.lm_head.weight.copy_(torch.arange(vocab * d, dtype=torch.float32).reshape(vocab, d) * 0.01)

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)


def _fake_worker(model: nn.Module) -> SimpleNamespace:
    return SimpleNamespace(model_runner=SimpleNamespace(model=model))


def test_worker_lens_readout_topk_shape_and_argmax():
    model = _FakeLM()
    resid = torch.randn(6, 4)  # 2 positions × 3 layers
    out = worker_lens_readout(
        _fake_worker(model),
        encode_tensor_payload(resid),
        top_n=3,
        softcap=None,
        word_mask_payload=None,
        rows_per_group=3,
    )
    top_idx = decode_tensor_payload(out["top_idx"])
    top_probs = decode_tensor_payload(out["top_probs"])
    assert top_idx.shape == (6, 3)
    assert top_probs.shape == (6, 3)
    assert top_idx.dtype == torch.int64
    assert torch.all(top_probs > 0)
    assert torch.all(top_probs <= 1.0 + 1e-5)

    with torch.no_grad():
        ref = model.compute_logits(resid.float())
        assert torch.equal(top_idx[:, 0], ref.argmax(dim=-1))
        log_z = ref.float().logsumexp(dim=-1)
        expected_top1_prob = (ref.float().max(dim=-1).values - log_z).exp()
        assert torch.allclose(top_probs[:, 0], expected_top1_prob, atol=1e-5)


def test_worker_lens_readout_applies_softcap_before_probs():
    model = _FakeLM()
    resid = torch.randn(3, 4) * 5.0  # large enough that tanh softcap moves mass
    softcap = 5.0
    out = worker_lens_readout(
        _fake_worker(model),
        encode_tensor_payload(resid),
        top_n=4,
        softcap=softcap,
        word_mask_payload=None,
        rows_per_group=3,
    )
    top_idx = decode_tensor_payload(out["top_idx"])
    top_probs = decode_tensor_payload(out["top_probs"])

    with torch.no_grad():
        raw = model.compute_logits(resid.float())
        capped = softcap * torch.tanh(raw / softcap)
        log_z = capped.logsumexp(dim=-1, keepdim=True)
        ref_idx = capped.topk(4, dim=-1).indices
        ref_probs = (capped.gather(-1, ref_idx) - log_z).exp()
    assert torch.equal(top_idx, ref_idx.to(torch.int64))
    assert torch.allclose(top_probs, ref_probs.to(torch.float32), atol=1e-5)


def test_worker_lens_readout_word_mask_preserves_final_row_top1():
    model = _FakeLM(vocab=16, d=4)
    resid = torch.randn(3, 4)
    with torch.no_grad():
        logits = model.compute_logits(resid.float())
        true_final_top1 = int(logits[2].argmax())

    # Mask out the final row's true top-1; intermediate rows must drop it,
    # final row must keep it (eager `_TypeReadoutState` contract).
    mask = torch.ones(16, dtype=torch.bool)
    mask[true_final_top1] = False
    out = worker_lens_readout(
        _fake_worker(model),
        encode_tensor_payload(resid),
        top_n=4,
        softcap=None,
        word_mask_payload=encode_tensor_payload(mask),
        rows_per_group=3,
    )
    top_idx = decode_tensor_payload(out["top_idx"])
    assert true_final_top1 not in {int(x) for x in top_idx[0]}
    assert true_final_top1 not in {int(x) for x in top_idx[1]}
    assert int(top_idx[2, 0]) == true_final_top1


def test_worker_lens_readout_pads_short_word_mask_to_logits_vocab():
    """Llama-3-style: tokenizer.vocab_size under-counts the padded embedding table."""
    model = _FakeLM(vocab=20, d=4)
    resid = torch.randn(3, 4)
    # Mask covers only the first 16 ids (like tokenizer.vocab_size=128000 vs 128256).
    mask = torch.ones(16, dtype=torch.bool)
    mask[0] = False
    out = worker_lens_readout(
        _fake_worker(model),
        encode_tensor_payload(resid),
        top_n=3,
        softcap=None,
        word_mask_payload=encode_tensor_payload(mask),
        rows_per_group=3,
    )
    top_idx = decode_tensor_payload(out["top_idx"])
    assert top_idx.shape == (3, 3)
    with torch.no_grad():
        final_top1 = int(model.compute_logits(resid.float())[2].argmax())
    # Padded slots (16..19) are non-word; intermediate rows must not surface them.
    for row in (0, 1):
        assert all(int(x) < 16 for x in top_idx[row])
    # Final row still preserves its true top-1 even if that id is a padded slot.
    assert int(top_idx[2, 0]) == final_top1


def _readout(model: nn.Module, resid: torch.Tensor):
    return worker_lens_readout(
        _fake_worker(model),
        encode_tensor_payload(resid),
        top_n=3,
        softcap=None,
        word_mask_payload=None,
        rows_per_group=3,
    )


def test_unit_logit_scale_is_accepted():
    # The overwhelmingly common case: vLLM passes scale=1.0, so eager and fused agree.
    model = _FakeLM()
    model.logits_processor = SimpleNamespace(scale=1.0, soft_cap=None)
    assert _readout(model, torch.randn(6, 4))["top_idx"] is not None


def test_applied_logit_scale_is_refused():
    """The unreconciled half of the Gemma-2 softcap problem.

    vLLM's LogitsProcessor applies `scale` inside compute_logits and the eager lm_head does
    not, so the same residual decodes differently on the two backends with nothing looking
    wrong on either. No fleet model sets a non-unit scale, so this refuses rather than
    silently picking a convention — which is exactly how the softcap got applied twice.
    """
    model = _FakeLM()
    model.logits_processor = SimpleNamespace(scale=0.5, soft_cap=None)
    with pytest.raises(RuntimeError, match="scale=0.5"):
        _readout(model, torch.randn(6, 4))


def test_missing_logits_processor_is_not_a_scale():
    # Families that never construct a LogitsProcessor must not trip the guard.
    assert _readout(_FakeLM(), torch.randn(6, 4))["top_idx"] is not None
