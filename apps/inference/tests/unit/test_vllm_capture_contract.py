"""Unit tests for the invariants that keep vLLM capture honest.

They cover a bug that shipped: prefix caching was left enabled for every backend
except the prompt_embeds (NLA) one. Capture reads the worker's forward activations,
so a KV-cache prefix hit meant the cached positions were never forwarded and never
captured, and the endpoints received a silently SHORT activation tensor. That
truncated /activation/topk-by-token responses to a plausible-looking 200 and, once
/activation/all indexed a token position past the short tensor on GPU, tripped a
device-side assert that poisoned the CUDA context and killed the server process.

Caching is now back ON, and the guarantee is enforced per request instead: a capture
request carries a ``cache_salt`` that no other request uses, which makes its block
hashes unique and so forces the full prefill it needs. What these tests pin is that
the engine-level setting and the per-request opt-out stay in step -- the endpoints
depend on the pair, and either one alone is the shipped bug again. The opt-out itself
is tested exhaustively in the engine's ``tests/test_vllm_kv_isolation.py``.

These are deliberately CPU-only. The GPU parity scripts could never have caught the
original bug: only full 16-token blocks are cacheable, and every prompt in that suite
is shorter than 16 tokens, on a freshly built engine each time. Reproducing the real
thing needs a long-lived server plus two >=16-token prompts sharing a 16-token
prefix, so the guarantee is asserted here on the construction/contract level instead.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
import torch
from interp_engine import Address, vllm_backend


class _Cfg:
    num_hidden_layers = 12
    hidden_size = 768


def build_kwargs(**overrides: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "dtype": "float32",
        "gpu_memory_utilization": 0.5,
        "max_model_len": 512,
        "enforce_eager": True,
        "trust_remote_code": False,
        "storage_path": "/tmp/hs",
        "enable_extraction": False,
        "enable_prompt_embeds": False,
        "tensor_parallel_size": 1,
        "extra_vllm_kwargs": None,
    }
    args.update(overrides)
    with patch("transformers.AutoConfig.from_pretrained", return_value=_Cfg()):
        kwargs, _, _, _, _ = vllm_backend._build_extract_engine_kwargs("gpt2", **args)
    return kwargs


def test_prefix_caching_is_enabled_for_an_ordinary_activation_pod():
    """On, because capture opts itself out per request and generation should not pay for it.

    Worth about 1.75x on time-to-first-token for a repeated long prefix, which is what chat
    traffic carrying a system prompt looks like.
    """
    assert build_kwargs()["enable_prefix_caching"] is True


def test_prefix_caching_is_disabled_for_the_prompt_embeds_backend():
    """The NLA path gives it up for a reason a salt cannot address.

    Prefix caching keys on token ids and an embeds prompt has none, which hangs engine init.
    Unlike the capture hazard, there is nothing to opt out OF -- so this whole engine goes without.
    """
    assert build_kwargs(enable_prompt_embeds=True)["enable_prefix_caching"] is False


@pytest.mark.parametrize("wanted", [True, False])
def test_explicit_extra_vllm_kwargs_still_wins(wanted: bool):
    """Both directions, since the interesting one is now False: it is how an operator who does not
    trust the per-request opt-out gets the old engine-wide behaviour back."""
    kwargs = build_kwargs(extra_vllm_kwargs={"enable_prefix_caching": wanted})
    assert kwargs["enable_prefix_caching"] is wanted


def test_short_capture_is_rejected():
    """A capture covering fewer positions than the prompt must raise, not return."""
    captured = {Address("resid_post", 5): torch.zeros(3, 8)}
    with pytest.raises(RuntimeError, match="19-token prompt"):
        vllm_backend._assert_full_prompt_captured(captured, 19)


def test_full_capture_is_accepted():
    captured = {
        Address("resid_post", 5): torch.zeros(19, 8),
        Address("mlp_in", 2): torch.zeros(19, 8),
    }
    vllm_backend._assert_full_prompt_captured(captured, 19)


def test_short_capture_reports_every_offending_point():
    """The message must name the points and their row counts, for a one-look diagnosis."""
    captured = {
        Address("resid_post", 5): torch.zeros(19, 8),
        Address("mlp_in", 2): torch.zeros(3, 8),
    }
    with pytest.raises(RuntimeError) as excinfo:
        vllm_backend._assert_full_prompt_captured(captured, 19)
    assert "mlp_in" in str(excinfo.value)
    assert "resid_post" not in str(excinfo.value)


class TestCaptureWidth:
    """Rows tell you about prefix caching; width tells you about GPU sharding.

    Under tensor parallelism the capture path reads rank 0's payload alone. That is right
    for the all-reduced points and wrong for the sharded ones, and a quarter-width residual
    reaching an SAE encode is either a confusing matmul error or, worse, plausible numbers
    for a quarter of the model.
    """

    def test_sharded_residual_is_rejected(self):
        captured = {Address("resid_post", 5): torch.zeros(19, 2048)}
        with pytest.raises(RuntimeError, match="8192 wide"):
            vllm_backend._assert_full_width_captured(captured, 8192)

    def test_full_width_is_accepted(self):
        captured = {
            Address("resid_post", 5): torch.zeros(19, 8192),
            Address("mlp_out", 2): torch.zeros(19, 8192),
        }
        vllm_backend._assert_full_width_captured(captured, 8192)

    def test_head_shaped_points_are_not_width_checked(self):
        # `z` is n_heads * head_dim, which is not hidden_size on every family (Gemma 3),
        # so there is no width to check it against; the served-point gate covers it.
        captured = {Address("z", 5): torch.zeros(19, 2048)}
        vllm_backend._assert_full_width_captured(captured, 8192)

    def test_unknown_hidden_size_skips_the_check(self):
        captured = {Address("resid_post", 5): torch.zeros(19, 2048)}
        vllm_backend._assert_full_width_captured(captured, 0)


class TestAttentionRecomputeSharding:
    def test_refuses_under_tensor_parallelism(self):
        # q/k/v are head-sharded, and `dims` describes the whole model, so the reshape
        # downstream would fail on the element count. Say why instead.
        with pytest.raises(RuntimeError, match="tensor_parallel_size=4"):
            vllm_backend.recompute_attn_from_payloads({}, [0], {}, 4)
