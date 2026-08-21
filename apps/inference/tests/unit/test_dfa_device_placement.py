"""DFA operands can arrive on different devices, and the math has to cope.

On the vLLM backend `value`/`attn_probs` come back from the worker as CPU tensors (they
travel as serialized payloads) while the SAE's `W_enc` stays on the serving GPU, so
`dfa_from_v_and_probs` was handed a genuine device mismatch and every `-att-` source 500'd
in the einsum with "Expected all tensors to be on the same device". The eager backend never
saw it: there, both sides are already on the model's device.

Cheap enough to run in CI: synthetic tensors only, no model or SAE weights.
"""

from __future__ import annotations

import math

import einops
import pytest
import torch

from neuronpedia_inference.engine_adapter import (
    _dfa_compute_device,
    dfa_from_v_and_probs,
)

N_HEADS = 4
N_KV_HEADS = 2
HEAD_DIM = 3
D_IN = N_HEADS * HEAD_DIM
D_SAE = 5
SEQ = 6
FEATURE_INDEX = 2
DEST_POS = 4


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    v = torch.randn(1, SEQ, N_KV_HEADS, HEAD_DIM, generator=generator)
    attn_weights = torch.rand(1, N_HEADS, SEQ, SEQ, generator=generator)
    W_enc = torch.randn(D_IN, D_SAE, generator=generator)
    return v, attn_weights, W_enc


def _dfa(v: torch.Tensor, attn_weights: torch.Tensor, W_enc: torch.Tensor):
    return dfa_from_v_and_probs(
        v,
        attn_weights,
        W_enc,
        FEATURE_INDEX,
        DEST_POS,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
    )


def _dfa_naive(v: torch.Tensor, attn_weights: torch.Tensor, W_enc: torch.Tensor) -> list[float]:
    """The obvious formulation, kept as the numeric oracle for the real one.

    Weight every head's value vector by its attention, concatenate, and only then project
    onto the encoder direction. Costs ``dest_pos x src_pos x d_model``, which is why
    `dfa_from_v_and_probs` contracts ``head_dim`` first instead -- this is the answer that
    rearrangement has to reproduce.
    """
    if N_KV_HEADS < N_HEADS:
        v = v.repeat_interleave(N_HEADS // N_KV_HEADS, dim=2)
    v_cat = einops.rearrange(v, "batch src n_heads d_head -> batch src (n_heads d_head)")
    attn_bcast = einops.repeat(
        attn_weights,
        "batch n_heads dest src -> batch dest src (n_heads d_head)",
        d_head=HEAD_DIM,
    )
    per_src_pos = einops.einsum(
        attn_bcast * v_cat.unsqueeze(1),
        W_enc[:, FEATURE_INDEX],
        "batch dest src d_model, d_model -> batch dest src",
    )
    return per_src_pos[0, DEST_POS].tolist()


def test_dfa_compute_device_prefers_an_accelerator_over_the_cpu():
    """A non-CPU operand decides the device, whichever position it arrives in.

    `meta` stands in for a GPU so this holds on CPU-only runners too; all that matters
    to the rule is that the device is not the CPU.
    """
    cpu = torch.zeros(2)
    not_cpu = torch.zeros(2, device="meta")

    assert _dfa_compute_device(not_cpu, cpu, cpu).type == "meta"
    assert _dfa_compute_device(cpu, not_cpu, cpu).type == "meta"
    assert _dfa_compute_device(cpu, cpu, cpu).type == "cpu"


def test_dfa_on_the_cpu_alone():
    """Baseline, and the shape contract: one value per source position."""
    result = _dfa(*_inputs())

    assert len(result["dfa_values"]) == SEQ
    assert all(math.isfinite(value) for value in result["dfa_values"])
    assert result["dfa_target_index"] == DEST_POS
    assert result["dfa_max_value"] == pytest.approx(max(result["dfa_values"]))


def test_dfa_agrees_with_the_naive_formulation():
    """Contracting head_dim before src_pos is an algebraic rearrangement, not an
    approximation, so the two must agree to float tolerance."""
    v, attn_weights, W_enc = _inputs()

    result = _dfa(v, attn_weights, W_enc)

    assert result["dfa_values"] == pytest.approx(_dfa_naive(v, attn_weights, W_enc), rel=1e-5)


def test_dfa_rejects_an_encoder_that_is_not_hook_z_shaped():
    """d_in must split evenly into (n_heads, head_dim); a silent reshape here would
    return plausible-looking wrong numbers rather than failing."""
    v, attn_weights, _ = _inputs()
    wrong_width = torch.randn(D_IN + 1, D_SAE)

    with pytest.raises(ValueError, match="hook_z encoder"):
        _dfa(v, attn_weights, wrong_width)


def _warm_up_cublas() -> None:
    """Pay the one-time cuBLAS workspace before any measurement, so it is not attributed to
    the call that happens to be first.

    PyTorch allocates that workspace through its own caching allocator on the first matmul
    on a device, so it lands in ``max_memory_allocated`` as a fixed cost of the first
    multiplication regardless of size -- 32 MiB on torch 2.13/cu130, for an ``8x4 @ 4x2``
    product. Left unpaid it dwarfs everything the DFA math itself allocates.
    """
    _ = torch.randn(8, 4, device="cuda") @ torch.randn(4, 2, device="cuda")
    torch.cuda.synchronize()


def _dfa_peak_overhead(seq: int) -> int:
    """Bytes `_dfa` adds to the allocator's peak beyond its own operands, at this length."""
    generator = torch.Generator().manual_seed(0)
    v = torch.randn(1, seq, N_KV_HEADS, HEAD_DIM, generator=generator).cuda()
    attn_weights = torch.rand(1, N_HEADS, seq, seq, generator=generator).cuda()
    W_enc = torch.randn(D_IN, D_SAE, generator=generator).cuda()

    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    _dfa(v, attn_weights, W_enc)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - baseline


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_dfa_peak_memory_does_not_scale_with_src_pos_squared():
    """The regression guard for the rewrite.

    The naive formulation materializes two ``[dest_pos, src_pos, d_model]`` tensors, so its
    peak grows quadratically in the prompt length -- gigabytes on a real prompt, once per
    result in `/activation/all`. Peak here must track ``src_pos``, not ``src_pos^2``.

    Read as growth across two prompt lengths, not as one absolute ceiling. A ceiling has to
    be set above whatever fixed CUDA overheads the process pays, which is a number about the
    torch build rather than about this math: the cuBLAS workspace above grew past the old
    budget and failed this test against an unchanged implementation. Doubling the length
    separates linear from quadratic whatever the constants are.
    """
    _warm_up_cublas()

    short = _dfa_peak_overhead(512)
    long = _dfa_peak_overhead(1024)

    # Linear doubles, quadratic quadruples. Comfortably either side of 3, and the observed
    # ratio is 2.00 to the pixel, because what is allocated here is a handful of
    # `[src_pos, n_heads]` tensors and nothing else.
    assert long < short * 3, f"peak grew {long / short:.2f}x for 2x the prompt: {short} -> {long} bytes"

    # And the level, not just the slope: the naive form's two [dest, src, d_model] tensors
    # are ~1000x what this allocates, so the bound stays loose enough not to flake.
    naive_overhead = 2 * 1024 * 1024 * D_IN * 4
    assert long < naive_overhead / 10, f"{long} bytes is within 10x of the naive {naive_overhead}"


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_dfa_with_cpu_captures_and_gpu_sae_weights():
    """The vLLM shape of this call: CPU captures, GPU SAE. Same answer as all-CPU."""
    v, attn_weights, W_enc = _inputs()
    expected = _dfa(v, attn_weights, W_enc)

    mixed = _dfa(v, attn_weights, W_enc.cuda())

    assert mixed["dfa_values"] == pytest.approx(expected["dfa_values"], rel=1e-4)
    assert mixed["dfa_max_value"] == pytest.approx(expected["dfa_max_value"], rel=1e-4)
    assert mixed["dfa_target_index"] == expected["dfa_target_index"]
