"""Where a jlens intervention is written, on a trunk that carries several residual streams.

Reading a stream stack and writing one used to be different problems, and the endpoint refused
every steer, ablation and swap on a hyper-connection model rather than guess. The engine can write
these points -- a spec names one, and ``resid_streams`` is steered by re-running the mHC kernel's
second half -- so what was missing was the endpoint saying which point it meant. A spec that says
nothing lands on ``resid_post``, and a hyper-connection trunk has no such tensor.

Three things have no later symptom and are pinned here:

- the spec that crosses to the worker carries the point the READ-OUT is taken at. A steer written
  somewhere else still produces fluent text and a complete read-out, and the two simply disagree.
- what a write to the whole stack does to the vector the lens decodes. Ablation and swap are linear
  in the residual, so applying them per stream moves the mixture by exactly what applying them to
  the mixture would -- the property that makes writing every stream the intervention rather than an
  approximation of it. It is asserted against the reduced row, not argued for in a comment.
- the BOS skip mask has the rank of the tensor it scopes. torch aligns from the right, so a flat
  mask lines its sequence axis up with the STREAM axis: at some prompt lengths that raises, and at
  others it silently protects the wrong positions.

No weights and no GPU: a stub trunk whose blocks return the stack is enough to run the eager
read-out end to end, and the vLLM half is the spec dict.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.nn as nn
from interp_engine import EagerModel, eager_residual_basis, vllm_residual_basis
from interp_engine.points import steer_refusal_reason

from neuronpedia_inference.endpoints.lens.prompt import (
    _apply_swap,
    _bos_skip_mask,
    _build_vllm_lens_specs,
    _iter_residuals_engine,
)
from neuronpedia_inference.endpoints.lens.residual_spec import (
    BLOCK_OUTPUT,
    CAPTURE_POINT_ADDRESSES,
    LensResidualSpec,
)

D_MODEL = 4
N_STREAMS = 3
N_LAYERS = 2
LAYERS = [0, 1]
DSV4 = "DeepseekV4ForCausalLM"

# Five, against three streams, so a flat mask cannot broadcast against the stack by accident.
BOS_ID = 9
PROMPT = [BOS_ID, 1, 2, 3, 4]

MEAN = LensResidualSpec(capture_point="block_output", stream_reduce="mean")
SELECT_1 = LensResidualSpec(capture_point="block_output", stream_reduce="select", stream_index=1)


def multi_stream():
    return eager_residual_basis(n_residual_streams=N_STREAMS, architecture=DSV4)


# --- a stub hyper-connection trunk --------------------------------------------------------------


class _Block(nn.Module):
    """A decoder block whose output is the whole ``[batch, seq, streams, d_model]`` stack.

    Which is what makes this trunk worth stubbing at all: on a conventional model element 0 of a
    block's return is one residual, and every shape downstream of the write hook is the same either
    way apart from that axis.
    """

    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = index

    def forward(self, stack: torch.Tensor) -> tuple[torch.Tensor, None]:
        return stack + float(self.index + 1), None


class _Trunk(nn.Module):
    """Runs the blocks in order and records each one's output after the hooks have run."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_Block(i) for i in range(N_LAYERS))
        self.written: list[torch.Tensor] = []

    def forward(self, tokens: torch.Tensor, past_key_values=None, use_cache: bool = True):
        seq = tokens.shape[1]
        torch.manual_seed(0)
        stack = torch.randn(1, seq, N_STREAMS, D_MODEL)
        self.written = []
        for block in self.layers:
            stack = block(stack)[0]
            self.written.append(stack.clone())
        return SimpleNamespace(logits=torch.zeros(1, seq, 11), past_key_values=None)


def _model():
    trunk = _Trunk()
    return SimpleNamespace(
        hf_model=trunk,
        arch=SimpleNamespace(decoder_layers=list(trunk.layers)),
        device=torch.device("cpu"),
        residual_basis=multi_stream(),
    )


def _prefill(model, residual: LensResidualSpec, **intervention) -> dict[int, torch.Tensor]:
    """The prompt positions' read-out rows, one ``[seq, d_model]`` block per layer."""
    batches = list(
        _iter_residuals_engine(
            cast(EagerModel, model),
            PROMPT,
            LAYERS,
            num_completion_tokens=0,
            temperature=0.0,
            eos_token_ids=None,
            bos_token_id=BOS_ID,
            residual=residual,
            **intervention,
        )
    )
    rows = batches[0]
    return {layer: torch.stack([row[2][layer] for row in rows], dim=0) for layer in LAYERS}


def _directions() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    return torch.randn(D_MODEL), torch.randn(D_MODEL)


# --- the vLLM spec ------------------------------------------------------------------------------


def test_a_conventional_trunk_still_sends_the_point_the_worker_already_assumed():
    """The default is not bypassed on the trunk it was written for; it is spelled out."""
    specs = _build_vllm_lens_specs({0: torch.ones(D_MODEL)}, 1.0, False, None, BLOCK_OUTPUT, 1)
    assert [s["point"] for s in specs] == ["resid_post"]
    assert all("stream" not in s for s in specs)


def test_a_hyper_connection_trunk_is_aimed_at_the_stream_stack():
    """`resid_post` does not exist here, so a spec that said nothing had nothing to aim at."""
    deltas = {0: torch.ones(D_MODEL), 5: torch.ones(D_MODEL)}
    specs = _build_vllm_lens_specs(deltas, 2.0, False, None, MEAN, N_STREAMS)
    assert [(s["layer"], s["op"], s["point"]) for s in specs] == [
        (0, "steer", "resid_streams"),
        (5, "steer", "resid_streams"),
    ]
    assert all("stream" not in s for s in specs), "the mean is a mixture, so every stream is written"


def test_a_lens_fitted_on_one_stream_writes_that_stream():
    """And the coordinate rides on every op, since which stream is written is not the op's business."""
    deltas = {3: torch.ones(D_MODEL)}
    ablate = _build_vllm_lens_specs(deltas, 0.0, True, None, SELECT_1, N_STREAMS)
    swap = _build_vllm_lens_specs(deltas, 0.0, False, {3: torch.ones(D_MODEL)}, SELECT_1, N_STREAMS)
    assert [(s["op"], s["point"], s["stream"]) for s in ablate] == [("ablate", "resid_streams", 1)]
    assert [(s["op"], s["point"], s["stream"]) for s in swap] == [("swap", "resid_streams", 1)]


def test_the_collapse_points_are_reached_by_the_same_table_the_read_out_uses():
    """A lens fitted at a sublayer's input: the point attention actually reads on such a trunk."""
    spec = LensResidualSpec(capture_point="attn_in", stream_reduce="mean")
    specs = _build_vllm_lens_specs({0: torch.ones(D_MODEL)}, 1.0, False, None, spec, N_STREAMS)
    assert specs[0]["point"] == "attn_stream_collapse"


def test_every_point_the_table_can_name_is_one_the_engine_agrees_is_writable():
    """The cross-repo half of the wiring, and the one a rename in either repo would break quietly.

    `steer_refusal_reason` is the engine's claim about the POINT, shared by its client gate and its
    worker registration, so a capture point that mapped onto a coefficient row would be refused
    several frames into an RPC rather than here.
    """
    for single, multi in CAPTURE_POINT_ADDRESSES.values():
        assert steer_refusal_reason(single) is None, single
        assert steer_refusal_reason(multi) is None, multi


# --- what the write does to the vector the lens decodes -----------------------------------------


def test_the_write_stream_is_the_one_the_lens_was_fitted_on_and_nothing_otherwise():
    assert MEAN.write_stream is None
    assert LensResidualSpec(capture_point="block_output", stream_reduce="sum").write_stream is None
    assert SELECT_1.write_stream == 1
    assert BLOCK_OUTPUT.write_stream is None


def test_swapping_every_stream_moves_the_mean_by_exactly_the_swap_on_the_mean():
    """The claim that makes writing the whole stack the intervention rather than an approximation.

    A swap is linear in the residual, so per-stream and on-the-mixture agree exactly. Asserted
    against the reduced row the read-out actually returns, at a non-BOS position.
    """
    src, tgt = _directions()
    baseline = _prefill(_model(), MEAN)
    swapped = _prefill(_model(), MEAN, steer_deltas={0: src}, swap_deltas={0: tgt})

    expected = _apply_swap(baseline[0], src, tgt)
    torch.testing.assert_close(swapped[0][1:], expected[1:])
    torch.testing.assert_close(swapped[1][1:], expected[1:] + 2.0, msg="layer 1 sees layer 0's write")


def test_ablation_removes_the_direction_from_the_mixture_and_not_merely_from_each_stream():
    direction, _ = _directions()
    ablated = _prefill(_model(), MEAN, steer_deltas={0: direction}, steer_strength=0.0, steer_ablate=True)
    unit = direction / torch.linalg.vector_norm(direction)
    projection = (ablated[0][1:] * unit).sum(dim=-1)
    torch.testing.assert_close(projection, torch.zeros_like(projection), atol=1e-6, rtol=0)


def test_a_select_lens_leaves_the_streams_it_did_not_name_untouched():
    """Invisible in the read-out, which returns only the selected stream -- so read off the trunk."""
    src, tgt = _directions()
    model = _model()
    _prefill(model, SELECT_1, steer_deltas={0: src}, swap_deltas={0: tgt})
    written = model.hf_model.written[0]

    baseline_model = _model()
    _prefill(baseline_model, SELECT_1)
    baseline = baseline_model.hf_model.written[0]

    for stream in (0, 2):
        torch.testing.assert_close(written[:, :, stream, :], baseline[:, :, stream, :])
    assert not torch.allclose(written[:, 1:, 1, :], baseline[:, 1:, 1, :])


# --- the BOS skip -------------------------------------------------------------------------------


def test_the_skip_mask_has_the_rank_of_the_stack_it_scopes():
    mask = _bos_skip_mask(PROMPT, BOS_ID, torch.device("cpu"))
    stacked = _bos_skip_mask(PROMPT, BOS_ID, torch.device("cpu"), stacked=True)
    assert mask is not None and stacked is not None
    assert mask.shape == (1, len(PROMPT), 1)
    assert stacked.shape == (1, len(PROMPT), 1, 1)
    with pytest.raises(RuntimeError):
        # What the flat mask does against a stack, and why the rank is not left to broadcasting.
        torch.where(cast(Any, mask), torch.zeros(1, len(PROMPT), N_STREAMS, D_MODEL), 1.0)


def test_the_bos_position_is_left_unwritten_across_every_stream():
    """Its attention-sink norm makes a norm-scaled intervention spuriously large there."""
    src, tgt = _directions()
    model = _model()
    _prefill(model, MEAN, steer_deltas={0: src}, swap_deltas={0: tgt})
    written = model.hf_model.written[0]

    baseline_model = _model()
    _prefill(baseline_model, MEAN)
    baseline = baseline_model.hf_model.written[0]

    torch.testing.assert_close(written[:, 0], baseline[:, 0], msg="BOS is position 0 of this prompt")
    assert not torch.allclose(written[:, 1:], baseline[:, 1:])


# --- the read-out is unchanged where it was already right ---------------------------------------


def test_an_unintervened_read_out_is_the_reduction_and_nothing_else():
    """The path that already worked, so the write plumbing must not have moved it."""
    rows = _prefill(_model(), MEAN)
    torch.manual_seed(0)
    stack = torch.randn(1, len(PROMPT), N_STREAMS, D_MODEL)
    torch.testing.assert_close(rows[0], (stack + 1.0).mean(dim=-2)[0])
    torch.testing.assert_close(rows[1], (stack + 3.0).mean(dim=-2)[0])


def test_a_conventional_trunk_keeps_its_flat_mask_and_its_unreduced_rows():
    """Nothing above is allowed to make the single-stream path pay for the stacked one."""
    assert not BLOCK_OUTPUT.reduces and BLOCK_OUTPUT.write_stream is None
    assert vllm_residual_basis(architecture="GPT2LMHeadModel").n_streams == 1
    mask = _bos_skip_mask(PROMPT, BOS_ID, torch.device("cpu"))
    assert mask is not None and mask.ndim == 3
