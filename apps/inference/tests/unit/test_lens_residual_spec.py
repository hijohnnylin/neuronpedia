"""Which activation a lens decodes, read off the artifact rather than assumed.

A ``J_bar`` is a square matrix and says nothing about what it multiplies. On a conventional trunk
that costs nothing -- there is one residual per layer -- and the read-out hard-coded ``resid_post``
for as long as every served model had one. On DeepSeek-V4's hyper-connection trunk the capture is
``[tokens, 4, d_model]``, and the stream mean, the stream sum and each of the four streams are all
``d_model``-wide: a lens fitted on one of them loads, runs, and reads out fluent tokens from the
wrong space. No shape check catches it and no output looks wrong.

So the two things worth pinning are the two that have no later symptom:

- a declaration in the file's ``provenance`` is what gets used, end to end, and reaches the point
  name and the reduction rather than being recorded and ignored (the loader dropped ``provenance``
  entirely before this).
- a multi-stream trunk with NOTHING declared is refused, and the refusal says where to write it.
  Falling back to a default here is the failure mode the whole mechanism exists to prevent.

No model and no weights: the question is which address and which reduction a declaration resolves to,
which is a property of the table and the stream count.
"""

from __future__ import annotations

import pytest
import torch
from interp_engine import ResidualBasisUnsupported, eager_residual_basis, vllm_residual_basis

from neuronpedia_inference.endpoints.lens.lens_loader import LoadedJacobianLens
from neuronpedia_inference.endpoints.lens.residual_spec import (
    BLOCK_OUTPUT,
    LensResidualSpec,
    LensSpaceUnknown,
    from_provenance,
    resolve_residual_spec,
)

DSV4 = "DeepseekV4ForCausalLM"
D_MODEL = 4


def single_stream():
    return eager_residual_basis(architecture="GPT2LMHeadModel")


def multi_stream():
    return vllm_residual_basis(n_residual_streams=4, architecture=DSV4)


def checkpoint(provenance: dict | None) -> dict:
    """A lens file's contents, as ``LoadedJacobianLens.load`` reads them."""
    return {
        "J": {0: torch.zeros(D_MODEL, D_MODEL, dtype=torch.float16)},
        "source_layers": [0],
        "n_prompts": 25,
        "d_model": D_MODEL,
        **({} if provenance is None else {"provenance": provenance}),
    }


# --- reading the declaration off the file -------------------------------------------------------


def test_the_cblank_deepseek_v4_provenance_is_read_as_the_stream_mean():
    """The artifact this mechanism was built for, with its provenance verbatim."""
    spec = from_provenance(
        {
            "model_id": "deepseek-ai/DeepSeek-V4-Flash",
            "dataset_id": "NeelNanda/pile-10k",
            "target_layer": 41,
            "capture_point": "block_output",
            "stream_reduce": "mean",
            "stream_index": None,
        }
    )
    assert spec == LensResidualSpec(capture_point="block_output", stream_reduce="mean")
    assert spec is not None and spec.reduces
    assert spec.point_name(4) == "resid_streams", "resid_post is refused outright on this trunk"
    assert spec.point_name(1) == "resid_post"


def test_a_lens_with_no_provenance_declares_nothing():
    """Every lens fitted before these fields existed, which is all of ours."""
    assert from_provenance(None) is None
    assert from_provenance({"model_id": "gpt2", "n_prompts": 1000}) is None


def test_a_declaration_that_cannot_be_used_is_refused_rather_than_defaulted():
    """It was written by something, so falling back would serve a space the file denies."""
    with pytest.raises(LensSpaceUnknown, match="unknown lens capture point"):
        from_provenance({"capture_point": "somewhere_else"})
    with pytest.raises(LensSpaceUnknown, match="unknown lens stream reduction"):
        from_provenance({"capture_point": "block_output", "stream_reduce": "median"})
    with pytest.raises(LensSpaceUnknown, match="disagree"):
        from_provenance({"stream_reduce": "select"})
    with pytest.raises(LensSpaceUnknown, match="disagree"):
        from_provenance({"stream_reduce": "mean", "stream_index": 2})


def test_the_loader_carries_the_declaration_off_the_file(tmp_path):
    """It used to drop `provenance` on load, so this is the link that did not exist."""
    path = tmp_path / "lens.pt"
    torch.save(checkpoint({"capture_point": "block_output", "stream_reduce": "mean"}), path)
    lens = LoadedJacobianLens.load(str(path), device_budget_bytes=0)
    assert lens.residual == LensResidualSpec(capture_point="block_output", stream_reduce="mean")
    assert lens.residual is not None
    assert "block_output" in lens.residual.describe() and "mean" in lens.residual.describe()

    torch.save(checkpoint(None), path)
    assert LoadedJacobianLens.load(str(path), device_budget_bytes=0).residual is None


# --- resolving it against the served model ------------------------------------------------------


def test_an_undeclared_lens_on_a_conventional_trunk_is_the_block_output():
    """Back-compatibility, and the only thing an old artifact could have meant."""
    assert resolve_residual_spec(None, single_stream()) is BLOCK_OUTPUT
    assert not BLOCK_OUTPUT.reduces


def test_an_undeclared_lens_on_a_hyper_connection_trunk_is_refused():
    """The case the module exists for: a guess here reads out confidently in the wrong space."""
    with pytest.raises(LensSpaceUnknown, match="4 parallel residual streams") as refused:
        resolve_residual_spec(None, multi_stream())
    message = str(refused.value)
    assert "convert-external-lens.py" in message, "the refusal must say where to record it"
    assert "--stream-reduce" in message


def test_a_declaration_that_contradicts_the_trunk_is_refused_both_ways():
    """Neither direction is a shape error later, so both have to be caught here."""
    with pytest.raises(ResidualBasisUnsupported, match="has no stream axis"):
        resolve_residual_spec(LensResidualSpec(stream_reduce="mean"), single_stream())
    with pytest.raises(ResidualBasisUnsupported, match="has to say which d_model vector"):
        resolve_residual_spec(BLOCK_OUTPUT, multi_stream())


def test_a_sublayer_capture_point_needs_no_reduction_even_on_a_hyper_connection_trunk():
    """`attn_out` is d_model-wide before it is scattered, so the stream count does not decide this."""
    spec = LensResidualSpec(capture_point="attn_out")
    assert resolve_residual_spec(spec, multi_stream()) is spec
    assert spec.point_name(4) == "attn_out" == spec.point_name(1)


def test_a_single_stream_lens_maps_onto_the_hyper_connection_equivalents():
    """The multi-stream column is the point that means the same thing, not a rename of the first."""
    assert LensResidualSpec(capture_point="attn_in").point_name(1) == "resid_pre"
    assert LensResidualSpec(capture_point="attn_in").point_name(4) == "attn_stream_collapse"
    assert LensResidualSpec(capture_point="mlp_in").point_name(1) == "resid_mid"
    assert LensResidualSpec(capture_point="mlp_in").point_name(4) == "mlp_stream_collapse"


# --- applying it --------------------------------------------------------------------------------


def test_the_reduction_applied_here_is_the_one_declared():
    stack = torch.arange(2 * 4 * D_MODEL, dtype=torch.float32).reshape(2, 4, D_MODEL)
    assert torch.equal(LensResidualSpec(stream_reduce="mean").reduce(stack, 4), stack.mean(dim=-2))
    assert torch.equal(LensResidualSpec(stream_reduce="sum").reduce(stack, 4), stack.sum(dim=-2))
    selected = LensResidualSpec(stream_reduce="select", stream_index=2)
    assert torch.equal(selected.reduce(stack, 4), stack[:, 2, :])


def test_no_reduction_leaves_a_flat_capture_alone():
    rows = torch.randn(3, D_MODEL)
    assert BLOCK_OUTPUT.reduce(rows, 1) is rows


def test_reducing_a_capture_whose_stream_axis_is_the_wrong_width_is_refused():
    """The axis is second-from-last, and averaging a different one returns a believable shape."""
    with pytest.raises(ValueError, match="expects 4 streams"):
        LensResidualSpec(stream_reduce="mean").reduce(torch.zeros(2, 5, D_MODEL), 4)
