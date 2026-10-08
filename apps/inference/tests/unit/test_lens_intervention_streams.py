"""Where a jlens intervention is written, on a trunk that carries several residual streams.

A spec that says nothing lands on ``resid_post``, and a hyper-connection trunk has no such tensor, so
the endpoint names the point its read-out is taken at. That has no later symptom -- a steer written
somewhere else still produces fluent text and a complete read-out, and the two simply disagree -- so
the spec that reaches the engine is pinned here. What the engine then does with it on each backend
(a full-stack write moving the mean exactly, one stream's row only, BOS unwritten) is pinned in the
engine's own tests, against the stack.
"""

from __future__ import annotations

import torch
from interp_engine import vllm_residual_basis
from interp_engine.points import steer_refusal_reason
from interp_engine.steer_specs import steering_spec_to_worker_specs

from neuronpedia_inference.endpoints.lens.prompt import (
    _build_lens_steering_spec,
)
from neuronpedia_inference.endpoints.lens.residual_spec import (
    BLOCK_OUTPUT,
    CAPTURE_POINT_ADDRESSES,
    LensResidualSpec,
)

D_MODEL = 4
N_STREAMS = 3
MEAN = LensResidualSpec(capture_point="block_output", stream_reduce="mean")
SELECT_1 = LensResidualSpec(capture_point="block_output", stream_reduce="select", stream_index=1)


# --- the spec that crosses to a worker ----------------------------------------------------------


def _worker_specs(steer_deltas, strength, ablate, swap_deltas, residual, n_streams) -> list[dict]:
    """What the engine registers against a vLLM request for this intervention.

    The endpoint builds one ``SteeringSpec`` and opens a ``steer()`` block with it; the engine
    flattens that spec for the worker. Pairing the two here pins the point and stream that reach
    the worker without a running engine.
    """
    spec = _build_lens_steering_spec(steer_deltas, strength, ablate, swap_deltas, residual, n_streams)
    assert spec is not None
    return steering_spec_to_worker_specs(spec)


def test_a_conventional_trunk_still_sends_the_point_the_worker_already_assumed():
    """The default is not bypassed on the trunk it was written for; it is spelled out."""
    specs = _worker_specs({0: torch.ones(D_MODEL)}, 1.0, False, None, BLOCK_OUTPUT, 1)
    assert [s["point"] for s in specs] == ["resid_post"]
    assert all(s["stream"] is None for s in specs)


def test_a_hyper_connection_trunk_is_aimed_at_the_stream_stack():
    """`resid_post` does not exist here, so a spec that said nothing had nothing to aim at."""
    deltas = {0: torch.ones(D_MODEL), 5: torch.ones(D_MODEL)}
    specs = _worker_specs(deltas, 2.0, False, None, MEAN, N_STREAMS)
    assert [(s["layer"], s["op"], s["point"]) for s in specs] == [
        (0, "norm_scaled_add", "resid_streams"),
        (5, "norm_scaled_add", "resid_streams"),
    ]
    assert all(s["stream"] is None for s in specs), "the mean is a mixture, so every stream is written"


def test_a_lens_fitted_on_one_stream_writes_that_stream():
    """And the coordinate rides on every op, since which stream is written is not the op's business."""
    deltas = {3: torch.ones(D_MODEL)}
    ablate = _worker_specs(deltas, 0.0, True, None, SELECT_1, N_STREAMS)
    swap = _worker_specs(deltas, 0.0, False, {3: torch.ones(D_MODEL)}, SELECT_1, N_STREAMS)
    assert [(s["op"], s["point"], s["stream"]) for s in ablate] == [("ablate", "resid_streams", 1)]
    assert [(s["op"], s["point"], s["stream"]) for s in swap] == [("swap", "resid_streams", 1)]


def test_the_collapse_points_are_reached_by_the_same_table_the_read_out_uses():
    """A lens fitted at a sublayer's input: the point attention actually reads on such a trunk."""
    spec = LensResidualSpec(capture_point="attn_in", stream_reduce="mean")
    specs = _worker_specs({0: torch.ones(D_MODEL)}, 1.0, False, None, spec, N_STREAMS)
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


def test_a_conventional_trunk_keeps_its_unreduced_rows():
    """Nothing above is allowed to make the single-stream path pay for the stacked one."""
    assert not BLOCK_OUTPUT.reduces and BLOCK_OUTPUT.write_stream is None
    assert vllm_residual_basis(architecture="GPT2LMHeadModel").n_streams == 1
