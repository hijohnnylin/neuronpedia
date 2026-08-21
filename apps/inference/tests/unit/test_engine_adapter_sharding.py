"""What a vLLM instance will serve depends on how its GPUs are sharded, not just on vLLM.

The capture path reads rank 0's payload alone. Tensor parallelism all-reduces the residual
and MLP points before the hook sees them, so rank 0 holds the whole vector there; it splits
attention heads, so rank 0 holds a fraction of `z` and of the q/k/v the pattern recompute
needs. Nothing downstream can tell the difference from the numbers, which is why the refusal
has to happen up here, where a `BackendUnsupported` becomes a 400 rather than wrong output.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from interp_engine import HOOK_CAPTURE_POINTS, Address

from neuronpedia_inference.engine_adapter import (
    BackendUnsupported,
    _assert_vllm_points_supported,
    _capture_points,
    vllm_attention_unsupported_reason,
    vllm_served_capture_points,
)


def _backend(tensor_parallel_size: int) -> SimpleNamespace:
    return SimpleNamespace(tensor_parallel_size=tensor_parallel_size)


class TestServedCapturePoints:
    def test_a_single_gpu_serves_everything_the_engine_declares_hookable(self):
        # The set is the engine's, not this app's: a copy here went stale for `attn_out` and
        # refused the attention-output SAEs with an error blaming the paged-attention kernel.
        assert vllm_served_capture_points(_backend(1)) == set(HOOK_CAPTURE_POINTS)

    def test_sharded_drops_the_head_sharded_points_and_keeps_the_rest(self):
        served = vllm_served_capture_points(_backend(4))
        assert not ({"z", "value", "mlp_act", "q_norm_in", "k_norm_out"} & served)
        assert {"resid_pre", "resid_mid", "resid_post", "mlp_in", "mlp_out", "attn_out"} <= served

    def test_a_replicated_point_survives_sharding(self):
        # `router_logits` is `n_experts` wide off a ReplicatedLinear gate, so every rank computes
        # the whole thing. It was refused while this set was "everything not `hidden_size` wide".
        assert "router_logits" in vllm_served_capture_points(_backend(4))

    def test_the_declared_points_are_the_served_set(self):
        model = SimpleNamespace(
            static_points=(Address("resid_post", 7), Address("mlp_out", 3)),
            tensor_parallel_size=1,
        )
        assert vllm_served_capture_points(model) == {"resid_post", "mlp_out"}

    def test_missing_attribute_is_treated_as_single_gpu(self):
        # Any backend predating the attribute is single-GPU by construction.
        assert "z" in vllm_served_capture_points(SimpleNamespace())


class TestPointAssertion:
    def test_residual_points_pass_when_sharded(self):
        _assert_vllm_points_supported(_backend(4), _capture_points(["blocks.50.hook_resid_post"]))

    def test_attention_output_passes_on_one_gpu(self):
        # The regression this file grew for: TransformerLens' block-level `hook_attn_out` is an
        # ordinary module output on the vLLM tree, hooked like `mlp_out` and unaffected by the
        # kernel that keeps `attn_probs` on the recompute path.
        points = _capture_points([f"blocks.{i}.hook_attn_out" for i in range(12)])
        _assert_vllm_points_supported(_backend(1), points)

    def test_z_is_refused_when_sharded(self):
        with pytest.raises(BackendUnsupported) as excinfo:
            _assert_vllm_points_supported(_backend(4), _capture_points(["blocks.5.attn.hook_z"]))
        message = str(excinfo.value)
        assert "hook_z" in message
        assert "sharded across 4 GPUs" in message

    def test_z_passes_on_one_gpu(self):
        _assert_vllm_points_supported(_backend(1), _capture_points(["blocks.5.attn.hook_z"]))

    def test_a_point_with_no_vllm_path_quotes_the_engines_own_reason(self):
        # Not a sharding fact, so the message must not blame the GPU layout -- it names what the
        # point table says, which for `mlp_pre` is that vLLM fuses the two input projections.
        with pytest.raises(BackendUnsupported) as excinfo:
            _assert_vllm_points_supported(_backend(1), _capture_points(["blocks.5.mlp.hook_pre"]))
        message = str(excinfo.value)
        assert "sharded" not in message
        assert "mlp_pre" in message

    def test_the_neuron_basis_passes_on_one_gpu(self):
        # `mlp_act` is the down projection's input, an ordinary module boundary on the vLLM tree.
        _assert_vllm_points_supported(_backend(1), _capture_points(["blocks.5.mlp.hook_post"]))

    def test_a_normalized_hook_is_judged_by_the_point_it_captures(self):
        # `blocks.19.ln2.hook_normalized` is served by capturing `resid_mid`, which every vLLM
        # instance serves, so the transcoder sources must pass even sharded -- the requested hook
        # name is not itself a point and would have no entry to look up.
        points = _capture_points([f"blocks.{i}.ln2.hook_normalized" for i in range(26)])
        assert {point.address for point in points} == {Address("resid_mid", i) for i in range(26)}
        _assert_vllm_points_supported(_backend(4), points)


class TestAttentionSharding:
    def test_available_on_one_gpu(self):
        assert vllm_attention_unsupported_reason(_backend(1)) is None

    def test_explained_when_sharded(self):
        reason = vllm_attention_unsupported_reason(_backend(4))
        assert reason is not None
        assert "4 GPUs" in reason
