"""The app asks the model which points it serves, and does not second-guess the answer.

This file used to hold the app's own copy of that answer, and the copy was wrong twice. First it
froze vLLM's set at five points while the engine grew five more, so the attention-output SAEs became
a 400 blaming the paged-attention kernel. Then, derived instead of copied, it narrowed the set by
tensor-parallel shard width -- and the engine's worker gathers the sharded points at collect, so a
multi-GPU pod was refusing points that work.

Both mistakes have the same shape: a second opinion about a capability, maintained by hand, in a
place that cannot see what the capture path sees. So what is tested here is no longer *which* points
are served -- that is the engine's, with its own tests against a live capture on each backend -- but
that this app forwards the question and reports the answer it gets.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import cast

import pytest
from interp_engine import Address
from interp_engine.describe import describe_model
from interp_engine.protocol import InterpModel
from interp_engine.residual_basis import ResidualBasis

from neuronpedia_inference.engine_adapter import (
    BackendUnsupported,
    _assert_points_served,
    _capture_points,
    backend_name,
)


def served_capture_points(model: InterpModel) -> set[str]:
    return set(model.describe().capture_points)


class _Double(SimpleNamespace):
    """A model that refuses exactly the point names it was built with, and serves the rest."""

    def __init__(self, refused: Mapping[str, str] | None = None, **kw) -> None:
        shape = {"hf_model_id": "double", "n_layers": 12, "d_model": 768, "n_heads": 12, "n_kv_heads": 12}
        super().__init__(
            hooks_available=True,
            graph_replay=False,
            static_points=(),
            static_writes=(),
            residual_basis=ResidualBasis(),
            head_dim=64,
            **shape,
            **kw,
        )
        self._refused = dict(refused or {})

    def describe(self):  # noqa: ANN201
        return describe_model(cast("InterpModel", self), "double")

    def refuses(self, point, layer=None):  # noqa: ANN001, ANN202
        address = point if isinstance(point, Address) else Address(str(point), layer)
        return self._refused.get(address.name)

    def serves(self, point, layer=None) -> bool:  # noqa: ANN001
        return self.refuses(point, layer) is None


def _Model(refused: Mapping[str, str] | None = None, **kw) -> InterpModel:  # noqa: N802
    """The double above, as the protocol the functions under test are typed against.

    Cast rather than filled in: these tests are about the capability query, and a double carrying
    weights and a tokenizer to satisfy the rest of the contract would say nothing more.
    """
    return cast("InterpModel", _Double(refused, **kw))


class TestPointAssertion:
    def test_a_point_the_model_serves_passes(self):
        _assert_points_served(_Model(), _capture_points(["blocks.50.hook_resid_post"]))

    def test_the_models_reason_is_what_the_caller_is_told(self):
        """Quoted rather than paraphrased: the reason is the only part that says what to do next."""
        reason = "vLLM fuses the two input projections into one matmul"
        with pytest.raises(BackendUnsupported) as excinfo:
            _assert_points_served(_Model({"mlp_pre": reason}), _capture_points(["blocks.5.mlp.hook_pre"]))
        message = str(excinfo.value)
        assert reason in message
        assert "blocks.5.mlp.hook_pre" in message, "the caller asked for a hook name, so name it back"

    def test_the_message_names_the_backend_that_refused(self):
        with pytest.raises(BackendUnsupported) as excinfo:
            _assert_points_served(_Model({"mlp_pre": "no"}), _capture_points(["blocks.5.mlp.hook_pre"]))
        assert backend_name(_Model()) in str(excinfo.value).lower() or "model" in str(excinfo.value).lower()

    def test_one_refused_point_does_not_hide_the_others(self):
        points = _capture_points(["blocks.5.mlp.hook_pre", "blocks.5.attn.hook_z"])
        with pytest.raises(BackendUnsupported) as excinfo:
            _assert_points_served(_Model({"mlp_pre": "fused", "z": "sharded"}), points)
        message = str(excinfo.value)
        assert "fused" in message and "sharded" in message

    def test_a_normalized_hook_is_judged_by_the_point_it_captures(self):
        """App-side arithmetic, so it stays here: the requested name is not a point at all.

        ``blocks.19.ln2.hook_normalized`` is served by capturing ``resid_mid`` and rescaling, so the
        verdict has to be asked about ``resid_mid``. Asking about the hook name would look up an
        entry that does not exist.
        """
        points = _capture_points([f"blocks.{i}.ln2.hook_normalized" for i in range(26)])
        assert {point.address for point in points} == {Address("resid_mid", i) for i in range(26)}
        _assert_points_served(_Model(), points)

    def test_a_refusal_is_raised_before_any_capture(self):
        """`BackendUnsupported` is a 400, so it must not cost a forward to discover."""
        model = _Model({"mlp_pre": "fused"})
        with pytest.raises(BackendUnsupported):
            _assert_points_served(model, _capture_points(["blocks.5.mlp.hook_pre"]))


class TestAdvertisedSet:
    def test_it_reports_what_the_model_says_it_serves(self):
        served = served_capture_points(_Model({"z": "sharded", "router_logits": "no router here"}))
        assert "resid_post" in served
        assert not ({"z", "router_logits"} & served)

    def test_an_architectures_own_gaps_are_not_advertised(self):
        """The half a backend-level table cannot know: this checkpoint has no QK-norm.

        A pod that advertised its backend's set announced these and then 500'd from the worker when
        someone believed it.
        """
        absent: dict[str, str] = dict.fromkeys(
            ("q_norm_in", "q_norm_out", "k_norm_in", "k_norm_out"), "gpt2 has no QK-norm"
        )
        served = served_capture_points(_Model(absent))
        assert not (set(absent) & served)
        assert "resid_post" in served


class TestAttentionIsTheEnginesAnswer:
    """The app had its own reason for refusing attention -- that a sharded pod has no rank holding
    every head -- and the engine now gathers the heads at collect, which made the app's copy refuse
    a pattern that works. So the app keeps no reason of its own: it asks about the point.
    """

    def test_a_sharded_pod_is_not_refused_by_the_app(self):
        sharded = _Model(tensor_parallel_size=8)
        assert sharded.serves(Address("attn_probs", 0)), "shard width is not the app's question"

    def test_the_engines_reason_is_what_reaches_the_caller(self):
        reason = "the off-kernel attention recompute cannot reproduce this model's configuration"
        model = _Model({"attn_probs": reason})
        assert model.refuses(Address("attn_probs", 0)) == reason
