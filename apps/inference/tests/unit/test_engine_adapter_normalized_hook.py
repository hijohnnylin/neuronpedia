"""`blocks.{i}.ln2.hook_normalized` is not a point, and the transcoders are trained there.

TransformerLens fires that hook inside the norm, on `x / scale` and before the learned gain, so the
engine's mapper refuses the name rather than answering with `mlp_in` -- the same tensor times that
gain. Every GemmaScope transcoder (`gemma-2-2b/*-gemmascope-transcoder-16k`) declares it as its
SAELens `hook_name`, so the capture path recomputes it from the norm's input.

The arithmetic itself is the engine's (`pre_gain_normalized`, pinned against a real RMSNorm module in
the engine's `tests/test_normalized_hook.py`), so what is worth pinning *here* is the routing: that
each hook name is resolved to the right point, that only the normalized ones are divided afterwards,
and that a model whose norms are not RMS norms is refused rather than served a centered tensor's
uncentered lookalike. A drift in any of those would not raise -- it would return a plausible tensor
and quietly change which features a shipping dashboard says fire.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from interp_engine import Address, UnmappedHook, pre_gain_normalized

from neuronpedia_inference.engine_adapter import (
    _capture_points,
    _finish,
    _rms_norm_eps,
)

EPS = 1e-6
D_MODEL = 16


@pytest.fixture(autouse=True)
def _forget_memoized_eps():
    """`_rms_norm_eps` caches per model id, and these tests hand it several fake models."""
    from neuronpedia_inference import engine_adapter

    engine_adapter._RMS_NORM_EPS.clear()
    yield
    engine_adapter._RMS_NORM_EPS.clear()


def _config(**fields: object) -> SimpleNamespace:
    """A config with enough dims for the engine's resolver, plus whatever the test is about."""
    return SimpleNamespace(
        architectures=["FakeForCausalLM"],
        num_hidden_layers=4,
        num_attention_heads=8,
        hidden_size=128,
        vocab_size=1000,
        **fields,
    )


def _model(**config_fields: object) -> SimpleNamespace:
    return SimpleNamespace(hf_model_id="fake/model", config=_config(**config_fields))


class TestHookResolution:
    def test_the_pre_mlp_norm_is_captured_at_its_input(self):
        (point,) = _capture_points(["blocks.19.ln2.hook_normalized"])
        assert point.address == Address("resid_mid", 19)
        assert point.normalize

    def test_the_pre_attention_norm_reads_the_residual_one_step_earlier(self):
        (point,) = _capture_points(["blocks.3.ln1.hook_normalized"])
        assert point.address == Address("resid_pre", 3)
        assert point.normalize

    def test_ordinary_hooks_are_untouched_and_owe_nothing(self):
        points = _capture_points(["blocks.5.hook_resid_post", "blocks.5.mlp.hook_in", "blocks.5.attn.hook_z"])
        assert [point.address for point in points] == [
            Address("resid_post", 5),
            Address("mlp_in", 5),
            Address("z", 5),
        ]
        assert not any(point.normalize for point in points)

    def test_a_post_sublayer_norm_still_refuses(self):
        # Gemma-2's sandwich norms carry a `hook_normalized` too, and its input is the sublayer's
        # output rather than the residual. Nothing is trained there, so it must not be guessed at.
        with pytest.raises(UnmappedHook):
            _capture_points(["blocks.19.ln2_post.hook_normalized"])

    def test_a_hook_that_is_neither_still_reports_the_mappers_refusal(self):
        with pytest.raises(UnmappedHook):
            _capture_points(["blocks.19.ln2.hook_scale"])


class TestEpsilon:
    def test_read_from_the_models_config(self):
        assert _rms_norm_eps(_model(rms_norm_eps=1e-5)) == pytest.approx(1e-5)

    def test_read_from_a_multimodal_configs_text_half(self):
        # Gemma-3 and friends keep it under the text sub-config; the top level has none.
        model = SimpleNamespace(
            hf_model_id="fake/multimodal",
            config=SimpleNamespace(
                architectures=["FakeForConditionalGeneration"],
                text_config=_config(rms_norm_eps=1e-4),
            ),
        )
        assert _rms_norm_eps(model) == pytest.approx(1e-4)

    def test_a_layernorm_family_is_refused_rather_than_approximated(self):
        # No `rms_norm_eps` means the norms subtract the mean, so TransformerLens' `hook_normalized`
        # is a centered tensor this arithmetic does not produce.
        with pytest.raises(UnmappedHook) as excinfo:
            _rms_norm_eps(_model(layer_norm_epsilon=1e-5))
        assert "rms_norm_eps" in str(excinfo.value)

    def test_the_answer_is_memoized_per_model(self):
        """The batch endpoints ask once per hook per request, and resolving walks a whole config."""
        from neuronpedia_inference import engine_adapter

        model = _model(rms_norm_eps=1e-5)
        assert _rms_norm_eps(model) == pytest.approx(1e-5)
        # Reaching for the config again would raise; the memo means it is not consulted twice.
        model.config = None
        assert _rms_norm_eps(model) == pytest.approx(1e-5)
        assert list(engine_adapter._RMS_NORM_EPS) == ["fake/model"]


class TestFinish:
    def test_only_the_normalized_points_are_divided(self):
        x = torch.randn(1, 4, D_MODEL) * 3.0
        model = _model(rms_norm_eps=EPS)
        (plain,) = _capture_points(["blocks.2.hook_resid_mid"])
        (normalized,) = _capture_points(["blocks.2.ln2.hook_normalized"])
        # The plain point is passed through untouched -- the same object, not an equal tensor.
        assert _finish(plain, x, model) is x
        assert torch.allclose(_finish(normalized, x, model), pre_gain_normalized(x, EPS))

    def test_the_models_own_epsilon_is_the_one_applied(self):
        """A hard-coded default would be within tolerance of every real config, so this uses one
        that is not: the two epsilons differ enough to distinguish which was used."""
        x = torch.randn(1, 4, D_MODEL) * 0.01
        (normalized,) = _capture_points(["blocks.2.ln2.hook_normalized"])
        finished = _finish(normalized, x, _model(rms_norm_eps=0.5))
        assert torch.allclose(finished, pre_gain_normalized(x, 0.5))
        assert not torch.allclose(finished, pre_gain_normalized(x, 1e-6), atol=1e-3)
