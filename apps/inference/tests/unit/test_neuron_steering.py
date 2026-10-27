"""Steering an MLP neuron: a one-hot vector at ``mlp_act``, and the requests a pod refuses."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from interp_engine.steer_specs import AddSpec

from neuronpedia_inference.endpoints.steer.completion import features_to_steering_specs
from neuronpedia_inference.inference_utils import steering
from neuronpedia_inference.inference_utils.steering import (
    NeuronSteerError,
    SteeringSettings,
    neuron_steering_vector,
    process_features_vectorized,
    refuse_split_neuron_steer,
)
from neuronpedia_inference.sae_manager import SAE_TYPE
from neuronpedia_inference.schemas import NPSteerFeature, NPSteerMethod

D_MLP = 6


class FakeSAEManager:
    """Two neuron layers and nothing else. ``get_sae`` fails, since a neuron has no SAE to load."""

    mlp_width = D_MLP
    mlp_width_uniform = True

    def get_sae_type(self, source: str) -> str | None:
        return SAE_TYPE.NEURONS if source in ("0", "1") else None

    def get_sae_hook(self, source: str) -> str:
        return f"blocks.{source}.mlp.hook_post"

    def get_sae(self, source: str):  # noqa: ANN201
        raise AssertionError(f"a neuron source has no SAE to load: {source}")


@pytest.fixture
def sae_manager(monkeypatch: pytest.MonkeyPatch) -> FakeSAEManager:
    manager = FakeSAEManager()
    monkeypatch.setattr(steering.SAEManager, "get_instance", staticmethod(lambda: manager))
    return manager


@pytest.fixture
def pod(monkeypatch: pytest.MonkeyPatch):
    def _set(backend: str, num_gpus: int) -> None:
        config = SimpleNamespace(backend=backend, num_gpus=num_gpus)
        monkeypatch.setattr(steering.Config, "get_instance", staticmethod(lambda: config))

    return _set


def _neuron(source: str, index: int, strength: float = 4.0) -> NPSteerFeature:
    return NPSteerFeature(model="m", source=source, index=index, strength=strength)


def test_a_neuron_steers_with_a_one_hot_vector() -> None:
    assert neuron_steering_vector(2, 4) == [0.0, 0.0, 1.0, 0.0]


@pytest.mark.parametrize(
    ("index", "d_mlp", "match"), [(6, 6, "out of range"), (-1, 6, "out of range"), (0, None, "width")]
)
def test_a_neuron_the_model_does_not_have_is_refused(index: int, d_mlp: int | None, match: str) -> None:
    with pytest.raises(NeuronSteerError, match=match):
        neuron_steering_vector(index, d_mlp)


def test_neuron_features_write_their_strength_at_mlp_act(sae_manager: FakeSAEManager) -> None:  # noqa: ARG001
    features = process_features_vectorized([_neuron("1", 3, strength=4.0), _neuron("0", 0, strength=-2.0)])
    specs = features_to_steering_specs(SteeringSettings(features=features, strength_multiplier=1.5))
    assert [(s.point, list(s.layers)) for s in specs] == [("mlp_act", [1]), ("mlp_act", [0])]
    (op,) = specs[0].layers[1].operations
    assert isinstance(op, AddSpec)
    assert op.scale == 6.0
    torch.testing.assert_close(torch.as_tensor(op.vector), torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]))


def test_a_model_whose_layers_differ_in_width_is_refused(sae_manager: FakeSAEManager) -> None:
    sae_manager.mlp_width_uniform = False
    with pytest.raises(NeuronSteerError, match="differs by layer"):
        process_features_vectorized([_neuron("0", 1)])


@pytest.mark.parametrize("method", [NPSteerMethod.ORTHOGONAL_DECOMP, NPSteerMethod.PROJECTION_CAP])
def test_a_split_mlp_takes_only_additive_neuron_steers(sae_manager, pod, method: NPSteerMethod) -> None:  # noqa: ANN001, ARG001
    pod("vllm", 2)
    with pytest.raises(NeuronSteerError, match="only be steered with SIMPLE_ADDITIVE"):
        refuse_split_neuron_steer([_neuron("0", 1)], method)
    refuse_split_neuron_steer([_neuron("0", 1)], NPSteerMethod.SIMPLE_ADDITIVE)


@pytest.mark.parametrize(("backend", "num_gpus"), [("vllm", 1), ("eager", 2)])
def test_a_whole_mlp_takes_every_neuron_steer_method(sae_manager, pod, backend: str, num_gpus: int) -> None:  # noqa: ANN001, ARG001
    pod(backend, num_gpus)
    refuse_split_neuron_steer([_neuron("0", 1)], NPSteerMethod.PROJECTION_CAP)
