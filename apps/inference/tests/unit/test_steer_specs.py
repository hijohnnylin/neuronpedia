"""The engine steering specs a steer request builds: one per feature, in request order."""

from __future__ import annotations

import pytest
import torch
from interp_engine import AddSpec, OrthogonalDecompSpec, ProjectionCapSpec

from neuronpedia_inference.endpoints.steer.completion import features_to_steering_specs, steer_write_targets
from neuronpedia_inference.inference_utils.steering import SteeringSettings
from neuronpedia_inference.schemas import NPSteerMethod, NPSteerVector


def _vector(hook: str, values: list[float], strength: float = 1.0) -> NPSteerVector:
    return NPSteerVector(steering_vector=values, strength=strength, hook=hook)


def test_each_feature_is_one_spec_at_its_point_in_request_order() -> None:
    features = [
        _vector("blocks.4.hook_resid_post", [3.0, 4.0], strength=2.0),
        _vector("blocks.2.attn.hook_z", [1.0, 0.0]),
        _vector("blocks.0.hook_resid_pre", [0.0, 1.0]),
    ]
    specs = features_to_steering_specs(SteeringSettings(features=features, strength_multiplier=0.5))
    assert [(s.point, list(s.layers)) for s in specs] == [("resid_post", [4]), ("z", [2]), ("embeddings", [0])]
    (op,) = specs[0].layers[4].operations
    assert isinstance(op, AddSpec) and op.scale == 1.0
    torch.testing.assert_close(torch.as_tensor(op.vector), torch.tensor([3.0, 4.0]))


def test_normalize_makes_each_vector_unit_length_first() -> None:
    settings = SteeringSettings(
        features=[_vector("blocks.1.hook_resid_post", [3.0, 4.0])], strength_multiplier=2.0, normalize_steering=True
    )
    (op,) = features_to_steering_specs(settings)[0].layers[1].operations
    torch.testing.assert_close(torch.as_tensor(op.vector), torch.tensor([0.6, 0.8]))


def test_normalize_refuses_a_zero_vector() -> None:
    settings = SteeringSettings(
        features=[_vector("blocks.1.hook_resid_post", [0.0, 0.0])], strength_multiplier=1.0, normalize_steering=True
    )
    with pytest.raises(ValueError, match="zero vector"):
        features_to_steering_specs(settings)


@pytest.mark.parametrize(
    ("method", "kind", "field"),
    [
        (NPSteerMethod.ORTHOGONAL_DECOMP, OrthogonalDecompSpec, "coeff"),
        (NPSteerMethod.PROJECTION_CAP, ProjectionCapSpec, "max"),
    ],
)
def test_the_method_picks_the_op_and_takes_the_scaled_strength(method: NPSteerMethod, kind: type, field: str) -> None:
    settings = SteeringSettings(
        features=[_vector("blocks.1.hook_resid_post", [1.0, 1.0], strength=3.0)],
        strength_multiplier=2.0,
        steer_method=method,
    )
    (op,) = features_to_steering_specs(settings)[0].layers[1].operations
    assert isinstance(op, kind) and getattr(op, field) == 6.0


def test_no_features_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one"):
        features_to_steering_specs(SteeringSettings(features=[], strength_multiplier=1.0))


def test_write_targets_are_sorted_and_deduplicated_per_point() -> None:
    features = [
        _vector("blocks.5.hook_resid_post", [1.0]),
        _vector("blocks.2.hook_resid_pre", [1.0]),
        _vector("blocks.5.hook_resid_post", [1.0]),
        _vector("blocks.3.attn.hook_z", [1.0]),
        _vector("ln_final.hook_normalized", [1.0]),
    ]
    assert steer_write_targets(features) == {"resid_post": [1, 5], "z": [3]}
