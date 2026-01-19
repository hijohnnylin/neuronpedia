"""Tests for legacy persona steering configuration loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from chatspace.generation import LegacyExperiment, load_legacy_role_trait_config


def _write_legacy_bundle(path: Path) -> None:
    payload = {
        "vectors": {
            "layer_0/foo": {
                "vector": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float16),
                "layer": 0,
            },
            "layer_1/bar": {
                "vector": torch.tensor([0.0, -3.0, 4.0], dtype=torch.float16),
                "layer": 1,
            },
        },
        "experiments": [
            {
                "id": "expA",
                "interventions": [
                    {"vector": "layer_0/foo", "cap": -0.5},
                    {"vector": "layer_1/bar", "cap": 0.75},
                ],
            },
            {
                "id": "expB",
                "interventions": [{"vector": "layer_1/bar", "cap": 1.25}],
            },
        ],
    }
    torch.save(payload, path)


def test_load_legacy_role_trait_config(tmp_path: Path) -> None:
    bundle_path = tmp_path / "legacy.pt"
    _write_legacy_bundle(bundle_path)

    experiments = load_legacy_role_trait_config(bundle_path)
    assert [exp.id for exp in experiments] == ["expA", "expB"]

    first = experiments[0]
    assert isinstance(first, LegacyExperiment)
    spec = first.spec
    assert set(spec.layers.keys()) == {0, 1}

    layer0 = spec.layers[0]
    assert len(layer0.operations) == 1
    cap0 = layer0.operations[0]
    assert cap0.min is None
    assert cap0.max == pytest.approx(-0.5)
    assert torch.allclose(cap0.vector, torch.tensor([1.0, 0.0, 0.0]))

    layer1 = spec.layers[1]
    assert len(layer1.operations) == 1
    cap1 = layer1.operations[0]
    assert cap1.min is None
    assert cap1.max == pytest.approx(0.75)
    assert torch.allclose(
        cap1.vector,
        torch.tensor([0.0, -0.6, 0.8]),
        atol=1e-6,
    )

    second = experiments[1].spec
    assert len(second.layers[1].operations) == 1
    cap_second = second.layers[1].operations[0]
    assert cap_second.min is None
    assert cap_second.max == pytest.approx(1.25)
    assert torch.allclose(
        cap_second.vector,
        torch.tensor([0.0, -0.6, 0.8]),
        atol=1e-6,
    )


def test_loader_rejects_unknown_vector(tmp_path: Path) -> None:
    bundle_path = tmp_path / "legacy.pt"
    payload = {
        "vectors": {"layer_0/foo": {"vector": torch.ones(2), "layer": 0}},
        "experiments": [
            {
                "id": "bad",
                "interventions": [{"vector": "missing"}],
            }
        ],
    }
    torch.save(payload, bundle_path)

    with pytest.raises(KeyError, match="unknown vector"):
        load_legacy_role_trait_config(bundle_path)


def test_loader_requires_cap(tmp_path: Path) -> None:
    bundle_path = tmp_path / "legacy.pt"
    payload = {
        "vectors": {"layer_0/foo": {"vector": torch.ones(2), "layer": 0}},
        "experiments": [
            {
                "id": "missing-cap",
                "interventions": [{"vector": "layer_0/foo"}],
            }
        ],
    }
    torch.save(payload, bundle_path)

    with pytest.raises(ValueError, match="missing 'cap' field"):
        load_legacy_role_trait_config(bundle_path)
