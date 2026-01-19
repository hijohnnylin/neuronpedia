"""Compatibility helpers for legacy steering configuration artifacts.

This module provides shims for loading historical ``.pt`` experiment bundles
that encode steering vectors alongside per-layer projection caps.  The legacy
format stores a dictionary with ``vectors`` (each entry containing the raw
tensor and target layer) and ``experiments`` (lists of ``{"vector": name,
"cap": value}`` interventions).  The helpers below translate that structure into
the modern :class:`SteeringSpec` representation used by :class:`VLLMSteerModel`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping
import math

import torch

from .vllm_steer_model import (
    AddSpec,
    LayerSteeringSpec,
    ProjectionCapSpec,
    SteeringSpec,
)


@dataclass(frozen=True)
class LegacyExperiment:
    """Container describing an experiment defined in the legacy ``.pt`` bundle."""

    id: str
    spec: SteeringSpec


def _prepare_vector(name: str, payload: Mapping[str, object]) -> tuple[int, torch.Tensor, float]:
    try:
        vector = payload["vector"]
        layer = payload["layer"]
    except KeyError as exc:
        raise KeyError(f"Legacy vector entry '{name}' missing key: {exc}") from exc

    if not isinstance(vector, torch.Tensor):
        raise TypeError(f"Legacy vector '{name}' must be a torch.Tensor, received {type(vector)}")

    tensor = vector.detach().to(dtype=torch.float32).contiguous()
    norm = float(torch.linalg.norm(tensor).item())
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"Legacy vector '{name}' has non-positive norm {norm}")
    unit = tensor / norm

    try:
        layer_idx = int(layer)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Legacy vector '{name}' layer must be convertible to int, received {layer!r}") from exc
    if layer_idx < 0:
        raise ValueError(f"Legacy vector '{name}' has negative layer index {layer_idx}")

    return layer_idx, unit, norm


def load_legacy_role_trait_config(path: str | Path) -> list[LegacyExperiment]:
    """Load steering experiments from a historical persona ``.pt`` bundle.

    Parameters
    ----------
    path :
        Filesystem path to the serialized configuration (typically
        ``role_trait_config.pt``).  The file must contain a dictionary with
        ``vectors`` and ``experiments`` keys mirroring the legacy format.

    Returns
    -------
    list[LegacyExperiment]
        Ordered collection preserving the experiment order encoded in the file.
        Each experiment is translated into a :class:`SteeringSpec` where the
        stored vector becomes an additive steering component and the legacy
        ``cap`` value (when present) is mapped to ``max`` on the
        associated :class:`ProjectionCapSpec`.
    """

    path = Path(path)
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"Legacy config at {path} must deserialize to a mapping, received {type(payload)}")

    try:
        vector_section = payload["vectors"]
        experiment_section = payload["experiments"]
    except KeyError as exc:
        raise KeyError(f"Legacy config missing required key: {exc}") from exc

    if not isinstance(vector_section, Mapping):
        raise TypeError(f"'vectors' entry must be a mapping, received {type(vector_section)}")
    if not isinstance(experiment_section, Iterable):
        raise TypeError(f"'experiments' entry must be iterable, received {type(experiment_section)}")

    prepared_vectors: dict[str, tuple[int, torch.Tensor, float]] = {}
    for name, vector_payload in vector_section.items():
        if not isinstance(name, str):
            raise TypeError(f"Legacy vector name must be a string, received {type(name)}")
        if not isinstance(vector_payload, Mapping):
            raise TypeError(f"Legacy vector '{name}' must map to a dict, received {type(vector_payload)}")
        prepared_vectors[name] = _prepare_vector(name, vector_payload)

    experiments: list[LegacyExperiment] = []
    for experiment in experiment_section:
        if not isinstance(experiment, Mapping):
            raise TypeError(f"Legacy experiment entries must be mappings, received {type(experiment)}")
        try:
            experiment_id = experiment["id"]
            interventions = experiment["interventions"]
        except KeyError as exc:
            raise KeyError(f"Legacy experiment missing key: {exc}") from exc
        if not isinstance(experiment_id, str):
            raise TypeError(f"Legacy experiment id must be a string, received {type(experiment_id)}")
        if not isinstance(interventions, Iterable):
            raise TypeError(f"Legacy experiment interventions must be iterable, received {type(interventions)}")

        layers: dict[int, LayerSteeringSpec] = {}
        for intervention in interventions:
            if not isinstance(intervention, Mapping):
                raise TypeError(f"Legacy intervention must be a mapping, received {type(intervention)}")
            try:
                vector_name = intervention["vector"]
            except KeyError as exc:
                raise KeyError(f"Legacy intervention missing 'vector': {exc}") from exc
            if not isinstance(vector_name, str):
                raise TypeError(f"Legacy intervention vector key must be a string, received {type(vector_name)}")
            if vector_name not in prepared_vectors:
                raise KeyError(f"Legacy intervention references unknown vector '{vector_name}'")
            layer_idx, unit_vector, magnitude = prepared_vectors[vector_name]

            projection_cap = None
            if "cap" in intervention and intervention["cap"] is not None:
                cap_value = float(intervention["cap"])
                projection_cap = ProjectionCapSpec(
                    vector=unit_vector.clone(),
                    min=None,
                    max=cap_value,
                )
            else:
                raise ValueError(f"Legacy intervention in experiment '{experiment_id}' missing 'cap' field")

            if layer_idx in layers:
                raise ValueError(
                    f"Legacy experiment '{experiment_id}' defines multiple interventions for layer {layer_idx}"
                )
            layers[layer_idx] = LayerSteeringSpec(operations=[projection_cap])

        experiments.append(LegacyExperiment(id=experiment_id, spec=SteeringSpec(layers=layers)))

    return experiments
