"""Steering vector utilities with optional training helpers."""

from __future__ import annotations

from .activations import load_activation_vector
from .constants import ERROR_FILENAME, SUMMARY_FILENAME
from .features import extract_layer_hidden_states
from .runs import collect_run_dirs, has_successful_run, latest_run_dir, list_trained_datasets

__all__ = [
    "ERROR_FILENAME",
    "SUMMARY_FILENAME",
    "collect_run_dirs",
    "has_successful_run",
    "extract_layer_hidden_states",
    "load_activation_vector",
    "latest_run_dir",
    "list_trained_datasets",
]

try:
    from .data import PersonaSteeringDatasetConfig, load_persona_steering_dataset, prepare_persona_token_budget
    from .model import QwenSteerModel, SteeringVectorConfig, TransformerSteerModel
except ImportError:
    pass
else:
    __all__.extend(
        [
            "PersonaSteeringDatasetConfig",
            "load_persona_steering_dataset",
            "prepare_persona_token_budget",
            "SteeringVectorConfig",
            "QwenSteerModel",
            "TransformerSteerModel",
        ]
    )
