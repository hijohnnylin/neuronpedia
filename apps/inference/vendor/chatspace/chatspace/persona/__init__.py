"""
Persona dataset utilities shared across training, evaluation, and CLI scripts.

This module centralises common helpers for:
- Discovering available persona datasets (roles and traits) from the raw dumps
- Building Hugging Face datasets from the JSONL sources with score/label filters
- Loading processed datasets saved under `/workspace/datasets/processed/persona`
- Applying lightweight column-based filters for classifier/evaluation workflows
"""

from .dataset import (
    DEFAULT_PERSONA_DATA_ROOT,
    DEFAULT_PROCESSED_PERSONA_ROOT,
    PersonaDatasetSpec,
    PersonaDatasetType,
    TokenBudgetRequest,
    TokenBudgetResult,
    build_prefixed_datasets,
    build_token_budget_splits,
    filter_persona_dataset,
    list_available,
    list_available_roles,
    list_available_traits,
    load_persona_dataset,
    load_processed_persona_dataset,
    parse_processed_dataset_spec,
    read_persona_name_file,
    resolve_persona_datasets,
    save_persona_dataset,
)

__all__ = [
    "DEFAULT_PERSONA_DATA_ROOT",
    "DEFAULT_PROCESSED_PERSONA_ROOT",
    "PersonaDatasetSpec",
    "PersonaDatasetType",
    "TokenBudgetRequest",
    "TokenBudgetResult",
    "build_prefixed_datasets",
    "build_token_budget_splits",
    "filter_persona_dataset",
    "list_available",
    "list_available_roles",
    "list_available_traits",
    "load_persona_dataset",
    "load_processed_persona_dataset",
    "parse_processed_dataset_spec",
    "read_persona_name_file",
    "resolve_persona_datasets",
    "save_persona_dataset",
]
