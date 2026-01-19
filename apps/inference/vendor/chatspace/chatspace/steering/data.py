"""Persona dataset utilities for steering vector training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from ..persona import (
    DEFAULT_PROCESSED_PERSONA_ROOT,
    TokenBudgetRequest,
    TokenBudgetResult,
    build_token_budget_splits,
)


@dataclass
class PersonaSteeringDatasetConfig:
    """Configuration for assembling persona datasets for steering training."""

    dataset_names: Sequence[str]
    dataset_root: Path = DEFAULT_PROCESSED_PERSONA_ROOT
    train_tokens: int = 100_000
    val_tokens: int = 0
    test_tokens: int = 0
    seed: int = 17
    tokenizer_name: str = "Qwen/Qwen3-32B"
    max_length: int = 4096
    role_min_score: int = 3
    trait_min_score: int = 75
    trait_positive_only: bool = True
    include_missing_scores: bool = False
    drop_system_messages: bool = True


def prepare_persona_token_budget(
    cfg: PersonaSteeringDatasetConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> TokenBudgetResult:
    """Prepare deterministic token-budget splits for steering datasets."""
    request = TokenBudgetRequest(
        train_tokens=cfg.train_tokens,
        val_tokens=cfg.val_tokens,
        test_tokens=cfg.test_tokens,
    )
    return build_token_budget_splits(
        cfg.dataset_names,
        tokenizer,
        budget=request,
        processed_root=cfg.dataset_root,
        max_length=cfg.max_length,
        seed=cfg.seed,
        role_min_score=cfg.role_min_score,
        trait_min_score=cfg.trait_min_score,
        trait_positive_only=cfg.trait_positive_only,
        include_missing_scores=cfg.include_missing_scores,
        drop_system_messages=cfg.drop_system_messages,
    )


def load_persona_steering_dataset(
    cfg: PersonaSteeringDatasetConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:
    """Retain the legacy API that returns only the training split."""
    result = prepare_persona_token_budget(cfg, tokenizer)
    return result.splits["train"]
