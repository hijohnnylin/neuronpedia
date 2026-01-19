"""Shared persona dataset loading and filtering utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import PreTrainedTokenizerBase

from ..constants import PERSONA_ROOT, PROCESSED_PERSONA_ROOT

PersonaDatasetType = Literal["role", "trait"]

DEFAULT_PERSONA_DATA_ROOT = PERSONA_ROOT
DEFAULT_PROCESSED_PERSONA_ROOT = PROCESSED_PERSONA_ROOT

_DATASET_SUBDIR = {
    "role": "roles_240",
    "trait": "traits_240",
}
_RESPONSES_SUBDIR = "responses"
_SCORES_SUBDIR = "extract_scores"


@dataclass(frozen=True)
class PersonaDatasetSpec:
    """Descriptor for a persona dataset (role or trait) tied to a specific model."""

    model: str
    dataset_type: PersonaDatasetType
    name: str

    def dataset_dir(self, root: Path) -> Path:
        """Return the directory containing raw persona files."""
        return root / self.model / _DATASET_SUBDIR[self.dataset_type]

    def responses_dir(self, root: Path) -> Path:
        return self.dataset_dir(root) / _RESPONSES_SUBDIR

    def scores_dir(self, root: Path) -> Path:
        return self.dataset_dir(root) / _SCORES_SUBDIR

    @property
    def label_column(self) -> str:
        return "role" if self.dataset_type == "role" else "trait"


def _score_key(record: dict[str, object]) -> str:
    return f"{record['label']}_p{record['prompt_index']}_q{record['question_index']}"


def list_available(
    model: str,
    dataset_type: PersonaDatasetType,
    persona_data_dir: Path = DEFAULT_PERSONA_DATA_ROOT,
) -> list[str]:
    """List persona dataset names available for a given model and type."""
    spec = PersonaDatasetSpec(model=model, dataset_type=dataset_type, name="")
    responses_dir = spec.responses_dir(persona_data_dir)
    if not responses_dir.exists():
        return []
    return sorted(path.stem for path in responses_dir.glob("*.jsonl"))


def list_available_roles(model: str, persona_data_dir: Path = DEFAULT_PERSONA_DATA_ROOT) -> list[str]:
    """List role datasets for the supplied model."""
    return list_available(model=model, dataset_type="role", persona_data_dir=persona_data_dir)


def list_available_traits(model: str, persona_data_dir: Path = DEFAULT_PERSONA_DATA_ROOT) -> list[str]:
    """List trait datasets for the supplied model."""
    return list_available(model=model, dataset_type="trait", persona_data_dir=persona_data_dir)


def read_persona_name_file(path: Path) -> list[str]:
    """Load persona names from a newline-delimited file, ignoring comments."""
    if not path.exists():
        return []
    entries: list[str] = []
    for line in path.read_text().splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        entries.append(value)
    return entries


def build_prefixed_datasets(names: Iterable[str], prefix: str) -> list[str]:
    """Apply a dataset prefix (e.g. qwen-3-32b__trait__) to persona names."""
    return [f"{prefix}{name}" for name in names]


def resolve_persona_datasets(
    *,
    explicit: Sequence[str] | None = None,
    traits_file: Path | None = None,
    roles_file: Path | None = None,
    trait_prefix: str = "qwen-3-32b__trait__",
    role_prefix: str = "qwen-3-32b__role__",
    include_traits: bool = True,
    include_roles: bool = False,
) -> list[str]:
    """Resolve persona dataset names from explicit overrides or trait/role lists."""
    if explicit:
        return list(explicit)

    datasets: list[str] = []
    if include_traits and traits_file is not None:
        names = read_persona_name_file(traits_file)
        datasets.extend(build_prefixed_datasets(names, trait_prefix))
    if include_roles and roles_file is not None:
        names = read_persona_name_file(roles_file)
        datasets.extend(build_prefixed_datasets(names, role_prefix))
    return datasets


def parse_processed_dataset_spec(selection: str) -> Tuple[str, Dict[str, str]]:
    """Parse a processed persona dataset selection with optional column filters.

    Examples
    --------
    "qwen-3-32b__role__accountant"
        -> ("qwen-3-32b__role__accountant", {})
    "qwen-3-32b__trait__analytical:label=pos"
        -> ("qwen-3-32b__trait__analytical", {"label": "pos"})
    """
    dataset_name, _, remainder = selection.partition(":")
    dataset_name = dataset_name.strip()
    filters: Dict[str, str] = {}
    remainder = remainder.strip()
    if not remainder:
        return dataset_name, filters

    for part in remainder.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)
        else:
            key, value = "role", part  # legacy shorthand
        filters[key.strip()] = value.strip()

    return dataset_name, filters


def _load_scores(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    with path.open() as handle:
        payload = json.load(handle)
    return {str(key): int(value) for key, value in payload.items()}


def _iter_persona_records(
    spec: PersonaDatasetSpec,
    *,
    persona_data_dir: Path,
    min_score: Optional[int],
    label_filter: Optional[str],
) -> Iterable[dict[str, object]]:
    responses_path = spec.responses_dir(persona_data_dir) / f"{spec.name}.jsonl"
    if not responses_path.exists():
        raise FileNotFoundError(f"Persona dataset not found: {responses_path}")

    scores = _load_scores(spec.scores_dir(persona_data_dir) / f"{spec.name}.json")
    label_column = spec.label_column

    with responses_path.open() as handle:
        for line in handle:
            record = json.loads(line)

            if label_filter is not None and record.get("label") != label_filter:
                continue

            score_value = scores.get(_score_key(record))
            if min_score is not None:
                # Some baseline/default datasets lack scores (None). Treat them as failing the filter.
                if score_value is None or score_value < min_score:
                    continue

            messages = record.get("conversation")
            if not messages:
                continue

            item = {
                "model": spec.model,
                label_column: spec.name,
                "dataset_type": spec.dataset_type,
                "system_prompt": record.get("system_prompt"),
                "label": record.get("label"),
                "prompt_index": record.get("prompt_index"),
                "question_index": record.get("question_index"),
                "question": record.get("question"),
                "messages": messages,
                "extract_score": score_value,
            }
            yield item


def load_persona_dataset(
    model: str,
    dataset_type: PersonaDatasetType,
    name: str,
    *,
    persona_data_dir: Path = DEFAULT_PERSONA_DATA_ROOT,
    min_score: Optional[int] = None,
    label_filter: Optional[str] = None,
) -> Dataset:
    """Build a Hugging Face dataset for persona records stored in JSONL form."""
    spec = PersonaDatasetSpec(model=model, dataset_type=dataset_type, name=name)
    records = list(
        _iter_persona_records(
            spec,
            persona_data_dir=persona_data_dir,
            min_score=min_score,
            label_filter=label_filter,
        )
    )
    if not records:
        raise ValueError(
            f"No persona records found for model={model}, type={dataset_type}, name={name} "
            f"(min_score={min_score}, label_filter={label_filter})"
        )
    return Dataset.from_list(records)


def save_persona_dataset(dataset: Dataset, output_dir: Path, name: str) -> None:
    """Persist a persona dataset as Hugging Face disk format plus optional Parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dict = DatasetDict({"train": dataset})
    dataset_path = output_dir / name
    dataset_dict.save_to_disk(str(dataset_path))

    parquet_path = output_dir / f"{name}.parquet"
    dataset.to_parquet(str(parquet_path))


def load_processed_persona_dataset(
    name: str,
    *,
    processed_root: Path = DEFAULT_PROCESSED_PERSONA_ROOT,
    columns: Optional[Sequence[str]] = None,
) -> Dataset:
    """Load a previously materialised persona dataset from disk or parquet."""
    dataset_path = processed_root / name
    if dataset_path.exists():
        dataset = load_from_disk(str(dataset_path))["train"]
    else:
        parquet_path = dataset_path.with_suffix(".parquet")
        if not parquet_path.exists():
            raise FileNotFoundError(f"Processed persona dataset not found: {dataset_path}")
        dataset = load_dataset("parquet", data_files=str(parquet_path))["train"]

    if columns is not None:
        missing = [col for col in columns if col not in dataset.column_names]
        if missing:
            raise ValueError(f"Requested columns not present in dataset {name}: {missing}")
        keep_set = set(columns)
        drop_columns = [col for col in dataset.column_names if col not in keep_set]
        if drop_columns:
            dataset = dataset.remove_columns(drop_columns)

    return dataset


def filter_persona_dataset(
    dataset: Dataset,
    *,
    min_score: Optional[int] = None,
    max_score: Optional[int] = None,
    labels: Optional[Sequence[str]] = None,
    role: Optional[str] = None,
    trait: Optional[str] = None,
    include_missing_scores: bool = False,
) -> Dataset:
    """Filter a persona dataset along common columns such as score, labels, role, or trait."""
    label_set = set(labels) if labels is not None else None

    def _predicate(row: dict[str, object]) -> bool:
        extract_score = row.get("extract_score")
        if extract_score is None:
            if not include_missing_scores and (min_score is not None or max_score is not None):
                return False
        else:
            try:
                value = int(extract_score)
            except (TypeError, ValueError):
                return False
            if min_score is not None and value < min_score:
                return False
            if max_score is not None and value > max_score:
                return False

        if label_set is not None:
            if row.get("label") not in label_set:
                return False

        if role is not None and row.get("role") != role:
            return False
        if trait is not None and row.get("trait") != trait:
            return False

        return True

    return dataset.filter(_predicate)


def _load_filtered_processed_dataset(
    dataset_name: str,
    filters: Mapping[str, str],
    *,
    processed_root: Path,
    role_min_score: int,
    trait_min_score: int,
    trait_positive_only: bool,
    include_missing_scores: bool,
) -> Dataset:
    dataset = load_processed_persona_dataset(dataset_name, processed_root=processed_root)
    columns = set(dataset.column_names)
    is_role_dataset = "role" in columns
    is_trait_dataset = "trait" in columns

    label_filter = filters.get("label")
    labels: Optional[Sequence[str]] = None
    if label_filter is not None:
        labels = [label_filter]
    elif is_trait_dataset and trait_positive_only:
        labels = ["pos"]

    filter_kwargs: dict[str, object] = {
        "labels": labels,
        "include_missing_scores": include_missing_scores,
    }

    if is_role_dataset:
        filter_kwargs["min_score"] = role_min_score
        if "role" in filters:
            filter_kwargs["role"] = filters["role"]
    elif is_trait_dataset:
        filter_kwargs["min_score"] = trait_min_score
        if "trait" in filters:
            filter_kwargs["trait"] = filters["trait"]
    else:
        filter_kwargs["min_score"] = None

    filtered = filter_persona_dataset(dataset, **filter_kwargs)
    if len(filtered) == 0:
        raise ValueError(f"No rows remain after filtering dataset '{dataset_name}' with {filters or 'default filters'}")
    return filtered


def _strip_system_messages(messages: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [dict(m) for m in messages if m.get("role") != "system"]


def _tokenize_length(
    messages: Sequence[Mapping[str, object]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> int:
    chat_text = tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=False,
    )
    tokenized = tokenizer(
        chat_text,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
    )
    return int(len(tokenized["input_ids"]))


@dataclass(frozen=True)
class TokenBudgetRequest:
    """Token budgets for persona dataset splits."""

    train_tokens: int
    val_tokens: int = 0
    test_tokens: int = 0

    def as_pairs(self) -> Tuple[Tuple[str, int], ...]:
        return (
            ("train", int(self.train_tokens)),
            ("val", int(self.val_tokens)),
            ("test", int(self.test_tokens)),
        )


@dataclass(frozen=True)
class TokenBudgetResult:
    """Result of token budget sampling across persona datasets."""

    splits: Mapping[str, Dataset]
    token_counts: Mapping[str, int]
    total_candidates: int
    remaining_candidates: int
    remaining_tokens: int


def _collect_tokenized_records(
    selections: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    *,
    processed_root: Path,
    max_length: int,
    seed: int,
    role_min_score: int,
    trait_min_score: int,
    trait_positive_only: bool,
    include_missing_scores: bool,
    drop_system_messages: bool,
) -> list[dict[str, object]]:
    if not selections:
        raise ValueError("No persona datasets provided.")

    records: list[dict[str, object]] = []
    for selection in selections:
        dataset_name, filters = parse_processed_dataset_spec(selection)
        dataset = _load_filtered_processed_dataset(
            dataset_name,
            filters,
            processed_root=processed_root,
            role_min_score=role_min_score,
            trait_min_score=trait_min_score,
            trait_positive_only=trait_positive_only,
            include_missing_scores=include_missing_scores,
        )

        for row in dataset:
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                continue
            if drop_system_messages:
                messages = _strip_system_messages(messages)
            if not messages:
                continue
            length = _tokenize_length(messages, tokenizer, max_length=max_length)
            record = dict(row)
            record["messages"] = messages
            record["length"] = length
            record["source_dataset"] = dataset_name
            record["dataset_selection"] = selection
            records.append(record)

    if not records:
        raise ValueError("No persona records satisfied the provided filters and tokenization constraints.")

    rng = np.random.default_rng(seed)
    rng.shuffle(records)
    return records


def build_token_budget_splits(
    selections: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    *,
    budget: TokenBudgetRequest,
    processed_root: Path = DEFAULT_PROCESSED_PERSONA_ROOT,
    max_length: int = 4096,
    seed: int = 17,
    role_min_score: int = 3,
    trait_min_score: int = 75,
    trait_positive_only: bool = True,
    include_missing_scores: bool = False,
    drop_system_messages: bool = True,
) -> TokenBudgetResult:
    """Tokenise persona conversations and allocate them into deterministic token-budget splits."""
    records = _collect_tokenized_records(
        selections,
        tokenizer,
        processed_root=processed_root,
        max_length=max_length,
        seed=seed,
        role_min_score=role_min_score,
        trait_min_score=trait_min_score,
        trait_positive_only=trait_positive_only,
        include_missing_scores=include_missing_scores,
        drop_system_messages=drop_system_messages,
    )

    lengths = [int(record["length"]) for record in records]
    full_dataset = Dataset.from_list(records)

    splits: dict[str, Dataset] = {}
    token_counts: dict[str, int] = {}
    cursor = 0

    for split_name, target_tokens in budget.as_pairs():
        if target_tokens <= 0:
            splits[split_name] = full_dataset.select([])
            token_counts[split_name] = 0
            continue

        start = cursor
        consumed = 0
        while cursor < len(lengths) and consumed < target_tokens:
            consumed += lengths[cursor]
            cursor += 1

        if cursor == start:
            raise ValueError(
                f"Insufficient persona data to satisfy the {split_name!r} token budget "
                f"({target_tokens} tokens requested, dataset exhausted)."
            )

        indices = list(range(start, cursor))
        splits[split_name] = full_dataset.select(indices)
        token_counts[split_name] = consumed

    leftover_indices = list(range(cursor, len(lengths)))
    remaining_tokens = int(sum(lengths[index] for index in leftover_indices))

    return TokenBudgetResult(
        splits=splits,
        token_counts=token_counts,
        total_candidates=len(lengths),
        remaining_candidates=len(leftover_indices),
        remaining_tokens=remaining_tokens,
    )
