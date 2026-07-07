from __future__ import annotations

import argparse
import gzip
import importlib
import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

DEFAULT_COLUMNAR_IMPORT_TABLES = frozenset({"Activation"})
DEFAULT_COLUMNAR_COPY_IMPORT_TABLES = frozenset({"Activation"})
SAEDASHBOARD_COLUMNAR_NEURON_TABLES = frozenset(
    {
        "activation_histograms",
        "feature_statistics",
        "feature_tables",
        "logits_histograms",
        "logits_tables",
    }
)
SAEDASHBOARD_COLUMNAR_ACTIVATION_TABLE_PRIORITY = (
    "activation_copy_rows",
    "activation_rows",
    "sequence_rows",
)
SAEDASHBOARD_COLUMNAR_ACTIVATION_ROW_COLUMNS = frozenset(
    {
        "feature_index",
        "sequence_index",
        "tokens",
        "max_value",
        "max_value_token_index",
        "min_value",
        "values",
        "bin_min",
        "bin_max",
        "bin_contains",
        "qualifying_token_index",
    }
)
SAEDASHBOARD_COLUMNAR_ACTIVATION_COPY_COLUMNS = [
    "id",
    "tokens",
    "dataIndex",
    "index",
    "layer",
    "modelId",
    "dataSource",
    "maxValue",
    "maxValueTokenIndex",
    "minValue",
    "values",
    "dfaValues",
    "dfaTargetIndex",
    "dfaMaxValue",
    "creatorId",
    "createdAt",
    "lossValues",
    "logitContributions",
    "binMin",
    "binMax",
    "binContains",
    "qualifyingTokenIndex",
]
DEFAULT_IMPORT_MODE_CONFIGS: dict[str, dict[str, Any]] = {
    "jsonl": {
        "prefer_arrow_for_tables": (),
        "prefer_copy_for_tables": (),
    },
    "activation_arrow_jsonb": {
        "prefer_arrow_for_tables": ("Activation",),
        "prefer_copy_for_tables": (),
        "artifact_format_by_table": {"Activation": "arrow"},
    },
    "activation_parquet_jsonb": {
        "prefer_arrow_for_tables": ("Activation",),
        "prefer_copy_for_tables": (),
        "artifact_format_by_table": {"Activation": "parquet"},
    },
    "activation_arrow_copy": {
        "prefer_arrow_for_tables": ("Activation",),
        "prefer_copy_for_tables": ("Activation",),
        "artifact_format_by_table": {"Activation": "arrow"},
    },
    "activation_arrow_copy_direct": {
        "prefer_arrow_for_tables": ("Activation",),
        "prefer_copy_for_tables": ("Activation",),
        "artifact_format_by_table": {"Activation": "arrow"},
        "copy_direct_for_tables": ("Activation",),
    },
    "activation_parquet_copy": {
        "prefer_arrow_for_tables": ("Activation",),
        "prefer_copy_for_tables": ("Activation",),
        "artifact_format_by_table": {"Activation": "parquet"},
    },
    "activation_parquet_copy_direct": {
        "prefer_arrow_for_tables": ("Activation",),
        "prefer_copy_for_tables": ("Activation",),
        "artifact_format_by_table": {"Activation": "parquet"},
        "copy_direct_for_tables": ("Activation",),
    },
}

_METADATA_TABLE_FILE_STEMS = (
    ("SourceRelease", "release"),
    ("Model", "model"),
    ("SourceSet", "sourceset"),
    ("Source", "source"),
)
_SHARDED_TABLE_DIRECTORIES = (
    ("features", "Neuron"),
    ("activations", "Activation"),
    ("explanations", "Explanation"),
)


class NeuronpediaLocalDBImportError(RuntimeError):
    """Raised when Neuronpedia local DB import or bundle inspection fails."""


@dataclass(frozen=True)
class NeuronpediaExportBundleSummary:
    """Row-count summary for a Neuronpedia export bundle or columnar sidecars."""

    export_root: Path
    table_row_counts: dict[str, int]
    processed_files: list[Path]
    elapsed_seconds: float
    table_load_seconds: dict[str, float]


@dataclass(frozen=True)
class NeuronpediaBundleSummaryParity:
    """Row-count parity summary between two export bundle views."""

    export_root: Path
    reference_row_counts: dict[str, int]
    candidate_row_counts: dict[str, int]
    row_count_mismatches: dict[str, dict[str, int]]


@dataclass(frozen=True)
class NeuronpediaLocalImportSummary:
    """Summary of one local Neuronpedia export-bundle import run.

    ``deduplicated_row_counts`` and ``zero_dropped_row_counts`` track rows removed by
    the opt-in ``dedup_activation_rows`` / ``drop_zero_activation_rows`` import-hygiene
    flags; both stay empty when the flags are disabled (the default), and
    ``bundle_row_counts`` always reflects post-filter counts.
    """

    export_root: Path
    bundle_row_counts: dict[str, int]
    imported_row_counts: dict[str, int]
    processed_files: list[Path]
    elapsed_seconds: float
    table_load_seconds: dict[str, float]
    table_import_seconds: dict[str, float]
    table_import_substage_seconds: dict[str, dict[str, float]] = field(
        default_factory=dict
    )
    deduplicated_row_counts: dict[str, int] = field(default_factory=dict)
    zero_dropped_row_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class NeuronpediaBundleImportParity:
    """Parity summary between an export bundle and one local import attempt."""

    export_root: Path
    bundle_row_counts: dict[str, int]
    import_attempted_row_counts: dict[str, int]
    imported_row_counts: dict[str, int]
    row_count_mismatches: dict[str, dict[str, int]]
    skipped_existing_row_counts: dict[str, int]


@dataclass(frozen=True)
class NeuronpediaLocalDBTableSampleSpec:
    """Stable local DB table sample definition for eager/lazy parity checks."""

    table_name: str
    sample_columns: tuple[str, ...]
    order_by: tuple[str, ...]


@dataclass(frozen=True)
class NeuronpediaLocalDBSnapshot:
    """Stable row-count and row-sample snapshot from one imported local DB target."""

    label: str
    table_row_counts: dict[str, int]
    table_samples: dict[str, list[dict[str, Any]]]
    elapsed_seconds: float


@dataclass(frozen=True)
class NeuronpediaLocalDBParity:
    """Parity summary between two imported local DB targets."""

    reference_label: str
    candidate_label: str
    reference_row_counts: dict[str, int]
    candidate_row_counts: dict[str, int]
    row_count_mismatches: dict[str, dict[str, int]]
    sample_mismatches: dict[str, dict[str, list[dict[str, Any]]]]


@dataclass(frozen=True)
class NeuronpediaBundleArtifact:
    """Importable export-bundle artifact and its target database table."""

    table_name: str
    path: Path
    format: str


DEFAULT_LOCAL_DB_PARITY_TABLE_SAMPLE_SPECS = (
    NeuronpediaLocalDBTableSampleSpec(
        table_name="SourceRelease",
        sample_columns=("name", "defaultSourceSetName", "createdAt"),
        order_by=("name",),
    ),
    NeuronpediaLocalDBTableSampleSpec(
        table_name="Model",
        sample_columns=(
            "id",
            "defaultSourceSetName",
            "defaultGraphSourceSetName",
            "defaultSourceId",
            "layers",
        ),
        order_by=("id",),
    ),
    NeuronpediaLocalDBTableSampleSpec(
        table_name="SourceSet",
        sample_columns=("modelId", "name", "releaseName", "defaultOfModelId"),
        order_by=("modelId", "name"),
    ),
    NeuronpediaLocalDBTableSampleSpec(
        table_name="Source",
        sample_columns=(
            "id",
            "modelId",
            "setName",
            "hfRepoId",
            "hfFolderId",
            "num_prompts",
            "num_tokens_in_prompt",
        ),
        order_by=("modelId", "setName", "id"),
    ),
    NeuronpediaLocalDBTableSampleSpec(
        table_name="Neuron",
        sample_columns=(
            "modelId",
            "layer",
            "index",
            "sourceSetName",
            "maxActApprox",
            "frac_nonzero",
            "freq_hist_data_bar_heights",
            "freq_hist_data_bar_values",
            "logits_hist_data_bar_heights",
            "logits_hist_data_bar_values",
            "pos_str",
            "pos_values",
            "neg_str",
            "neg_values",
        ),
        order_by=("modelId", "layer", "index"),
    ),
    NeuronpediaLocalDBTableSampleSpec(
        table_name="Activation",
        sample_columns=(
            "id",
            "modelId",
            "layer",
            "index",
            "tokens",
            "values",
            "maxValue",
            "maxValueTokenIndex",
            "binMin",
            "binMax",
            "binContains",
            "qualifyingTokenIndex",
        ),
        order_by=("modelId", "layer", "index", "id"),
    ),
)


DEFERRED_MODEL_FK_COLUMNS = (
    "defaultSourceSetName",
    "defaultGraphSourceSetName",
    "defaultSourceId",
)


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _count_jsonl_records(path: Path) -> int:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _load_pyarrow_modules() -> tuple[Any, Any, Any]:
    try:
        pyarrow = importlib.import_module("pyarrow")
        pyarrow_compute = importlib.import_module("pyarrow.compute")
        pyarrow_ipc = importlib.import_module("pyarrow.ipc")
    except (
        ImportError
    ) as exc:  # pragma: no cover - exercised when Arrow support is unavailable
        raise NeuronpediaLocalDBImportError(
            "Arrow bundle support requires pyarrow to be installed."
        ) from exc
    return pyarrow, pyarrow_compute, pyarrow_ipc


def _load_pyarrow_ipc() -> Any:
    try:
        return importlib.import_module("pyarrow.ipc")
    except (
        ImportError
    ) as exc:  # pragma: no cover - exercised when Arrow support is unavailable
        raise NeuronpediaLocalDBImportError(
            "Arrow bundle support requires pyarrow to be installed."
        ) from exc


def _load_pyarrow_parquet() -> Any:
    try:
        return importlib.import_module("pyarrow.parquet")
    except (
        ImportError
    ) as exc:  # pragma: no cover - exercised when Parquet support is unavailable
        raise NeuronpediaLocalDBImportError(
            "Parquet bundle support requires pyarrow to be installed."
        ) from exc


def _load_pgpq_arrow_encoder() -> Any:
    try:
        return importlib.import_module("pgpq").ArrowToPostgresBinaryEncoder
    except ImportError as exc:  # pragma: no cover - exercised when pgpq is unavailable
        raise NeuronpediaLocalDBImportError(
            "Binary Arrow COPY support requires pgpq to be installed for the current environment."
        ) from exc


def count_arrow_records(path: Path) -> int:
    pa_ipc = _load_pyarrow_ipc()

    with path.open("rb") as handle:
        reader = pa_ipc.RecordBatchFileReader(handle)
        return sum(
            reader.get_batch(batch_idx).num_rows
            for batch_idx in range(reader.num_record_batches)
        )


def count_parquet_records(path: Path) -> int:
    pyarrow_parquet = _load_pyarrow_parquet()

    return int(pyarrow_parquet.read_metadata(str(path)).num_rows)


def _metadata_bundle_path(export_root: Path, file_stem: str) -> Path | None:
    for suffix in (".jsonl", ".jsonl.gz"):
        candidate = export_root / f"{file_stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def arrow_sidecar_path(path: Path) -> Path:
    if path.name.endswith(".jsonl.gz"):
        return path.with_suffix("").with_suffix(".arrow")
    return path.with_suffix(".arrow")


def parquet_sidecar_path(path: Path) -> Path:
    if path.name.endswith(".jsonl.gz"):
        return path.with_suffix("").with_suffix(".parquet")
    return path.with_suffix(".parquet")


def bundle_artifact_logical_path(path: Path) -> Path:
    if path.name.endswith(".jsonl.gz"):
        return path.with_suffix("").with_suffix("")
    if path.suffix in {".arrow", ".parquet"}:
        return path.with_suffix("")
    return path


def export_bundle_jsonl_paths(export_root: Path | str) -> list[tuple[str, Path]]:
    export_root_path = Path(export_root)
    table_paths: list[tuple[str, Path]] = []
    for table_name, file_stem in _METADATA_TABLE_FILE_STEMS:
        path = _metadata_bundle_path(export_root_path, file_stem)
        if path is not None:
            table_paths.append((table_name, path))
    for dir_name, table_name in _SHARDED_TABLE_DIRECTORIES:
        directory = export_root_path / dir_name
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.jsonl.gz")):
            table_paths.append((table_name, path))
    return table_paths


def export_bundle_arrow_paths(export_root: Path | str) -> list[tuple[str, Path]]:
    export_root_path = Path(export_root)
    table_paths: list[tuple[str, Path]] = []
    for table_name, file_stem in _METADATA_TABLE_FILE_STEMS:
        path = export_root_path / f"{file_stem}.arrow"
        if path.exists():
            table_paths.append((table_name, path))
    for dir_name, table_name in _SHARDED_TABLE_DIRECTORIES:
        directory = export_root_path / dir_name
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.arrow")):
            table_paths.append((table_name, path))
    return table_paths


def export_bundle_parquet_paths(export_root: Path | str) -> list[tuple[str, Path]]:
    export_root_path = Path(export_root)
    table_paths: list[tuple[str, Path]] = []
    for table_name, file_stem in _METADATA_TABLE_FILE_STEMS:
        path = export_root_path / f"{file_stem}.parquet"
        if path.exists():
            table_paths.append((table_name, path))
    for dir_name, table_name in _SHARDED_TABLE_DIRECTORIES:
        directory = export_root_path / dir_name
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.parquet")):
            table_paths.append((table_name, path))
    return table_paths


def discover_export_bundle_import_artifacts(
    export_root: Path | str,
    *,
    prefer_arrow_for_tables: Iterable[str] = (),
    prefer_copy_for_tables: Iterable[str] = (),
    artifact_format_by_table: dict[str, str] | None = None,
) -> list[NeuronpediaBundleArtifact]:
    export_root_path = Path(export_root)
    preferred_tables = (
        set(prefer_arrow_for_tables) | set(prefer_copy_for_tables)
    ) & set(DEFAULT_COLUMNAR_IMPORT_TABLES)
    forced_formats = artifact_format_by_table or {}
    artifacts: list[NeuronpediaBundleArtifact] = []
    for table_name, path in export_bundle_jsonl_paths(export_root_path):
        if table_name in preferred_tables:
            forced_format = forced_formats.get(table_name)
            candidate_paths = (
                ((forced_format, arrow_sidecar_path(path)),)
                if forced_format == "arrow"
                else ((forced_format, parquet_sidecar_path(path)),)
                if forced_format == "parquet"
                else (
                    ("parquet", parquet_sidecar_path(path)),
                    ("arrow", arrow_sidecar_path(path)),
                )
            )
            for candidate_format, candidate_path in candidate_paths:
                if candidate_path.exists():
                    artifacts.append(
                        NeuronpediaBundleArtifact(
                            table_name=table_name,
                            path=candidate_path,
                            format=cast(str, candidate_format),
                        )
                    )
                    break
            else:
                if forced_format is not None:
                    raise NeuronpediaLocalDBImportError(
                        f"Requested {forced_format} artifact for {table_name}, but no sidecar exists for {path}."
                    )
                artifacts.append(
                    NeuronpediaBundleArtifact(
                        table_name=table_name, path=path, format="jsonl"
                    )
                )
                continue
            continue
        artifacts.append(
            NeuronpediaBundleArtifact(table_name=table_name, path=path, format="jsonl")
        )
    discovered_paths = {artifact.path for artifact in artifacts}
    discovered_shards = {
        (artifact.table_name, bundle_artifact_logical_path(artifact.path))
        for artifact in artifacts
    }
    if preferred_tables:
        for table_name, path in export_bundle_parquet_paths(export_root_path):
            shard_key = (table_name, bundle_artifact_logical_path(path))
            if (
                table_name in preferred_tables
                and path not in discovered_paths
                and shard_key not in discovered_shards
            ):
                artifacts.append(
                    NeuronpediaBundleArtifact(
                        table_name=table_name, path=path, format="parquet"
                    )
                )
                discovered_paths.add(path)
                discovered_shards.add(shard_key)
        for table_name, path in export_bundle_arrow_paths(export_root_path):
            shard_key = (table_name, bundle_artifact_logical_path(path))
            if (
                table_name in preferred_tables
                and path not in discovered_paths
                and shard_key not in discovered_shards
            ):
                artifacts.append(
                    NeuronpediaBundleArtifact(
                        table_name=table_name, path=path, format="arrow"
                    )
                )
                discovered_paths.add(path)
                discovered_shards.add(shard_key)
    return artifacts


def _load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise NeuronpediaLocalDBImportError(f"Expected JSON object in {path}.")
    return payload


def discover_saedashboard_columnar_manifest_paths(
    columnar_root: Path | str,
) -> list[Path]:
    """Discover feature-batch manifests under a SAEDashboard columnar artifact root."""

    columnar_root_path = Path(columnar_root)
    if not columnar_root_path.exists():
        raise NeuronpediaLocalDBImportError(
            f"Columnar artifact root does not exist: {columnar_root_path}"
        )

    manifest_paths: set[Path] = set()
    root_manifest_path = columnar_root_path / "manifest.json"
    if root_manifest_path.exists():
        root_manifest = _load_json_file(root_manifest_path)
        if "row_counts" in root_manifest and "tables" in root_manifest:
            manifest_paths.add(root_manifest_path)
        for batch in root_manifest.get(
            "batches", ()
        ):  # root batch manifests point to feature_batch_* dirs
            if isinstance(batch, dict) and (artifact_dir := batch.get("artifact_dir")):
                candidate = columnar_root_path / str(artifact_dir) / "manifest.json"
                if candidate.exists():
                    manifest_paths.add(candidate)

    for candidate in columnar_root_path.rglob("manifest.json"):
        if candidate.parent.name.startswith("feature_batch_"):
            manifest_paths.add(candidate)

    if not manifest_paths:
        raise NeuronpediaLocalDBImportError(
            f"No SAEDashboard columnar feature-batch manifests found under {columnar_root_path}."
        )
    return sorted(manifest_paths)


def _count_columnar_table_records(path: Path) -> int:
    if path.suffix == ".arrow":
        return count_arrow_records(path)
    if path.suffix == ".parquet":
        return count_parquet_records(path)
    raise NeuronpediaLocalDBImportError(
        f"Unsupported SAEDashboard columnar table format for {path}; expected .arrow or .parquet."
    )


def _read_columnar_table(path: Path) -> Any:
    if path.suffix == ".arrow":
        pyarrow_ipc = _load_pyarrow_ipc()
        with path.open("rb") as handle:
            return pyarrow_ipc.RecordBatchFileReader(handle).read_all()
    if path.suffix == ".parquet":
        pyarrow_parquet = _load_pyarrow_parquet()
        return pyarrow_parquet.read_table(str(path))
    raise NeuronpediaLocalDBImportError(
        f"Unsupported SAEDashboard columnar table format for {path}; expected .arrow or .parquet."
    )


def _iter_columnar_record_batches(
    path: Path, *, batch_size: int = 65000, coalesce_arrow: bool = False
) -> Iterable[Any]:
    if path.suffix == ".arrow":
        pyarrow_ipc = _load_pyarrow_ipc()
        with path.open("rb") as handle:
            reader = pyarrow_ipc.RecordBatchFileReader(handle)
            if coalesce_arrow:
                yield from (
                    reader.read_all()
                    .combine_chunks()
                    .to_batches(max_chunksize=batch_size)
                )
                return
            for batch_idx in range(reader.num_record_batches):
                yield reader.get_batch(batch_idx)
        return
    if path.suffix == ".parquet":
        pyarrow_parquet = _load_pyarrow_parquet()
        parquet_file = pyarrow_parquet.ParquetFile(str(path))
        yield from parquet_file.iter_batches(batch_size=batch_size)
        return
    raise NeuronpediaLocalDBImportError(
        f"Unsupported SAEDashboard columnar table format for {path}; expected .arrow or .parquet."
    )


def _feature_indexed_rows(table_name: str, path: Path) -> dict[int, dict[str, Any]]:
    table = _read_columnar_table(path)
    if "feature_index" not in table.schema.names:
        raise NeuronpediaLocalDBImportError(
            f"Columnar table {table_name} at {path} must contain feature_index."
        )
    indexed_rows: dict[int, dict[str, Any]] = {}
    for row in cast(list[dict[str, Any]], table.to_pylist()):
        feature_index = int(row["feature_index"])
        if feature_index in indexed_rows:
            raise NeuronpediaLocalDBImportError(
                f"Columnar table {table_name} at {path} contains duplicate feature_index={feature_index}."
            )
        indexed_rows[feature_index] = row
    return indexed_rows


def _round_float_list(values: Any, *, precision: int = 3) -> list[float]:
    if values is None:
        return []
    return [round(float(value), precision) for value in values]


def _float_or_zero(value: Any) -> float:
    return 0.0 if value is None else float(value)


def _int_list(values: Any) -> list[int]:
    if values is None:
        return []
    return [int(value) for value in values]


def _decode_columnar_token_ids(
    decode_token_ids: Callable[[list[int]], list[str]], token_ids: Any
) -> list[str]:
    token_id_list = _int_list(token_ids)
    decoded_tokens = decode_token_ids(token_id_list)
    if len(decoded_tokens) != len(token_id_list):
        raise NeuronpediaLocalDBImportError(
            f"Token decoder returned {len(decoded_tokens)} tokens for {len(token_id_list)} ids."
        )
    return list(decoded_tokens)


def _created_at_value(created_at: datetime | str | None) -> str:
    if created_at is None:
        return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    if isinstance(created_at, datetime):
        if created_at.tzinfo is not None:
            created_at = created_at.astimezone(timezone.utc).replace(tzinfo=None)
        return created_at.isoformat()
    return created_at


def _required_columnar_table_paths(
    manifest_path: Path,
    required_tables: frozenset[str] = SAEDASHBOARD_COLUMNAR_NEURON_TABLES,
    *,
    logical_table_name: str = "Neuron",
) -> dict[str, Path]:
    manifest = _load_json_file(manifest_path)
    tables = manifest.get("tables")
    if not isinstance(tables, dict):
        raise NeuronpediaLocalDBImportError(
            f"Columnar manifest {manifest_path} must contain a tables object."
        )

    missing_tables = sorted(required_tables - set(tables))
    if missing_tables:
        raise NeuronpediaLocalDBImportError(
            f"Columnar manifest {manifest_path} is missing {logical_table_name} import tables: {missing_tables}."
        )

    table_paths = {
        table_name: manifest_path.parent / str(tables[table_name])
        for table_name in required_tables
    }
    for table_name, table_path in table_paths.items():
        if not table_path.exists():
            raise NeuronpediaLocalDBImportError(
                f"Columnar table {table_path} for {table_name} referenced by {manifest_path} does not exist."
            )
    return table_paths


def _preferred_activation_table_path(manifest_path: Path) -> tuple[str, Path]:
    manifest = _load_json_file(manifest_path)
    tables = manifest.get("tables")
    if not isinstance(tables, dict):
        raise NeuronpediaLocalDBImportError(
            f"Columnar manifest {manifest_path} must contain a tables object."
        )

    for table_name in SAEDASHBOARD_COLUMNAR_ACTIVATION_TABLE_PRIORITY:
        if table_name not in tables:
            continue
        table_path = manifest_path.parent / str(tables[table_name])
        if not table_path.exists():
            raise NeuronpediaLocalDBImportError(
                f"Columnar table {table_path} for {table_name} referenced by {manifest_path} does not exist."
            )
        return table_name, table_path

    raise NeuronpediaLocalDBImportError(
        f"Columnar manifest {manifest_path} is missing Activation import tables: "
        f"{list(SAEDASHBOARD_COLUMNAR_ACTIVATION_TABLE_PRIORITY)}."
    )


def _build_saedashboard_columnar_neuron_records_with_metadata(
    columnar_root: Path | str,
    *,
    model_id: str,
    layer: str,
    creator_id: str,
    decode_token_ids: Callable[[list[int]], list[str]],
    created_at: datetime | str | None = None,
    hook_name: str | None = None,
    source_set_name: str | None = None,
) -> tuple[list[dict[str, Any]], list[Path], float]:
    created_at_record_value = _created_at_value(created_at)
    records: list[dict[str, Any]] = []
    processed_files: list[Path] = []
    table_load_seconds = 0.0

    for manifest_path in discover_saedashboard_columnar_manifest_paths(columnar_root):
        table_paths = _required_columnar_table_paths(manifest_path)
        load_start_time = time.perf_counter()
        feature_statistics = _feature_indexed_rows(
            "feature_statistics", table_paths["feature_statistics"]
        )
        feature_tables = _feature_indexed_rows(
            "feature_tables", table_paths["feature_tables"]
        )
        logits_histograms = _feature_indexed_rows(
            "logits_histograms", table_paths["logits_histograms"]
        )
        activation_histograms = _feature_indexed_rows(
            "activation_histograms", table_paths["activation_histograms"]
        )
        logits_tables = _feature_indexed_rows(
            "logits_tables", table_paths["logits_tables"]
        )
        table_load_seconds += time.perf_counter() - load_start_time
        processed_files.extend(
            table_paths[table_name] for table_name in sorted(table_paths)
        )

        feature_indices = sorted(feature_statistics)
        for feature_index in feature_indices:
            missing_feature_tables = [
                table_name
                for table_name, indexed_rows in (
                    ("feature_tables", feature_tables),
                    ("logits_histograms", logits_histograms),
                    ("activation_histograms", activation_histograms),
                    ("logits_tables", logits_tables),
                )
                if feature_index not in indexed_rows
            ]
            if missing_feature_tables:
                raise NeuronpediaLocalDBImportError(
                    f"feature_index={feature_index} is missing rows in {missing_feature_tables}."
                )

            statistics_row = feature_statistics[feature_index]
            feature_table_row = feature_tables[feature_index]
            logits_histogram_row = logits_histograms[feature_index]
            activation_histogram_row = activation_histograms[feature_index]
            logits_table_row = logits_tables[feature_index]
            max_act_approx = float(statistics_row.get("max") or 0.0)

            records.append(
                {
                    "modelId": model_id,
                    "layer": layer,
                    "index": feature_index,
                    "sourceSetName": source_set_name,
                    "creatorId": creator_id,
                    "createdAt": created_at_record_value,
                    "maxActApprox": max_act_approx,
                    "hasVector": False,
                    "vector": [],
                    "vectorLabel": None,
                    "vectorDefaultSteerStrength": max_act_approx,
                    "hookName": hook_name,
                    "topkCosSimIndices": [],
                    "topkCosSimValues": [],
                    "neuron_alignment_indices": _int_list(
                        feature_table_row.get("neuron_alignment_indices")
                    ),
                    "neuron_alignment_values": _round_float_list(
                        feature_table_row.get("neuron_alignment_values")
                    ),
                    "neuron_alignment_l1": _round_float_list(
                        feature_table_row.get("neuron_alignment_l1")
                    ),
                    "correlated_neurons_indices": _int_list(
                        feature_table_row.get("correlated_neurons_indices")
                    ),
                    "correlated_neurons_pearson": _round_float_list(
                        feature_table_row.get("correlated_neurons_pearson")
                    ),
                    "correlated_neurons_l1": _round_float_list(
                        feature_table_row.get("correlated_neurons_cossim")
                    ),
                    "correlated_features_indices": _int_list(
                        feature_table_row.get("correlated_features_indices")
                    ),
                    "correlated_features_pearson": _round_float_list(
                        feature_table_row.get("correlated_features_pearson")
                    ),
                    "correlated_features_l1": _round_float_list(
                        feature_table_row.get("correlated_features_cossim")
                    ),
                    "neg_str": _decode_columnar_token_ids(
                        decode_token_ids, logits_table_row.get("bottom_token_ids")
                    ),
                    "neg_values": _round_float_list(
                        logits_table_row.get("bottom_logits")
                    ),
                    "pos_str": _decode_columnar_token_ids(
                        decode_token_ids, logits_table_row.get("top_token_ids")
                    ),
                    "pos_values": _round_float_list(logits_table_row.get("top_logits")),
                    "frac_nonzero": _float_or_zero(
                        statistics_row.get("positive_density")
                        if statistics_row.get("positive_density") is not None
                        else statistics_row.get("frac_nonzero")
                    ),
                    "freq_hist_data_bar_heights": _round_float_list(
                        activation_histogram_row.get("bar_heights")
                    ),
                    "freq_hist_data_bar_values": _round_float_list(
                        activation_histogram_row.get("bar_values")
                    ),
                    "logits_hist_data_bar_heights": _round_float_list(
                        logits_histogram_row.get("bar_heights")
                    ),
                    "logits_hist_data_bar_values": _round_float_list(
                        logits_histogram_row.get("bar_values")
                    ),
                    "decoder_weights_dist": [],
                }
            )

    return records, processed_files, table_load_seconds


def _parse_sequence_group_name(group_name: str) -> tuple[float, float, float]:
    bin_min, bin_max, bin_contains = 0.0, 0.0, 0.0
    if "TOP ACTIVATIONS" in group_name:
        bin_min, bin_max, bin_contains = -1.0, 99.0, -1.0
        try:
            bin_max = float(group_name.split(" = ")[-1])
        except ValueError as exc:
            raise NeuronpediaLocalDBImportError(
                f"Could not parse top-activation group name: {group_name!r}"
            ) from exc
    elif "INTERVAL" in group_name:
        try:
            split = group_name.split("<br>")
            first_split = split[0].split(" ")
            second_split = split[1].split(" ")
            bin_min = float(first_split[1])
            bin_max = float(first_split[-1])
            bin_contains = float(second_split[-1].rstrip("%")) / 100
        except (IndexError, ValueError) as exc:
            raise NeuronpediaLocalDBImportError(
                f"Could not parse interval group name: {group_name!r}"
            ) from exc
    return bin_min, bin_max, bin_contains


def _trim_trailing_pad_token_rows(
    token_ids: list[int],
    values: list[float],
    token_logits: list[float],
    pad_token_id: int | None,
) -> tuple[list[int], list[float], list[float]]:
    if pad_token_id is None or not token_ids:
        return token_ids, values, token_logits
    trimmed_len = len(token_ids)
    while trimmed_len > 0 and token_ids[trimmed_len - 1] == pad_token_id:
        trimmed_len -= 1
    return token_ids[:trimmed_len], values[:trimmed_len], token_logits[:trimmed_len]


def _build_activation_record_from_sequence_rows(
    *,
    rows: list[dict[str, Any]],
    model_id: str,
    layer: str,
    creator_id: str,
    created_at: str,
    decode_token_ids: Callable[[list[int]], list[str]],
    activation_id_prefix: str,
    pad_token_id: int | None,
) -> dict[str, Any]:
    sorted_rows = sorted(rows, key=lambda row: int(row["context_token_index"]))
    first_row = sorted_rows[0]
    feature_index = int(first_row["feature_index"])
    sequence_index = int(first_row["sequence_index"])
    group_name = str(first_row.get("group_name") or "")
    bin_min, bin_max, bin_contains = _parse_sequence_group_name(group_name)
    token_ids = [int(row["token_id"]) for row in sorted_rows]
    values = _round_float_list([row["feat_act"] for row in sorted_rows])
    token_logits = _round_float_list([row["token_logit"] for row in sorted_rows])
    token_ids, values, token_logits = _trim_trailing_pad_token_rows(
        token_ids, values, token_logits, pad_token_id
    )
    tokens = _decode_columnar_token_ids(decode_token_ids, token_ids)
    max_value = max(values) if values else 0.0
    min_value = min(values) if values else 0.0
    max_value_token_index = values.index(max_value) if values else 0
    return {
        "id": f"{activation_id_prefix}-{feature_index}-{sequence_index}",
        "tokens": tokens,
        "dataIndex": None,
        "index": feature_index,
        "layer": layer,
        "modelId": model_id,
        "dataSource": None,
        "maxValue": max_value,
        "maxValueTokenIndex": max_value_token_index,
        "minValue": min_value,
        "values": values,
        "dfaValues": [],
        "dfaTargetIndex": None,
        "dfaMaxValue": None,
        "creatorId": creator_id,
        "createdAt": created_at,
        "lossValues": [],
        "logitContributions": None,
        "binMin": bin_min,
        "binMax": bin_max,
        "binContains": bin_contains,
        "qualifyingTokenIndex": int(first_row["qualifying_token_index"]) - 1,
    }


def _build_activation_record_from_activation_row(
    *,
    row: Mapping[str, Any],
    model_id: str,
    layer: str,
    creator_id: str,
    created_at: str,
    activation_id_prefix: str,
) -> dict[str, Any]:
    feature_index = int(row["feature_index"])
    sequence_index = int(row["sequence_index"])
    return {
        "id": f"{activation_id_prefix}-{feature_index}-{sequence_index}",
        "tokens": list(cast(list[str], row["tokens"])),
        "dataIndex": None,
        "index": feature_index,
        "layer": layer,
        "modelId": model_id,
        "dataSource": None,
        "maxValue": float(row["max_value"]),
        "maxValueTokenIndex": int(row["max_value_token_index"]),
        "minValue": float(row["min_value"]),
        "values": list(cast(list[float], row["values"])),
        "dfaValues": [],
        "dfaTargetIndex": None,
        "dfaMaxValue": None,
        "creatorId": creator_id,
        "createdAt": created_at,
        "lossValues": [],
        "logitContributions": None,
        "binMin": float(row["bin_min"]),
        "binMax": float(row["bin_max"]),
        "binContains": float(row["bin_contains"]),
        "qualifyingTokenIndex": int(row["qualifying_token_index"]),
    }


def _is_top_activation_record(record: Mapping[str, Any]) -> bool:
    return float(record["binMin"]) == -1.0 and float(record["binContains"]) == -1.0


class _ActivationRecordHygieneFilter:
    """Opt-in per-feature dedup and zero-activation filtering for Activation records.

    Records must arrive with each feature's rows contiguous (all three columnar
    Activation sources emit rows grouped by ``feature_index``). Zero-drop is applied
    before dedup; the order is immaterial because duplicate records share identical
    ``values`` and therefore identical ``maxValue``.

    Dedup keys records on (feature_index, tokens tuple, values tuple), keeping the
    TOP-ACTIVATIONS record (binMin=-1/binContains=-1) in preference to interval-group
    records when both exist; among interval duplicates the first record wins.
    """

    def __init__(
        self,
        *,
        dedup_activation_rows: bool,
        drop_zero_activation_rows: bool,
        hygiene_counters: dict[str, int] | None = None,
    ) -> None:
        self._dedup_activation_rows = dedup_activation_rows
        self._drop_zero_activation_rows = drop_zero_activation_rows
        self._hygiene_counters = (
            hygiene_counters if hygiene_counters is not None else {}
        )
        if dedup_activation_rows:
            self._hygiene_counters.setdefault("deduplicated_rows", 0)
        if drop_zero_activation_rows:
            self._hygiene_counters.setdefault("zero_rows_dropped", 0)
        self._current_feature_index: int | None = None
        self._kept_records_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}

    def push(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        if self._drop_zero_activation_rows and float(record["maxValue"]) == 0.0:
            self._hygiene_counters["zero_rows_dropped"] += 1
            return []
        if not self._dedup_activation_rows:
            return [record]

        feature_index = int(record["index"])
        emitted_records: list[dict[str, Any]] = []
        if (
            self._current_feature_index is not None
            and feature_index != self._current_feature_index
        ):
            emitted_records = list(self._kept_records_by_key.values())
            self._kept_records_by_key = {}
        self._current_feature_index = feature_index

        record_key = (tuple(record["tokens"]), tuple(record["values"]))
        kept_record = self._kept_records_by_key.get(record_key)
        if kept_record is None:
            self._kept_records_by_key[record_key] = record
        else:
            self._hygiene_counters["deduplicated_rows"] += 1
            if _is_top_activation_record(record) and not _is_top_activation_record(
                kept_record
            ):
                self._kept_records_by_key[record_key] = record
        return emitted_records

    def flush(self) -> list[dict[str, Any]]:
        emitted_records = list(self._kept_records_by_key.values())
        self._kept_records_by_key = {}
        self._current_feature_index = None
        return emitted_records


def _iter_hygiene_filtered_activation_records(
    records: Iterable[dict[str, Any]],
    *,
    dedup_activation_rows: bool,
    drop_zero_activation_rows: bool,
    hygiene_counters: dict[str, int] | None = None,
) -> Iterable[dict[str, Any]]:
    if not (dedup_activation_rows or drop_zero_activation_rows):
        yield from records
        return
    record_filter = _ActivationRecordHygieneFilter(
        dedup_activation_rows=dedup_activation_rows,
        drop_zero_activation_rows=drop_zero_activation_rows,
        hygiene_counters=hygiene_counters,
    )
    for record in records:
        yield from record_filter.push(record)
    yield from record_filter.flush()


def _iter_hygiene_filtered_activation_record_batches(
    batches: Iterable[Any],
    *,
    dedup_activation_rows: bool,
    drop_zero_activation_rows: bool,
    hygiene_counters: dict[str, int] | None = None,
) -> Iterable[Any]:
    if not (dedup_activation_rows or drop_zero_activation_rows):
        yield from batches
        return
    pyarrow, _, _ = _load_pyarrow_modules()
    record_filter = _ActivationRecordHygieneFilter(
        dedup_activation_rows=dedup_activation_rows,
        drop_zero_activation_rows=drop_zero_activation_rows,
        hygiene_counters=hygiene_counters,
    )
    batch_schema = None
    pending_records: list[dict[str, Any]] = []
    for batch in batches:
        batch_schema = batch.schema
        for record in cast(list[dict[str, Any]], batch.to_pylist()):
            pending_records.extend(record_filter.push(record))
        if pending_records:
            yield pyarrow.RecordBatch.from_pylist(pending_records, schema=batch_schema)
            pending_records = []
    pending_records.extend(record_filter.flush())
    if pending_records and batch_schema is not None:
        yield pyarrow.RecordBatch.from_pylist(pending_records, schema=batch_schema)


def _iter_activation_records_from_sequence_row_batches(
    batches: Iterable[Any],
    *,
    model_id: str,
    layer: str,
    creator_id: str,
    created_at: str,
    decode_token_ids: Callable[[list[int]], list[str]],
    activation_id_prefix: str,
    pad_token_id: int | None,
) -> Iterable[dict[str, Any]]:
    current_key: tuple[int, int] | None = None
    current_rows: list[dict[str, Any]] = []
    previous_key: tuple[int, int] | None = None

    for batch in batches:
        for row in cast(list[dict[str, Any]], batch.to_pylist()):
            row_key = (int(row["feature_index"]), int(row["sequence_index"]))
            if previous_key is not None and row_key < previous_key:
                raise NeuronpediaLocalDBImportError(
                    "Columnar sequence_rows must be sorted by feature_index and sequence_index for streamed "
                    f"Activation conversion; observed {row_key} after {previous_key}."
                )
            if current_key is None:
                current_key = row_key
            elif row_key != current_key:
                yield _build_activation_record_from_sequence_rows(
                    rows=current_rows,
                    model_id=model_id,
                    layer=layer,
                    creator_id=creator_id,
                    created_at=created_at,
                    decode_token_ids=decode_token_ids,
                    activation_id_prefix=activation_id_prefix,
                    pad_token_id=pad_token_id,
                )
                current_key = row_key
                current_rows = []
            current_rows.append(row)
            previous_key = row_key

    if current_rows:
        yield _build_activation_record_from_sequence_rows(
            rows=current_rows,
            model_id=model_id,
            layer=layer,
            creator_id=creator_id,
            created_at=created_at,
            decode_token_ids=decode_token_ids,
            activation_id_prefix=activation_id_prefix,
            pad_token_id=pad_token_id,
        )


def _iter_activation_records_from_activation_row_batches(
    batches: Iterable[Any],
    *,
    model_id: str,
    layer: str,
    creator_id: str,
    created_at: str,
    activation_id_prefix: str,
) -> Iterable[dict[str, Any]]:
    for batch in batches:
        missing_columns = SAEDASHBOARD_COLUMNAR_ACTIVATION_ROW_COLUMNS - set(
            batch.schema.names
        )
        if missing_columns:
            raise NeuronpediaLocalDBImportError(
                f"Columnar activation_rows batch is missing required Activation columns: {sorted(missing_columns)}."
            )
        for row in cast(list[dict[str, Any]], batch.to_pylist()):
            yield _build_activation_record_from_activation_row(
                row=row,
                model_id=model_id,
                layer=layer,
                creator_id=creator_id,
                created_at=created_at,
                activation_id_prefix=activation_id_prefix,
            )


def _empty_activation_batch_columns() -> dict[str, list[Any]]:
    return {
        column_name: [] for column_name in SAEDASHBOARD_COLUMNAR_ACTIVATION_COPY_COLUMNS
    }


def _append_activation_columns_from_sequence_rows(
    columns: dict[str, list[Any]],
    *,
    rows: list[tuple[int, int, float, float]],
    feature_index: int,
    sequence_index: int,
    group_name: str,
    qualifying_token_index: int,
    model_id: str,
    layer: str,
    creator_id: str,
    created_at: str,
    decode_token_ids: Callable[[list[int]], list[str]],
    activation_id_prefix: str,
    pad_token_id: int | None,
) -> None:
    sorted_rows = sorted(rows, key=lambda row: row[0])
    bin_min, bin_max, bin_contains = _parse_sequence_group_name(group_name)
    token_ids = [int(row[1]) for row in sorted_rows]
    values = _round_float_list([row[2] for row in sorted_rows])
    token_logits = _round_float_list([row[3] for row in sorted_rows])
    token_ids, values, _ = _trim_trailing_pad_token_rows(
        token_ids, values, token_logits, pad_token_id
    )
    tokens = _decode_columnar_token_ids(decode_token_ids, token_ids)
    max_value = max(values) if values else 0.0
    min_value = min(values) if values else 0.0
    max_value_token_index = values.index(max_value) if values else 0

    columns["id"].append(f"{activation_id_prefix}-{feature_index}-{sequence_index}")
    columns["tokens"].append(tokens)
    columns["dataIndex"].append(None)
    columns["index"].append(feature_index)
    columns["layer"].append(layer)
    columns["modelId"].append(model_id)
    columns["dataSource"].append(None)
    columns["maxValue"].append(max_value)
    columns["maxValueTokenIndex"].append(max_value_token_index)
    columns["minValue"].append(min_value)
    columns["values"].append(values)
    columns["dfaValues"].append([])
    columns["dfaTargetIndex"].append(None)
    columns["dfaMaxValue"].append(None)
    columns["creatorId"].append(creator_id)
    columns["createdAt"].append(created_at)
    columns["lossValues"].append([])
    columns["logitContributions"].append(None)
    columns["binMin"].append(bin_min)
    columns["binMax"].append(bin_max)
    columns["binContains"].append(bin_contains)
    columns["qualifyingTokenIndex"].append(qualifying_token_index - 1)


def _activation_record_batch_from_columns(columns: dict[str, list[Any]]) -> Any:
    pyarrow, _, _ = _load_pyarrow_modules()
    row_count = len(columns["id"])
    arrays = [
        pyarrow.array(columns["id"], type=pyarrow.string()),
        pyarrow.array(columns["tokens"], type=pyarrow.list_(pyarrow.string())),
        pyarrow.nulls(row_count),
        pyarrow.array(columns["index"], type=pyarrow.int64()),
        pyarrow.array(columns["layer"], type=pyarrow.string()),
        pyarrow.array(columns["modelId"], type=pyarrow.string()),
        pyarrow.nulls(row_count),
        pyarrow.array(columns["maxValue"], type=pyarrow.float64()),
        pyarrow.array(columns["maxValueTokenIndex"], type=pyarrow.int64()),
        pyarrow.array(columns["minValue"], type=pyarrow.float64()),
        pyarrow.array(columns["values"], type=pyarrow.list_(pyarrow.float64())),
        pyarrow.array(columns["dfaValues"], type=pyarrow.list_(pyarrow.float64())),
        pyarrow.nulls(row_count),
        pyarrow.nulls(row_count),
        pyarrow.array(columns["creatorId"], type=pyarrow.string()),
        pyarrow.array(columns["createdAt"], type=pyarrow.string()),
        pyarrow.array(columns["lossValues"], type=pyarrow.list_(pyarrow.float64())),
        pyarrow.nulls(row_count),
        pyarrow.array(columns["binMin"], type=pyarrow.float64()),
        pyarrow.array(columns["binMax"], type=pyarrow.float64()),
        pyarrow.array(columns["binContains"], type=pyarrow.float64()),
        pyarrow.array(columns["qualifyingTokenIndex"], type=pyarrow.int64()),
    ]
    return pyarrow.RecordBatch.from_arrays(
        arrays, names=SAEDASHBOARD_COLUMNAR_ACTIVATION_COPY_COLUMNS
    )


def _empty_list_array(pyarrow: Any, *, row_count: int, value_type: Any) -> Any:
    return pyarrow.ListArray.from_arrays(
        pyarrow.array([0] * (row_count + 1), type=pyarrow.int32()),
        pyarrow.array([], type=value_type),
    )


def _activation_record_batch_from_activation_row_batch(
    batch: Any,
    *,
    model_id: str,
    layer: str,
    creator_id: str,
    created_at: str,
    activation_id_prefix: str,
) -> Any:
    pyarrow, pyarrow_compute, _ = _load_pyarrow_modules()
    row_count = batch.num_rows
    feature_index = batch.column(batch.schema.get_field_index("feature_index"))
    sequence_index = batch.column(batch.schema.get_field_index("sequence_index"))
    id_column = pyarrow_compute.binary_join_element_wise(
        pyarrow.array([activation_id_prefix] * row_count, type=pyarrow.string()),
        pyarrow_compute.cast(feature_index, pyarrow.string()),
        pyarrow_compute.cast(sequence_index, pyarrow.string()),
        "-",
    )
    arrays = [
        id_column,
        batch.column(batch.schema.get_field_index("tokens")),
        pyarrow.nulls(row_count),
        feature_index,
        pyarrow.array([layer] * row_count, type=pyarrow.string()),
        pyarrow.array([model_id] * row_count, type=pyarrow.string()),
        pyarrow.nulls(row_count),
        batch.column(batch.schema.get_field_index("max_value")),
        batch.column(batch.schema.get_field_index("max_value_token_index")),
        batch.column(batch.schema.get_field_index("min_value")),
        batch.column(batch.schema.get_field_index("values")),
        _empty_list_array(pyarrow, row_count=row_count, value_type=pyarrow.float64()),
        pyarrow.nulls(row_count),
        pyarrow.nulls(row_count),
        pyarrow.array([creator_id] * row_count, type=pyarrow.string()),
        pyarrow.array([created_at] * row_count, type=pyarrow.string()),
        _empty_list_array(pyarrow, row_count=row_count, value_type=pyarrow.float64()),
        pyarrow.nulls(row_count),
        batch.column(batch.schema.get_field_index("bin_min")),
        batch.column(batch.schema.get_field_index("bin_max")),
        batch.column(batch.schema.get_field_index("bin_contains")),
        batch.column(batch.schema.get_field_index("qualifying_token_index")),
    ]
    return pyarrow.RecordBatch.from_arrays(
        arrays, names=SAEDASHBOARD_COLUMNAR_ACTIVATION_COPY_COLUMNS
    )


def _append_activation_columns_from_activation_row(
    columns: dict[str, list[Any]],
    *,
    row: Mapping[str, Any],
    model_id: str,
    layer: str,
    creator_id: str,
    created_at: str,
    activation_id_prefix: str,
) -> None:
    record = _build_activation_record_from_activation_row(
        row=row,
        model_id=model_id,
        layer=layer,
        creator_id=creator_id,
        created_at=created_at,
        activation_id_prefix=activation_id_prefix,
    )
    for column_name in SAEDASHBOARD_COLUMNAR_ACTIVATION_COPY_COLUMNS:
        columns[column_name].append(record[column_name])


def _activation_copy_sequence_index(
    activation_id: str,
    feature_index: int,
) -> int:
    try:
        _prefix, feature_suffix, sequence_suffix = activation_id.rsplit("-", 2)
    except ValueError as exc:
        raise NeuronpediaLocalDBImportError(
            "Columnar activation_copy_rows ids must end with '-{feature_index}-{sequence_index}'; "
            f"got {activation_id!r}."
        ) from exc

    try:
        feature_from_id = int(feature_suffix)
        sequence_index = int(sequence_suffix)
    except ValueError as exc:
        raise NeuronpediaLocalDBImportError(
            "Columnar activation_copy_rows ids must end with integer feature and sequence indexes; "
            f"got {activation_id!r}."
        ) from exc

    if feature_from_id != feature_index:
        raise NeuronpediaLocalDBImportError(
            "Columnar activation_copy_rows id feature index must match the row index column; "
            f"got id={activation_id!r} index={feature_index}."
        )

    return sequence_index


def _rewrite_activation_copy_row_batch(
    batch: Any,
    *,
    model_id: str,
    layer: str,
    creator_id: str,
    created_at: str,
    activation_id_prefix: str,
) -> Any:
    pyarrow, _, _ = _load_pyarrow_modules()

    id_index = batch.schema.get_field_index("id")
    feature_index_index = batch.schema.get_field_index("index")
    layer_index = batch.schema.get_field_index("layer")
    model_id_index = batch.schema.get_field_index("modelId")
    creator_id_index = batch.schema.get_field_index("creatorId")
    created_at_index = batch.schema.get_field_index("createdAt")

    feature_indices = cast(list[int], batch.column(feature_index_index).to_pylist())
    activation_ids = cast(list[str], batch.column(id_index).to_pylist())
    row_count = len(feature_indices)

    rewritten_ids = [
        f"{activation_id_prefix}-{feature_index}-{_activation_copy_sequence_index(activation_id, feature_index)}"
        for activation_id, feature_index in zip(
            activation_ids, feature_indices, strict=True
        )
    ]

    rewritten_columns = []
    for column_name in SAEDASHBOARD_COLUMNAR_ACTIVATION_COPY_COLUMNS:
        column_index = batch.schema.get_field_index(column_name)
        column = batch.column(column_index)
        if column_name == "id":
            rewritten_columns.append(
                pyarrow.array(rewritten_ids, type=batch.schema.field(id_index).type)
            )
        elif column_name == "layer":
            rewritten_columns.append(
                pyarrow.array(
                    [layer] * row_count, type=batch.schema.field(layer_index).type
                )
            )
        elif column_name == "modelId":
            rewritten_columns.append(
                pyarrow.array(
                    [model_id] * row_count, type=batch.schema.field(model_id_index).type
                )
            )
        elif column_name == "creatorId":
            rewritten_columns.append(
                pyarrow.array(
                    [creator_id] * row_count,
                    type=batch.schema.field(creator_id_index).type,
                )
            )
        elif column_name == "createdAt":
            rewritten_columns.append(
                pyarrow.array(
                    [created_at] * row_count,
                    type=batch.schema.field(created_at_index).type,
                )
            )
        else:
            rewritten_columns.append(column)

    return pyarrow.RecordBatch.from_arrays(
        rewritten_columns,
        names=SAEDASHBOARD_COLUMNAR_ACTIVATION_COPY_COLUMNS,
    )


def _iter_activation_record_batches_from_sequence_row_batches(
    batches: Iterable[Any],
    *,
    model_id: str,
    layer: str,
    creator_id: str,
    created_at: str,
    decode_token_ids: Callable[[list[int]], list[str]],
    activation_id_prefix: str,
    pad_token_id: int | None,
    output_batch_size: int = 65000,
) -> Iterable[Any]:
    current_key: tuple[int, int] | None = None
    current_rows: list[tuple[int, int, float, float]] = []
    current_group_name = ""
    current_qualifying_token_index = 0
    previous_key: tuple[int, int] | None = None
    columns = _empty_activation_batch_columns()

    def flush_current_group() -> None:
        nonlocal columns, current_rows
        if current_key is None or not current_rows:
            return
        feature_index, sequence_index = current_key
        _append_activation_columns_from_sequence_rows(
            columns,
            rows=current_rows,
            feature_index=feature_index,
            sequence_index=sequence_index,
            group_name=current_group_name,
            qualifying_token_index=current_qualifying_token_index,
            model_id=model_id,
            layer=layer,
            creator_id=creator_id,
            created_at=created_at,
            decode_token_ids=decode_token_ids,
            activation_id_prefix=activation_id_prefix,
            pad_token_id=pad_token_id,
        )
        current_rows = []

    required_columns = {
        "feature_index",
        "sequence_index",
        "context_token_index",
        "group_name",
        "qualifying_token_index",
        "token_id",
        "feat_act",
        "token_logit",
    }

    for batch in batches:
        missing_columns = required_columns - set(batch.schema.names)
        if missing_columns:
            raise NeuronpediaLocalDBImportError(
                f"Columnar sequence_rows batch is missing required Activation columns: {sorted(missing_columns)}."
            )
        batch_columns = {
            column_name: batch.column(
                batch.schema.get_field_index(column_name)
            ).to_pylist()
            for column_name in required_columns
        }
        for row_idx in range(batch.num_rows):
            row_key = (
                int(batch_columns["feature_index"][row_idx]),
                int(batch_columns["sequence_index"][row_idx]),
            )
            if previous_key is not None and row_key < previous_key:
                raise NeuronpediaLocalDBImportError(
                    "Columnar sequence_rows must be sorted by feature_index and sequence_index for streamed "
                    f"Activation conversion; observed {row_key} after {previous_key}."
                )
            if current_key is None:
                current_key = row_key
                current_group_name = str(batch_columns["group_name"][row_idx] or "")
                current_qualifying_token_index = int(
                    batch_columns["qualifying_token_index"][row_idx]
                )
            elif row_key != current_key:
                flush_current_group()
                if len(columns["id"]) >= output_batch_size:
                    yield _activation_record_batch_from_columns(columns)
                    columns = _empty_activation_batch_columns()
                current_key = row_key
                current_group_name = str(batch_columns["group_name"][row_idx] or "")
                current_qualifying_token_index = int(
                    batch_columns["qualifying_token_index"][row_idx]
                )
            current_rows.append(
                (
                    int(batch_columns["context_token_index"][row_idx]),
                    int(batch_columns["token_id"][row_idx]),
                    float(batch_columns["feat_act"][row_idx]),
                    float(batch_columns["token_logit"][row_idx]),
                )
            )
            previous_key = row_key

    flush_current_group()
    if columns["id"]:
        yield _activation_record_batch_from_columns(columns)


def _iter_activation_record_batches_from_activation_row_batches(
    batches: Iterable[Any],
    *,
    model_id: str,
    layer: str,
    creator_id: str,
    created_at: str,
    activation_id_prefix: str,
    output_batch_size: int = 65000,
) -> Iterable[Any]:
    for batch in batches:
        missing_columns = SAEDASHBOARD_COLUMNAR_ACTIVATION_ROW_COLUMNS - set(
            batch.schema.names
        )
        if missing_columns:
            raise NeuronpediaLocalDBImportError(
                f"Columnar activation_rows batch is missing required Activation columns: {sorted(missing_columns)}."
            )
        for start_idx in range(0, batch.num_rows, output_batch_size):
            batch_slice = batch.slice(
                start_idx, min(output_batch_size, batch.num_rows - start_idx)
            )
            yield _activation_record_batch_from_activation_row_batch(
                batch_slice,
                model_id=model_id,
                layer=layer,
                creator_id=creator_id,
                created_at=created_at,
                activation_id_prefix=activation_id_prefix,
            )


def _iter_activation_record_batches_from_activation_copy_row_batches(
    batches: Iterable[Any],
    *,
    model_id: str,
    layer: str,
    creator_id: str,
    created_at: str,
    activation_id_prefix: str,
) -> Iterable[Any]:
    required_columns = set(SAEDASHBOARD_COLUMNAR_ACTIVATION_COPY_COLUMNS)
    for batch in batches:
        missing_columns = required_columns - set(batch.schema.names)
        if missing_columns:
            raise NeuronpediaLocalDBImportError(
                "Columnar activation_copy_rows batch is missing required Activation COPY columns: "
                f"{sorted(missing_columns)}."
            )
        yield _rewrite_activation_copy_row_batch(
            batch,
            model_id=model_id,
            layer=layer,
            creator_id=creator_id,
            created_at=created_at,
            activation_id_prefix=activation_id_prefix,
        )


def _iter_saedashboard_columnar_activation_record_batches_with_metadata(
    columnar_root: Path | str,
    *,
    model_id: str,
    layer: str,
    creator_id: str,
    decode_token_ids: Callable[[list[int]], list[str]],
    created_at: datetime | str | None = None,
    activation_id_prefix: str = "columnar-activation",
    pad_token_id: int | None = None,
    read_batch_size: int = 65000,
    output_batch_size: int = 65000,
    dedup_activation_rows: bool = False,
    drop_zero_activation_rows: bool = False,
    hygiene_counters: dict[str, int] | None = None,
) -> tuple[Iterable[Any], list[Path]]:
    created_at_record_value = _created_at_value(created_at)
    processed_files: list[Path] = []

    def _record_batches() -> Iterable[Any]:
        for manifest_path in discover_saedashboard_columnar_manifest_paths(
            columnar_root
        ):
            table_name, table_path = _preferred_activation_table_path(manifest_path)
            processed_files.append(table_path)
            if table_name == "activation_copy_rows":
                yield from _iter_activation_record_batches_from_activation_copy_row_batches(
                    _iter_columnar_record_batches(
                        table_path, batch_size=read_batch_size
                    ),
                    model_id=model_id,
                    layer=layer,
                    creator_id=creator_id,
                    created_at=created_at_record_value,
                    activation_id_prefix=activation_id_prefix,
                )
            elif table_name == "activation_rows":
                yield from _iter_activation_record_batches_from_activation_row_batches(
                    _iter_columnar_record_batches(
                        table_path, batch_size=read_batch_size, coalesce_arrow=True
                    ),
                    model_id=model_id,
                    layer=layer,
                    creator_id=creator_id,
                    created_at=created_at_record_value,
                    activation_id_prefix=activation_id_prefix,
                    output_batch_size=output_batch_size,
                )
            else:
                yield from _iter_activation_record_batches_from_sequence_row_batches(
                    _iter_columnar_record_batches(
                        table_path, batch_size=read_batch_size
                    ),
                    model_id=model_id,
                    layer=layer,
                    creator_id=creator_id,
                    created_at=created_at_record_value,
                    decode_token_ids=decode_token_ids,
                    activation_id_prefix=activation_id_prefix,
                    pad_token_id=pad_token_id,
                    output_batch_size=output_batch_size,
                )

    filtered_record_batches = _iter_hygiene_filtered_activation_record_batches(
        _record_batches(),
        dedup_activation_rows=dedup_activation_rows,
        drop_zero_activation_rows=drop_zero_activation_rows,
        hygiene_counters=hygiene_counters,
    )
    return filtered_record_batches, processed_files


def _build_saedashboard_columnar_activation_records_with_metadata(
    columnar_root: Path | str,
    *,
    model_id: str,
    layer: str,
    creator_id: str,
    decode_token_ids: Callable[[list[int]], list[str]],
    created_at: datetime | str | None = None,
    activation_id_prefix: str = "columnar-activation",
    pad_token_id: int | None = None,
    read_batch_size: int = 65000,
    dedup_activation_rows: bool = False,
    drop_zero_activation_rows: bool = False,
    hygiene_counters: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[Path], float]:
    created_at_record_value = _created_at_value(created_at)
    records: list[dict[str, Any]] = []
    processed_files: list[Path] = []
    table_load_seconds = 0.0

    def _iter_activation_copy_row_records(table_path: Path) -> Iterable[dict[str, Any]]:
        for batch in _iter_activation_record_batches_from_activation_copy_row_batches(
            _iter_columnar_record_batches(table_path, batch_size=read_batch_size),
            model_id=model_id,
            layer=layer,
            creator_id=creator_id,
            created_at=created_at_record_value,
            activation_id_prefix=activation_id_prefix,
        ):
            yield from cast(list[dict[str, Any]], batch.to_pylist())

    for manifest_path in discover_saedashboard_columnar_manifest_paths(columnar_root):
        table_name, table_path = _preferred_activation_table_path(manifest_path)
        processed_files.append(table_path)
        load_start_time = time.perf_counter()
        if table_name == "activation_copy_rows":
            table_records: Iterable[dict[str, Any]] = _iter_activation_copy_row_records(
                table_path
            )
        elif table_name == "activation_rows":
            table_records = _iter_activation_records_from_activation_row_batches(
                _iter_columnar_record_batches(table_path, batch_size=read_batch_size),
                model_id=model_id,
                layer=layer,
                creator_id=creator_id,
                created_at=created_at_record_value,
                activation_id_prefix=activation_id_prefix,
            )
        else:
            table_records = _iter_activation_records_from_sequence_row_batches(
                _iter_columnar_record_batches(table_path, batch_size=read_batch_size),
                model_id=model_id,
                layer=layer,
                creator_id=creator_id,
                created_at=created_at_record_value,
                decode_token_ids=decode_token_ids,
                activation_id_prefix=activation_id_prefix,
                pad_token_id=pad_token_id,
            )
        records.extend(
            _iter_hygiene_filtered_activation_records(
                table_records,
                dedup_activation_rows=dedup_activation_rows,
                drop_zero_activation_rows=drop_zero_activation_rows,
                hygiene_counters=hygiene_counters,
            )
        )
        table_load_seconds += time.perf_counter() - load_start_time

    return records, processed_files, table_load_seconds


def build_saedashboard_columnar_activation_records(
    columnar_root: Path | str,
    *,
    model_id: str,
    layer: str,
    creator_id: str,
    decode_token_ids: Callable[[list[int]], list[str]],
    created_at: datetime | str | None = None,
    activation_id_prefix: str = "columnar-activation",
    pad_token_id: int | None = None,
    dedup_activation_rows: bool = False,
    drop_zero_activation_rows: bool = False,
) -> list[dict[str, Any]]:
    """Build legacy-compatible Activation rows from SAEDashboard activation_rows artifacts or sequence_rows fallback."""

    records, _, _ = _build_saedashboard_columnar_activation_records_with_metadata(
        columnar_root,
        model_id=model_id,
        layer=layer,
        creator_id=creator_id,
        decode_token_ids=decode_token_ids,
        created_at=created_at,
        activation_id_prefix=activation_id_prefix,
        pad_token_id=pad_token_id,
        dedup_activation_rows=dedup_activation_rows,
        drop_zero_activation_rows=drop_zero_activation_rows,
    )
    return records


def import_saedashboard_columnar_activations_local_db_with_connection(
    connection: Any,
    columnar_root: Path | str,
    *,
    model_id: str,
    layer: str,
    creator_id: str,
    decode_token_ids: Callable[[list[int]], list[str]],
    created_at: datetime | str | None = None,
    activation_id_prefix: str = "columnar-activation",
    pad_token_id: int | None = None,
    chunk_size: int = 65000,
    use_copy: bool = True,
    use_stage_table: bool = True,
    dedup_activation_rows: bool = False,
    drop_zero_activation_rows: bool = False,
) -> NeuronpediaLocalImportSummary:
    """Import SAEDashboard columnar activation_rows artifacts into Activation, falling back to sequence_rows."""

    import_start_time = time.perf_counter()
    hygiene_counters: dict[str, int] = {}

    def _hygiene_row_counts(counter_name: str, enabled: bool) -> dict[str, int]:
        if not enabled:
            return {}
        return {"Activation": hygiene_counters.get(counter_name, 0)}

    if use_copy:
        record_batches, processed_files = (
            _iter_saedashboard_columnar_activation_record_batches_with_metadata(
                columnar_root,
                model_id=model_id,
                layer=layer,
                creator_id=creator_id,
                decode_token_ids=decode_token_ids,
                created_at=created_at,
                activation_id_prefix=activation_id_prefix,
                pad_token_id=pad_token_id,
                output_batch_size=chunk_size,
                dedup_activation_rows=dedup_activation_rows,
                drop_zero_activation_rows=drop_zero_activation_rows,
                hygiene_counters=hygiene_counters,
            )
        )
        copy_substage_seconds: dict[str, float] = {}
        bundle_count, imported_count, load_seconds, import_seconds = (
            import_arrow_record_batches_copy_local_db(
                connection,
                "Activation",
                SAEDASHBOARD_COLUMNAR_ACTIVATION_COPY_COLUMNS,
                record_batches,
                chunk_size=chunk_size,
                substage_seconds=copy_substage_seconds,
                use_stage_table=use_stage_table,
            )
        )
        return NeuronpediaLocalImportSummary(
            export_root=Path(columnar_root),
            bundle_row_counts={"Activation": bundle_count},
            imported_row_counts={"Activation": imported_count},
            processed_files=processed_files,
            elapsed_seconds=time.perf_counter() - import_start_time,
            table_load_seconds={"Activation": load_seconds},
            table_import_seconds={"Activation": import_seconds},
            table_import_substage_seconds={"Activation": copy_substage_seconds},
            deduplicated_row_counts=_hygiene_row_counts(
                "deduplicated_rows", dedup_activation_rows
            ),
            zero_dropped_row_counts=_hygiene_row_counts(
                "zero_rows_dropped", drop_zero_activation_rows
            ),
        )

    records, processed_files, load_seconds = (
        _build_saedashboard_columnar_activation_records_with_metadata(
            columnar_root,
            model_id=model_id,
            layer=layer,
            creator_id=creator_id,
            decode_token_ids=decode_token_ids,
            created_at=created_at,
            activation_id_prefix=activation_id_prefix,
            pad_token_id=pad_token_id,
            dedup_activation_rows=dedup_activation_rows,
            drop_zero_activation_rows=drop_zero_activation_rows,
            hygiene_counters=hygiene_counters,
        )
    )
    table_import_start_time = time.perf_counter()
    imported_count = import_records_local_db(
        connection,
        "Activation",
        records,
        chunk_size=chunk_size,
    )
    import_seconds = time.perf_counter() - table_import_start_time
    return NeuronpediaLocalImportSummary(
        export_root=Path(columnar_root),
        bundle_row_counts={"Activation": len(records)},
        imported_row_counts={"Activation": imported_count},
        processed_files=processed_files,
        elapsed_seconds=time.perf_counter() - import_start_time,
        table_load_seconds={"Activation": load_seconds},
        table_import_seconds={"Activation": import_seconds},
        deduplicated_row_counts=_hygiene_row_counts(
            "deduplicated_rows", dedup_activation_rows
        ),
        zero_dropped_row_counts=_hygiene_row_counts(
            "zero_rows_dropped", drop_zero_activation_rows
        ),
    )


def _set_optional_record_field(
    record: dict[str, Any], field_name: str, value: Any
) -> None:
    if value is not None:
        record[field_name] = value


def build_saedashboard_columnar_metadata_records(
    *,
    model_id: str,
    source_set_name: str,
    source_id: str,
    creator_id: str,
    creator_name: str = "",
    release_name: str | None = None,
    release_description: str = "",
    release_description_short: str | None = None,
    release_urls: Iterable[str] | None = None,
    model_display_name: str | None = None,
    model_display_name_short: str | None = None,
    model_layers: int = 0,
    model_neurons_per_layer: int = 0,
    model_owner: str = "",
    model_visibility: str = "PUBLIC",
    model_instruct: bool | None = None,
    source_set_description: str = "",
    source_set_type: str = "",
    source_set_visibility: str = "PUBLIC",
    source_visibility: str = "PUBLIC",
    source_dataset: str | None = None,
    source_hf_repo_id: str | None = None,
    source_hf_folder_id: str | None = None,
    source_saelens_config: str | None = None,
    source_saelens_release: str | None = None,
    source_saelens_sae_id: str | None = None,
    num_prompts: int | None = None,
    num_tokens_in_prompt: int | None = None,
    created_at: datetime | str | None = None,
    updated_at: datetime | str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build metadata/source rows needed before importing SAEDashboard columnar rows."""

    created_at_record_value = _created_at_value(created_at)
    updated_at_record_value = _created_at_value(
        updated_at if updated_at is not None else created_at_record_value
    )
    display_name = model_display_name or model_id
    display_name_short = model_display_name_short or display_name
    table_records: dict[str, list[dict[str, Any]]] = {}

    if release_name is not None:
        release_record: dict[str, Any] = {
            "name": release_name,
            "description": release_description or source_set_description,
            "descriptionShort": release_description_short
            or release_description
            or source_set_description,
            "urls": list(release_urls or ()),
            "creatorNameShort": creator_name,
            "creatorName": creator_name,
            "creatorId": creator_id,
            "defaultSourceSetName": source_set_name,
            "createdAt": created_at_record_value,
            "visibility": "PUBLIC",
            "isNewUi": False,
            "featured": False,
        }
        table_records["SourceRelease"] = [release_record]

    table_records["Model"] = [
        {
            "id": model_id,
            "instruct": model_id.endswith("-it")
            if model_instruct is None
            else model_instruct,
            "creatorId": creator_id,
            "displayNameShort": display_name_short,
            "displayName": display_name,
            "visibility": model_visibility,
            "defaultSourceSetName": source_set_name,
            "defaultGraphSourceSetName": source_set_name,
            "defaultSourceId": source_id,
            "inferenceEnabled": True,
            "layers": model_layers,
            "neuronsPerLayer": model_neurons_per_layer,
            "owner": model_owner,
            "createdAt": created_at_record_value,
            "updatedAt": updated_at_record_value,
        }
    ]
    table_records["SourceSet"] = [
        {
            "modelId": model_id,
            "name": source_set_name,
            "creatorId": creator_id,
            "hasDashboards": True,
            "visibility": source_set_visibility,
            "description": source_set_description,
            "type": source_set_type,
            "creatorName": creator_name,
            "urls": [],
            "releaseName": release_name,
            "defaultOfModelId": model_id,
            "defaultRange": 1,
            "defaultShowBreaks": True,
            "showDfa": False,
            "showCorrelated": False,
            "showHeadAttribution": False,
            "showUmap": False,
            "createdAt": created_at_record_value,
        }
    ]
    source_record: dict[str, Any] = {
        "id": source_id,
        "modelId": model_id,
        "setName": source_set_name,
        "creatorId": creator_id,
        "hasDashboards": True,
        "inferenceEnabled": False,
        "inferenceHosts": [],
        "visibility": source_visibility,
        "defaultOfModelId": model_id,
        "hasUmap": False,
        "hasUmapLogSparsity": False,
        "hasUmapClusters": False,
        "createdAt": created_at_record_value,
    }
    _set_optional_record_field(source_record, "saelensConfig", source_saelens_config)
    _set_optional_record_field(source_record, "saelensRelease", source_saelens_release)
    _set_optional_record_field(source_record, "saelensSaeId", source_saelens_sae_id)
    _set_optional_record_field(source_record, "hfRepoId", source_hf_repo_id)
    _set_optional_record_field(source_record, "hfFolderId", source_hf_folder_id)
    _set_optional_record_field(source_record, "num_prompts", num_prompts)
    _set_optional_record_field(
        source_record, "num_tokens_in_prompt", num_tokens_in_prompt
    )
    _set_optional_record_field(source_record, "dataset", source_dataset)
    table_records["Source"] = [source_record]
    return table_records


def import_saedashboard_columnar_metadata_local_db_with_connection(
    connection: Any,
    columnar_root: Path | str,
    *,
    model_id: str,
    source_set_name: str,
    source_id: str,
    creator_id: str,
    chunk_size: int = 65000,
    **metadata_kwargs: Any,
) -> NeuronpediaLocalImportSummary:
    """Import metadata/source rows needed by a SAEDashboard columnar artifact."""

    import_start_time = time.perf_counter()
    table_records = build_saedashboard_columnar_metadata_records(
        model_id=model_id,
        source_set_name=source_set_name,
        source_id=source_id,
        creator_id=creator_id,
        **metadata_kwargs,
    )
    bundle_row_counts: dict[str, int] = {}
    imported_row_counts: dict[str, int] = {}
    table_load_seconds: dict[str, float] = {}
    table_import_seconds: dict[str, float] = {}
    inserted_model_ids: set[str] = set()
    deferred_model_fk_updates: list[dict[str, Any]] = []

    for table_name, records in table_records.items():
        table_load_seconds[table_name] = 0.0
        bundle_row_counts[table_name] = len(records)
        table_import_start_time = time.perf_counter()
        if table_name == "Model":
            imported_count, inserted_ids, model_fk_updates = (
                import_model_records_local_db(
                    connection,
                    records,
                    chunk_size=chunk_size,
                )
            )
            inserted_model_ids.update(inserted_ids)
            deferred_model_fk_updates.extend(model_fk_updates)
        else:
            imported_count = import_records_local_db(
                connection,
                table_name,
                records,
                chunk_size=chunk_size,
            )
        table_import_seconds[table_name] = time.perf_counter() - table_import_start_time
        imported_row_counts[table_name] = imported_count

    apply_deferred_model_fk_updates(
        connection, deferred_model_fk_updates, inserted_model_ids
    )

    return NeuronpediaLocalImportSummary(
        export_root=Path(columnar_root),
        bundle_row_counts=bundle_row_counts,
        imported_row_counts=imported_row_counts,
        processed_files=[],
        elapsed_seconds=time.perf_counter() - import_start_time,
        table_load_seconds=table_load_seconds,
        table_import_seconds=table_import_seconds,
    )


def build_saedashboard_columnar_neuron_records(
    columnar_root: Path | str,
    *,
    model_id: str,
    layer: str,
    creator_id: str,
    decode_token_ids: Callable[[list[int]], list[str]],
    created_at: datetime | str | None = None,
    hook_name: str | None = None,
    source_set_name: str | None = None,
) -> list[dict[str, Any]]:
    """Build legacy-compatible Neuron rows from SAEDashboard columnar feature tables."""

    records, _, _ = _build_saedashboard_columnar_neuron_records_with_metadata(
        columnar_root,
        model_id=model_id,
        layer=layer,
        creator_id=creator_id,
        decode_token_ids=decode_token_ids,
        created_at=created_at,
        hook_name=hook_name,
        source_set_name=source_set_name,
    )
    return records


def import_saedashboard_columnar_neurons_local_db_with_connection(
    connection: Any,
    columnar_root: Path | str,
    *,
    model_id: str,
    layer: str,
    creator_id: str,
    decode_token_ids: Callable[[list[int]], list[str]],
    created_at: datetime | str | None = None,
    hook_name: str | None = None,
    source_set_name: str | None = None,
    chunk_size: int = 65000,
) -> NeuronpediaLocalImportSummary:
    """Import SAEDashboard columnar feature-level tables into the Neuron table."""

    import_start_time = time.perf_counter()
    records, processed_files, load_seconds = (
        _build_saedashboard_columnar_neuron_records_with_metadata(
            columnar_root,
            model_id=model_id,
            layer=layer,
            creator_id=creator_id,
            decode_token_ids=decode_token_ids,
            created_at=created_at,
            hook_name=hook_name,
            source_set_name=source_set_name,
        )
    )
    table_import_start_time = time.perf_counter()
    imported_count = import_records_local_db(
        connection,
        "Neuron",
        records,
        chunk_size=chunk_size,
    )
    import_seconds = time.perf_counter() - table_import_start_time
    return NeuronpediaLocalImportSummary(
        export_root=Path(columnar_root),
        bundle_row_counts={"Neuron": len(records)},
        imported_row_counts={"Neuron": imported_count},
        processed_files=processed_files,
        elapsed_seconds=time.perf_counter() - import_start_time,
        table_load_seconds={"Neuron": load_seconds},
        table_import_seconds={"Neuron": import_seconds},
    )


def _combine_local_import_summaries(
    export_root: Path | str,
    summaries: Iterable[NeuronpediaLocalImportSummary],
    *,
    elapsed_seconds: float,
) -> NeuronpediaLocalImportSummary:
    bundle_row_counts: dict[str, int] = {}
    imported_row_counts: dict[str, int] = {}
    processed_files: list[Path] = []
    table_load_seconds: dict[str, float] = {}
    table_import_seconds: dict[str, float] = {}
    table_import_substage_seconds: dict[str, dict[str, float]] = {}
    deduplicated_row_counts: dict[str, int] = {}
    zero_dropped_row_counts: dict[str, int] = {}

    for summary in summaries:
        for table_name, row_count in summary.bundle_row_counts.items():
            bundle_row_counts[table_name] = (
                bundle_row_counts.get(table_name, 0) + row_count
            )
        for table_name, row_count in summary.imported_row_counts.items():
            imported_row_counts[table_name] = (
                imported_row_counts.get(table_name, 0) + row_count
            )
        for table_name, seconds in summary.table_load_seconds.items():
            table_load_seconds[table_name] = (
                table_load_seconds.get(table_name, 0.0) + seconds
            )
        for table_name, seconds in summary.table_import_seconds.items():
            table_import_seconds[table_name] = (
                table_import_seconds.get(table_name, 0.0) + seconds
            )
        for table_name, substages in summary.table_import_substage_seconds.items():
            combined_substages = table_import_substage_seconds.setdefault(
                table_name, {}
            )
            for substage_name, seconds in substages.items():
                combined_substages[substage_name] = (
                    combined_substages.get(substage_name, 0.0) + seconds
                )
        for table_name, row_count in summary.deduplicated_row_counts.items():
            deduplicated_row_counts[table_name] = (
                deduplicated_row_counts.get(table_name, 0) + row_count
            )
        for table_name, row_count in summary.zero_dropped_row_counts.items():
            zero_dropped_row_counts[table_name] = (
                zero_dropped_row_counts.get(table_name, 0) + row_count
            )
        processed_files.extend(summary.processed_files)

    return NeuronpediaLocalImportSummary(
        export_root=Path(export_root),
        bundle_row_counts=bundle_row_counts,
        imported_row_counts=imported_row_counts,
        processed_files=processed_files,
        elapsed_seconds=elapsed_seconds,
        table_load_seconds=table_load_seconds,
        table_import_seconds=table_import_seconds,
        table_import_substage_seconds=table_import_substage_seconds,
        deduplicated_row_counts=deduplicated_row_counts,
        zero_dropped_row_counts=zero_dropped_row_counts,
    )


def import_saedashboard_columnar_bundle_local_db_with_connection(
    connection: Any,
    columnar_root: Path | str,
    *,
    model_id: str,
    source_set_name: str,
    source_id: str,
    creator_id: str,
    decode_token_ids: Callable[[list[int]], list[str]],
    created_at: datetime | str | None = None,
    activation_id_prefix: str = "columnar-activation",
    pad_token_id: int | None = None,
    hook_name: str | None = None,
    chunk_size: int = 65000,
    activation_use_stage_table: bool = True,
    dedup_activation_rows: bool = False,
    drop_zero_activation_rows: bool = False,
    **metadata_kwargs: Any,
) -> NeuronpediaLocalImportSummary:
    """Import SAEDashboard columnar metadata, Neuron rows, and Activation rows in dependency order.

    ``dedup_activation_rows`` and ``drop_zero_activation_rows`` are opt-in Activation
    import-hygiene filters; both default to off, preserving the existing import
    contract exactly.
    """

    import_start_time = time.perf_counter()
    shared_metadata_kwargs = {"created_at": created_at, **metadata_kwargs}
    metadata_summary = import_saedashboard_columnar_metadata_local_db_with_connection(
        connection,
        columnar_root,
        model_id=model_id,
        source_set_name=source_set_name,
        source_id=source_id,
        creator_id=creator_id,
        chunk_size=chunk_size,
        **shared_metadata_kwargs,
    )
    neuron_summary = import_saedashboard_columnar_neurons_local_db_with_connection(
        connection,
        columnar_root,
        model_id=model_id,
        layer=source_id,
        creator_id=creator_id,
        decode_token_ids=decode_token_ids,
        created_at=created_at,
        hook_name=hook_name,
        source_set_name=source_set_name,
        chunk_size=chunk_size,
    )
    activation_summary = (
        import_saedashboard_columnar_activations_local_db_with_connection(
            connection,
            columnar_root,
            model_id=model_id,
            layer=source_id,
            creator_id=creator_id,
            decode_token_ids=decode_token_ids,
            created_at=created_at,
            activation_id_prefix=activation_id_prefix,
            pad_token_id=pad_token_id,
            chunk_size=chunk_size,
            use_stage_table=activation_use_stage_table,
            dedup_activation_rows=dedup_activation_rows,
            drop_zero_activation_rows=drop_zero_activation_rows,
        )
    )
    return _combine_local_import_summaries(
        columnar_root,
        (metadata_summary, neuron_summary, activation_summary),
        elapsed_seconds=time.perf_counter() - import_start_time,
    )


def import_saedashboard_columnar_bundle_local_db(
    columnar_root: Path | str,
    *,
    local_db_url: str,
    model_id: str,
    source_set_name: str,
    source_id: str,
    creator_id: str,
    decode_token_ids: Callable[[list[int]], list[str]],
    created_at: datetime | str | None = None,
    activation_id_prefix: str = "columnar-activation",
    pad_token_id: int | None = None,
    hook_name: str | None = None,
    chunk_size: int = 65000,
    activation_use_stage_table: bool = True,
    dedup_activation_rows: bool = False,
    drop_zero_activation_rows: bool = False,
    **metadata_kwargs: Any,
) -> NeuronpediaLocalImportSummary:
    """Import a SAEDashboard columnar artifact into a local Neuronpedia DB.

    ``dedup_activation_rows`` and ``drop_zero_activation_rows`` are opt-in Activation
    import-hygiene filters; both default to off, preserving the existing import
    contract exactly.
    """

    psycopg = importlib.import_module("psycopg")
    columnar_root_path = Path(columnar_root)
    if not columnar_root_path.exists():
        raise NeuronpediaLocalDBImportError(
            f"Columnar artifact root does not exist: {columnar_root_path}"
        )

    with psycopg.connect(local_db_url) as connection:
        summary = import_saedashboard_columnar_bundle_local_db_with_connection(
            connection,
            columnar_root_path,
            model_id=model_id,
            source_set_name=source_set_name,
            source_id=source_id,
            creator_id=creator_id,
            decode_token_ids=decode_token_ids,
            created_at=created_at,
            activation_id_prefix=activation_id_prefix,
            pad_token_id=pad_token_id,
            hook_name=hook_name,
            chunk_size=chunk_size,
            activation_use_stage_table=activation_use_stage_table,
            dedup_activation_rows=dedup_activation_rows,
            drop_zero_activation_rows=drop_zero_activation_rows,
            **metadata_kwargs,
        )
        connection.commit()
    return summary


def summarize_saedashboard_columnar_artifacts(
    columnar_root: Path | str,
) -> NeuronpediaExportBundleSummary:
    """Summarize SAEDashboard columnar artifact row counts from feature-batch manifests."""

    columnar_root_path = Path(columnar_root)
    summary_start_time = time.perf_counter()
    table_row_counts: dict[str, int] = {}
    processed_files: list[Path] = []
    table_load_seconds: dict[str, float] = {}
    for manifest_path in discover_saedashboard_columnar_manifest_paths(
        columnar_root_path
    ):
        manifest = _load_json_file(manifest_path)
        row_counts = manifest.get("row_counts")
        tables = manifest.get("tables")
        if not isinstance(row_counts, dict) or not isinstance(tables, dict):
            raise NeuronpediaLocalDBImportError(
                f"Columnar manifest {manifest_path} must contain row_counts and tables objects."
            )
        for table_name, relative_table_path in sorted(tables.items()):
            table_path = manifest_path.parent / str(relative_table_path)
            if not table_path.exists():
                raise NeuronpediaLocalDBImportError(
                    f"Columnar table {table_path} referenced by {manifest_path} does not exist."
                )
            load_start_time = time.perf_counter()
            record_count = _count_columnar_table_records(table_path)
            table_load_seconds[table_name] = table_load_seconds.get(table_name, 0.0) + (
                time.perf_counter() - load_start_time
            )
            expected_count = row_counts.get(table_name)
            if expected_count is not None and int(expected_count) != record_count:
                raise NeuronpediaLocalDBImportError(
                    f"Columnar table {table_path} has {record_count} rows, but manifest expects {expected_count}."
                )
            table_row_counts[table_name] = (
                table_row_counts.get(table_name, 0) + record_count
            )
            processed_files.append(table_path)

    return NeuronpediaExportBundleSummary(
        export_root=columnar_root_path,
        table_row_counts=table_row_counts,
        processed_files=processed_files,
        elapsed_seconds=time.perf_counter() - summary_start_time,
        table_load_seconds=table_load_seconds,
    )


def summarize_neuronpedia_export_bundle(
    export_root: Path | str,
) -> NeuronpediaExportBundleSummary:
    export_root_path = Path(export_root)
    if not export_root_path.exists():
        raise NeuronpediaLocalDBImportError(
            f"Export root does not exist: {export_root_path}"
        )

    table_paths = export_bundle_jsonl_paths(export_root_path)
    if not table_paths:
        raise NeuronpediaLocalDBImportError(
            f"No importable Neuronpedia bundle files found under {export_root_path}"
        )

    return _summarize_table_paths(export_root_path, table_paths, _count_jsonl_records)


def summarize_neuronpedia_export_bundle_arrow(
    export_root: Path | str,
) -> NeuronpediaExportBundleSummary:
    export_root_path = Path(export_root)
    if not export_root_path.exists():
        raise NeuronpediaLocalDBImportError(
            f"Export root does not exist: {export_root_path}"
        )

    table_paths = export_bundle_arrow_paths(export_root_path)
    if not table_paths:
        raise NeuronpediaLocalDBImportError(
            f"No Arrow sidecars found under {export_root_path}"
        )

    return _summarize_table_paths(export_root_path, table_paths, count_arrow_records)


def summarize_neuronpedia_export_bundle_parquet(
    export_root: Path | str,
) -> NeuronpediaExportBundleSummary:
    export_root_path = Path(export_root)
    if not export_root_path.exists():
        raise NeuronpediaLocalDBImportError(
            f"Export root does not exist: {export_root_path}"
        )

    table_paths = export_bundle_parquet_paths(export_root_path)
    if not table_paths:
        raise NeuronpediaLocalDBImportError(
            f"No Parquet sidecars found under {export_root_path}"
        )

    return _summarize_table_paths(export_root_path, table_paths, count_parquet_records)


def _summarize_table_paths(
    export_root: Path,
    table_paths: list[tuple[str, Path]],
    count_records: Any,
) -> NeuronpediaExportBundleSummary:
    summary_start_time = time.perf_counter()
    table_row_counts: dict[str, int] = {}
    processed_files: list[Path] = []
    table_load_seconds: dict[str, float] = {}
    for table_name, path in table_paths:
        load_start_time = time.perf_counter()
        record_count = int(count_records(path))
        table_load_seconds[table_name] = table_load_seconds.get(table_name, 0.0) + (
            time.perf_counter() - load_start_time
        )
        table_row_counts[table_name] = (
            table_row_counts.get(table_name, 0) + record_count
        )
        processed_files.append(path)

    return NeuronpediaExportBundleSummary(
        export_root=export_root,
        table_row_counts=table_row_counts,
        processed_files=processed_files,
        elapsed_seconds=time.perf_counter() - summary_start_time,
        table_load_seconds=table_load_seconds,
    )


def compare_neuronpedia_export_bundle_summaries(
    reference_summary: NeuronpediaExportBundleSummary,
    candidate_summary: NeuronpediaExportBundleSummary,
) -> NeuronpediaBundleSummaryParity:
    row_count_mismatches: dict[str, dict[str, int]] = {}
    all_tables = sorted(
        set(reference_summary.table_row_counts)
        | set(candidate_summary.table_row_counts)
    )
    for table_name in all_tables:
        reference_rows = int(reference_summary.table_row_counts.get(table_name, 0))
        candidate_rows = int(candidate_summary.table_row_counts.get(table_name, 0))
        if reference_rows != candidate_rows:
            row_count_mismatches[table_name] = {
                "reference_rows": reference_rows,
                "candidate_rows": candidate_rows,
            }

    return NeuronpediaBundleSummaryParity(
        export_root=reference_summary.export_root,
        reference_row_counts=dict(reference_summary.table_row_counts),
        candidate_row_counts=dict(candidate_summary.table_row_counts),
        row_count_mismatches=row_count_mismatches,
    )


def compare_neuronpedia_export_bundle_to_import_summary(
    bundle_summary: NeuronpediaExportBundleSummary,
    import_summary: NeuronpediaLocalImportSummary,
) -> NeuronpediaBundleImportParity:
    """Compare bundle row counts with one local import attempt.

    Inserted rows can be lower than bundle rows when the target database
    already contains matching records and the importer uses `ON CONFLICT DO
    NOTHING`. The attempted row counts should still match the bundle summary.
    """

    row_count_mismatches: dict[str, dict[str, int]] = {}
    skipped_existing_row_counts: dict[str, int] = {}
    all_tables = sorted(
        set(bundle_summary.table_row_counts) | set(import_summary.bundle_row_counts)
    )
    for table_name in all_tables:
        bundle_rows = int(bundle_summary.table_row_counts.get(table_name, 0))
        attempted_rows = int(import_summary.bundle_row_counts.get(table_name, 0))
        imported_rows = int(import_summary.imported_row_counts.get(table_name, 0))
        if bundle_rows != attempted_rows:
            row_count_mismatches[table_name] = {
                "bundle_rows": bundle_rows,
                "attempted_rows": attempted_rows,
            }
        skipped_existing_rows = max(0, attempted_rows - imported_rows)
        if skipped_existing_rows > 0:
            skipped_existing_row_counts[table_name] = skipped_existing_rows

    return NeuronpediaBundleImportParity(
        export_root=bundle_summary.export_root,
        bundle_row_counts=dict(bundle_summary.table_row_counts),
        import_attempted_row_counts=dict(import_summary.bundle_row_counts),
        imported_row_counts=dict(import_summary.imported_row_counts),
        row_count_mismatches=row_count_mismatches,
        skipped_existing_row_counts=skipped_existing_row_counts,
    )


def _normalize_local_db_parity_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return _created_at_value(value)
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_local_db_parity_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_local_db_parity_value(item) for item in value]
    return value


def _cursor_description_column_names(cursor: Any) -> list[str]:
    column_names: list[str] = []
    for description_item in cursor.description or ():
        if hasattr(description_item, "name"):
            column_names.append(str(description_item.name))
        else:
            column_names.append(str(description_item[0]))
    return column_names


def _cursor_rows_to_dicts(cursor: Any, rows: Iterable[Any]) -> list[dict[str, Any]]:
    row_list = list(rows)
    if not row_list:
        return []
    if isinstance(row_list[0], Mapping):
        return [
            {
                str(key): _normalize_local_db_parity_value(value)
                for key, value in cast(Mapping[Any, Any], row).items()
            }
            for row in row_list
        ]

    column_names = _cursor_description_column_names(cursor)
    return [
        {
            column_name: _normalize_local_db_parity_value(value)
            for column_name, value in zip(column_names, row, strict=False)
        }
        for row in row_list
    ]


def _first_cursor_row_value(row: Any) -> Any:
    if isinstance(row, Mapping):
        return next(iter(row.values()))
    return row[0]


def capture_neuronpedia_local_db_snapshot(
    connection: Any,
    *,
    label: str,
    table_sample_specs: Iterable[
        NeuronpediaLocalDBTableSampleSpec
    ] = DEFAULT_LOCAL_DB_PARITY_TABLE_SAMPLE_SPECS,
    sample_limit: int = 5,
) -> NeuronpediaLocalDBSnapshot:
    """Capture stable local DB row counts and samples for eager/lazy import parity."""

    if sample_limit < 1:
        raise ValueError("sample_limit must be >= 1")

    psycopg_sql = importlib.import_module("psycopg.sql")
    snapshot_start_time = time.perf_counter()
    table_row_counts: dict[str, int] = {}
    table_samples: dict[str, list[dict[str, Any]]] = {}

    with connection.cursor() as cursor:
        for spec in table_sample_specs:
            count_query = psycopg_sql.SQL(
                "SELECT COUNT(*) AS row_count FROM {table_name}"
            ).format(
                table_name=psycopg_sql.Identifier(spec.table_name),
            )
            cursor.execute(count_query)
            count_row = cursor.fetchone()
            table_row_counts[spec.table_name] = int(_first_cursor_row_value(count_row))

            column_list = psycopg_sql.SQL(", ").join(
                psycopg_sql.Identifier(column_name)
                for column_name in spec.sample_columns
            )
            order_by = psycopg_sql.SQL(", ").join(
                psycopg_sql.Identifier(column_name) for column_name in spec.order_by
            )
            sample_query = psycopg_sql.SQL(
                "SELECT {column_list} FROM {table_name} ORDER BY {order_by} LIMIT %s"
            ).format(
                column_list=column_list,
                table_name=psycopg_sql.Identifier(spec.table_name),
                order_by=order_by,
            )
            cursor.execute(sample_query, (sample_limit,))
            table_samples[spec.table_name] = _cursor_rows_to_dicts(
                cursor, cursor.fetchall()
            )

    return NeuronpediaLocalDBSnapshot(
        label=label,
        table_row_counts=table_row_counts,
        table_samples=table_samples,
        elapsed_seconds=time.perf_counter() - snapshot_start_time,
    )


def compare_neuronpedia_local_db_snapshots(
    reference_snapshot: NeuronpediaLocalDBSnapshot,
    candidate_snapshot: NeuronpediaLocalDBSnapshot,
) -> NeuronpediaLocalDBParity:
    """Compare two local DB snapshots from eager/export-bundle and lazy/columnar imports."""

    row_count_mismatches: dict[str, dict[str, int]] = {}
    all_count_tables = sorted(
        set(reference_snapshot.table_row_counts)
        | set(candidate_snapshot.table_row_counts)
    )
    for table_name in all_count_tables:
        reference_rows = int(reference_snapshot.table_row_counts.get(table_name, 0))
        candidate_rows = int(candidate_snapshot.table_row_counts.get(table_name, 0))
        if reference_rows != candidate_rows:
            row_count_mismatches[table_name] = {
                "reference_rows": reference_rows,
                "candidate_rows": candidate_rows,
            }

    sample_mismatches: dict[str, dict[str, list[dict[str, Any]]]] = {}
    all_sample_tables = sorted(
        set(reference_snapshot.table_samples) | set(candidate_snapshot.table_samples)
    )
    for table_name in all_sample_tables:
        reference_sample_rows = reference_snapshot.table_samples.get(table_name, [])
        candidate_sample_rows = candidate_snapshot.table_samples.get(table_name, [])
        if reference_sample_rows != candidate_sample_rows:
            sample_mismatches[table_name] = {
                "reference_rows": reference_sample_rows,
                "candidate_rows": candidate_sample_rows,
            }

    return NeuronpediaLocalDBParity(
        reference_label=reference_snapshot.label,
        candidate_label=candidate_snapshot.label,
        reference_row_counts=dict(reference_snapshot.table_row_counts),
        candidate_row_counts=dict(candidate_snapshot.table_row_counts),
        row_count_mismatches=row_count_mismatches,
        sample_mismatches=sample_mismatches,
    )


def _is_arrow_string_like(pyarrow: Any, arrow_type: Any) -> bool:
    return any(
        predicate(arrow_type)
        for predicate in (
            pyarrow.types.is_string,
            pyarrow.types.is_large_string,
            getattr(pyarrow.types, "is_string_view", lambda _value: False),
        )
    )


def _parse_iso_timestamp_for_copy(value: str | None) -> datetime | None:
    if value is None:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def cast_arrow_column_for_postgres_copy(
    pyarrow: Any, pyarrow_compute: Any, column: Any, pg_type: str
) -> Any:
    if pg_type == "text":
        return pyarrow_compute.cast(column, pyarrow.string())
    if pg_type == "text[]":
        return pyarrow_compute.cast(column, pyarrow.list_(pyarrow.string()))
    if pg_type == "double precision":
        return pyarrow_compute.cast(column, pyarrow.float64())
    if pg_type == "double precision[]":
        return pyarrow_compute.cast(column, pyarrow.list_(pyarrow.float64()))
    if pg_type == "integer":
        return pyarrow_compute.cast(column, pyarrow.int32())
    if pg_type == "bigint":
        return pyarrow_compute.cast(column, pyarrow.int64())
    if pg_type == "boolean":
        return pyarrow_compute.cast(column, pyarrow.bool_())
    if pg_type.startswith("timestamp(") and pg_type.endswith("without time zone"):
        values = column.to_pylist()
        return pyarrow.array(
            [
                _parse_iso_timestamp_for_copy(cast(str | None, value))
                for value in values
            ],
            type=pyarrow.timestamp("us"),
        )
    raise NeuronpediaLocalDBImportError(
        f"Unsupported Postgres COPY type for Arrow binary import: {pg_type}"
    )


def sanitize_arrow_column_for_copy(
    pyarrow: Any, pyarrow_compute: Any, column: Any
) -> Any:
    if _is_arrow_string_like(pyarrow, column.type):
        return pyarrow_compute.replace_substring(column, "\x00", " ")
    if pyarrow.types.is_list(column.type) and _is_arrow_string_like(
        pyarrow, column.type.value_type
    ):
        flattened_values = pyarrow_compute.list_flatten(column)
        if (
            flattened_values.null_count == 0
            and not pyarrow_compute.any(
                pyarrow_compute.match_substring(flattened_values, "\x00")
            ).as_py()
        ):
            return column
        values = column.to_pylist()
        sanitized_values = [
            [
                " " if item is None else cast(str, item).replace("\x00", " ")
                for item in row
            ]
            if row is not None
            else None
            for row in values
        ]
        return pyarrow.array(sanitized_values, type=pyarrow.list_(pyarrow.string()))
    return column


def align_arrow_batch_for_postgres_copy(
    batch: Any, column_defs: list[tuple[str, str]]
) -> Any:
    pyarrow, pyarrow_compute, _ = _load_pyarrow_modules()

    aligned_columns: list[Any] = []
    batch_schema_names = set(batch.schema.names)
    for column_name, pg_type in column_defs:
        if column_name in batch_schema_names:
            column = batch.column(batch.schema.get_field_index(column_name))
        else:
            column = pyarrow.nulls(batch.num_rows)
        aligned_column = cast_arrow_column_for_postgres_copy(
            pyarrow, pyarrow_compute, column, pg_type
        )
        aligned_columns.append(
            sanitize_arrow_column_for_copy(pyarrow, pyarrow_compute, aligned_column)
        )

    return pyarrow.RecordBatch.from_arrays(
        aligned_columns,
        names=[column_name for column_name, _ in column_defs],
    )


def resolve_table_column_defs(
    cursor: Any, table_name: str, available_columns: Iterable[str]
) -> list[tuple[str, str]]:
    column_names = list(dict.fromkeys(available_columns))
    cursor.execute(
        """
        SELECT a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod)
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relname = %s
          AND n.nspname = current_schema()
          AND a.attnum > 0
          AND NOT a.attisdropped
          AND a.attname = ANY(%s)
        ORDER BY a.attnum
        """,
        (table_name, column_names),
    )
    return [(row[0], row[1]) for row in cursor.fetchall()]


# Postgres caps the total size of jsonb array elements at 268,435,455 bytes; keep the
# serialized `jsonb_to_recordset` payloads comfortably below that so wide-context
# activation records (e.g. 2048 features x 319-token prompts) import without tripping
# `ProgramLimitExceeded`.
MAX_JSONB_PAYLOAD_BYTES = 192 * 1024 * 1024


def iter_jsonb_record_payloads(
    records: list[dict[str, Any]],
    *,
    chunk_size: int,
    max_payload_bytes: int = MAX_JSONB_PAYLOAD_BYTES,
) -> Any:
    """Yield serialized jsonb payload strings for record chunks.

    Chunks first follow the row-count ``chunk_size``; any chunk whose serialized payload
    exceeds ``max_payload_bytes`` is recursively halved (preserving record order) so each
    executed statement stays under the Postgres jsonb array size limit. ``json.dumps``
    escapes non-ASCII by default, so ``len(payload)`` equals the byte length.
    """

    pending: list[list[dict[str, Any]]] = [
        records[start_idx : start_idx + chunk_size]
        for start_idx in range(0, len(records), max(1, chunk_size))
    ]
    pending.reverse()
    while pending:
        chunk = pending.pop()
        payload = json.dumps(chunk).replace("\\u0000", " ")
        if len(payload) > max_payload_bytes and len(chunk) > 1:
            midpoint = len(chunk) // 2
            pending.append(chunk[midpoint:])
            pending.append(chunk[:midpoint])
            continue
        if len(payload) > max_payload_bytes:
            raise NeuronpediaLocalDBImportError(
                "A single record serializes to "
                f"{len(payload)} bytes, exceeding the jsonb payload limit of "
                f"{max_payload_bytes} bytes."
            )
        yield payload


def import_records_local_db(
    connection: Any,
    table_name: str,
    records: list[dict[str, Any]],
    *,
    chunk_size: int = 65000,
) -> int:
    if not records:
        return 0

    psycopg_sql = importlib.import_module("psycopg.sql")

    available_columns: list[str] = []
    for record in records:
        for key in record:
            if key not in available_columns:
                available_columns.append(key)

    with connection.cursor() as cursor:
        column_defs = resolve_table_column_defs(cursor, table_name, available_columns)
        if not column_defs:
            raise NeuronpediaLocalDBImportError(
                f"Could not resolve columns for table {table_name}."
            )
        column_names = [column_name for column_name, _ in column_defs]
        query = psycopg_sql.SQL(
            "INSERT INTO {table_name} ({column_list}) "
            "SELECT {column_list} FROM jsonb_to_recordset(%s::jsonb) AS t({column_defs}) "
            "ON CONFLICT DO NOTHING"
        ).format(
            table_name=psycopg_sql.Identifier(table_name),
            column_list=psycopg_sql.SQL(", ").join(
                psycopg_sql.Identifier(column_name) for column_name in column_names
            ),
            column_defs=psycopg_sql.SQL(", ").join(
                psycopg_sql.SQL("{} {}").format(
                    psycopg_sql.Identifier(column_name),
                    psycopg_sql.SQL(cast(Any, column_type)),
                )
                for column_name, column_type in column_defs
            ),
        )

        imported_rows = 0
        for payload in iter_jsonb_record_payloads(records, chunk_size=chunk_size):
            cursor.execute(query, (payload,))
            if cursor.rowcount > 0:
                imported_rows += cursor.rowcount
    return imported_rows


def prepare_model_records_for_insert(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prepared_records: list[dict[str, Any]] = []
    deferred_updates: list[dict[str, Any]] = []
    for record in records:
        prepared_record = dict(record)
        deferred_update = {"id": record["id"]}
        should_defer = False
        for column_name in DEFERRED_MODEL_FK_COLUMNS:
            column_value = record.get(column_name)
            deferred_update[column_name] = column_value
            if column_name in prepared_record:
                prepared_record[column_name] = None
            should_defer = should_defer or column_value is not None
        prepared_records.append(prepared_record)
        if should_defer:
            deferred_updates.append(deferred_update)
    return prepared_records, deferred_updates


def import_model_records_local_db(
    connection: Any,
    records: list[dict[str, Any]],
    *,
    chunk_size: int = 65000,
) -> tuple[int, set[str], list[dict[str, Any]]]:
    if not records:
        return 0, set(), []

    psycopg_sql = importlib.import_module("psycopg.sql")
    prepared_records, deferred_updates = prepare_model_records_for_insert(records)

    available_columns: list[str] = []
    for record in prepared_records:
        for key in record:
            if key not in available_columns:
                available_columns.append(key)

    inserted_ids: set[str] = set()
    with connection.cursor() as cursor:
        column_defs = resolve_table_column_defs(cursor, "Model", available_columns)
        if not column_defs:
            raise NeuronpediaLocalDBImportError(
                "Could not resolve columns for table Model."
            )
        column_names = [column_name for column_name, _ in column_defs]
        query = psycopg_sql.SQL(
            "INSERT INTO {table_name} ({column_list}) "
            "SELECT {column_list} FROM jsonb_to_recordset(%s::jsonb) AS t({column_defs}) "
            "ON CONFLICT DO NOTHING RETURNING id"
        ).format(
            table_name=psycopg_sql.Identifier("Model"),
            column_list=psycopg_sql.SQL(", ").join(
                psycopg_sql.Identifier(column_name) for column_name in column_names
            ),
            column_defs=psycopg_sql.SQL(", ").join(
                psycopg_sql.SQL("{} {}").format(
                    psycopg_sql.Identifier(column_name),
                    psycopg_sql.SQL(cast(Any, column_type)),
                )
                for column_name, column_type in column_defs
            ),
        )

        for start_idx in range(0, len(prepared_records), chunk_size):
            chunk = prepared_records[start_idx : start_idx + chunk_size]
            payload = json.dumps(chunk).replace("\\u0000", " ")
            cursor.execute(query, (payload,))
            inserted_ids.update(row[0] for row in cursor.fetchall())

    return len(inserted_ids), inserted_ids, deferred_updates


def apply_deferred_model_fk_updates(
    connection: Any,
    deferred_updates: list[dict[str, Any]],
    inserted_ids: set[str],
) -> None:
    if not deferred_updates or not inserted_ids:
        return

    applicable_updates = [
        update for update in deferred_updates if update["id"] in inserted_ids
    ]
    if not applicable_updates:
        return

    with connection.cursor() as cursor:
        cursor.executemany(
            'UPDATE "Model" '
            'SET "defaultSourceSetName" = %(defaultSourceSetName)s, '
            '    "defaultGraphSourceSetName" = %(defaultGraphSourceSetName)s, '
            '    "defaultSourceId" = %(defaultSourceId)s '
            'WHERE "id" = %(id)s',
            applicable_updates,
        )


def import_arrow_records_local_db(
    connection: Any,
    table_name: str,
    path: Path,
    *,
    chunk_size: int = 65000,
) -> tuple[int, int, float, float]:
    _, _, pa_ipc = _load_pyarrow_modules()

    bundle_row_count = 0
    imported_row_count = 0
    load_seconds = 0.0
    import_seconds = 0.0

    with path.open("rb") as handle:
        reader = pa_ipc.RecordBatchFileReader(handle)
        for batch_idx in range(reader.num_record_batches):
            load_start_time = time.perf_counter()
            batch = reader.get_batch(batch_idx)
            records = cast(list[dict[str, Any]], batch.to_pylist())
            load_seconds += time.perf_counter() - load_start_time
            bundle_row_count += len(records)
            for start_idx in range(0, len(records), chunk_size):
                chunk = records[start_idx : start_idx + chunk_size]
                table_import_start_time = time.perf_counter()
                imported_row_count += import_records_local_db(
                    connection,
                    table_name,
                    chunk,
                    chunk_size=chunk_size,
                )
                import_seconds += time.perf_counter() - table_import_start_time

    return bundle_row_count, imported_row_count, load_seconds, import_seconds


def import_parquet_records_local_db(
    connection: Any,
    table_name: str,
    path: Path,
    *,
    chunk_size: int = 65000,
) -> tuple[int, int, float, float]:
    pyarrow_parquet = _load_pyarrow_parquet()

    bundle_row_count = 0
    imported_row_count = 0
    load_seconds = 0.0
    import_seconds = 0.0

    parquet_file = pyarrow_parquet.ParquetFile(str(path))
    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        load_start_time = time.perf_counter()
        records = cast(list[dict[str, Any]], batch.to_pylist())
        load_seconds += time.perf_counter() - load_start_time
        bundle_row_count += len(records)
        table_import_start_time = time.perf_counter()
        imported_row_count += import_records_local_db(
            connection,
            table_name,
            records,
            chunk_size=chunk_size,
        )
        import_seconds += time.perf_counter() - table_import_start_time

    return bundle_row_count, imported_row_count, load_seconds, import_seconds


def import_arrow_record_batches_copy_local_db(
    connection: Any,
    table_name: str,
    available_columns: list[str],
    record_batches: Iterable[Any],
    *,
    chunk_size: int = 65000,
    substage_seconds: dict[str, float] | None = None,
    use_stage_table: bool = True,
) -> tuple[int, int, float, float]:
    pgpq_arrow_encoder = _load_pgpq_arrow_encoder()
    psycopg_sql = importlib.import_module("psycopg.sql")

    bundle_row_count = 0
    imported_row_count = 0
    load_seconds = 0.0
    import_seconds = 0.0

    with connection.cursor() as cursor:
        resolve_columns_start_time = time.perf_counter()
        column_defs = resolve_table_column_defs(cursor, table_name, available_columns)
        if substage_seconds is not None:
            substage_seconds["resolve_columns"] = substage_seconds.get(
                "resolve_columns", 0.0
            ) + (time.perf_counter() - resolve_columns_start_time)
        if not column_defs:
            raise NeuronpediaLocalDBImportError(
                f"Could not resolve columns for table {table_name}."
            )
        column_names = [column_name for column_name, _ in column_defs]
        copy_target_name = table_name
        create_stage_query = None
        insert_query = None
        truncate_query = None
        if use_stage_table:
            copy_target_name = f"__np_stage_{table_name.lower()}_{time.time_ns()}"
            create_stage_query = psycopg_sql.SQL(
                "CREATE TEMP TABLE {stage_table} (LIKE {table_name} INCLUDING DEFAULTS) ON COMMIT DROP"
            ).format(
                stage_table=psycopg_sql.Identifier(copy_target_name),
                table_name=psycopg_sql.Identifier(table_name),
            )
        copy_query = psycopg_sql.SQL(
            "COPY {copy_target} ({column_list}) FROM STDIN WITH (FORMAT BINARY)"
        ).format(
            copy_target=psycopg_sql.Identifier(copy_target_name),
            column_list=psycopg_sql.SQL(", ").join(
                psycopg_sql.Identifier(column_name) for column_name in column_names
            ),
        )
        if use_stage_table:
            insert_query = psycopg_sql.SQL(
                "INSERT INTO {table_name} ({column_list}) SELECT {column_list} FROM {stage_table} ON CONFLICT DO NOTHING"
            ).format(
                table_name=psycopg_sql.Identifier(table_name),
                column_list=psycopg_sql.SQL(", ").join(
                    psycopg_sql.Identifier(column_name) for column_name in column_names
                ),
                stage_table=psycopg_sql.Identifier(copy_target_name),
            )
            truncate_query = psycopg_sql.SQL("TRUNCATE {stage_table}").format(
                stage_table=psycopg_sql.Identifier(copy_target_name)
            )
            assert create_stage_query is not None
            create_stage_start_time = time.perf_counter()
            cursor.execute(create_stage_query)
            if substage_seconds is not None:
                substage_seconds["create_stage_table"] = substage_seconds.get(
                    "create_stage_table", 0.0
                ) + (time.perf_counter() - create_stage_start_time)

        table_import_start_time: float | None = None
        encoder = None
        copy_schema = None
        copy = None
        copy_context = None
        try:
            load_start_time = time.perf_counter()
            record_batch_iterator = iter(record_batches)
            while True:
                next_batch_start_time = time.perf_counter()
                try:
                    batch = next(record_batch_iterator)
                except StopIteration:
                    break
                if substage_seconds is not None:
                    substage_seconds["record_batch_next"] = substage_seconds.get(
                        "record_batch_next", 0.0
                    ) + (time.perf_counter() - next_batch_start_time)
                align_start_time = time.perf_counter()
                aligned_batch = align_arrow_batch_for_postgres_copy(batch, column_defs)
                if substage_seconds is not None:
                    substage_seconds["align_arrow_batch"] = substage_seconds.get(
                        "align_arrow_batch", 0.0
                    ) + (time.perf_counter() - align_start_time)
                load_seconds += time.perf_counter() - load_start_time
                bundle_row_count += batch.num_rows

                if encoder is None:
                    table_import_start_time = time.perf_counter()
                    encoder_setup_start_time = time.perf_counter()
                    copy_schema = aligned_batch.schema
                    encoder = pgpq_arrow_encoder(copy_schema)
                    copy_context = cursor.copy(copy_query)
                    copy = copy_context.__enter__()
                    header_payload = encoder.write_header()
                    if substage_seconds is not None:
                        substage_seconds["pgpq_encoder_setup"] = substage_seconds.get(
                            "pgpq_encoder_setup", 0.0
                        ) + (time.perf_counter() - encoder_setup_start_time)
                    copy_write_start_time = time.perf_counter()
                    copy.write(header_payload)
                    if substage_seconds is not None:
                        substage_seconds["copy_write"] = substage_seconds.get(
                            "copy_write", 0.0
                        ) + (time.perf_counter() - copy_write_start_time)
                elif aligned_batch.schema != copy_schema:
                    raise NeuronpediaLocalDBImportError(
                        f"COPY record batch schema changed while importing table {table_name}."
                    )

                for start_idx in range(0, batch.num_rows, chunk_size):
                    end_idx = min(start_idx + chunk_size, batch.num_rows)
                    batch_slice = aligned_batch.slice(start_idx, end_idx - start_idx)
                    assert copy is not None
                    encode_start_time = time.perf_counter()
                    encoded_batch = encoder.write_batch(batch_slice)
                    if substage_seconds is not None:
                        substage_seconds["pgpq_encode_batch"] = substage_seconds.get(
                            "pgpq_encode_batch", 0.0
                        ) + (time.perf_counter() - encode_start_time)
                    copy_write_start_time = time.perf_counter()
                    copy.write(encoded_batch)
                    if substage_seconds is not None:
                        substage_seconds["copy_write"] = substage_seconds.get(
                            "copy_write", 0.0
                        ) + (time.perf_counter() - copy_write_start_time)
                load_start_time = time.perf_counter()
            if encoder is not None:
                assert copy is not None
                finish_start_time = time.perf_counter()
                finish_payload = encoder.finish()
                if substage_seconds is not None:
                    substage_seconds["pgpq_finish"] = substage_seconds.get(
                        "pgpq_finish", 0.0
                    ) + (time.perf_counter() - finish_start_time)
                copy_write_start_time = time.perf_counter()
                copy.write(finish_payload)
                if substage_seconds is not None:
                    substage_seconds["copy_write"] = substage_seconds.get(
                        "copy_write", 0.0
                    ) + (time.perf_counter() - copy_write_start_time)
        except BaseException as exc:
            if copy_context is not None:
                close_start_time = time.perf_counter()
                copy_context.__exit__(type(exc), exc, exc.__traceback__)
                if substage_seconds is not None:
                    substage_seconds["copy_stream_close"] = substage_seconds.get(
                        "copy_stream_close", 0.0
                    ) + (time.perf_counter() - close_start_time)
            raise
        else:
            if copy_context is not None:
                close_start_time = time.perf_counter()
                copy_context.__exit__(None, None, None)
                if substage_seconds is not None:
                    substage_seconds["copy_stream_close"] = substage_seconds.get(
                        "copy_stream_close", 0.0
                    ) + (time.perf_counter() - close_start_time)

        if encoder is not None:
            assert table_import_start_time is not None
            if use_stage_table:
                assert insert_query is not None
                assert truncate_query is not None
                insert_start_time = time.perf_counter()
                cursor.execute(insert_query)
                if substage_seconds is not None:
                    substage_seconds["insert_from_stage"] = substage_seconds.get(
                        "insert_from_stage", 0.0
                    ) + (time.perf_counter() - insert_start_time)
                if cursor.rowcount > 0:
                    imported_row_count += cursor.rowcount
                truncate_start_time = time.perf_counter()
                cursor.execute(truncate_query)
                if substage_seconds is not None:
                    substage_seconds["truncate_stage"] = substage_seconds.get(
                        "truncate_stage", 0.0
                    ) + (time.perf_counter() - truncate_start_time)
            else:
                imported_row_count += bundle_row_count
            import_seconds += time.perf_counter() - table_import_start_time

    return bundle_row_count, imported_row_count, load_seconds, import_seconds


def import_arrow_records_copy_local_db(
    connection: Any,
    table_name: str,
    path: Path,
    *,
    chunk_size: int = 65000,
    substage_seconds: dict[str, float] | None = None,
    use_stage_table: bool = True,
) -> tuple[int, int, float, float]:
    _, _, pa_ipc = _load_pyarrow_modules()

    with path.open("rb") as handle:
        reader = pa_ipc.RecordBatchFileReader(handle)
        return import_arrow_record_batches_copy_local_db(
            connection,
            table_name,
            list(reader.schema.names),
            (
                reader.get_batch(batch_idx)
                for batch_idx in range(reader.num_record_batches)
            ),
            chunk_size=chunk_size,
            substage_seconds=substage_seconds,
            use_stage_table=use_stage_table,
        )


def import_parquet_records_copy_local_db(
    connection: Any,
    table_name: str,
    path: Path,
    *,
    chunk_size: int = 65000,
    substage_seconds: dict[str, float] | None = None,
    use_stage_table: bool = True,
) -> tuple[int, int, float, float]:
    pyarrow_parquet = _load_pyarrow_parquet()

    parquet_file = pyarrow_parquet.ParquetFile(str(path))
    return import_arrow_record_batches_copy_local_db(
        connection,
        table_name,
        list(parquet_file.schema_arrow.names),
        parquet_file.iter_batches(batch_size=chunk_size),
        chunk_size=chunk_size,
        substage_seconds=substage_seconds,
        use_stage_table=use_stage_table,
    )


def import_neuronpedia_export_bundle_local_db_with_connection(
    connection: Any,
    export_root: Path | str,
    *,
    prefer_arrow_for_tables: Iterable[str] = (),
    prefer_copy_for_tables: Iterable[str] = (),
    artifact_format_by_table: dict[str, str] | None = None,
    copy_direct_for_tables: Iterable[str] = (),
) -> NeuronpediaLocalImportSummary:
    export_root_path = Path(export_root)
    artifacts = discover_export_bundle_import_artifacts(
        export_root_path,
        prefer_arrow_for_tables=prefer_arrow_for_tables,
        prefer_copy_for_tables=prefer_copy_for_tables,
        artifact_format_by_table=artifact_format_by_table,
    )
    if not artifacts:
        raise NeuronpediaLocalDBImportError(
            f"No importable Neuronpedia bundle files found under {export_root_path}"
        )

    import_start_time = time.perf_counter()
    bundle_row_counts: dict[str, int] = {}
    imported_row_counts: dict[str, int] = {}
    processed_files: list[Path] = []
    table_load_seconds: dict[str, float] = {}
    table_import_seconds: dict[str, float] = {}
    table_import_substage_seconds: dict[str, dict[str, float]] = {}
    inserted_model_ids: set[str] = set()
    deferred_model_fk_updates: list[dict[str, Any]] = []
    copy_tables = set(prefer_copy_for_tables) & set(DEFAULT_COLUMNAR_COPY_IMPORT_TABLES)
    direct_copy_tables = set(copy_direct_for_tables) & copy_tables

    for artifact in artifacts:
        table_name = artifact.table_name
        if artifact.format in {"arrow", "parquet"}:
            use_copy = table_name in copy_tables
            if use_copy:
                copy_importer = (
                    import_arrow_records_copy_local_db
                    if artifact.format == "arrow"
                    else import_parquet_records_copy_local_db
                )
                copy_substage_seconds: dict[str, float] = {}
                bundle_count, imported_count, load_seconds, import_seconds = (
                    copy_importer(
                        connection,
                        table_name,
                        artifact.path,
                        substage_seconds=copy_substage_seconds,
                        use_stage_table=table_name not in direct_copy_tables,
                    )
                )
                table_substages = table_import_substage_seconds.setdefault(
                    table_name, {}
                )
                for substage_name, seconds in copy_substage_seconds.items():
                    table_substages[substage_name] = (
                        table_substages.get(substage_name, 0.0) + seconds
                    )
            else:
                row_importer = (
                    import_arrow_records_local_db
                    if artifact.format == "arrow"
                    else import_parquet_records_local_db
                )
                bundle_count, imported_count, load_seconds, import_seconds = (
                    row_importer(
                        connection,
                        table_name,
                        artifact.path,
                    )
                )
            table_load_seconds[table_name] = (
                table_load_seconds.get(table_name, 0.0) + load_seconds
            )
            table_import_seconds[table_name] = (
                table_import_seconds.get(table_name, 0.0) + import_seconds
            )
            bundle_row_counts[table_name] = (
                bundle_row_counts.get(table_name, 0) + bundle_count
            )
        else:
            load_start_time = time.perf_counter()
            records = _load_jsonl_records(artifact.path)
            table_load_seconds[table_name] = table_load_seconds.get(table_name, 0.0) + (
                time.perf_counter() - load_start_time
            )
            bundle_row_counts[table_name] = bundle_row_counts.get(table_name, 0) + len(
                records
            )
            table_import_start_time = time.perf_counter()
            if table_name == "Model":
                imported_count, inserted_ids, model_fk_updates = (
                    import_model_records_local_db(connection, records)
                )
                inserted_model_ids.update(inserted_ids)
                deferred_model_fk_updates.extend(model_fk_updates)
            else:
                imported_count = import_records_local_db(
                    connection, table_name, records
                )
            table_import_seconds[table_name] = table_import_seconds.get(
                table_name, 0.0
            ) + (time.perf_counter() - table_import_start_time)
        imported_row_counts[table_name] = (
            imported_row_counts.get(table_name, 0) + imported_count
        )
        processed_files.append(artifact.path)

    apply_deferred_model_fk_updates(
        connection, deferred_model_fk_updates, inserted_model_ids
    )

    return NeuronpediaLocalImportSummary(
        export_root=export_root_path,
        bundle_row_counts=bundle_row_counts,
        imported_row_counts=imported_row_counts,
        processed_files=processed_files,
        elapsed_seconds=time.perf_counter() - import_start_time,
        table_load_seconds=table_load_seconds,
        table_import_seconds=table_import_seconds,
        table_import_substage_seconds=table_import_substage_seconds,
    )


def import_neuronpedia_export_bundle_local_db(
    export_root: Path | str,
    *,
    local_db_url: str,
    prefer_arrow_for_tables: Iterable[str] = (),
    prefer_copy_for_tables: Iterable[str] = (),
    artifact_format_by_table: dict[str, str] | None = None,
    copy_direct_for_tables: Iterable[str] = (),
) -> NeuronpediaLocalImportSummary:
    psycopg = importlib.import_module("psycopg")

    export_root_path = Path(export_root)
    if not export_root_path.exists():
        raise NeuronpediaLocalDBImportError(
            f"Export root does not exist: {export_root_path}"
        )

    with psycopg.connect(local_db_url) as connection:
        summary = import_neuronpedia_export_bundle_local_db_with_connection(
            connection,
            export_root_path,
            prefer_arrow_for_tables=prefer_arrow_for_tables,
            prefer_copy_for_tables=prefer_copy_for_tables,
            artifact_format_by_table=artifact_format_by_table,
            copy_direct_for_tables=copy_direct_for_tables,
        )
        connection.commit()
    return summary


def benchmark_neuronpedia_export_bundle_local_db_modes(
    export_root: Path | str,
    *,
    local_db_url: str,
    import_modes: dict[str, dict[str, Iterable[str]]] | None = None,
    rollback_each_mode: bool = True,
) -> dict[str, NeuronpediaLocalImportSummary]:
    """Benchmark local import modes against the same bundle.

    Use ``rollback_each_mode=False`` only for one import mode against a clean
    database/schema when measuring realistic no-rollback COPY throughput.
    """

    psycopg = importlib.import_module("psycopg")

    export_root_path = Path(export_root)
    if not export_root_path.exists():
        raise NeuronpediaLocalDBImportError(
            f"Export root does not exist: {export_root_path}"
        )

    requested_modes = import_modes or DEFAULT_IMPORT_MODE_CONFIGS
    if not rollback_each_mode and len(requested_modes) != 1:
        raise ValueError(
            "No-rollback import benchmarking requires exactly one import mode and a clean target DB/schema."
        )

    benchmark_summaries: dict[str, NeuronpediaLocalImportSummary] = {}
    with psycopg.connect(local_db_url) as connection:
        for mode_name, mode_config in requested_modes.items():
            summary = import_neuronpedia_export_bundle_local_db_with_connection(
                connection,
                export_root_path,
                prefer_arrow_for_tables=mode_config.get("prefer_arrow_for_tables", ()),
                prefer_copy_for_tables=mode_config.get("prefer_copy_for_tables", ()),
                artifact_format_by_table=cast(
                    dict[str, str] | None, mode_config.get("artifact_format_by_table")
                ),
                copy_direct_for_tables=mode_config.get("copy_direct_for_tables", ()),
            )
            if rollback_each_mode:
                connection.rollback()
            else:
                connection.commit()
            benchmark_summaries[mode_name] = summary
    return benchmark_summaries


def import_mode_configs_for_names(
    mode_names: Iterable[str],
) -> dict[str, dict[str, Any]]:
    requested_modes: dict[str, dict[str, Any]] = {}
    for mode_name in mode_names:
        if mode_name not in DEFAULT_IMPORT_MODE_CONFIGS:
            raise ValueError(
                f"Unknown import mode {mode_name!r}; expected one of {sorted(DEFAULT_IMPORT_MODE_CONFIGS)}."
            )
        requested_modes[mode_name] = DEFAULT_IMPORT_MODE_CONFIGS[mode_name]
    return requested_modes


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _jsonify(asdict(cast(Any, value)))
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect, import, or benchmark local Neuronpedia export bundles."
    )
    parser.add_argument("--export-root", required=True, type=Path)
    parser.add_argument(
        "--operation",
        choices=("summary", "columnar-summary", "import", "benchmark"),
        default="summary",
    )
    parser.add_argument(
        "--local-db-url", help="Postgres URL for import or benchmark operations."
    )
    parser.add_argument(
        "--import-mode",
        action="append",
        choices=tuple(DEFAULT_IMPORT_MODE_CONFIGS),
        help="Import mode to run. Repeatable for rollback-protected benchmarks.",
    )
    parser.add_argument(
        "--no-rollback",
        action="store_true",
        help="Commit exactly one benchmark import mode into a clean target DB/schema.",
    )
    parser.add_argument(
        "--include-columnar-summaries",
        action="store_true",
        help="For summary operations, also summarize Arrow and Parquet sidecars when present.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    export_root = args.export_root

    if args.operation == "summary":
        payload: dict[str, Any] = {
            "jsonl": summarize_neuronpedia_export_bundle(export_root),
        }
        if args.include_columnar_summaries:
            try:
                payload["arrow"] = summarize_neuronpedia_export_bundle_arrow(
                    export_root
                )
            except NeuronpediaLocalDBImportError as exc:
                payload["arrow_error"] = str(exc)
            try:
                payload["parquet"] = summarize_neuronpedia_export_bundle_parquet(
                    export_root
                )
            except NeuronpediaLocalDBImportError as exc:
                payload["parquet_error"] = str(exc)
        print(json.dumps(_jsonify(payload), indent=2))
        return 0

    if args.operation == "columnar-summary":
        print(
            json.dumps(
                _jsonify(summarize_saedashboard_columnar_artifacts(export_root)),
                indent=2,
            )
        )
        return 0

    if not args.local_db_url:
        raise SystemExit(
            "--local-db-url is required for import and benchmark operations."
        )

    mode_names = args.import_mode or ["jsonl"]
    mode_configs = import_mode_configs_for_names(mode_names)
    if args.operation == "import":
        if len(mode_configs) != 1:
            raise SystemExit("Import operation requires exactly one --import-mode.")
        mode_config = next(iter(mode_configs.values()))
        summary = import_neuronpedia_export_bundle_local_db(
            export_root,
            local_db_url=args.local_db_url,
            prefer_arrow_for_tables=mode_config.get("prefer_arrow_for_tables", ()),
            prefer_copy_for_tables=mode_config.get("prefer_copy_for_tables", ()),
            artifact_format_by_table=cast(
                dict[str, str] | None, mode_config.get("artifact_format_by_table")
            ),
        )
        print(json.dumps(_jsonify(summary), indent=2))
        return 0

    summaries = benchmark_neuronpedia_export_bundle_local_db_modes(
        export_root,
        local_db_url=args.local_db_url,
        import_modes=mode_configs,
        rollback_each_mode=not args.no_rollback,
    )
    print(json.dumps(_jsonify(summaries), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
