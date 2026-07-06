from __future__ import annotations

# ruff: noqa: E402
import gzip
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
LEGACY_BASELINE_CONTRACT_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "legacy_dashboard_gen_baseline"
    / "preserved_baseline_contract.json"
)
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

db_import: Any = importlib.import_module("neuronpedia_utils.local_db_import")


def _load_export_data_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.chdir(tmp_path)
    module_path = PACKAGE_ROOT / "neuronpedia_utils" / "export-data.py"
    spec = importlib.util.spec_from_file_location(
        "neuronpedia_utils_export_data_test", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_jsonl_gz(path: Path, records: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def _load_legacy_baseline_contract() -> dict[str, Any]:
    return json.loads(LEGACY_BASELINE_CONTRACT_PATH.read_text(encoding="utf-8"))


def test_export_data_uses_psycopg_dict_row_connections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export_data = _load_export_data_module(tmp_path, monkeypatch)
    monkeypatch.setenv("DATABASE_NAME", "np_db")
    monkeypatch.setenv("DATABASE_USERNAME", "np_user")
    monkeypatch.setenv("DATABASE_PASSWORD", "np_password")
    monkeypatch.setenv("DATABASE_HOST", "localhost")
    monkeypatch.setenv("DATABASE_PORT", "5433")
    connect_calls: list[dict[str, Any]] = []

    class _Cursor:
        def __init__(self) -> None:
            self.executed: list[str] = []

        def __enter__(self) -> _Cursor:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def execute(self, query: str) -> None:
            self.executed.append(query)

    class _Connection:
        def __init__(self) -> None:
            self.cursor_instance = _Cursor()
            self.committed = False

        def cursor(self) -> _Cursor:
            return self.cursor_instance

        def commit(self) -> None:
            self.committed = True

    def _fake_connect(**kwargs: Any) -> _Connection:
        connect_calls.append(kwargs)
        return _Connection()

    monkeypatch.setattr(export_data.psycopg, "connect", _fake_connect)

    connection = export_data.create_connection()

    assert connect_calls == [
        {
            "dbname": "np_db",
            "user": "np_user",
            "password": "np_password",
            "host": "localhost",
            "port": "5433",
            "row_factory": export_data.dict_row,
        }
    ]
    assert connection.cursor_instance.executed == ["SET work_mem = '2048MB'"]
    assert connection.committed is True


def test_summarize_bundle_arrow_and_parquet_match_jsonl_counts(tmp_path: Path) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")
    pyarrow_ipc = pytest.importorskip("pyarrow.ipc")

    export_root = tmp_path / "bundle"
    export_root.mkdir()
    activations_dir = export_root / "activations"
    activations_dir.mkdir()
    activation_jsonl_path = activations_dir / "batch-0.jsonl.gz"
    activation_records = [
        {"id": "activation-1", "index": "1", "tokens": ["a"], "values": [1.0]},
        {"id": "activation-2", "index": "2", "tokens": ["b"], "values": [2.0]},
    ]
    _write_jsonl_gz(activation_jsonl_path, activation_records)

    table = pyarrow.Table.from_pylist(activation_records)
    with (activations_dir / "batch-0.arrow").open("wb") as handle:
        with pyarrow_ipc.new_file(handle, table.schema) as writer:
            writer.write_table(table)
    pyarrow_parquet.write_table(table, str(activations_dir / "batch-0.parquet"))

    jsonl_summary = db_import.summarize_neuronpedia_export_bundle(export_root)
    arrow_summary = db_import.summarize_neuronpedia_export_bundle_arrow(export_root)
    parquet_summary = db_import.summarize_neuronpedia_export_bundle_parquet(export_root)

    assert jsonl_summary.table_row_counts == {"Activation": 2}
    assert arrow_summary.table_row_counts == {"Activation": 2}
    assert parquet_summary.table_row_counts == {"Activation": 2}
    assert (
        db_import.compare_neuronpedia_export_bundle_summaries(
            jsonl_summary, arrow_summary
        ).row_count_mismatches
        == {}
    )
    assert (
        db_import.compare_neuronpedia_export_bundle_summaries(
            jsonl_summary, parquet_summary
        ).row_count_mismatches
        == {}
    )


def test_legacy_preserved_baseline_contract_keeps_jsonl_import_mode() -> None:
    baseline = _load_legacy_baseline_contract()

    assert baseline["preserved_baseline_lineage"] == {
        "saedashboard": "7886eaa",
        "saelens": "3eea6552",
        "neuronpedia": "5a33f17",
    }
    assert db_import.DEFAULT_IMPORT_MODE_CONFIGS["jsonl"] == {
        "prefer_arrow_for_tables": (),
        "prefer_copy_for_tables": (),
    }

    for scenario in baseline["scenarios"].values():
        import_contract = scenario["import_contract"]
        legacy_contract = scenario["legacy_contract"]
        assert legacy_contract["export_bundle_format"] == "legacy_jsonl_gzip"
        assert legacy_contract["sequence_selection_backend"] == "legacy_json_cpu"
        assert import_contract["import_mode"] == "jsonl"
        assert import_contract["prefer_arrow_for_tables"] == []
        assert import_contract["prefer_copy_for_tables"] == []
        assert scenario["preserved_baseline_result"]["imported_activation_rows"] > 0


def test_discover_import_artifacts_prefers_parquet_then_arrow_for_columnar_tables(
    tmp_path: Path,
) -> None:
    export_root = tmp_path / "bundle"
    export_root.mkdir()
    (export_root / "model.jsonl.gz").write_bytes(b"{}\n")
    activations_dir = export_root / "activations"
    activations_dir.mkdir()
    activation_jsonl_path = activations_dir / "batch-0.jsonl.gz"
    activation_jsonl_path.write_bytes(b"{}\n")
    activation_arrow_path = activations_dir / "batch-0.arrow"
    activation_arrow_path.write_bytes(b"arrow")
    activation_parquet_path = activations_dir / "batch-0.parquet"
    activation_parquet_path.write_bytes(b"parquet")

    artifacts = db_import.discover_export_bundle_import_artifacts(
        export_root,
        prefer_arrow_for_tables=("Activation",),
        prefer_copy_for_tables=("Activation",),
    )

    assert artifacts == [
        db_import.NeuronpediaBundleArtifact(
            "Model", export_root / "model.jsonl.gz", "jsonl"
        ),
        db_import.NeuronpediaBundleArtifact(
            "Activation", activation_parquet_path, "parquet"
        ),
    ]
    assert activation_arrow_path not in [artifact.path for artifact in artifacts]
    assert activation_jsonl_path not in [artifact.path for artifact in artifacts]


def test_discover_import_artifacts_can_force_arrow_when_parquet_exists(
    tmp_path: Path,
) -> None:
    export_root = tmp_path / "bundle"
    export_root.mkdir()
    activations_dir = export_root / "activations"
    activations_dir.mkdir()
    activation_jsonl_path = activations_dir / "batch-0.jsonl.gz"
    activation_jsonl_path.write_bytes(b"{}\n")
    activation_arrow_path = activations_dir / "batch-0.arrow"
    activation_arrow_path.write_bytes(b"arrow")
    activation_parquet_path = activations_dir / "batch-0.parquet"
    activation_parquet_path.write_bytes(b"parquet")

    artifacts = db_import.discover_export_bundle_import_artifacts(
        export_root,
        prefer_arrow_for_tables=("Activation",),
        artifact_format_by_table={"Activation": "arrow"},
    )

    assert artifacts == [
        db_import.NeuronpediaBundleArtifact(
            "Activation", activation_arrow_path, "arrow"
        )
    ]
    assert activation_parquet_path not in [artifact.path for artifact in artifacts]


def test_summarize_saedashboard_columnar_artifacts_validates_manifest_row_counts(
    tmp_path: Path,
) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    pyarrow_ipc = pytest.importorskip("pyarrow.ipc")
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")

    columnar_root = tmp_path / "batch-0.columnar"
    feature_batch_dir = columnar_root / "feature_batch_0"
    feature_batch_dir.mkdir(parents=True)
    feature_statistics_path = feature_batch_dir / "feature_statistics.parquet"
    sequence_rows_path = feature_batch_dir / "sequence_rows.arrow"

    pyarrow_parquet.write_table(
        pyarrow.table({"feature_index": [0, 1], "max": [1.0, 2.0]}),
        str(feature_statistics_path),
    )
    sequence_rows = pyarrow.table(
        {
            "feature_index": [0, 0, 1],
            "sequence_index": [0, 1, 0],
            "token_id": [10, 11, 12],
        }
    )
    with sequence_rows_path.open("wb") as handle:
        with pyarrow_ipc.new_file(handle, sequence_rows.schema) as writer:
            writer.write_table(sequence_rows)

    (columnar_root / "manifest.json").write_text(
        json.dumps(
            {
                "dashboard_output_format": "columnar",
                "batches": [
                    {"artifact_dir": "feature_batch_0", "feature_batch_index": 0}
                ],
            }
        ),
        encoding="utf-8",
    )
    feature_batch_manifest = feature_batch_dir / "manifest.json"
    feature_batch_manifest.write_text(
        json.dumps(
            {
                "dashboard_output_format": "columnar",
                "feature_batch_index": 0,
                "row_counts": {"feature_statistics": 2, "sequence_rows": 3},
                "tables": {
                    "feature_statistics": "feature_statistics.parquet",
                    "sequence_rows": "sequence_rows.arrow",
                },
            }
        ),
        encoding="utf-8",
    )

    assert db_import.discover_saedashboard_columnar_manifest_paths(columnar_root) == [
        feature_batch_manifest
    ]

    summary = db_import.summarize_saedashboard_columnar_artifacts(columnar_root)

    assert summary.export_root == columnar_root
    assert summary.table_row_counts == {"feature_statistics": 2, "sequence_rows": 3}
    assert summary.processed_files == [feature_statistics_path, sequence_rows_path]
    assert summary.table_load_seconds.keys() == {"feature_statistics", "sequence_rows"}


def test_summarize_saedashboard_columnar_artifacts_rejects_manifest_row_count_mismatch(
    tmp_path: Path,
) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")

    feature_batch_dir = tmp_path / "feature_batch_0"
    feature_batch_dir.mkdir()
    pyarrow_parquet.write_table(
        pyarrow.table({"feature_index": [0, 1], "max": [1.0, 2.0]}),
        str(feature_batch_dir / "feature_statistics.parquet"),
    )
    (feature_batch_dir / "manifest.json").write_text(
        json.dumps(
            {
                "row_counts": {"feature_statistics": 3},
                "tables": {"feature_statistics": "feature_statistics.parquet"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        db_import.NeuronpediaLocalDBImportError, match="manifest expects 3"
    ):
        db_import.summarize_saedashboard_columnar_artifacts(feature_batch_dir)


def _write_columnar_batch_for_neuron_records(tmp_path: Path) -> Path:
    pyarrow = pytest.importorskip("pyarrow")
    pyarrow_ipc = pytest.importorskip("pyarrow.ipc")

    columnar_root = tmp_path / "batch-0.columnar"
    feature_batch_dir = columnar_root / "feature_batch_0"
    feature_batch_dir.mkdir(parents=True)
    tables = {
        "feature_statistics": pyarrow.table(
            {
                "feature_index": [5, 6],
                "max": [12.3456, 9.0],
                "frac_nonzero": [0.4, 0.2],
                "positive_density": [0.123456, 0.05],
            }
        ),
        "feature_tables": pyarrow.table(
            {
                "feature_index": [5, 6],
                "neuron_alignment_indices": [[1, 2], [3]],
                "neuron_alignment_values": [[0.1119, 0.2222], [0.3333]],
                "neuron_alignment_l1": [[0.0109, 0.0201], [0.03]],
                "correlated_neurons_indices": [[7], [8]],
                "correlated_neurons_pearson": [[0.4444], [0.5555]],
                "correlated_neurons_cossim": [[0.6666], [0.7777]],
                "correlated_features_indices": [[9], [10]],
                "correlated_features_pearson": [[0.8888], [0.9999]],
                "correlated_features_cossim": [[0.1234], [0.2345]],
            }
        ),
        "logits_histograms": pyarrow.table(
            {
                "feature_index": [5, 6],
                "bar_heights": [[1, 2], [3, 4]],
                "bar_values": [[-0.1234, 0.4567], [0.5, 0.6]],
            }
        ),
        "activation_histograms": pyarrow.table(
            {
                "feature_index": [5, 6],
                "bar_heights": [[5, 6], [7, 8]],
                "bar_values": [[1.2345, 2.3456], [3.0, 4.0]],
            }
        ),
        "logits_tables": pyarrow.table(
            {
                "feature_index": [5, 6],
                "top_logits": [[0.1234, 0.5678], [0.9]],
                "top_token_ids": [[101, 102], [103]],
                "bottom_logits": [[-0.9876, -0.5432], [-0.1]],
                "bottom_token_ids": [[201, 202], [203]],
            }
        ),
    }
    table_files: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    for table_name, table in tables.items():
        table_files[table_name] = f"{table_name}.arrow"
        row_counts[table_name] = table.num_rows
        with (feature_batch_dir / table_files[table_name]).open("wb") as handle:
            with pyarrow_ipc.new_file(handle, table.schema) as writer:
                writer.write_table(table)

    (columnar_root / "manifest.json").write_text(
        json.dumps(
            {
                "dashboard_output_format": "columnar",
                "batches": [
                    {"artifact_dir": "feature_batch_0", "feature_batch_index": 0}
                ],
            }
        ),
        encoding="utf-8",
    )
    (feature_batch_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dashboard_output_format": "columnar",
                "feature_batch_index": 0,
                "row_counts": row_counts,
                "tables": table_files,
            }
        ),
        encoding="utf-8",
    )
    return columnar_root


def test_build_saedashboard_columnar_neuron_records_maps_feature_tables(
    tmp_path: Path,
) -> None:
    columnar_root = _write_columnar_batch_for_neuron_records(tmp_path)

    records = db_import.build_saedashboard_columnar_neuron_records(
        columnar_root,
        model_id="model-a",
        layer="9-source-a",
        creator_id="creator-a",
        created_at="2026-01-02T03:04:05",
        hook_name="blocks.9.hook_mlp_in",
        source_set_name="source-set-a",
        decode_token_ids=lambda token_ids: [
            f"tok_{token_id}" for token_id in token_ids
        ],
    )

    assert records[0] == {
        "modelId": "model-a",
        "layer": "9-source-a",
        "index": 5,
        "sourceSetName": "source-set-a",
        "creatorId": "creator-a",
        "createdAt": "2026-01-02T03:04:05",
        "maxActApprox": 12.3456,
        "hasVector": False,
        "vector": [],
        "vectorLabel": None,
        "vectorDefaultSteerStrength": 12.3456,
        "hookName": "blocks.9.hook_mlp_in",
        "topkCosSimIndices": [],
        "topkCosSimValues": [],
        "neuron_alignment_indices": [1, 2],
        "neuron_alignment_values": [0.112, 0.222],
        "neuron_alignment_l1": [0.011, 0.02],
        "correlated_neurons_indices": [7],
        "correlated_neurons_pearson": [0.444],
        "correlated_neurons_l1": [0.667],
        "correlated_features_indices": [9],
        "correlated_features_pearson": [0.889],
        "correlated_features_l1": [0.123],
        "neg_str": ["tok_201", "tok_202"],
        "neg_values": [-0.988, -0.543],
        "pos_str": ["tok_101", "tok_102"],
        "pos_values": [0.123, 0.568],
        "frac_nonzero": 0.123456,
        "freq_hist_data_bar_heights": [5.0, 6.0],
        "freq_hist_data_bar_values": [1.234, 2.346],
        "logits_hist_data_bar_heights": [1.0, 2.0],
        "logits_hist_data_bar_values": [-0.123, 0.457],
        "decoder_weights_dist": [],
    }
    assert [record["index"] for record in records] == [5, 6]


def test_import_saedashboard_columnar_neurons_local_db_uses_neuron_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    columnar_root = _write_columnar_batch_for_neuron_records(tmp_path)
    imported_calls: list[tuple[str, list[dict[str, Any]], int]] = []

    def _fake_import_records(
        connection: Any,
        table_name: str,
        records: list[dict[str, Any]],
        *,
        chunk_size: int = 65000,
    ) -> int:
        imported_calls.append((table_name, records, chunk_size))
        return len(records)

    monkeypatch.setattr(db_import, "import_records_local_db", _fake_import_records)

    summary = db_import.import_saedashboard_columnar_neurons_local_db_with_connection(
        object(),
        columnar_root,
        model_id="model-a",
        layer="9-source-a",
        creator_id="creator-a",
        decode_token_ids=lambda token_ids: [str(token_id) for token_id in token_ids],
        chunk_size=1,
    )

    assert imported_calls[0][0] == "Neuron"
    assert len(imported_calls[0][1]) == 2
    assert imported_calls[0][2] == 1
    assert summary.bundle_row_counts == {"Neuron": 2}
    assert summary.imported_row_counts == {"Neuron": 2}
    assert summary.table_load_seconds.keys() == {"Neuron"}
    assert summary.table_import_seconds.keys() == {"Neuron"}


def test_build_saedashboard_columnar_neuron_records_rejects_missing_feature_table(
    tmp_path: Path,
) -> None:
    columnar_root = _write_columnar_batch_for_neuron_records(tmp_path)
    manifest_path = columnar_root / "feature_batch_0" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["tables"]["logits_tables"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        db_import.NeuronpediaLocalDBImportError, match="missing Neuron import tables"
    ):
        db_import.build_saedashboard_columnar_neuron_records(
            columnar_root,
            model_id="model-a",
            layer="9-source-a",
            creator_id="creator-a",
            decode_token_ids=lambda token_ids: [
                str(token_id) for token_id in token_ids
            ],
        )


def _write_columnar_batch_for_activation_records(
    tmp_path: Path,
    *,
    include_sequence_rows: bool = True,
    include_activation_rows: bool = False,
    include_activation_copy_rows: bool = False,
) -> Path:
    pyarrow = pytest.importorskip("pyarrow")
    pyarrow_ipc = pytest.importorskip("pyarrow.ipc")

    columnar_root = tmp_path / "batch-0.columnar"
    feature_batch_dir = columnar_root / "feature_batch_0"
    feature_batch_dir.mkdir(parents=True)
    row_counts: dict[str, int] = {}
    tables: dict[str, str] = {}

    if include_sequence_rows:
        sequence_rows = pyarrow.table(
            {
                "feature_index": [5, 5, 5, 5, 6],
                "sequence_index": [0, 0, 1, 1, 0],
                "group_index": [0, 0, 1, 1, 0],
                "group_name": [
                    "TOP ACTIVATIONS<br>MAX = 12.346",
                    "TOP ACTIVATIONS<br>MAX = 12.346",
                    "INTERVAL 0.100 - 0.300<br>CONTAINS 12.5%",
                    "INTERVAL 0.100 - 0.300<br>CONTAINS 12.5%",
                    "TOP ACTIVATIONS<br>MAX = 7.000",
                ],
                "group_sequence_index": [0, 0, 0, 0, 0],
                "context_token_index": [1, 0, 0, 1, 0],
                "original_index": [42, 42, 43, 43, 44],
                "qualifying_token_index": [3, 3, 4, 4, 1],
                "source_token_index": [2, 1, 1, 2, 1],
                "token_id": [102, 101, 103, 0, 104],
                "feat_act": [0.5555, 0.1234, 0.2, 0.0, 7.0],
                "token_logit": [1.1111, 2.2222, 3.3333, 4.4444, 5.0],
            }
        )
        sequence_rows_path = feature_batch_dir / "sequence_rows.arrow"
        with sequence_rows_path.open("wb") as handle:
            with pyarrow_ipc.new_file(handle, sequence_rows.schema) as writer:
                writer.write_table(sequence_rows)
        row_counts["sequence_rows"] = sequence_rows.num_rows
        tables["sequence_rows"] = "sequence_rows.arrow"

    if include_activation_rows:
        activation_rows = pyarrow.table(
            {
                "feature_index": [5, 5, 6],
                "sequence_index": [0, 1, 0],
                "tokens": [["tok_101", "tok_102"], ["tok_103"], ["tok_104"]],
                "max_value": [0.555, 0.2, 7.0],
                "max_value_token_index": [1, 0, 0],
                "min_value": [0.123, 0.2, 7.0],
                "values": [[0.123, 0.555], [0.2], [7.0]],
                "bin_min": [-1.0, 0.1, -1.0],
                "bin_max": [12.346, 0.3, 7.0],
                "bin_contains": [-1.0, 0.125, -1.0],
                "qualifying_token_index": [2, 3, 0],
            }
        )
        activation_rows_path = feature_batch_dir / "activation_rows.arrow"
        with activation_rows_path.open("wb") as handle:
            with pyarrow_ipc.new_file(handle, activation_rows.schema) as writer:
                writer.write_table(activation_rows)
        row_counts["activation_rows"] = activation_rows.num_rows
        tables["activation_rows"] = "activation_rows.arrow"

    if include_activation_copy_rows:
        activation_copy_rows = pyarrow.table(
            {
                "id": ["copy-5-0", "copy-5-1", "copy-6-0"],
                "tokens": [["tok_101", "tok_102"], ["tok_103"], ["tok_104"]],
                "dataIndex": [None, None, None],
                "index": [5, 5, 6],
                "layer": ["9-source-a", "9-source-a", "9-source-a"],
                "modelId": ["model-a", "model-a", "model-a"],
                "dataSource": [None, None, None],
                "maxValue": [0.555, 0.2, 7.0],
                "maxValueTokenIndex": [1, 0, 0],
                "minValue": [0.123, 0.2, 7.0],
                "values": [[0.123, 0.555], [0.2], [7.0]],
                "dfaValues": [[], [], []],
                "dfaTargetIndex": [None, None, None],
                "dfaMaxValue": [None, None, None],
                "creatorId": ["creator-a", "creator-a", "creator-a"],
                "createdAt": [
                    "2026-01-02T03:04:05",
                    "2026-01-02T03:04:05",
                    "2026-01-02T03:04:05",
                ],
                "lossValues": [[], [], []],
                "logitContributions": [None, None, None],
                "binMin": [-1.0, 0.1, -1.0],
                "binMax": [12.346, 0.3, 7.0],
                "binContains": [-1.0, 0.125, -1.0],
                "qualifyingTokenIndex": [2, 3, 0],
            }
        )
        activation_copy_rows_path = feature_batch_dir / "activation_copy_rows.arrow"
        with activation_copy_rows_path.open("wb") as handle:
            with pyarrow_ipc.new_file(handle, activation_copy_rows.schema) as writer:
                writer.write_table(activation_copy_rows)
        row_counts["activation_copy_rows"] = activation_copy_rows.num_rows
        tables["activation_copy_rows"] = "activation_copy_rows.arrow"

    (columnar_root / "manifest.json").write_text(
        json.dumps(
            {
                "dashboard_output_format": "columnar",
                "batches": [
                    {"artifact_dir": "feature_batch_0", "feature_batch_index": 0}
                ],
            }
        ),
        encoding="utf-8",
    )
    (feature_batch_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dashboard_output_format": "columnar",
                "feature_batch_index": 0,
                "row_counts": row_counts,
                "tables": tables,
            }
        ),
        encoding="utf-8",
    )
    return columnar_root


def test_build_saedashboard_columnar_activation_records_maps_sequence_rows(
    tmp_path: Path,
) -> None:
    columnar_root = _write_columnar_batch_for_activation_records(
        tmp_path,
        include_sequence_rows=True,
        include_activation_rows=False,
    )

    records = db_import.build_saedashboard_columnar_activation_records(
        columnar_root,
        model_id="model-a",
        layer="9-source-a",
        creator_id="creator-a",
        created_at="2026-01-02T03:04:05",
        activation_id_prefix="act",
        pad_token_id=0,
        decode_token_ids=lambda token_ids: [
            f"tok_{token_id}" for token_id in token_ids
        ],
    )

    assert records[0] == {
        "id": "act-5-0",
        "tokens": ["tok_101", "tok_102"],
        "dataIndex": None,
        "index": 5,
        "layer": "9-source-a",
        "modelId": "model-a",
        "dataSource": None,
        "maxValue": 0.555,
        "maxValueTokenIndex": 1,
        "minValue": 0.123,
        "values": [0.123, 0.555],
        "dfaValues": [],
        "dfaTargetIndex": None,
        "dfaMaxValue": None,
        "creatorId": "creator-a",
        "createdAt": "2026-01-02T03:04:05",
        "lossValues": [],
        "logitContributions": None,
        "binMin": -1.0,
        "binMax": 12.346,
        "binContains": -1.0,
        "qualifyingTokenIndex": 2,
    }
    assert records[1]["id"] == "act-5-1"
    assert records[1]["tokens"] == ["tok_103"]
    assert records[1]["values"] == [0.2]
    assert records[1]["binMin"] == 0.1
    assert records[1]["binMax"] == 0.3
    assert records[1]["binContains"] == 0.125
    assert [record["id"] for record in records] == ["act-5-0", "act-5-1", "act-6-0"]


def test_build_saedashboard_columnar_activation_records_prefers_activation_rows(
    tmp_path: Path,
) -> None:
    columnar_root = _write_columnar_batch_for_activation_records(
        tmp_path,
        include_sequence_rows=True,
        include_activation_rows=True,
    )

    records = db_import.build_saedashboard_columnar_activation_records(
        columnar_root,
        model_id="model-a",
        layer="9-source-a",
        creator_id="creator-a",
        created_at="2026-01-02T03:04:05",
        activation_id_prefix="act",
        decode_token_ids=lambda _token_ids: (_ for _ in ()).throw(
            AssertionError("activation_rows should bypass token decoding")
        ),
    )

    assert [record["id"] for record in records] == ["act-5-0", "act-5-1", "act-6-0"]
    assert records[0]["tokens"] == ["tok_101", "tok_102"]
    assert records[0]["values"] == [0.123, 0.555]


def test_build_saedashboard_columnar_activation_records_prefers_activation_copy_rows(
    tmp_path: Path,
) -> None:
    columnar_root = _write_columnar_batch_for_activation_records(
        tmp_path,
        include_sequence_rows=True,
        include_activation_rows=True,
        include_activation_copy_rows=True,
    )

    records = db_import.build_saedashboard_columnar_activation_records(
        columnar_root,
        model_id="override-model",
        layer="override-layer",
        creator_id="override-creator",
        created_at="2026-04-05T06:07:08",
        activation_id_prefix="override-act",
        decode_token_ids=lambda _token_ids: (_ for _ in ()).throw(
            AssertionError("activation_copy_rows should bypass token decoding")
        ),
    )

    assert [record["id"] for record in records] == [
        "override-act-5-0",
        "override-act-5-1",
        "override-act-6-0",
    ]
    assert records[0]["modelId"] == "override-model"
    assert records[0]["layer"] == "override-layer"
    assert records[0]["creatorId"] == "override-creator"
    assert records[0]["createdAt"] == "2026-04-05T06:07:08"


def test_import_saedashboard_columnar_activations_local_db_streams_copy_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    columnar_root = _write_columnar_batch_for_activation_records(
        tmp_path,
        include_sequence_rows=True,
        include_activation_rows=True,
        include_activation_copy_rows=True,
    )
    copied_batches: list[Any] = []
    captured_use_stage_table: list[bool] = []

    def _fail_import_records(*args: Any, **kwargs: Any) -> int:
        raise AssertionError(
            "streaming Activation import should not use legacy JSONB records"
        )

    def _fake_copy_import(
        connection: Any,
        table_name: str,
        available_columns: list[str],
        record_batches: Any,
        *,
        chunk_size: int = 65000,
        substage_seconds: dict[str, float] | None = None,
        use_stage_table: bool = True,
    ) -> tuple[int, int, float, float]:
        assert table_name == "Activation"
        assert (
            available_columns == db_import.SAEDASHBOARD_COLUMNAR_ACTIVATION_COPY_COLUMNS
        )
        assert chunk_size == 2
        captured_use_stage_table.append(use_stage_table)
        copied_batches.extend(list(record_batches))
        if substage_seconds is not None:
            substage_seconds["copy_write"] = 0.25
            substage_seconds["insert_from_stage"] = 0.125
        row_count = sum(batch.num_rows for batch in copied_batches)
        return row_count, row_count, 0.123, 0.456

    monkeypatch.setattr(db_import, "import_records_local_db", _fail_import_records)
    monkeypatch.setattr(
        db_import, "import_arrow_record_batches_copy_local_db", _fake_copy_import
    )

    summary = (
        db_import.import_saedashboard_columnar_activations_local_db_with_connection(
            object(),
            columnar_root,
            model_id="override-model",
            layer="override-layer",
            creator_id="override-creator",
            created_at="2026-04-05T06:07:08",
            activation_id_prefix="override-act",
            decode_token_ids=lambda _token_ids: (_ for _ in ()).throw(
                AssertionError("activation_rows should bypass token decoding")
            ),
            pad_token_id=0,
            chunk_size=2,
        )
    )

    copied_rows: list[dict[str, Any]] = []
    for batch in copied_batches:
        copied_rows.extend(batch.to_pylist())
    assert copied_rows[0]["id"] == "override-act-5-0"
    assert copied_rows[0]["tokens"] == ["tok_101", "tok_102"]
    assert copied_rows[0]["values"] == [0.123, 0.555]
    assert copied_rows[0]["modelId"] == "override-model"
    assert copied_rows[0]["layer"] == "override-layer"
    assert copied_rows[0]["creatorId"] == "override-creator"
    assert copied_rows[0]["createdAt"] == "2026-04-05T06:07:08"
    assert copied_rows[1]["id"] == "override-act-5-1"
    assert copied_rows[1]["tokens"] == ["tok_103"]
    assert [row["id"] for row in copied_rows] == [
        "override-act-5-0",
        "override-act-5-1",
        "override-act-6-0",
    ]
    assert captured_use_stage_table == [True]
    assert summary.bundle_row_counts == {"Activation": 3}
    assert summary.imported_row_counts == {"Activation": 3}
    assert summary.processed_files == [
        columnar_root / "feature_batch_0" / "activation_copy_rows.arrow"
    ]
    assert summary.table_load_seconds.keys() == {"Activation"}
    assert summary.table_import_seconds.keys() == {"Activation"}
    assert summary.table_load_seconds["Activation"] == 0.123
    assert summary.table_import_seconds["Activation"] == 0.456
    assert summary.table_import_substage_seconds == {
        "Activation": {"copy_write": 0.25, "insert_from_stage": 0.125}
    }


def test_import_saedashboard_columnar_activations_local_db_can_disable_stage_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    columnar_root = _write_columnar_batch_for_activation_records(
        tmp_path,
        include_sequence_rows=True,
        include_activation_rows=True,
        include_activation_copy_rows=True,
    )
    captured_use_stage_table: list[bool] = []

    def _fake_copy_import(
        connection: Any,
        table_name: str,
        available_columns: list[str],
        record_batches: Any,
        *,
        chunk_size: int = 65000,
        substage_seconds: dict[str, float] | None = None,
        use_stage_table: bool = True,
    ) -> tuple[int, int, float, float]:
        captured_use_stage_table.append(use_stage_table)
        list(record_batches)
        if substage_seconds is not None:
            substage_seconds["copy_write"] = 0.25
        return 3, 3, 0.123, 0.456

    monkeypatch.setattr(
        db_import, "import_arrow_record_batches_copy_local_db", _fake_copy_import
    )

    summary = (
        db_import.import_saedashboard_columnar_activations_local_db_with_connection(
            object(),
            columnar_root,
            model_id="model-a",
            layer="9-source-a",
            creator_id="creator-a",
            decode_token_ids=lambda token_ids: [
                str(token_id) for token_id in token_ids
            ],
            chunk_size=2,
            use_stage_table=False,
        )
    )

    assert captured_use_stage_table == [False]
    assert summary.bundle_row_counts == {"Activation": 3}
    assert summary.imported_row_counts == {"Activation": 3}


def test_import_saedashboard_columnar_activations_local_db_can_use_legacy_jsonb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    columnar_root = _write_columnar_batch_for_activation_records(
        tmp_path,
        include_sequence_rows=True,
        include_activation_rows=True,
    )
    imported_calls: list[tuple[str, list[dict[str, Any]], int]] = []

    def _fake_import_records(
        connection: Any,
        table_name: str,
        records: list[dict[str, Any]],
        *,
        chunk_size: int = 65000,
    ) -> int:
        imported_calls.append((table_name, records, chunk_size))
        return len(records)

    monkeypatch.setattr(db_import, "import_records_local_db", _fake_import_records)

    summary = (
        db_import.import_saedashboard_columnar_activations_local_db_with_connection(
            object(),
            columnar_root,
            model_id="model-a",
            layer="9-source-a",
            creator_id="creator-a",
            decode_token_ids=lambda _token_ids: (_ for _ in ()).throw(
                AssertionError("activation_rows should bypass token decoding")
            ),
            chunk_size=1,
            use_copy=False,
        )
    )

    assert imported_calls[0][0] == "Activation"
    assert len(imported_calls[0][1]) == 3
    assert imported_calls[0][2] == 1
    assert summary.bundle_row_counts == {"Activation": 3}
    assert summary.imported_row_counts == {"Activation": 3}


def test_import_saedashboard_columnar_bundle_local_db_combines_ordered_summaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    columnar_root = tmp_path / "batch-0.columnar"
    import_calls: list[str] = []

    def _summary(table_name: str, row_count: int, processed_file: Path | None) -> Any:
        return db_import.NeuronpediaLocalImportSummary(
            export_root=columnar_root,
            bundle_row_counts={table_name: row_count},
            imported_row_counts={table_name: row_count},
            processed_files=[] if processed_file is None else [processed_file],
            elapsed_seconds=0.1,
            table_load_seconds={table_name: 0.01},
            table_import_seconds={table_name: 0.02},
        )

    def _fake_metadata_import(connection: Any, root: Path, **kwargs: Any) -> Any:
        import_calls.append("metadata")
        assert kwargs["source_id"] == "9-source-set-a"
        assert kwargs["source_set_name"] == "source-set-a"
        return _summary("Source", 1, None)

    def _fake_neuron_import(connection: Any, root: Path, **kwargs: Any) -> Any:
        import_calls.append("neuron")
        assert kwargs["layer"] == "9-source-set-a"
        assert kwargs["source_set_name"] == "source-set-a"
        return _summary("Neuron", 2, root / "feature_statistics.arrow")

    def _fake_activation_import(connection: Any, root: Path, **kwargs: Any) -> Any:
        import_calls.append("activation")
        assert kwargs["layer"] == "9-source-set-a"
        assert kwargs["activation_id_prefix"] == "act"
        assert kwargs["use_stage_table"] is True
        return _summary("Activation", 3, root / "sequence_rows.arrow")

    monkeypatch.setattr(
        db_import,
        "import_saedashboard_columnar_metadata_local_db_with_connection",
        _fake_metadata_import,
    )
    monkeypatch.setattr(
        db_import,
        "import_saedashboard_columnar_neurons_local_db_with_connection",
        _fake_neuron_import,
    )
    monkeypatch.setattr(
        db_import,
        "import_saedashboard_columnar_activations_local_db_with_connection",
        _fake_activation_import,
    )

    summary = db_import.import_saedashboard_columnar_bundle_local_db_with_connection(
        object(),
        columnar_root,
        model_id="model-a",
        source_set_name="source-set-a",
        source_id="9-source-set-a",
        creator_id="creator-a",
        decode_token_ids=lambda token_ids: [str(token_id) for token_id in token_ids],
        activation_id_prefix="act",
    )

    assert import_calls == ["metadata", "neuron", "activation"]
    assert summary.bundle_row_counts == {"Source": 1, "Neuron": 2, "Activation": 3}
    assert summary.imported_row_counts == summary.bundle_row_counts
    assert summary.processed_files == [
        columnar_root / "feature_statistics.arrow",
        columnar_root / "sequence_rows.arrow",
    ]
    assert summary.table_load_seconds == {
        "Source": 0.01,
        "Neuron": 0.01,
        "Activation": 0.01,
    }
    assert summary.table_import_seconds == {
        "Source": 0.02,
        "Neuron": 0.02,
        "Activation": 0.02,
    }


def test_import_saedashboard_columnar_bundle_local_db_threads_activation_use_stage_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    columnar_root = tmp_path / "batch-0.columnar"
    received_activation_kwargs: dict[str, Any] = {}

    def _summary(table_name: str) -> Any:
        return db_import.NeuronpediaLocalImportSummary(
            export_root=columnar_root,
            bundle_row_counts={table_name: 1},
            imported_row_counts={table_name: 1},
            processed_files=[],
            elapsed_seconds=0.1,
            table_load_seconds={table_name: 0.01},
            table_import_seconds={table_name: 0.02},
        )

    monkeypatch.setattr(
        db_import,
        "import_saedashboard_columnar_metadata_local_db_with_connection",
        lambda *args, **kwargs: _summary("Source"),
    )
    monkeypatch.setattr(
        db_import,
        "import_saedashboard_columnar_neurons_local_db_with_connection",
        lambda *args, **kwargs: _summary("Neuron"),
    )

    def _fake_activation_import(connection: Any, root: Path, **kwargs: Any) -> Any:
        received_activation_kwargs.update(kwargs)
        return _summary("Activation")

    monkeypatch.setattr(
        db_import,
        "import_saedashboard_columnar_activations_local_db_with_connection",
        _fake_activation_import,
    )

    db_import.import_saedashboard_columnar_bundle_local_db_with_connection(
        object(),
        columnar_root,
        model_id="model-a",
        source_set_name="source-set-a",
        source_id="9-source-set-a",
        creator_id="creator-a",
        decode_token_ids=lambda token_ids: [str(token_id) for token_id in token_ids],
        activation_use_stage_table=False,
    )

    assert received_activation_kwargs["use_stage_table"] is False


def test_import_saedashboard_columnar_bundle_local_db_commits_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    columnar_root = tmp_path / "batch-0.columnar"
    columnar_root.mkdir()

    class _Connection:
        def __init__(self) -> None:
            self.commit_calls = 0

        def commit(self) -> None:
            self.commit_calls += 1

        def __enter__(self) -> _Connection:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

    connection = _Connection()
    received: dict[str, Any] = {}

    def _fake_bundle_import_with_connection(
        passed_connection: Any,
        passed_root: Path,
        **kwargs: Any,
    ) -> Any:
        received["connection"] = passed_connection
        received["root"] = passed_root
        received["kwargs"] = kwargs
        return db_import.NeuronpediaLocalImportSummary(
            export_root=passed_root,
            bundle_row_counts={"Neuron": 1},
            imported_row_counts={"Neuron": 1},
            processed_files=[],
            elapsed_seconds=0.1,
            table_load_seconds={"Neuron": 0.0},
            table_import_seconds={"Neuron": 0.1},
        )

    monkeypatch.setattr(
        db_import,
        "import_saedashboard_columnar_bundle_local_db_with_connection",
        _fake_bundle_import_with_connection,
    )
    monkeypatch.setitem(
        sys.modules,
        "psycopg",
        SimpleNamespace(connect=lambda *_args, **_kwargs: connection),
    )

    summary = db_import.import_saedashboard_columnar_bundle_local_db(
        columnar_root,
        local_db_url="postgres://postgres:postgres@127.0.0.1:5433/postgres",
        model_id="model-a",
        source_set_name="source-set-a",
        source_id="9-source-set-a",
        creator_id="creator-a",
        decode_token_ids=lambda token_ids: [str(token_id) for token_id in token_ids],
        source_set_description="Source set A",
    )

    assert connection.commit_calls == 1
    assert received["connection"] is connection
    assert received["root"] == columnar_root
    assert received["kwargs"]["source_set_description"] == "Source set A"
    assert summary.imported_row_counts == {"Neuron": 1}


def test_build_saedashboard_columnar_activation_records_rejects_bad_group_name(
    tmp_path: Path,
) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    pyarrow_ipc = pytest.importorskip("pyarrow.ipc")

    columnar_root = tmp_path / "batch-0.columnar"
    feature_batch_dir = columnar_root / "feature_batch_0"
    feature_batch_dir.mkdir(parents=True)
    sequence_rows = pyarrow.table(
        {
            "feature_index": [0],
            "sequence_index": [0],
            "context_token_index": [0],
            "group_name": ["INTERVAL not-parseable"],
            "qualifying_token_index": [1],
            "token_id": [10],
            "feat_act": [1.0],
            "token_logit": [0.5],
        }
    )
    with (feature_batch_dir / "sequence_rows.arrow").open("wb") as handle:
        with pyarrow_ipc.new_file(handle, sequence_rows.schema) as writer:
            writer.write_table(sequence_rows)
    (feature_batch_dir / "manifest.json").write_text(
        json.dumps(
            {
                "row_counts": {"sequence_rows": 1},
                "tables": {"sequence_rows": "sequence_rows.arrow"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        db_import.NeuronpediaLocalDBImportError, match="Could not parse interval"
    ):
        db_import.build_saedashboard_columnar_activation_records(
            columnar_root,
            model_id="model-a",
            layer="9-source-a",
            creator_id="creator-a",
            decode_token_ids=lambda token_ids: [
                str(token_id) for token_id in token_ids
            ],
        )


def test_align_arrow_batch_for_postgres_copy_casts_and_sanitizes_strings() -> None:
    pyarrow = pytest.importorskip("pyarrow")

    batch = pyarrow.record_batch(
        [
            pyarrow.array([1, 2], type=pyarrow.int64()),
            pyarrow.array(["alpha\x00", "beta"], type=pyarrow.string_view()),
            pyarrow.array(
                ["2026-05-05T12:36:33.999798", None], type=pyarrow.string_view()
            ),
            pyarrow.array(
                [["x\x00", "y"], None], type=pyarrow.large_list(pyarrow.string_view())
            ),
            pyarrow.nulls(2),
        ],
        names=["index", "creatorId", "createdAt", "tokens", "logitContributions"],
    )

    aligned = db_import.align_arrow_batch_for_postgres_copy(
        batch,
        [
            ("index", "text"),
            ("creatorId", "text"),
            ("createdAt", "timestamp(3) without time zone"),
            ("tokens", "text[]"),
            ("logitContributions", "text"),
        ],
    )

    assert str(aligned.schema.field("index").type) == "string"
    assert str(aligned.schema.field("createdAt").type) == "timestamp[us]"
    assert str(aligned.schema.field("tokens").type) == "list<item: string>"
    assert aligned.column(aligned.schema.get_field_index("creatorId")).to_pylist() == [
        "alpha ",
        "beta",
    ]
    assert aligned.column(aligned.schema.get_field_index("tokens")).to_pylist() == [
        ["x ", "y"],
        None,
    ]
    assert aligned.column(
        aligned.schema.get_field_index("logitContributions")
    ).to_pylist() == [None, None]


def test_import_arrow_record_batches_copy_local_db_uses_one_binary_copy_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    pytest.importorskip("psycopg")

    batches = [
        pyarrow.record_batch(
            [
                pyarrow.array(["activation-1"]),
                pyarrow.array([["a", "b"]]),
                pyarrow.array([[1.0, 0.5]]),
            ],
            names=["id", "tokens", "values"],
        ),
        pyarrow.record_batch(
            [
                pyarrow.array(["activation-2"]),
                pyarrow.array([["c"]]),
                pyarrow.array([[2.0]]),
            ],
            names=["id", "tokens", "values"],
        ),
    ]
    copy_writes: list[bytes] = []
    encoded_schema_names: list[list[str]] = []

    class _Encoder:
        def __init__(self, schema: Any) -> None:
            encoded_schema_names.append(list(schema.names))

        def write_header(self) -> bytes:
            return b"header"

        def write_batch(self, batch: Any) -> bytes:
            return f"batch:{batch.num_rows}".encode()

        def finish(self) -> bytes:
            return b"finish"

    class _Copy:
        def __enter__(self) -> _Copy:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def write(self, payload: bytes) -> None:
            copy_writes.append(payload)

    class _Cursor:
        def __init__(self) -> None:
            self.execute_calls = 0
            self.rowcount = 0
            self.copy_calls = 0

        def __enter__(self) -> _Cursor:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def execute(self, query: object, params: object | None = None) -> None:
            self.execute_calls += 1
            self.rowcount = 2 if self.execute_calls == 2 else 0

        def fetchall(self) -> list[tuple[str, str]]:
            return []

        def copy(self, query: object) -> _Copy:
            self.copy_calls += 1
            return _Copy()

    class _Connection:
        def __init__(self) -> None:
            self.cursor_instance = _Cursor()

        def cursor(self) -> _Cursor:
            return self.cursor_instance

    connection = _Connection()

    monkeypatch.setattr(
        db_import,
        "resolve_table_column_defs",
        lambda cursor, table_name, available_columns: [
            ("id", "text"),
            ("tokens", "text[]"),
            ("values", "double precision[]"),
        ],
    )
    monkeypatch.setattr(db_import, "_load_pgpq_arrow_encoder", lambda: _Encoder)

    bundle_count, imported_count, load_seconds, import_seconds = (
        db_import.import_arrow_record_batches_copy_local_db(
            connection,
            "Activation",
            ["id", "tokens", "values"],
            batches,
            chunk_size=1,
        )
    )

    assert bundle_count == 2
    assert imported_count == 2
    assert load_seconds >= 0.0
    assert import_seconds >= 0.0
    assert connection.cursor_instance.copy_calls == 1
    assert copy_writes == [b"header", b"batch:1", b"batch:1", b"finish"]
    assert encoded_schema_names == [["id", "tokens", "values"]]


def test_import_neuronpedia_export_bundle_local_db_orders_metadata_and_defers_model_fks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_root = tmp_path / "bundle"
    export_root.mkdir()
    (export_root / "release.jsonl").write_text(
        json.dumps({"name": "release-1"}) + "\n", encoding="utf-8"
    )
    (export_root / "model.jsonl").write_text(
        json.dumps({"id": "model-1", "defaultSourceSetName": "set-1"}) + "\n",
        encoding="utf-8",
    )
    (export_root / "sourceset.jsonl").write_text(
        json.dumps({"name": "set-1", "modelId": "model-1"}) + "\n",
        encoding="utf-8",
    )
    (export_root / "source.jsonl").write_text(
        json.dumps({"id": "source-1", "modelId": "model-1", "setName": "set-1"}) + "\n",
        encoding="utf-8",
    )

    imported_tables: list[str] = []
    imported_records: dict[str, list[dict[str, Any]]] = {}
    deferred_updates: list[dict[str, Any]] = []

    def _fake_import_records(
        connection: Any,
        table_name: str,
        records: list[dict[str, Any]],
        *,
        chunk_size: int = 65000,
    ) -> int:
        imported_tables.append(table_name)
        imported_records[table_name] = records
        return len(records)

    def _fake_import_model_records(
        connection: Any,
        records: list[dict[str, Any]],
        *,
        chunk_size: int = 65000,
    ) -> tuple[int, set[str], list[dict[str, Any]]]:
        prepared_records, model_updates = db_import.prepare_model_records_for_insert(
            records
        )
        imported_tables.append("Model")
        imported_records["Model"] = prepared_records
        return len(prepared_records), {prepared_records[0]["id"]}, model_updates

    monkeypatch.setattr(db_import, "import_records_local_db", _fake_import_records)
    monkeypatch.setattr(
        db_import, "import_model_records_local_db", _fake_import_model_records
    )
    monkeypatch.setattr(
        db_import,
        "apply_deferred_model_fk_updates",
        lambda connection, updates, inserted_ids: deferred_updates.extend(updates),
    )

    summary = db_import.import_neuronpedia_export_bundle_local_db_with_connection(
        object(), export_root
    )

    assert imported_tables == ["SourceRelease", "Model", "SourceSet", "Source"]
    assert imported_records["Model"] == [
        {"id": "model-1", "defaultSourceSetName": None}
    ]
    assert deferred_updates == [
        {
            "id": "model-1",
            "defaultSourceSetName": "set-1",
            "defaultGraphSourceSetName": None,
            "defaultSourceId": None,
        }
    ]
    assert summary.bundle_row_counts == {
        "SourceRelease": 1,
        "Model": 1,
        "SourceSet": 1,
        "Source": 1,
    }
    assert summary.imported_row_counts == {
        "SourceRelease": 1,
        "Model": 1,
        "SourceSet": 1,
        "Source": 1,
    }


def test_build_saedashboard_columnar_metadata_records_maps_source_rows() -> None:
    records_by_table = db_import.build_saedashboard_columnar_metadata_records(
        model_id="model-a-it",
        source_set_name="source-set-a",
        source_id="9-source-set-a",
        creator_id="creator-a",
        creator_name="Creator A",
        release_name="release-a",
        release_description="Release A",
        release_urls=("https://example.com/release",),
        model_layers=18,
        model_neurons_per_layer=0,
        model_owner="",
        source_set_description="Source set A",
        source_dataset="dataset-a",
        source_hf_repo_id="hf/repo-a",
        source_hf_folder_id="folder-a",
        num_prompts=10,
        num_tokens_in_prompt=512,
        created_at="2026-01-02T03:04:05",
    )

    assert list(records_by_table) == ["SourceRelease", "Model", "SourceSet", "Source"]
    assert records_by_table["SourceRelease"] == [
        {
            "name": "release-a",
            "description": "Release A",
            "descriptionShort": "Release A",
            "urls": ["https://example.com/release"],
            "creatorNameShort": "Creator A",
            "creatorName": "Creator A",
            "creatorId": "creator-a",
            "defaultSourceSetName": "source-set-a",
            "createdAt": "2026-01-02T03:04:05",
            "visibility": "PUBLIC",
            "isNewUi": False,
            "featured": False,
        }
    ]
    assert records_by_table["Model"][0] == {
        "id": "model-a-it",
        "instruct": True,
        "creatorId": "creator-a",
        "displayNameShort": "model-a-it",
        "displayName": "model-a-it",
        "visibility": "PUBLIC",
        "defaultSourceSetName": "source-set-a",
        "defaultGraphSourceSetName": "source-set-a",
        "defaultSourceId": "9-source-set-a",
        "inferenceEnabled": True,
        "layers": 18,
        "neuronsPerLayer": 0,
        "owner": "",
        "createdAt": "2026-01-02T03:04:05",
        "updatedAt": "2026-01-02T03:04:05",
    }
    assert records_by_table["SourceSet"][0]["releaseName"] == "release-a"
    assert records_by_table["SourceSet"][0]["defaultOfModelId"] == "model-a-it"
    assert records_by_table["Source"][0] == {
        "id": "9-source-set-a",
        "modelId": "model-a-it",
        "setName": "source-set-a",
        "creatorId": "creator-a",
        "hasDashboards": True,
        "inferenceEnabled": False,
        "inferenceHosts": [],
        "visibility": "PUBLIC",
        "defaultOfModelId": "model-a-it",
        "hasUmap": False,
        "hasUmapLogSparsity": False,
        "hasUmapClusters": False,
        "createdAt": "2026-01-02T03:04:05",
        "hfRepoId": "hf/repo-a",
        "hfFolderId": "folder-a",
        "num_prompts": 10,
        "num_tokens_in_prompt": 512,
        "dataset": "dataset-a",
    }


def test_import_saedashboard_columnar_metadata_local_db_orders_and_defers_model_fks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported_tables: list[str] = []
    imported_records: dict[str, list[dict[str, Any]]] = {}
    deferred_updates: list[dict[str, Any]] = []

    def _fake_import_records(
        connection: Any,
        table_name: str,
        records: list[dict[str, Any]],
        *,
        chunk_size: int = 65000,
    ) -> int:
        imported_tables.append(table_name)
        imported_records[table_name] = records
        return len(records)

    def _fake_import_model_records(
        connection: Any,
        records: list[dict[str, Any]],
        *,
        chunk_size: int = 65000,
    ) -> tuple[int, set[str], list[dict[str, Any]]]:
        prepared_records, model_updates = db_import.prepare_model_records_for_insert(
            records
        )
        imported_tables.append("Model")
        imported_records["Model"] = prepared_records
        return len(prepared_records), {prepared_records[0]["id"]}, model_updates

    monkeypatch.setattr(db_import, "import_records_local_db", _fake_import_records)
    monkeypatch.setattr(
        db_import, "import_model_records_local_db", _fake_import_model_records
    )
    monkeypatch.setattr(
        db_import,
        "apply_deferred_model_fk_updates",
        lambda connection, updates, inserted_ids: deferred_updates.extend(updates),
    )

    summary = db_import.import_saedashboard_columnar_metadata_local_db_with_connection(
        object(),
        tmp_path / "batch-0.columnar",
        model_id="model-a",
        source_set_name="source-set-a",
        source_id="9-source-set-a",
        creator_id="creator-a",
        release_name="release-a",
        created_at="2026-01-02T03:04:05",
    )

    assert imported_tables == ["SourceRelease", "Model", "SourceSet", "Source"]
    assert imported_records["Model"][0]["defaultSourceSetName"] is None
    assert deferred_updates == [
        {
            "id": "model-a",
            "defaultSourceSetName": "source-set-a",
            "defaultGraphSourceSetName": "source-set-a",
            "defaultSourceId": "9-source-set-a",
        }
    ]
    assert summary.bundle_row_counts == {
        "SourceRelease": 1,
        "Model": 1,
        "SourceSet": 1,
        "Source": 1,
    }
    assert summary.imported_row_counts == summary.bundle_row_counts


def test_compare_neuronpedia_export_bundle_to_import_summary_tracks_attempted_rows() -> (
    None
):
    export_root = Path("/tmp/reference_bundle")
    bundle_summary = db_import.NeuronpediaExportBundleSummary(
        export_root=export_root,
        table_row_counts={"Model": 1, "Activation": 2},
        processed_files=[
            export_root / "model.jsonl",
            export_root / "activations" / "batch.jsonl.gz",
        ],
        elapsed_seconds=0.25,
        table_load_seconds={"Model": 0.01, "Activation": 0.02},
    )
    import_summary = db_import.NeuronpediaLocalImportSummary(
        export_root=export_root,
        bundle_row_counts={"Model": 1, "Activation": 2},
        imported_row_counts={"Activation": 1},
        processed_files=[
            export_root / "model.jsonl",
            export_root / "activations" / "batch.jsonl.gz",
        ],
        elapsed_seconds=0.5,
        table_load_seconds={"Model": 0.01, "Activation": 0.02},
        table_import_seconds={"Model": 0.03, "Activation": 0.04},
    )

    parity = db_import.compare_neuronpedia_export_bundle_to_import_summary(
        bundle_summary, import_summary
    )

    assert parity.import_attempted_row_counts == {"Model": 1, "Activation": 2}
    assert parity.imported_row_counts == {"Activation": 1}
    assert parity.row_count_mismatches == {}
    assert parity.skipped_existing_row_counts == {"Model": 1, "Activation": 1}


def test_capture_neuronpedia_local_db_snapshot_collects_counts_and_stable_samples() -> (
    None
):
    class _Cursor:
        def __init__(self) -> None:
            self.operations: list[dict[str, Any]] = [
                {"fetchone": {"row_count": 2}},
                {
                    "description": [
                        SimpleNamespace(name="modelId"),
                        SimpleNamespace(name="layer"),
                        SimpleNamespace(name="index"),
                        SimpleNamespace(name="frac_nonzero"),
                    ],
                    "fetchall": [("model-a", "9-source-a", 0, 0.125)],
                },
                {"fetchone": {"row_count": 1}},
                {
                    "fetchall": [
                        {
                            "id": "act-0-0",
                            "modelId": "model-a",
                            "layer": "9-source-a",
                            "index": 0,
                            "tokens": ["tok_1"],
                            "values": [0.5],
                        }
                    ]
                },
            ]
            self.execute_calls: list[tuple[Any, Any]] = []
            self.current_operation: dict[str, Any] = {}
            self.description: list[Any] | None = None

        def __enter__(self) -> _Cursor:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def execute(self, query: object, params: object | None = None) -> None:
            self.execute_calls.append((query, params))
            self.current_operation = self.operations[len(self.execute_calls) - 1]
            self.description = self.current_operation.get("description")

        def fetchone(self) -> Any:
            return self.current_operation["fetchone"]

        def fetchall(self) -> list[Any]:
            return self.current_operation["fetchall"]

    class _Connection:
        def __init__(self) -> None:
            self.cursor_instance = _Cursor()

        def cursor(self) -> _Cursor:
            return self.cursor_instance

    sample_specs = (
        db_import.NeuronpediaLocalDBTableSampleSpec(
            table_name="Neuron",
            sample_columns=("modelId", "layer", "index", "frac_nonzero"),
            order_by=("modelId", "layer", "index"),
        ),
        db_import.NeuronpediaLocalDBTableSampleSpec(
            table_name="Activation",
            sample_columns=("id", "modelId", "layer", "index", "tokens", "values"),
            order_by=("modelId", "layer", "index", "id"),
        ),
    )
    connection = _Connection()

    snapshot = db_import.capture_neuronpedia_local_db_snapshot(
        connection,
        label="lazy-columnar",
        table_sample_specs=sample_specs,
        sample_limit=3,
    )

    assert snapshot.label == "lazy-columnar"
    assert snapshot.table_row_counts == {"Neuron": 2, "Activation": 1}
    assert snapshot.table_samples == {
        "Neuron": [
            {
                "modelId": "model-a",
                "layer": "9-source-a",
                "index": 0,
                "frac_nonzero": 0.125,
            }
        ],
        "Activation": [
            {
                "id": "act-0-0",
                "modelId": "model-a",
                "layer": "9-source-a",
                "index": 0,
                "tokens": ["tok_1"],
                "values": [0.5],
            }
        ],
    }
    assert connection.cursor_instance.execute_calls[1][1] == (3,)
    assert connection.cursor_instance.execute_calls[3][1] == (3,)


def test_compare_neuronpedia_local_db_snapshots_reports_count_and_sample_mismatches() -> (
    None
):
    reference_snapshot = db_import.NeuronpediaLocalDBSnapshot(
        label="eager-legacy",
        table_row_counts={"Neuron": 2, "Activation": 1},
        table_samples={
            "Neuron": [{"modelId": "model-a", "layer": "9-source-a", "index": 0}],
            "Activation": [{"id": "act-0-0", "values": [0.5]}],
        },
        elapsed_seconds=0.1,
    )
    candidate_snapshot = db_import.NeuronpediaLocalDBSnapshot(
        label="lazy-columnar",
        table_row_counts={"Neuron": 3, "Activation": 1},
        table_samples={
            "Neuron": [{"modelId": "model-a", "layer": "9-source-a", "index": 1}],
            "Activation": [{"id": "act-0-0", "values": [0.5]}],
        },
        elapsed_seconds=0.1,
    )

    parity = db_import.compare_neuronpedia_local_db_snapshots(
        reference_snapshot, candidate_snapshot
    )

    assert parity.reference_label == "eager-legacy"
    assert parity.candidate_label == "lazy-columnar"
    assert parity.row_count_mismatches == {
        "Neuron": {"reference_rows": 2, "candidate_rows": 3}
    }
    assert parity.sample_mismatches == {
        "Neuron": {
            "reference_rows": [
                {"modelId": "model-a", "layer": "9-source-a", "index": 0}
            ],
            "candidate_rows": [
                {"modelId": "model-a", "layer": "9-source-a", "index": 1}
            ],
        }
    }


def test_benchmark_neuronpedia_export_bundle_local_db_modes_covers_arrow_parquet_and_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_root = tmp_path / "bundle"
    export_root.mkdir()
    _write_jsonl_gz(export_root / "model.jsonl.gz", [{"id": "model-1"}])
    activations_dir = export_root / "activations"
    activations_dir.mkdir()
    activation_jsonl_path = activations_dir / "batch-0.jsonl.gz"
    _write_jsonl_gz(
        activation_jsonl_path, [{"id": "activation-1"}, {"id": "activation-2"}]
    )
    activation_arrow_path = activations_dir / "batch-0.arrow"
    activation_arrow_path.write_bytes(b"arrow")
    activation_parquet_path = activations_dir / "batch-0.parquet"
    activation_parquet_path.write_bytes(b"parquet")

    imported_calls: list[tuple[str, object]] = []

    class _Connection:
        def __init__(self) -> None:
            self.rollback_calls = 0
            self.commit_calls = 0

        def rollback(self) -> None:
            self.rollback_calls += 1

        def commit(self) -> None:
            self.commit_calls += 1

        def __enter__(self) -> _Connection:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

    connection = _Connection()

    def _fake_import_records(
        connection: Any,
        table_name: str,
        records: list[dict[str, Any]],
        *,
        chunk_size: int = 65000,
    ) -> int:
        imported_calls.append((table_name, records))
        return len(records)

    def _fake_import_model_records(
        connection: Any,
        records: list[dict[str, Any]],
        *,
        chunk_size: int = 65000,
    ) -> tuple[int, set[str], list[dict[str, Any]]]:
        imported_calls.append(("Model", records))
        return len(records), {record["id"] for record in records}, []

    def _fake_arrow_jsonb_import(
        connection: Any,
        table_name: str,
        path: Path,
        *,
        chunk_size: int = 65000,
    ) -> tuple[int, int, float, float]:
        imported_calls.append((f"{table_name}:arrow_jsonb", path))
        return 2, 2, 0.01, 0.02

    def _fake_parquet_jsonb_import(
        connection: Any,
        table_name: str,
        path: Path,
        *,
        chunk_size: int = 65000,
    ) -> tuple[int, int, float, float]:
        imported_calls.append((f"{table_name}:parquet_jsonb", path))
        return 2, 2, 0.01, 0.02

    def _fake_arrow_copy_import(
        connection: Any,
        table_name: str,
        path: Path,
        *,
        chunk_size: int = 65000,
        substage_seconds: dict[str, float] | None = None,
        use_stage_table: bool = True,
    ) -> tuple[int, int, float, float]:
        imported_calls.append((f"{table_name}:arrow_copy:{use_stage_table}", path))
        if substage_seconds is not None:
            substage_seconds["copy_write"] = 0.03
        return 2, 2, 0.01, 0.02

    def _fake_parquet_copy_import(
        connection: Any,
        table_name: str,
        path: Path,
        *,
        chunk_size: int = 65000,
        substage_seconds: dict[str, float] | None = None,
        use_stage_table: bool = True,
    ) -> tuple[int, int, float, float]:
        imported_calls.append((f"{table_name}:parquet_copy:{use_stage_table}", path))
        if substage_seconds is not None:
            substage_seconds["copy_write"] = 0.04
        return 2, 2, 0.01, 0.02

    monkeypatch.setattr(db_import, "import_records_local_db", _fake_import_records)
    monkeypatch.setattr(
        db_import, "import_model_records_local_db", _fake_import_model_records
    )
    monkeypatch.setattr(
        db_import, "import_arrow_records_local_db", _fake_arrow_jsonb_import
    )
    monkeypatch.setattr(
        db_import, "import_parquet_records_local_db", _fake_parquet_jsonb_import
    )
    monkeypatch.setattr(
        db_import, "import_arrow_records_copy_local_db", _fake_arrow_copy_import
    )
    monkeypatch.setattr(
        db_import, "import_parquet_records_copy_local_db", _fake_parquet_copy_import
    )
    monkeypatch.setitem(
        sys.modules,
        "psycopg",
        SimpleNamespace(connect=lambda *_args, **_kwargs: connection),
    )

    summaries = db_import.benchmark_neuronpedia_export_bundle_local_db_modes(
        export_root,
        local_db_url="postgres://postgres:postgres@127.0.0.1:5433/postgres",
    )

    assert list(summaries) == [
        "jsonl",
        "activation_arrow_jsonb",
        "activation_parquet_jsonb",
        "activation_arrow_copy",
        "activation_arrow_copy_direct",
        "activation_parquet_copy",
        "activation_parquet_copy_direct",
    ]
    assert connection.rollback_calls == 7
    assert connection.commit_calls == 0
    assert ("Activation:arrow_jsonb", activation_arrow_path) in imported_calls
    assert ("Activation:parquet_jsonb", activation_parquet_path) in imported_calls
    assert ("Activation:arrow_copy:True", activation_arrow_path) in imported_calls
    assert ("Activation:arrow_copy:False", activation_arrow_path) in imported_calls
    assert ("Activation:parquet_copy:True", activation_parquet_path) in imported_calls
    assert ("Activation:parquet_copy:False", activation_parquet_path) in imported_calls
    assert summaries["activation_arrow_copy"].table_import_substage_seconds == {
        "Activation": {"copy_write": 0.03}
    }
    assert summaries["activation_parquet_copy"].table_import_substage_seconds == {
        "Activation": {"copy_write": 0.04}
    }
    assert summaries["activation_arrow_copy_direct"].table_import_substage_seconds == {
        "Activation": {"copy_write": 0.03}
    }
    assert summaries[
        "activation_parquet_copy_direct"
    ].table_import_substage_seconds == {"Activation": {"copy_write": 0.04}}


def test_benchmark_neuronpedia_export_bundle_local_db_modes_can_commit_single_clean_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_root = tmp_path / "bundle"
    export_root.mkdir()
    _write_jsonl_gz(export_root / "model.jsonl.gz", [{"id": "model-1"}])

    class _Connection:
        def __init__(self) -> None:
            self.rollback_calls = 0
            self.commit_calls = 0

        def rollback(self) -> None:
            self.rollback_calls += 1

        def commit(self) -> None:
            self.commit_calls += 1

        def __enter__(self) -> _Connection:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

    connection = _Connection()
    monkeypatch.setattr(
        db_import,
        "import_model_records_local_db",
        lambda connection, records, *, chunk_size=65000: (
            len(records),
            {record["id"] for record in records},
            [],
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "psycopg",
        SimpleNamespace(connect=lambda *_args, **_kwargs: connection),
    )

    summaries = db_import.benchmark_neuronpedia_export_bundle_local_db_modes(
        export_root,
        local_db_url="postgres://postgres:postgres@127.0.0.1:5433/postgres",
        import_modes={
            "jsonl_clean_commit": {
                "prefer_arrow_for_tables": (),
                "prefer_copy_for_tables": (),
            }
        },
        rollback_each_mode=False,
    )

    assert list(summaries) == ["jsonl_clean_commit"]
    assert connection.rollback_calls == 0
    assert connection.commit_calls == 1


def test_benchmark_neuronpedia_export_bundle_local_db_modes_rejects_multi_mode_no_rollback(
    tmp_path: Path,
) -> None:
    export_root = tmp_path / "bundle"
    export_root.mkdir()

    with pytest.raises(ValueError, match="requires exactly one import mode"):
        db_import.benchmark_neuronpedia_export_bundle_local_db_modes(
            export_root,
            local_db_url="postgres://postgres:postgres@127.0.0.1:5433/postgres",
            rollback_each_mode=False,
        )


def test_iter_jsonb_record_payloads_splits_oversized_chunks() -> None:
    records = [{"id": index, "values": "x" * 40} for index in range(8)]

    default_payloads = list(
        db_import.iter_jsonb_record_payloads(records, chunk_size=4)
    )
    assert len(default_payloads) == 2
    combined = [record for payload in default_payloads for record in json.loads(payload)]
    assert combined == records

    # Force byte-limit splitting: each 4-record chunk serializes to ~260 bytes.
    split_payloads = list(
        db_import.iter_jsonb_record_payloads(
            records, chunk_size=4, max_payload_bytes=150
        )
    )
    assert len(split_payloads) == 4
    for payload in split_payloads:
        assert len(payload) <= 150
    combined = [record for payload in split_payloads for record in json.loads(payload)]
    assert combined == records


def test_iter_jsonb_record_payloads_raises_for_single_oversized_record() -> None:
    records = [{"id": 0, "values": "x" * 500}]

    with pytest.raises(db_import.NeuronpediaLocalDBImportError, match="jsonb payload limit"):
        list(
            db_import.iter_jsonb_record_payloads(
                records, chunk_size=10, max_payload_bytes=100
            )
        )
