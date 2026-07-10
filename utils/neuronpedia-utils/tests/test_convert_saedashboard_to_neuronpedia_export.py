from __future__ import annotations

import gzip
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    import polars as pl
except ImportError:
    pl = None

# polars backs only the converter's opt-in --emit-arrow legacy-sidecar mode (default off; the
# production columnar lane writes/reads Arrow via pyarrow). Tests exercising that mode skip
# cleanly in environments without polars installed.
requires_polars = pytest.mark.skipif(
    pl is None, reason="polars not installed (only needed for the optional --emit-arrow mode)"
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    PACKAGE_ROOT / "neuronpedia_utils" / "convert-saedashboard-to-neuronpedia-export.py"
)


def _load_converter_module():
    os.environ.setdefault("DEFAULT_CREATOR_ID", "test-creator")
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    spec = importlib.util.spec_from_file_location(
        "convert_saedashboard_export", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_converter_uses_fallback_creator_id_outside_package_cwd(tmp_path: Path) -> None:
    previous_creator_id = os.environ.pop("DEFAULT_CREATOR_ID", None)
    previous_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        if str(PACKAGE_ROOT) not in sys.path:
            sys.path.insert(0, str(PACKAGE_ROOT))
        spec = importlib.util.spec_from_file_location(
            "convert_saedashboard_export_fallback", MODULE_PATH
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.DEFAULT_CREATOR_ID == "clkht01d40000jv08hvalcvly"
    finally:
        os.chdir(previous_cwd)
        if previous_creator_id is None:
            os.environ.pop("DEFAULT_CREATOR_ID", None)
        else:
            os.environ["DEFAULT_CREATOR_ID"] = previous_creator_id


@requires_polars
def test_converter_cli_runs_outside_package_cwd(tmp_path: Path) -> None:
    module = _load_converter_module()

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    export_root = tmp_path / "custom_exports"
    (input_dir / "batch-0.json").write_text(
        json.dumps(_sample_batch_data()),
        encoding="utf-8",
    )

    args = {
        "saedashboard_output_dir": str(input_dir),
        "export_root": str(export_root),
        "creator_name": "Google DeepMind",
        "release_id": "gemma-scope-2",
        "release_title": "Gemma Scope 2",
        "url": "https://huggingface.co/google/gemma-scope-2-1b-it",
        "model_name": "gemma-3-1b-it",
        "model_layers": 26,
        "neuronpedia_source_set_id": "gemmascope-2-transcoder-262k-rte",
        "neuronpedia_source_set_description": "Transcoder - 262k (RTE)",
        "hf_weights_repo_id": "google/gemma-scope-2-1b-it",
        "hf_weights_path": "transcoder_all/weights.safetensors",
        "hook_point": module.HOOK_POINT_TYPE_CHOICES.hook_mlp_in,
        "layer_num": 9,
        "prompts_huggingface_dataset_path": "aps/super_glue:rte[train]#pretokenized=/tmp/pretok",
        "n_prompts_total": 2490,
        "n_tokens_in_prompt": 128,
        "zero_out_bos_token": False,
        "emit_arrow": True,
    }
    previous_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        module.main(SimpleNamespace(params=args), **args)
    finally:
        os.chdir(previous_cwd)

    expected_export_root = (
        export_root / "gemma-3-1b-it" / "9-gemmascope-2-transcoder-262k-rte"
    )
    assert (expected_export_root / "features" / "batch-0.arrow").exists()
    assert (expected_export_root / "activations" / "batch-0.arrow").exists()


@requires_polars
def test_converter_cli_prefers_neuronpedia_env_export_root(
    tmp_path: Path, monkeypatch
) -> None:
    export_root = tmp_path / "np_exports"
    monkeypatch.setenv("NEURONPEDIA_EXPORT_ROOT", str(export_root))
    module = _load_converter_module()

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "batch-0.json").write_text(
        json.dumps(_sample_batch_data()),
        encoding="utf-8",
    )

    args = {
        "saedashboard_output_dir": str(input_dir),
        "export_root": None,
        "creator_name": "Google DeepMind",
        "release_id": "gemma-scope-2",
        "release_title": "Gemma Scope 2",
        "url": "https://huggingface.co/google/gemma-scope-2-1b-it",
        "model_name": "gemma-3-1b-it",
        "model_layers": 26,
        "neuronpedia_source_set_id": "gemmascope-2-transcoder-262k-rte",
        "neuronpedia_source_set_description": "Transcoder - 262k (RTE)",
        "hf_weights_repo_id": "google/gemma-scope-2-1b-it",
        "hf_weights_path": "transcoder_all/weights.safetensors",
        "hook_point": module.HOOK_POINT_TYPE_CHOICES.hook_mlp_in,
        "layer_num": 9,
        "prompts_huggingface_dataset_path": "aps/super_glue:rte[train]#pretokenized=/tmp/pretok",
        "n_prompts_total": 2490,
        "n_tokens_in_prompt": 128,
        "zero_out_bos_token": False,
        "emit_arrow": True,
    }
    previous_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        module.main(SimpleNamespace(params=args), **args)
    finally:
        os.chdir(previous_cwd)

    expected_export_root = (
        export_root / "gemma-3-1b-it" / "9-gemmascope-2-transcoder-262k-rte"
    )
    assert (expected_export_root / "features" / "batch-0.arrow").exists()
    assert (expected_export_root / "activations" / "batch-0.arrow").exists()


def _read_jsonl_gz(path: Path) -> list[dict[str, object]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _without_keys(rows: list[dict[str, object]], *keys: str) -> list[dict[str, object]]:
    excluded = set(keys)
    return [
        {key: value for key, value in row.items() if key not in excluded}
        for row in rows
    ]


def _sample_batch_data() -> dict[str, object]:
    return {
        "model_id": "gemma-3-1b-it",
        "layer": 9,
        "sae_id_suffix": "",
        "features": [
            {
                "feature_index": 12,
                "vector": [0.25, -0.5],
                "neuron_alignment_indices": [1, 2],
                "neuron_alignment_values": [0.1, 0.2],
                "neuron_alignment_l1": [0.3, 0.4],
                "correlated_neurons_indices": [4],
                "correlated_neurons_pearson": [0.7],
                "correlated_neurons_l1": [0.8],
                "correlated_features_indices": [9],
                "correlated_features_pearson": [0.6],
                "correlated_features_l1": [0.5],
                "neg_str": ["bad"],
                "neg_values": [-1.2],
                "pos_str": ["good"],
                "pos_values": [1.4],
                "frac_nonzero": 0.25,
                "freq_hist_data_bar_heights": [1.0, 2.0],
                "freq_hist_data_bar_values": [0.1, 0.2],
                "logits_hist_data_bar_heights": [3.0, 4.0],
                "logits_hist_data_bar_values": [0.3, 0.4],
                "decoder_weights_dist": [0.9, 1.1],
                "activations": [
                    {
                        "tokens": ["hello", "world"],
                        "values": [0.25, 1.5],
                        "bin_contains": [0, 1],
                        "bin_max": 1.5,
                        "bin_min": 0.0,
                        "qualifying_token_index": 1,
                        "dfa_values": [0.0, 0.1],
                        "loss_values": [1.2, 0.8],
                        "logit_contributions": [0.01, 0.02],
                    }
                ],
            }
        ],
    }


@requires_polars
def test_write_flat_table_rows_arrow_matches_legacy_metadata_row(
    tmp_path: Path,
) -> None:
    module = _load_converter_module()
    row = module._normalize_export_row(
        {
            "id": "gemma-3-1b-it",
            "createdAt": datetime(2026, 5, 5, 12, 30, 0),
            "layers": 26,
            "defaultGraphSourceSetName": "gemmascope-2-transcoder-262k-rte",
        }
    )

    legacy_jsonl_path = tmp_path / "legacy" / "model.jsonl"
    legacy_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    arrow_jsonl_path = tmp_path / "arrow" / "model.jsonl"
    arrow_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    module._write_flat_table_rows([row], str(legacy_jsonl_path), emit_arrow=False)
    _, arrow_path = module._write_flat_table_rows(
        [row], str(arrow_jsonl_path), emit_arrow=True
    )

    legacy_rows = _read_jsonl_gz(legacy_jsonl_path.with_suffix(".jsonl.gz"))
    arrow_rows = _read_jsonl_gz(arrow_jsonl_path.with_suffix(".jsonl.gz"))
    assert arrow_rows == legacy_rows
    assert arrow_path is not None
    assert [
        module._normalize_export_row(record)
        for record in pl.read_ipc(arrow_path).to_dicts()
    ] == legacy_rows


@requires_polars
def test_write_flat_table_rows_removes_stale_arrow_when_disabled(tmp_path: Path) -> None:
    module = _load_converter_module()
    row = module._normalize_export_row({"id": "gemma-3-1b-it", "layers": 26})
    jsonl_path = tmp_path / "model.jsonl"
    stale_batch_arrow = tmp_path / "activations" / "batch-9.arrow"
    stale_batch_arrow.parent.mkdir()
    stale_batch_arrow.write_bytes(b"stale")

    _, arrow_path = module._write_flat_table_rows([row], str(jsonl_path), emit_arrow=True)
    assert arrow_path is not None
    assert Path(arrow_path).exists()

    module._remove_arrow_sidecars(str(tmp_path))
    _, disabled_arrow_path = module._write_flat_table_rows([row], str(jsonl_path), emit_arrow=False)

    assert disabled_arrow_path is None
    assert not Path(arrow_path).exists()
    assert not stale_batch_arrow.exists()


@requires_polars
def test_process_data_arrow_jsonl_matches_legacy_rows(tmp_path: Path) -> None:
    module = _load_converter_module()
    module.DEFAULT_CREATOR_ID = "test-creator"
    module.created_at = datetime(2026, 5, 5, 10, 0, 0)
    module.ZERO_OUT_BOS_TOKEN = False
    module.VECTOR_STEER_HOOK_NAME = "hook_mlp_in"

    batch_data = _sample_batch_data()

    legacy_root = tmp_path / "legacy"
    module.OUTPUT_PATH_BASE = str(legacy_root)
    module.process_data(
        batch_data, "gemmascope-2-transcoder-262k-rte", "batch-0", emit_arrow=False
    )

    arrow_root = tmp_path / "arrow"
    module.OUTPUT_PATH_BASE = str(arrow_root)
    module.process_data(
        batch_data, "gemmascope-2-transcoder-262k-rte", "batch-0", emit_arrow=True
    )

    source_id = "9-gemmascope-2-transcoder-262k-rte"
    legacy_source_dir = legacy_root / source_id
    arrow_source_dir = arrow_root / source_id

    legacy_feature_rows = _read_jsonl_gz(
        legacy_source_dir / "features" / "batch-0.jsonl.gz"
    )
    arrow_feature_rows = _read_jsonl_gz(
        arrow_source_dir / "features" / "batch-0.jsonl.gz"
    )
    legacy_activation_rows = _read_jsonl_gz(
        legacy_source_dir / "activations" / "batch-0.jsonl.gz"
    )
    arrow_activation_rows = _read_jsonl_gz(
        arrow_source_dir / "activations" / "batch-0.jsonl.gz"
    )

    assert arrow_feature_rows == legacy_feature_rows
    assert _without_keys(arrow_activation_rows, "id") == _without_keys(
        legacy_activation_rows, "id"
    )

    feature_arrow_path = arrow_source_dir / "features" / "batch-0.arrow"
    activation_arrow_path = arrow_source_dir / "activations" / "batch-0.arrow"
    assert feature_arrow_path.exists()
    assert activation_arrow_path.exists()
    assert [
        module._normalize_export_row(record)
        for record in pl.read_ipc(feature_arrow_path).to_dicts()
    ] == legacy_feature_rows
    assert [
        module._normalize_export_row(record)
        for record in pl.read_ipc(activation_arrow_path).to_dicts()
    ] == arrow_activation_rows


@requires_polars
def test_benchmark_flat_table_write_modes_exposes_pre_db_arrow_timing(
    tmp_path: Path,
) -> None:
    module = _load_converter_module()
    row = module._normalize_export_row(
        {
            "id": "gemma-3-1b-it",
            "createdAt": datetime(2026, 5, 5, 12, 30, 0),
            "layers": 26,
            "defaultGraphSourceSetName": "gemmascope-2-transcoder-262k-rte",
        }
    )

    results = module.benchmark_flat_table_write_modes([row], tmp_path, "model")

    assert results["row_count"] == 1
    assert set(results["modes"]) == {
        "legacy_jsonl_gzip",
        "arrow_ipc_only",
        "arrow_bundle",
    }
    legacy_path = results["modes"]["legacy_jsonl_gzip"]["path"]
    arrow_only_path = results["modes"]["arrow_ipc_only"]["path"]
    bundle_arrow_path = results["modes"]["arrow_bundle"]["arrow_path"]
    bundle_jsonl_path = results["modes"]["arrow_bundle"]["jsonl_gzip_path"]

    assert legacy_path.exists()
    assert arrow_only_path.exists()
    assert bundle_arrow_path.exists()
    assert bundle_jsonl_path.exists()

    legacy_rows = _read_jsonl_gz(legacy_path)
    bundle_rows = _read_jsonl_gz(bundle_jsonl_path)

    assert legacy_rows == bundle_rows
    assert [
        module._normalize_export_row(record)
        for record in pl.read_ipc(arrow_only_path).to_dicts()
    ] == legacy_rows
    assert [
        module._normalize_export_row(record)
        for record in pl.read_ipc(bundle_arrow_path).to_dicts()
    ] == legacy_rows
