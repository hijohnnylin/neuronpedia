import argparse
import json
import math
import os
from dataclasses import astuple
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dotenv
import psycopg2
from cuid2 import Cuid
from model_head_metrics import ModelHeadMetrics
from psycopg2.extras import Json, execute_values

dotenv.load_dotenv(".env.default")
dotenv.load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EXPORTS_DIR = SCRIPT_DIR.parent / "exports"
DEFAULT_BATCH_SIZE = 1000

POSTGRES_PLACEHOLDERS = {
    "dbname": "TODO_DATABASE_NAME",
    "user": "TODO_DATABASE_USERNAME",
    "password": "TODO_DATABASE_PASSWORD",
    "host": "TODO_DATABASE_HOST",
    "port": "TODO_DATABASE_PORT",
}

CUID_GENERATOR: Cuid = Cuid(length=25)

CONFIG_KEYS = (
    "model_name",
    "dataset_name",
    "n_sequences",
    "seq_len",
    "dtype",
    "attn_implementation",
)

METRIC_KEYS = (
    "self_attention_score",
    "prev_token_score",
    "pattern_entropy",
    "qk_distance",
    "qk_distance_variance",
    "induction_score",
)

COLUMNS = (
    "id",
    "modelId",
    "layer",
    "headIndex",
    "modelName",
    "datasetName",
    "nSequences",
    "seqLen",
    "dtype",
    "attnImplementation",
    "selfAttentionScore",
    "prevTokenScore",
    "patternEntropy",
    "qkDistance",
    "qkDistanceVariance",
    "inductionScore",
    "qkDistanceHistogram",
    "topQueryTokens",
    "topKeyTokens",
    "activationHistogram",
    "headStatistics",
    "createdAt",
    "updatedAt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load HeadVis-style attention head metrics into Postgres. Walks "
            "<exports-dir>/<np_model_id>/headvis/<dataset>/ for each run "
            "written by compute-head-metrics.py and upserts ModelHeadMetrics."
        )
    )
    parser.add_argument(
        "exports_dir",
        nargs="?",
        default=str(DEFAULT_EXPORTS_DIR),
        help=(
            "Root exports directory containing <np_model_id>/headvis/<dataset> "
            "trees written by compute-head-metrics.py."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of rows to upsert per database batch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse the metrics files and print the planned import without writing.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    return args


def create_connection():
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return psycopg2.connect(database_url)

    return psycopg2.connect(
        dbname=os.getenv("DATABASE_NAME", POSTGRES_PLACEHOLDERS["dbname"]),
        user=os.getenv("DATABASE_USERNAME", POSTGRES_PLACEHOLDERS["user"]),
        password=os.getenv("DATABASE_PASSWORD", POSTGRES_PLACEHOLDERS["password"]),
        host=os.getenv("DATABASE_HOST", POSTGRES_PLACEHOLDERS["host"]),
        port=os.getenv("DATABASE_PORT", POSTGRES_PLACEHOLDERS["port"]),
    )


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def discover_run_dirs(exports_dir: Path) -> list[Path]:
    """Find <exports_dir>/<np_id>/headvis/<dataset> trees with config + scatter_data."""
    if not exports_dir.is_dir():
        raise ValueError(f"Exports directory does not exist: {exports_dir}")

    run_dirs: list[Path] = []
    for np_id_dir in sorted(exports_dir.iterdir()):
        if not np_id_dir.is_dir():
            continue
        headvis_dir = np_id_dir / "headvis"
        if not headvis_dir.is_dir():
            continue
        for dataset_dir in sorted(headvis_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            if not (dataset_dir / "config.json").is_file():
                continue
            if not (dataset_dir / "scatter_data.json").is_file():
                continue
            run_dirs.append(dataset_dir)

    if not run_dirs:
        raise ValueError(
            f"No HeadVis run directories found under {exports_dir}. "
            "Expected <exports-dir>/<np_id>/headvis/<dataset>/."
        )
    return run_dirs


def require_config(config: dict[str, Any], key: str) -> Any:
    value = config.get(key)
    if value is None:
        raise ValueError(f"Metrics config is missing required key '{key}'.")
    return value


def finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _normalize_pipeline_config(config: dict[str, Any]) -> dict[str, Any]:
    """Flatten the run config so DB import works regardless of where keys live."""
    pipeline = config.get("pipeline_config")
    if isinstance(pipeline, dict):
        merged: dict[str, Any] = dict(pipeline)
        for key, value in config.items():
            if key == "pipeline_config":
                continue
            merged.setdefault(key, value)
        return merged
    return dict(config)


def _read_head_json(run_dir: Path, layer: int, head: int) -> dict[str, Any] | None:
    head_path = run_dir / "heads" / f"L{layer}H{head}.json"
    if not head_path.is_file():
        return None
    payload = load_json(head_path)
    if not isinstance(payload, dict):
        raise ValueError(f"{head_path} must contain a JSON object.")
    return payload


def _resolve_np_model_id(run_dir: Path, config: dict[str, Any]) -> str:
    """The directory name two levels up is the np_model_id by construction."""
    explicit = config.get("np_model_id")
    if isinstance(explicit, str) and explicit:
        return explicit
    return run_dir.parent.parent.name


def build_rows_from_run_dir(run_dir: Path) -> list[ModelHeadMetrics]:
    config_raw = load_json(run_dir / "config.json")
    if not isinstance(config_raw, dict):
        raise ValueError(f"{run_dir / 'config.json'} must contain a JSON object.")
    config = _normalize_pipeline_config(config_raw)

    scatter_rows = load_json(run_dir / "scatter_data.json")
    if not isinstance(scatter_rows, list):
        raise ValueError(
            f"{run_dir / 'scatter_data.json'} must contain a JSON list of head rows."
        )

    for key in CONFIG_KEYS:
        require_config(config, key)

    hf_model_name = str(config["model_name"])
    np_model_id = _resolve_np_model_id(run_dir, config)
    now = datetime.now(timezone.utc)

    rows: list[ModelHeadMetrics] = []
    for entry in scatter_rows:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{run_dir / 'scatter_data.json'} rows must be JSON objects."
            )
        if "layer" not in entry or "head" not in entry:
            raise ValueError(
                f"{run_dir / 'scatter_data.json'} row missing 'layer' or 'head': {entry}"
            )
        layer_index = int(entry["layer"])
        head_index = int(entry["head"])

        head_json = _read_head_json(run_dir, layer_index, head_index)
        qk_hist = head_json.get("qk_distance_histogram") if head_json else None
        top_q = head_json.get("top_query_tokens") if head_json else None
        top_k = head_json.get("top_key_tokens") if head_json else None
        activation_hist = head_json.get("histogram") if head_json else None
        statistics = head_json.get("statistics") if head_json else None

        rows.append(
            ModelHeadMetrics(
                id=CUID_GENERATOR.generate(),
                modelId=np_model_id,
                layer=layer_index,
                headIndex=head_index,
                modelName=hf_model_name,
                datasetName=str(config["dataset_name"]),
                nSequences=int(config["n_sequences"]),
                seqLen=int(config["seq_len"]),
                dtype=str(config["dtype"]),
                attnImplementation=str(config["attn_implementation"]),
                selfAttentionScore=finite_float_or_none(entry.get("self_attention_score")),
                prevTokenScore=finite_float_or_none(entry.get("prev_token_score")),
                patternEntropy=finite_float_or_none(entry.get("pattern_entropy")),
                qkDistance=finite_float_or_none(entry.get("qk_distance")),
                qkDistanceVariance=finite_float_or_none(entry.get("qk_distance_variance")),
                inductionScore=finite_float_or_none(entry.get("induction_score")),
                qkDistanceHistogram=qk_hist,
                topQueryTokens=top_q,
                topKeyTokens=top_k,
                activationHistogram=activation_hist,
                headStatistics=statistics,
                createdAt=now,
                updatedAt=now,
            )
        )
    return rows


def _row_to_pg_tuple(row: ModelHeadMetrics) -> tuple[Any, ...]:
    """Convert a dataclass row into the tuple shape expected by execute_values.

    Json columns are wrapped in psycopg2's Json adapter so dicts/lists serialize.
    """
    raw = astuple(row)
    json_indices = {COLUMNS.index(name) for name in (
        "qkDistanceHistogram",
        "topQueryTokens",
        "topKeyTokens",
        "activationHistogram",
        "headStatistics",
    )}
    return tuple(
        Json(value) if i in json_indices and value is not None else value
        for i, value in enumerate(raw)
    )


def upsert_rows(rows: list[ModelHeadMetrics], batch_size: int) -> None:
    if not rows:
        print("No rows to import.")
        return

    quoted_columns = ", ".join(f'"{column}"' for column in COLUMNS)
    update_columns = (
        "modelName",
        "selfAttentionScore",
        "prevTokenScore",
        "patternEntropy",
        "qkDistance",
        "qkDistanceVariance",
        "inductionScore",
        "qkDistanceHistogram",
        "topQueryTokens",
        "topKeyTokens",
        "activationHistogram",
        "headStatistics",
        "updatedAt",
    )
    update_assignments = ", ".join(
        f'"{column}" = EXCLUDED."{column}"' for column in update_columns
    )
    query = f"""
        INSERT INTO "ModelHeadMetrics" ({quoted_columns})
        VALUES %s
        ON CONFLICT (
            "modelId",
            "datasetName",
            "nSequences",
            "seqLen",
            "dtype",
            "attnImplementation",
            "layer",
            "headIndex"
        )
        DO UPDATE SET {update_assignments}
    """

    conn = create_connection()
    try:
        with conn.cursor() as cur:
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                execute_values(cur, query, [_row_to_pg_tuple(row) for row in batch])
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main() -> None:
    args = parse_args()
    exports_dir = Path(args.exports_dir).expanduser().resolve()
    run_dirs = discover_run_dirs(exports_dir)

    rows: list[ModelHeadMetrics] = []
    for run_dir in run_dirs:
        dir_rows = build_rows_from_run_dir(run_dir)
        rows.extend(dir_rows)

        if dir_rows:
            first_row = dir_rows[0]
            print(
                "Prepared "
                f"{len(dir_rows)} rows from {run_dir.relative_to(exports_dir)} for "
                f"modelId={first_row.modelId}, modelName={first_row.modelName}, "
                f"datasetName={first_row.datasetName}."
            )
        else:
            print(f"Prepared 0 rows from {run_dir.relative_to(exports_dir)}.")

    if rows:
        print(f"Prepared {len(rows)} total rows from {len(run_dirs)} run directories.")
    else:
        print(f"Prepared 0 total rows from {len(run_dirs)} run directories.")

    if args.dry_run:
        print("Dry run complete; no database writes performed.")
        return

    upsert_rows(rows, args.batch_size)
    print(f"Imported {len(rows)} model head metric rows from {len(run_dirs)} run directories.")


if __name__ == "__main__":
    main()
