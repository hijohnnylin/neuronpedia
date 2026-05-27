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
from psycopg2.extras import execute_values

dotenv.load_dotenv(".env.default")
dotenv.load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_MAP_PATH = SCRIPT_DIR / "np_model_to_hf.json"
DEFAULT_METRICS_DIR = SCRIPT_DIR / "head-metrics"
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
    "createdAt",
    "updatedAt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load HeadVis-style attention head metrics JSON into Postgres."
    )
    parser.add_argument(
        "metrics_dir",
        nargs="?",
        default=str(DEFAULT_METRICS_DIR),
        help="Directory containing JSON files written by compute-head-metrics.py.",
    )
    parser.add_argument(
        "--model-map",
        default=str(DEFAULT_MODEL_MAP_PATH),
        help="Path to np_model_to_hf.json.",
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


def discover_metrics_files(metrics_dir: Path) -> list[Path]:
    if not metrics_dir.is_dir():
        raise ValueError(f"Metrics directory does not exist: {metrics_dir}")

    metrics_files = sorted(metrics_dir.glob("*.json"))
    if not metrics_files:
        raise ValueError(f"No metrics JSON files found in {metrics_dir}")
    return metrics_files


def resolve_model_id(hf_model_name: str, model_map_path: Path) -> str:
    model_map = load_json(model_map_path)
    if not isinstance(model_map, dict):
        raise ValueError(f"{model_map_path} must contain a JSON object.")

    matches = [
        np_model_id
        for np_model_id, mapped_hf_model_name in model_map.items()
        if mapped_hf_model_name == hf_model_name
    ]
    if not matches:
        raise ValueError(
            f"Could not find Hugging Face model '{hf_model_name}' in {model_map_path} values."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Found multiple Neuronpedia model ids for Hugging Face model '{hf_model_name}': {matches}"
        )
    return matches[0]


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


def validate_metric_matrix(metric_name: str, value: Any) -> list[list[Any]]:
    if not isinstance(value, list) or not all(isinstance(row, list) for row in value):
        raise ValueError(f"metrics.{metric_name} must be a 2D array.")

    expected_heads = len(value[0]) if value else 0
    for layer_index, row in enumerate(value):
        if len(row) != expected_heads:
            raise ValueError(
                f"metrics.{metric_name}[{layer_index}] has {len(row)} heads, "
                f"expected {expected_heads}."
            )
    return value


def build_rows(
    metrics_payload: dict[str, Any], model_map_path: Path
) -> list[ModelHeadMetrics]:
    config = metrics_payload.get("config")
    metrics = metrics_payload.get("metrics")
    if not isinstance(config, dict):
        raise ValueError("Metrics file must contain a config object.")
    if not isinstance(metrics, dict):
        raise ValueError("Metrics file must contain a metrics object.")

    for key in CONFIG_KEYS:
        require_config(config, key)

    metric_matrices = {
        metric_name: validate_metric_matrix(metric_name, metrics.get(metric_name))
        for metric_name in METRIC_KEYS
    }

    first_metric = metric_matrices[METRIC_KEYS[0]]
    n_layers = len(first_metric)
    n_heads = len(first_metric[0]) if first_metric else 0
    for metric_name, matrix in metric_matrices.items():
        if len(matrix) != n_layers:
            raise ValueError(
                f"metrics.{metric_name} has {len(matrix)} layers, expected {n_layers}."
            )
        for layer_index, row in enumerate(matrix):
            if len(row) != n_heads:
                raise ValueError(
                    f"metrics.{metric_name}[{layer_index}] has {len(row)} heads, "
                    f"expected {n_heads}."
                )

    hf_model_name = str(config["model_name"])
    model_id = resolve_model_id(hf_model_name, model_map_path)
    now = datetime.now(timezone.utc)

    rows: list[ModelHeadMetrics] = []
    for layer_index in range(n_layers):
        for head_index in range(n_heads):
            rows.append(
                ModelHeadMetrics(
                    id=CUID_GENERATOR.generate(),
                    modelId=model_id,
                    layer=layer_index,
                    headIndex=head_index,
                    modelName=hf_model_name,
                    datasetName=str(config["dataset_name"]),
                    nSequences=int(config["n_sequences"]),
                    seqLen=int(config["seq_len"]),
                    dtype=str(config["dtype"]),
                    attnImplementation=str(config["attn_implementation"]),
                    selfAttentionScore=finite_float_or_none(
                        metric_matrices["self_attention_score"][layer_index][head_index]
                    ),
                    prevTokenScore=finite_float_or_none(
                        metric_matrices["prev_token_score"][layer_index][head_index]
                    ),
                    patternEntropy=finite_float_or_none(
                        metric_matrices["pattern_entropy"][layer_index][head_index]
                    ),
                    qkDistance=finite_float_or_none(
                        metric_matrices["qk_distance"][layer_index][head_index]
                    ),
                    qkDistanceVariance=finite_float_or_none(
                        metric_matrices["qk_distance_variance"][layer_index][head_index]
                    ),
                    inductionScore=finite_float_or_none(
                        metric_matrices["induction_score"][layer_index][head_index]
                    ),
                    createdAt=now,
                    updatedAt=now,
                )
            )
    return rows


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
                execute_values(cur, query, [astuple(row) for row in batch])
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main() -> None:
    args = parse_args()
    metrics_dir = Path(args.metrics_dir).expanduser().resolve()
    model_map_path = Path(args.model_map).expanduser().resolve()
    metrics_files = discover_metrics_files(metrics_dir)

    rows: list[ModelHeadMetrics] = []
    for metrics_file in metrics_files:
        metrics_payload = load_json(metrics_file)
        file_rows = build_rows(metrics_payload, model_map_path)
        rows.extend(file_rows)

        if file_rows:
            first_row = file_rows[0]
            print(
                "Prepared "
                f"{len(file_rows)} rows from {metrics_file.name} for "
                f"modelId={first_row.modelId}, modelName={first_row.modelName}, "
                f"datasetName={first_row.datasetName}."
            )
        else:
            print(f"Prepared 0 rows from {metrics_file.name}.")

    if rows:
        print(f"Prepared {len(rows)} total rows from {len(metrics_files)} metrics files.")
    else:
        print(f"Prepared 0 total rows from {len(metrics_files)} metrics files.")

    if args.dry_run:
        print("Dry run complete; no database writes performed.")
        return

    upsert_rows(rows, args.batch_size)
    print(f"Imported {len(rows)} model head metric rows from {len(metrics_files)} metrics files.")


if __name__ == "__main__":
    main()
