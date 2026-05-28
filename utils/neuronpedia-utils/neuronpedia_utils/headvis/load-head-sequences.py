import argparse
import json
import os
import re
from dataclasses import astuple
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dotenv
import psycopg2
from cuid2 import Cuid
from model_head_sequence import ModelHeadSequence
from psycopg2.extras import execute_values

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
    "interval",
    "tokens",
    "attentionIndices",
    "attentionValues",
    "maxActivation",
    "createdAt",
    "updatedAt",
)

HEAD_FILENAME_RE = re.compile(r"^L(\d+)H(\d+)\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load HeadVis-style sampled sequences into Postgres. Walks "
            "<exports-dir>/<np_model_id>/headvis/<dataset>/heads/L*H*.json "
            "and inserts ModelHeadSequence rows. Errors if any rows already "
            "exist for a (model, dataset, run-config, layer, head) combination."
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
        help="Number of rows to insert per database batch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse files and run safety checks without writing.",
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
            heads_dir = dataset_dir / "heads"
            if not heads_dir.is_dir():
                continue
            run_dirs.append(dataset_dir)

    if not run_dirs:
        raise ValueError(
            f"No HeadVis run directories found under {exports_dir}. "
            "Expected <exports-dir>/<np_id>/headvis/<dataset>/."
        )
    return run_dirs


def _normalize_pipeline_config(config: dict[str, Any]) -> dict[str, Any]:
    pipeline = config.get("pipeline_config")
    if isinstance(pipeline, dict):
        merged: dict[str, Any] = dict(pipeline)
        for key, value in config.items():
            if key == "pipeline_config":
                continue
            merged.setdefault(key, value)
        return merged
    return dict(config)


def require_config(config: dict[str, Any], key: str) -> Any:
    value = config.get(key)
    if value is None:
        raise ValueError(f"Run config is missing required key '{key}'.")
    return value


def _resolve_np_model_id(run_dir: Path, config: dict[str, Any]) -> str:
    explicit = config.get("np_model_id")
    if isinstance(explicit, str) and explicit:
        return explicit
    return run_dir.parent.parent.name


def _iter_head_files(run_dir: Path) -> list[tuple[int, int, Path]]:
    heads_dir = run_dir / "heads"
    out: list[tuple[int, int, Path]] = []
    for path in sorted(heads_dir.glob("L*H*.json")):
        match = HEAD_FILENAME_RE.match(path.name)
        if not match:
            continue
        out.append((int(match.group(1)), int(match.group(2)), path))
    return out


def _build_rows_for_head(
    config: dict[str, Any],
    np_model_id: str,
    layer: int,
    head: int,
    head_payload: dict[str, Any],
    now: datetime,
) -> list[ModelHeadSequence]:
    sequences = head_payload.get("sequences")
    if not isinstance(sequences, list):
        return []

    rows: list[ModelHeadSequence] = []
    for seq in sequences:
        if not isinstance(seq, dict):
            continue
        tokens = seq.get("tokens")
        attention_indices = seq.get("attention_indices")
        attention_values = seq.get("attention_values")
        if (
            not isinstance(tokens, list)
            or not isinstance(attention_indices, list)
            or not isinstance(attention_values, list)
        ):
            continue
        if len(attention_indices) != len(attention_values):
            raise ValueError(
                f"attention_indices/values length mismatch in L{layer}H{head} "
                f"(seq interval={seq.get('interval')})"
            )

        rows.append(
            ModelHeadSequence(
                id=CUID_GENERATOR.generate(),
                modelId=np_model_id,
                layer=layer,
                headIndex=head,
                modelName=str(config["model_name"]),
                datasetName=str(config["dataset_name"]),
                nSequences=int(config["n_sequences"]),
                seqLen=int(config["seq_len"]),
                dtype=str(config["dtype"]),
                attnImplementation=str(config["attn_implementation"]),
                interval=int(seq.get("interval", 0)),
                tokens=[str(token) for token in tokens],
                attentionIndices=[int(idx) for idx in attention_indices],
                attentionValues=[float(value) for value in attention_values],
                maxActivation=float(seq.get("max_activation", 0.0)),
                createdAt=now,
                updatedAt=now,
            )
        )
    return rows


def build_rows_from_run_dir(run_dir: Path) -> tuple[dict[str, Any], str, list[ModelHeadSequence]]:
    config_raw = load_json(run_dir / "config.json")
    if not isinstance(config_raw, dict):
        raise ValueError(f"{run_dir / 'config.json'} must contain a JSON object.")
    config = _normalize_pipeline_config(config_raw)

    for key in CONFIG_KEYS:
        require_config(config, key)

    np_model_id = _resolve_np_model_id(run_dir, config)
    now = datetime.now(timezone.utc)

    rows: list[ModelHeadSequence] = []
    for layer, head, path in _iter_head_files(run_dir):
        payload = load_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a JSON object.")
        rows.extend(
            _build_rows_for_head(config, np_model_id, layer, head, payload, now)
        )
    return config, np_model_id, rows


def precheck_no_existing_rows(
    cur,
    np_model_id: str,
    config: dict[str, Any],
) -> int:
    """Return the count of pre-existing rows for this (model, dataset, run-config)."""
    cur.execute(
        """
        SELECT COUNT(*) FROM "ModelHeadSequence"
        WHERE "modelId" = %s
          AND "datasetName" = %s
          AND "nSequences" = %s
          AND "seqLen" = %s
          AND "dtype" = %s
          AND "attnImplementation" = %s
        """,
        (
            np_model_id,
            str(config["dataset_name"]),
            int(config["n_sequences"]),
            int(config["seq_len"]),
            str(config["dtype"]),
            str(config["attn_implementation"]),
        ),
    )
    (count,) = cur.fetchone()
    return int(count)


def insert_rows(rows: list[ModelHeadSequence], batch_size: int) -> None:
    if not rows:
        print("No rows to insert.")
        return

    quoted_columns = ", ".join(f'"{column}"' for column in COLUMNS)
    query = f'INSERT INTO "ModelHeadSequence" ({quoted_columns}) VALUES %s'

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
    exports_dir = Path(args.exports_dir).expanduser().resolve()
    run_dirs = discover_run_dirs(exports_dir)

    runs: list[tuple[Path, dict[str, Any], str, list[ModelHeadSequence]]] = []
    total_rows = 0
    for run_dir in run_dirs:
        config, np_model_id, rows = build_rows_from_run_dir(run_dir)
        runs.append((run_dir, config, np_model_id, rows))
        total_rows += len(rows)
        print(
            f"Prepared {len(rows)} sequence rows from "
            f"{run_dir.relative_to(exports_dir)} "
            f"(modelId={np_model_id}, dataset={config['dataset_name']})."
        )

    print(f"Prepared {total_rows} total sequence rows from {len(run_dirs)} runs.")

    if total_rows == 0:
        print("Nothing to import.")
        return

    conn = create_connection()
    try:
        with conn.cursor() as cur:
            for run_dir, config, np_model_id, _rows in runs:
                existing = precheck_no_existing_rows(cur, np_model_id, config)
                if existing > 0:
                    raise RuntimeError(
                        f"Refusing to import {run_dir.relative_to(exports_dir)}: "
                        f"{existing} ModelHeadSequence rows already exist for "
                        f"modelId={np_model_id}, datasetName={config['dataset_name']}, "
                        f"nSequences={config['n_sequences']}, seqLen={config['seq_len']}, "
                        f"dtype={config['dtype']}, attnImplementation={config['attn_implementation']}. "
                        "Delete those rows first or skip this run."
                    )
    finally:
        conn.close()

    if args.dry_run:
        print("Dry run complete; no database writes performed.")
        return

    all_rows: list[ModelHeadSequence] = []
    for _run_dir, _config, _np, rows in runs:
        all_rows.extend(rows)

    insert_rows(all_rows, args.batch_size)
    print(
        f"Imported {len(all_rows)} ModelHeadSequence rows from "
        f"{len(runs)} run directories."
    )


if __name__ == "__main__":
    main()
