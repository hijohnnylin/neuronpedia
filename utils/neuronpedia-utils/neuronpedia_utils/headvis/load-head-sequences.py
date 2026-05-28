import argparse
import json
import os
import random
import re
import string
import time
from dataclasses import astuple
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dotenv
import psycopg2
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

class _FastCuid:
    """Drop-in replacement for cuid2.Cuid that's ~100x faster.

    Real cuid2 uses SHA-3-512 hashing per ID, which dominates CPU time during
    bulk imports (one ID per sequence row, often tens of thousands per run).
    This produces IDs with the same surface format (lowercase alphanumeric,
    starting with a letter) using random.choices. At length 25 we have ~124
    bits of entropy, so collisions are astronomically unlikely for our
    import scale.
    """

    _FIRST = string.ascii_lowercase
    _REST = string.ascii_lowercase + string.digits

    def __init__(self, length: int = 24) -> None:
        if length < 2:
            raise ValueError("length must be >= 2")
        self._rest_len = length - 1

    def generate(self) -> str:
        return random.choice(self._FIRST) + "".join(
            random.choices(self._REST, k=self._rest_len)
        )


CUID_GENERATOR = _FastCuid(length=25)

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
            "and inserts ModelHeadSequence rows. Skips with a warning any "
            "(model, dataset, run-config, layer, head) combination that "
            "already has rows; a summary of skipped runs is printed at the end."
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


try:
    import orjson  # type: ignore[import-not-found]

    def load_json(path: Path) -> Any:
        with open(path, "rb") as f:
            return orjson.loads(f.read())
except ImportError:

    def load_json(path: Path) -> Any:
        with open(path, "rb") as f:
            return json.loads(f.read())


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def discover_run_dirs(exports_dir: Path) -> list[Path]:
    if not exports_dir.is_dir():
        raise ValueError(f"Exports directory does not exist: {exports_dir}")

    _log(f"Scanning exports directory: {exports_dir}")
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
    _log(f"Discovered {len(run_dirs)} run director{'y' if len(run_dirs) == 1 else 'ies'}.")
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
    cfg_model_name: str,
    cfg_dataset_name: str,
    cfg_n_sequences: int,
    cfg_seq_len: int,
    cfg_dtype: str,
    cfg_attn_implementation: str,
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
    generate = CUID_GENERATOR.generate
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
                id=generate(),
                modelId=np_model_id,
                layer=layer,
                headIndex=head,
                modelName=cfg_model_name,
                datasetName=cfg_dataset_name,
                nSequences=cfg_n_sequences,
                seqLen=cfg_seq_len,
                dtype=cfg_dtype,
                attnImplementation=cfg_attn_implementation,
                interval=int(seq.get("interval", 0)),
                tokens=tokens,
                attentionIndices=attention_indices,
                attentionValues=attention_values,
                maxActivation=float(seq.get("max_activation", 0.0)),
                createdAt=now,
                updatedAt=now,
            )
        )
    return rows


def _read_run_config(run_dir: Path) -> tuple[dict[str, Any], str]:
    """Parse config.json for a run dir and resolve the np_model_id."""
    config_raw = load_json(run_dir / "config.json")
    if not isinstance(config_raw, dict):
        raise ValueError(f"{run_dir / 'config.json'} must contain a JSON object.")
    config = _normalize_pipeline_config(config_raw)
    for key in CONFIG_KEYS:
        require_config(config, key)
    np_model_id = _resolve_np_model_id(run_dir, config)
    return config, np_model_id


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


def _insert_query() -> str:
    quoted_columns = ", ".join(f'"{column}"' for column in COLUMNS)
    return f'INSERT INTO "ModelHeadSequence" ({quoted_columns}) VALUES %s'


def _stream_insert_run(
    cur,
    query: str,
    run_dir: Path,
    config: dict[str, Any],
    np_model_id: str,
    batch_size: int,
    dry_run: bool,
) -> int:
    """Stream rows for one run dir into the DB in batches.

    Reads head files one at a time, accumulates rows in a buffer of
    `batch_size`, flushes to the DB whenever full, then drops the batch.
    Returns the number of rows inserted (or that would be inserted, when
    `dry_run` is True).
    """
    cfg_model_name = str(config["model_name"])
    cfg_dataset_name = str(config["dataset_name"])
    cfg_n_sequences = int(config["n_sequences"])
    cfg_seq_len = int(config["seq_len"])
    cfg_dtype = str(config["dtype"])
    cfg_attn_implementation = str(config["attn_implementation"])
    now = datetime.now(timezone.utc)

    head_files = _iter_head_files(run_dir)
    total_heads = len(head_files)
    if total_heads:
        _log(
            f"  Streaming {total_heads} head JSON file(s) from "
            f"{(run_dir / 'heads').name}/ ..."
        )
    progress_step = max(1, total_heads // 10) if total_heads else 1

    inserted = 0
    buffer: list[ModelHeadSequence] = []

    def _flush_full_batches() -> None:
        nonlocal inserted
        while len(buffer) >= batch_size:
            batch = buffer[:batch_size]
            if not dry_run:
                execute_values(cur, query, [astuple(row) for row in batch])
            del buffer[:batch_size]
            inserted += len(batch)
            _log(
                f"    {'Would insert' if dry_run else 'Inserted'} "
                f"batch of {len(batch)} (run total so far: {inserted})."
            )

    for idx, (layer, head, path) in enumerate(head_files, start=1):
        payload = load_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a JSON object.")
        buffer.extend(
            _build_rows_for_head(
                cfg_model_name,
                cfg_dataset_name,
                cfg_n_sequences,
                cfg_seq_len,
                cfg_dtype,
                cfg_attn_implementation,
                np_model_id,
                layer,
                head,
                payload,
                now,
            )
        )
        _flush_full_batches()

        if total_heads and (idx % progress_step == 0 or idx == total_heads):
            _log(
                f"    ...read {idx}/{total_heads} head files "
                f"(buffered={len(buffer)}, inserted so far={inserted})."
            )

    if buffer:
        if not dry_run:
            execute_values(cur, query, [astuple(row) for row in buffer])
        inserted += len(buffer)
        _log(
            f"    {'Would insert' if dry_run else 'Inserted'} "
            f"final batch of {len(buffer)} (run total: {inserted})."
        )
        buffer.clear()

    return inserted


def main() -> None:
    overall_start = time.monotonic()
    args = parse_args()
    exports_dir = Path(args.exports_dir).expanduser().resolve()
    _log(
        f"Starting load-head-sequences (dry_run={args.dry_run}, "
        f"batch_size={args.batch_size})."
    )
    run_dirs = discover_run_dirs(exports_dir)

    _log(
        "Reading run configs and pre-checking for existing rows in the "
        "database (no head JSON files read yet)..."
    )
    warnings: list[str] = []
    runs_to_import: list[tuple[Path, dict[str, Any], str]] = []
    conn = create_connection()
    try:
        with conn.cursor() as cur:
            for index, run_dir in enumerate(run_dirs, start=1):
                _log(
                    f"  [{index}/{len(run_dirs)}] Checking "
                    f"{run_dir.relative_to(exports_dir)} ..."
                )
                config, np_model_id = _read_run_config(run_dir)
                existing = precheck_no_existing_rows(cur, np_model_id, config)
                if existing > 0:
                    warning = (
                        f"Skipping {run_dir.relative_to(exports_dir)}: "
                        f"{existing} ModelHeadSequence rows already exist for "
                        f"modelId={np_model_id}, datasetName={config['dataset_name']}, "
                        f"nSequences={config['n_sequences']}, seqLen={config['seq_len']}, "
                        f"dtype={config['dtype']}, attnImplementation={config['attn_implementation']}."
                    )
                    _log(f"  WARNING: {warning}")
                    warnings.append(warning)
                    continue
                _log(f"  [{index}/{len(run_dirs)}] OK; queued for streaming insert.")
                runs_to_import.append((run_dir, config, np_model_id))
    finally:
        conn.close()

    _log(
        f"Pre-check complete: {len(runs_to_import)} run(s) to import, "
        f"{len(warnings)} skipped."
    )

    if not runs_to_import:
        _log("No runs to import.")
        _print_warning_summary(warnings)
        return

    query = _insert_query()
    grand_total = 0

    if args.dry_run:
        _log("Dry run: streaming through head files without writing to DB...")
        for run_index, (run_dir, config, np_model_id) in enumerate(runs_to_import, start=1):
            run_start = time.monotonic()
            _log(
                f"[{run_index}/{len(runs_to_import)}] Validating "
                f"{run_dir.relative_to(exports_dir)} "
                f"(modelId={np_model_id}, dataset={config['dataset_name']})..."
            )
            inserted = _stream_insert_run(
                cur=None,
                query=query,
                run_dir=run_dir,
                config=config,
                np_model_id=np_model_id,
                batch_size=args.batch_size,
                dry_run=True,
            )
            grand_total += inserted
            _log(
                f"[{run_index}/{len(runs_to_import)}] Would insert {inserted} rows "
                f"({time.monotonic() - run_start:.2f}s). Running total: {grand_total}."
            )
        _log(
            f"Dry run complete in {time.monotonic() - overall_start:.2f}s; "
            f"would have inserted {grand_total} ModelHeadSequence rows from "
            f"{len(runs_to_import)} run directories."
        )
        _print_warning_summary(warnings)
        return

    _log(
        f"Streaming inserts for {len(runs_to_import)} run dir(s) "
        f"(batch_size={args.batch_size}, commit per run dir)..."
    )
    conn = create_connection()
    try:
        with conn.cursor() as cur:
            for run_index, (run_dir, config, np_model_id) in enumerate(runs_to_import, start=1):
                run_start = time.monotonic()
                _log(
                    f"[{run_index}/{len(runs_to_import)}] Processing "
                    f"{run_dir.relative_to(exports_dir)} "
                    f"(modelId={np_model_id}, dataset={config['dataset_name']})..."
                )
                try:
                    inserted = _stream_insert_run(
                        cur=cur,
                        query=query,
                        run_dir=run_dir,
                        config=config,
                        np_model_id=np_model_id,
                        batch_size=args.batch_size,
                        dry_run=False,
                    )
                except Exception:
                    _log(
                        f"[{run_index}/{len(runs_to_import)}] Error mid-stream; "
                        "rolling back this run dir and aborting."
                    )
                    conn.rollback()
                    raise
                _log(
                    f"[{run_index}/{len(runs_to_import)}] Committing "
                    f"{inserted} row(s) for "
                    f"{run_dir.relative_to(exports_dir)}..."
                )
                conn.commit()
                grand_total += inserted
                _log(
                    f"[{run_index}/{len(runs_to_import)}] Done in "
                    f"{time.monotonic() - run_start:.2f}s. "
                    f"Cumulative inserted: {grand_total}."
                )
    finally:
        conn.close()

    _log(
        f"Imported {grand_total} ModelHeadSequence rows from "
        f"{len(runs_to_import)} run directories in "
        f"{time.monotonic() - overall_start:.2f}s."
    )
    _print_warning_summary(warnings)


def _print_warning_summary(warnings: list[str]) -> None:
    if not warnings:
        return
    print("")
    print(f"Summary: skipped {len(warnings)} run(s) due to pre-existing rows:")
    for warning in warnings:
        print(f"  - {warning}")


if __name__ == "__main__":
    main()
