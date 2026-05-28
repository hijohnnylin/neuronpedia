import argparse
import json
import math
import os
import random
import string
import time
from dataclasses import astuple
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dotenv
import psycopg2
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

class _FastCuid:
    """Drop-in replacement for cuid2.Cuid that's ~100x faster.

    Real cuid2 uses SHA-3-512 hashing per ID, which dominates CPU time during
    bulk imports (tens of thousands of rows per run). This produces IDs with
    the same surface format (lowercase alphanumeric, starting with a letter)
    using random.choices. At length 25 we have ~124 bits of entropy, so
    collisions are astronomically unlikely for our import scale.
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
            "written by compute-head-metrics.py and inserts ModelHeadMetrics. "
            "Skips with a warning any (modelId, datasetName, nSequences, "
            "seqLen, dtype, attnImplementation, layer, headIndex) row that "
            "already exists. Streams head files into the DB in batches and "
            "commits per run dir."
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
    """Find <exports_dir>/<np_id>/headvis/<dataset> trees with config + scatter_data."""
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
            if not (dataset_dir / "scatter_data.json").is_file():
                continue
            run_dirs.append(dataset_dir)

    if not run_dirs:
        raise ValueError(
            f"No HeadVis run directories found under {exports_dir}. "
            "Expected <exports-dir>/<np_id>/headvis/<dataset>/."
        )
    _log(f"Discovered {len(run_dirs)} run director{'y' if len(run_dirs) == 1 else 'ies'}.")
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


def _existing_layer_head_pairs(
    cur,
    np_model_id: str,
    config: dict[str, Any],
) -> set[tuple[int, int]]:
    """Return the set of (layer, headIndex) pairs already in ModelHeadMetrics
    for this (modelId, datasetName, nSequences, seqLen, dtype, attnImplementation).

    The table's unique constraint extends this with (layer, headIndex), so any
    pair returned here would conflict on insert and must be skipped.
    """
    cur.execute(
        """
        SELECT "layer", "headIndex" FROM "ModelHeadMetrics"
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
    return {(int(layer), int(head)) for layer, head in cur.fetchall()}


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


def _insert_query() -> str:
    """INSERT with ON CONFLICT DO NOTHING as a safety net.

    The precheck against the unique key already filters out (layer, head)
    pairs that exist, so DO NOTHING only fires under unusual races.
    """
    quoted_columns = ", ".join(f'"{column}"' for column in COLUMNS)
    return f"""
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
        DO NOTHING
    """


def _stream_insert_run(
    cur,
    query: str,
    run_dir: Path,
    config: dict[str, Any],
    np_model_id: str,
    existing_pairs: set[tuple[int, int]],
    batch_size: int,
    dry_run: bool,
) -> tuple[int, list[tuple[int, int]]]:
    """Stream rows for one run dir into the DB in batches.

    For each (layer, head) entry in scatter_data.json: if it's in
    `existing_pairs` skip it (and record the skip), otherwise read the
    head JSON, build a ModelHeadMetrics row, append to a buffer of size
    `batch_size`, flush whenever full, then drop the batch.

    Returns (inserted_count, skipped_layer_head_pairs).
    """
    scatter_rows = load_json(run_dir / "scatter_data.json")
    if not isinstance(scatter_rows, list):
        raise ValueError(
            f"{run_dir / 'scatter_data.json'} must contain a JSON list of head rows."
        )

    hf_model_name = str(config["model_name"])
    cfg_dataset_name = str(config["dataset_name"])
    cfg_n_sequences = int(config["n_sequences"])
    cfg_seq_len = int(config["seq_len"])
    cfg_dtype = str(config["dtype"])
    cfg_attn_implementation = str(config["attn_implementation"])
    now = datetime.now(timezone.utc)

    total_entries = len(scatter_rows)
    if total_entries:
        _log(
            f"  Streaming {total_entries} scatter entr{'y' if total_entries == 1 else 'ies'} "
            f"from {run_dir.name}/ ..."
        )
    progress_step = max(1, total_entries // 10) if total_entries else 1

    inserted = 0
    skipped: list[tuple[int, int]] = []
    buffer: list[ModelHeadMetrics] = []
    generate = CUID_GENERATOR.generate

    def _flush_full_batches() -> None:
        nonlocal inserted
        while len(buffer) >= batch_size:
            batch = buffer[:batch_size]
            if not dry_run:
                execute_values(cur, query, [_row_to_pg_tuple(r) for r in batch])
            del buffer[:batch_size]
            inserted += len(batch)
            _log(
                f"    {'Would insert' if dry_run else 'Inserted'} "
                f"batch of {len(batch)} (run total so far: {inserted})."
            )

    for idx, entry in enumerate(scatter_rows, start=1):
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

        if (layer_index, head_index) in existing_pairs:
            skipped.append((layer_index, head_index))
        else:
            head_json = _read_head_json(run_dir, layer_index, head_index)
            qk_hist = head_json.get("qk_distance_histogram") if head_json else None
            top_q = head_json.get("top_query_tokens") if head_json else None
            top_k = head_json.get("top_key_tokens") if head_json else None
            activation_hist = head_json.get("histogram") if head_json else None
            statistics = head_json.get("statistics") if head_json else None

            buffer.append(
                ModelHeadMetrics(
                    id=generate(),
                    modelId=np_model_id,
                    layer=layer_index,
                    headIndex=head_index,
                    modelName=hf_model_name,
                    datasetName=cfg_dataset_name,
                    nSequences=cfg_n_sequences,
                    seqLen=cfg_seq_len,
                    dtype=cfg_dtype,
                    attnImplementation=cfg_attn_implementation,
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
            _flush_full_batches()

        if total_entries and (idx % progress_step == 0 or idx == total_entries):
            _log(
                f"    ...processed {idx}/{total_entries} entries "
                f"(buffered={len(buffer)}, inserted so far={inserted}, "
                f"skipped so far={len(skipped)})."
            )

    if buffer:
        if not dry_run:
            execute_values(cur, query, [_row_to_pg_tuple(r) for r in buffer])
        inserted += len(buffer)
        _log(
            f"    {'Would insert' if dry_run else 'Inserted'} "
            f"final batch of {len(buffer)} (run total: {inserted})."
        )
        buffer.clear()

    return inserted, skipped


def _format_skipped_warning(
    run_dir: Path,
    exports_dir: Path,
    skipped: list[tuple[int, int]],
) -> str:
    sample = ", ".join(f"L{layer}H{head}" for layer, head in skipped[:10])
    suffix = "" if len(skipped) <= 10 else f", ... (+{len(skipped) - 10} more)"
    return (
        f"In {run_dir.relative_to(exports_dir)}: skipped {len(skipped)} "
        f"(layer, head) row(s) that already existed in ModelHeadMetrics "
        f"[{sample}{suffix}]."
    )


def _print_warning_summary(warnings: list[str]) -> None:
    if not warnings:
        return
    print("")
    print(f"Summary: {len(warnings)} run(s) had pre-existing rows that were skipped:")
    for warning in warnings:
        print(f"  - {warning}")


def main() -> None:
    overall_start = time.monotonic()
    args = parse_args()
    exports_dir = Path(args.exports_dir).expanduser().resolve()
    _log(
        f"Starting load-head-metrics (dry_run={args.dry_run}, "
        f"batch_size={args.batch_size})."
    )
    run_dirs = discover_run_dirs(exports_dir)

    _log(
        "Reading run configs and querying existing (layer, head) rows in the "
        "database (no head JSON files read yet)..."
    )
    plans: list[tuple[Path, dict[str, Any], str, set[tuple[int, int]]]] = []
    conn = create_connection()
    try:
        with conn.cursor() as cur:
            for index, run_dir in enumerate(run_dirs, start=1):
                _log(
                    f"  [{index}/{len(run_dirs)}] Checking "
                    f"{run_dir.relative_to(exports_dir)} ..."
                )
                config, np_model_id = _read_run_config(run_dir)
                existing = _existing_layer_head_pairs(cur, np_model_id, config)
                if existing:
                    _log(
                        f"  [{index}/{len(run_dirs)}] {len(existing)} existing "
                        "(layer, head) row(s) found; those will be skipped."
                    )
                else:
                    _log(
                        f"  [{index}/{len(run_dirs)}] No existing rows; "
                        "queued for streaming insert."
                    )
                plans.append((run_dir, config, np_model_id, existing))
    finally:
        conn.close()

    _log(f"Pre-check complete for {len(plans)} run(s).")

    query = _insert_query()
    grand_inserted = 0
    grand_skipped = 0
    skip_warnings: list[str] = []

    if args.dry_run:
        _log("Dry run: streaming through scatter entries without writing to DB...")
        for run_index, (run_dir, config, np_model_id, existing_pairs) in enumerate(plans, start=1):
            run_start = time.monotonic()
            _log(
                f"[{run_index}/{len(plans)}] Validating "
                f"{run_dir.relative_to(exports_dir)} "
                f"(modelId={np_model_id}, dataset={config['dataset_name']})..."
            )
            inserted, skipped = _stream_insert_run(
                cur=None,
                query=query,
                run_dir=run_dir,
                config=config,
                np_model_id=np_model_id,
                existing_pairs=existing_pairs,
                batch_size=args.batch_size,
                dry_run=True,
            )
            grand_inserted += inserted
            grand_skipped += len(skipped)
            if skipped:
                warning = _format_skipped_warning(run_dir, exports_dir, skipped)
                _log(f"  WARNING: {warning}")
                skip_warnings.append(warning)
            _log(
                f"[{run_index}/{len(plans)}] Would insert {inserted} row(s), "
                f"skip {len(skipped)} ({time.monotonic() - run_start:.2f}s). "
                f"Running totals: inserted={grand_inserted}, skipped={grand_skipped}."
            )
        _log(
            f"Dry run complete in {time.monotonic() - overall_start:.2f}s; "
            f"would insert {grand_inserted} row(s), skip {grand_skipped} "
            f"existing row(s) across {len(plans)} run directories."
        )
        _print_warning_summary(skip_warnings)
        return

    _log(
        f"Streaming inserts for {len(plans)} run dir(s) "
        f"(batch_size={args.batch_size}, commit per run dir)..."
    )
    conn = create_connection()
    try:
        with conn.cursor() as cur:
            for run_index, (run_dir, config, np_model_id, existing_pairs) in enumerate(plans, start=1):
                run_start = time.monotonic()
                _log(
                    f"[{run_index}/{len(plans)}] Processing "
                    f"{run_dir.relative_to(exports_dir)} "
                    f"(modelId={np_model_id}, dataset={config['dataset_name']})..."
                )
                try:
                    inserted, skipped = _stream_insert_run(
                        cur=cur,
                        query=query,
                        run_dir=run_dir,
                        config=config,
                        np_model_id=np_model_id,
                        existing_pairs=existing_pairs,
                        batch_size=args.batch_size,
                        dry_run=False,
                    )
                except Exception:
                    _log(
                        f"[{run_index}/{len(plans)}] Error mid-stream; "
                        "rolling back this run dir and aborting."
                    )
                    conn.rollback()
                    raise

                if skipped:
                    warning = _format_skipped_warning(run_dir, exports_dir, skipped)
                    _log(f"  WARNING: {warning}")
                    skip_warnings.append(warning)

                _log(
                    f"[{run_index}/{len(plans)}] Committing {inserted} row(s) "
                    f"for {run_dir.relative_to(exports_dir)}..."
                )
                conn.commit()
                grand_inserted += inserted
                grand_skipped += len(skipped)
                _log(
                    f"[{run_index}/{len(plans)}] Done in "
                    f"{time.monotonic() - run_start:.2f}s. "
                    f"Cumulative: inserted={grand_inserted}, skipped={grand_skipped}."
                )
    finally:
        conn.close()

    _log(
        f"Imported {grand_inserted} ModelHeadMetrics row(s), skipped "
        f"{grand_skipped} existing row(s) from {len(plans)} run directories "
        f"in {time.monotonic() - overall_start:.2f}s."
    )
    _print_warning_summary(skip_warnings)


if __name__ == "__main__":
    main()
