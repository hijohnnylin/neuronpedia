"""Decodes byte-level BPE token strings in ICA Lens dashboard data.

The dashboard export kept GPT-2/Qwen byte-level vocab strings (e.g. "èĻ½çĦ¶") and replaced only
the space and newline marks. This turns them into real text (e.g. "虽然"). A token that holds
part of a split UTF-8 character gets U+FFFD, as the tokenizer's decode() does.

    files: rewrites the export files, keeps the originals in BACKUP, and writes the changed rows
           to CHANGES.
    db:    applies CHANGES to the database in DB_URL, in small batches.
"""

import argparse
import gzip
import json
import os
import shutil
from multiprocessing import Pool
from pathlib import Path

EXPORTS = Path("/data/exports")
BACKUP = Path("/data/exports-bpe-orig")
CHANGES = Path("/data/icalens-bpe-fix")
MODELS = ["gpt2-small", "qwen3.5-2b-pt", "qwen3.5-9b-pt", "qwen3.6-27b"]
BATCH_ROWS = 2000


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's map from each byte to a printable unicode character."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs], strict=True))


BYTE_OF = {v: k for k, v in bytes_to_unicode().items()}
# The export already turned the space and newline marks into real characters.
BYTE_OF[" "] = 0x20
BYTE_OF["\n"] = 0x0A


def fix_token(token: str) -> str:
    """Returns the real text of a byte-level token, or the token unchanged if it is not one."""
    if not any(0xA1 <= ord(c) <= 0x143 for c in token):
        return token
    try:
        raw = bytes(BYTE_OF[c] for c in token)
    except KeyError:
        return token
    text = raw.decode("utf-8", errors="replace")
    # Postgres text cannot hold NUL.
    return token if "\x00" in text else text


def fix_tokens(tokens: list[str] | None) -> list[str] | None:
    """Fixes each token in a list."""
    return None if tokens is None else [fix_token(t) for t in tokens]


def rewrite_folder(src: Path, dst: Path, kind: str, changes_out) -> tuple[int, int]:
    """Writes fixed copies of every batch file in src to dst; returns (rows, changed rows)."""
    dst.mkdir(parents=True, exist_ok=True)
    rows = changed = 0
    for f in sorted(src.glob("*.jsonl.gz")):
        with (
            gzip.open(f, "rt") as fin,
            gzip.open(dst / f.name, "wt", compresslevel=6) as fout,
        ):
            for line in fin:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows += 1
                if kind == "activations":
                    new = fix_tokens(row["tokens"])
                    if new != row["tokens"]:
                        row["tokens"] = new
                        changed += 1
                        changes_out.write(
                            json.dumps(
                                {"id": row["id"], "tokens": new}, ensure_ascii=False
                            )
                            + "\n"
                        )
                else:
                    pos, neg = (
                        fix_tokens(row.get("pos_str")),
                        fix_tokens(row.get("neg_str")),
                    )
                    if pos != row.get("pos_str") or neg != row.get("neg_str"):
                        row["pos_str"], row["neg_str"] = pos, neg
                        changed += 1
                        changes_out.write(
                            json.dumps(
                                {
                                    "index": str(row["index"]),
                                    "pos_str": pos,
                                    "neg_str": neg,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows, changed


def fix_source(source_dir: Path) -> str:
    """Fixes one source's files. The originals move to BACKUP, so a second run is a no-op."""
    rel = source_dir.relative_to(EXPORTS)
    backup = BACKUP / rel
    out = CHANGES / rel
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for kind in ("features", "activations"):
        live = source_dir / kind
        orig = backup / kind
        if not orig.exists():
            orig.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(live, orig)
        tmp = source_dir / f"{kind}.fixing"
        if tmp.exists():
            shutil.rmtree(tmp)
        with gzip.open(out / f"{kind}.jsonl.gz", "wt", compresslevel=6) as changes_out:
            rows, changed = rewrite_folder(orig, tmp, kind, changes_out)
        shutil.rmtree(live)
        tmp.rename(live)
        report.append(f"{kind} {changed}/{rows}")
    return f"{rel}: " + ", ".join(report)


def run_files() -> None:
    """Fixes every affected source's export files."""
    dirs = [d for m in MODELS for d in sorted((EXPORTS / m).glob("*-icalens-res"))]
    print(f"{len(dirs)} sources", flush=True)
    with Pool(12) as pool:
        for msg in pool.imap_unordered(fix_source, dirs):
            print(msg, flush=True)


def batches(path: Path):
    """Yields lists of up to BATCH_ROWS change rows from a changes file."""
    batch = []
    with gzip.open(path, "rt") as fin:
        for line in fin:
            batch.append(json.loads(line))
            if len(batch) >= BATCH_ROWS:
                yield batch
                batch = []
    if batch:
        yield batch


def run_db() -> None:
    """Applies the recorded changes to the database in DB_URL."""
    import psycopg

    activation_sql = (
        'UPDATE "Activation" a SET tokens = v.tokens '
        "FROM jsonb_to_recordset(%s::jsonb) AS v(id text, tokens text[]) "
        'WHERE a.id = v.id AND a."modelId" = %s AND a.layer = %s'
    )
    neuron_sql = (
        'UPDATE "Neuron" n SET pos_str = v.pos_str, neg_str = v.neg_str '
        "FROM jsonb_to_recordset(%s::jsonb) AS v(index text, pos_str text[], neg_str text[]) "
        'WHERE n."modelId" = %s AND n.layer = %s AND n.index = v.index'
    )
    with psycopg.connect(os.environ["DB_URL"], autocommit=True) as conn:
        for model in MODELS:
            for source_dir in sorted((CHANGES / model).glob("*-icalens-res")):
                layer = source_dir.name
                counts = {}
                for kind, sql in (
                    ("features", neuron_sql),
                    ("activations", activation_sql),
                ):
                    expected = updated = 0
                    for batch in batches(source_dir / f"{kind}.jsonl.gz"):
                        cur = conn.execute(
                            sql, (json.dumps(batch, ensure_ascii=False), model, layer)
                        )
                        expected += len(batch)
                        updated += cur.rowcount
                    counts[kind] = (updated, expected)
                print(
                    f"{model} {layer}: "
                    + ", ".join(f"{k} {u}/{e}" for k, (u, e) in counts.items()),
                    flush=True,
                )
                if any(u != e for u, e in counts.values()):
                    raise SystemExit(f"row count mismatch for {model} {layer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("step", choices=["files", "db"])
    args = parser.parse_args()
    run_files() if args.step == "files" else run_db()
