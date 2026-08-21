"""Backfill per-token chat spans into stored J-lens share blobs.

Every share created before the inference overhaul stored its token stream with no span metadata
(``role`` / ``section`` / ``channel`` / ``message_index``), because the concept did not exist yet.
The webapp used to rebuild chat bubbles from the token strings using per-model-family tables; it now
reads the spans the server computes, so a span-less ``chat`` share renders as plain, uninteractive
text. This script backfills those spans in place.

The spans are not guessed. For each blob we re-render its stored ``messages`` through the model's
real chat template with the engine's ``Tokenize.message_spans`` -- the same call the live lens
endpoint makes -- and require the rendered token ids to equal the blob's stored ids exactly. That
equality is the proof: a blob either aligns, in which case its prompt spans are the ones the server
would send today, or it is skipped and reported. Positions past the prompt are the model's own
generation, which no template can know, so they go through ``GeneratedTurnSpans`` exactly as the
endpoint streams them.

Two values the lens request carries were never persisted with the share (``enable_thinking``, and
whether the final assistant turn was a prefill the run continued rather than one it generated), so
we try the small set of shapes the endpoint can send and let the id alignment pick the one that was
actually used.

**The rewrite is strictly additive**, and ``_assert_additive`` enforces that before anything is
uploaded: the only difference between the old and new blob is the four span keys on each token. This
matters because the deployed webapp is still on the pre-overhaul commit -- it validates a share blob
as "has ``kind``, has a ``tokens`` array" and groups from token strings, so fields it has never heard
of are inert there. A backfilled blob renders identically on the deployed site and correctly on the
new code, which is what makes this safe to run before the deploy rather than with it.

``is_generated`` and the DB's ``numPromptTokens`` are deliberately left alone: both ARE read by the
deployed code (it rebuilds ``is_generated`` from ``numPromptTokens``), so writing them would change
what production renders today. The prompt/generated boundary that alignment discovers is reported
instead, under ``prompt_len``.

Usage (see README.md for the full environment setup):

    # dry run: compute, verify, write the rewritten blobs locally, upload nothing
    cd interp-engine && uv run --with boto3 --with 'psycopg[binary]' \
        python ../utils/jlens-share-spans/backfill_spans.py --out-dir /tmp/jlens-spans

    # same, then upload the verified blobs over their existing S3 keys
    ... python ../utils/jlens-share-spans/backfill_spans.py --out-dir /tmp/jlens-spans --apply
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import os
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from interp_engine.tokenize import GeneratedTurnSpans, Tokenize, TokenSpan

# The four keys this script adds to every token. Nothing else in the blob is touched.
SPAN_KEYS = ("message_index", "role", "channel", "section")

# `meta.model` in a blob is whatever model id the inference server ran under, which for some
# models is a short name rather than a Hugging Face repo id. Anything containing a "/" is used
# as-is; these are the shorthands that need expanding. Extend with --model-map rather than editing.
SHORT_MODEL_IDS = {
    "gemma-2-2b-it": "google/gemma-2-2b-it",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "gemma-2-2b": "google/gemma-2-2b",
    "gemma-2-9b": "google/gemma-2-9b",
    "gpt2-small": "gpt2",
}

# Only chat runs group into bubbles; completion shares render every token as a chip already and
# have no messages to render a template from.
SHARE_KIND = "chat"

HTTP_TIMEOUT_SECONDS = 60

# How far either side of a share's creation date to look for the date its template rendered with,
# covering the timezone gap between the database timestamp and the inference server's clock.
DATE_SKEW_DAYS = 1


class FrozenNow:
    """A ``strftime_now`` stand-in pinned to a fixed day.

    A template that stamps the current date into its system preamble (gpt-oss writes
    ``Current date: YYYY-MM-DD``) renders differently every day, so nothing rendered now could ever
    reproduce a run captured months ago. Jinja resolves context variables ahead of environment
    globals, so passing this through as a template kwarg pins the date the template sees. Which day
    to pin comes from the share's creation timestamp, and the id alignment is what confirms it.
    """

    def __init__(self, day: date):
        self.day = day

    def __call__(self, fmt: str) -> str:
        return self.day.strftime(fmt)

    def __str__(self) -> str:
        return self.day.isoformat()


def json_safe(template_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Template kwargs rendered for the report (a `FrozenNow` becomes its date)."""
    return {
        key: value if isinstance(value, str | int | float | bool) or value is None else str(value)
        for key, value in template_kwargs.items()
    }


@dataclass
class Share:
    """One row of `JlensShare` (read-only; this script never writes to the database)."""

    id: str
    model_id: str
    url: str
    # When the run was captured. Needed because some chat templates stamp "today" into their
    # preamble, so reproducing the stored ids means rendering as of this date (see `FrozenNow`).
    created_at: datetime


@dataclass
class PromptRender:
    """An accepted chat-template render: the prompt spans plus the shape that produced them."""

    spans: list[TokenSpan]
    is_prefill: bool
    add_generation_prompt: bool
    template_kwargs: dict[str, Any]
    num_messages: int

    @property
    def prompt_len(self) -> int:
        return len(self.spans)


@dataclass
class Result:
    """Per-share outcome, written to the JSON report."""

    id: str
    model_id: str
    status: str
    detail: str = ""
    num_tokens: int = 0
    prompt_len: int | None = None
    template_kwargs: dict[str, Any] = field(default_factory=dict)
    is_prefill: bool | None = None
    steer_prompt_len: int | None = None
    bytes_before: int = 0
    bytes_after: int = 0


# --------------------------------------------------------------------------- #
# Loading shares and blobs
# --------------------------------------------------------------------------- #


def list_shares(database_url: str, model_ids: list[str], share_ids: list[str], limit: int | None) -> list[Share]:
    """Read the chat shares to consider, oldest first. Read-only."""
    try:
        import psycopg
    except ImportError as error:  # pragma: no cover - environment problem, not logic
        raise SystemExit(
            "psycopg is required to list shares. Run this script with:\n"
            "  uv run --with boto3 --with 'psycopg[binary]' python <this script>"
        ) from error

    sql = ['SELECT id, "modelId", url, "createdAt" FROM "JlensShare" WHERE kind = %(kind)s']
    params: dict[str, Any] = {"kind": SHARE_KIND}
    if model_ids:
        sql.append('AND "modelId" = ANY(%(model_ids)s)')
        params["model_ids"] = model_ids
    if share_ids:
        sql.append("AND id = ANY(%(share_ids)s)")
        params["share_ids"] = share_ids
    sql.append('ORDER BY "createdAt"')
    if limit is not None:
        sql.append("LIMIT %(limit)s")
        params["limit"] = limit

    with psycopg.connect(database_url) as connection, connection.cursor() as cursor:
        cursor.execute(" ".join(sql), params)  # pyright: ignore[reportArgumentType]
        return [Share(id=row[0], model_id=row[1], url=row[2], created_at=row[3]) for row in cursor.fetchall()]


def fetch_blob(url: str) -> tuple[dict[str, Any], int]:
    """Fetch a share blob, returning ``(parsed, raw_byte_count)``.

    The objects are stored gzipped with ``Content-Encoding: gzip`` (so a browser gets plain JSON),
    which ``urllib`` does not transparently decode -- hence the magic-number check rather than
    trusting the header.
    """
    with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_SECONDS) as response:  # noqa: S310 - fixed S3 host
        raw = response.read()
    body = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
    return json.loads(body), len(body)


def parse_s3_url(url: str) -> tuple[str, str, str]:
    """Split a stored blob url into ``(bucket, region, key)``.

    Written by the share route as ``https://{bucket}.s3.{region}.amazonaws.com/{key}``, and the key
    must round-trip exactly so the backfilled object replaces the original rather than orphaning it.
    """
    parsed = urllib.parse.urlparse(url)
    host_parts = parsed.netloc.split(".")
    if len(host_parts) < 4 or host_parts[1] != "s3":
        raise ValueError(f"Unrecognized share url: {url}")
    bucket = host_parts[0]
    region = host_parts[2]
    key = urllib.parse.unquote(parsed.path.lstrip("/"))
    return bucket, region, key


# --------------------------------------------------------------------------- #
# Tokenizers
# --------------------------------------------------------------------------- #


class TokenizerCache:
    """Loads one tokenizer per model id (no weights, no GPU) and keeps it for reuse."""

    def __init__(self, model_map: dict[str, str]):
        self._model_map = model_map
        self._cache: dict[str, tuple[Any, Tokenize]] = {}

    def resolve_repo_id(self, model: str) -> str:
        if model in self._model_map:
            return self._model_map[model]
        if "/" in model:
            return model
        raise ValueError(f"No Hugging Face repo id known for model {model!r} (pass --model-map to add one)")

    def get(self, model: str) -> tuple[Any, Tokenize]:
        repo_id = self.resolve_repo_id(model)
        if repo_id not in self._cache:
            from transformers import AutoTokenizer

            print(f"  loading tokenizer {repo_id}", file=sys.stderr)
            tokenizer = AutoTokenizer.from_pretrained(repo_id)
            self._cache[repo_id] = (tokenizer, Tokenize(tokenizer))
        return self._cache[repo_id]


# --------------------------------------------------------------------------- #
# Span computation
# --------------------------------------------------------------------------- #


def template_kwarg_variants(tokenizer: Any, created_at: datetime) -> list[dict[str, Any]]:
    """Template kwargs to try, gated on what the template actually references.

    Mirrors the lens endpoint's `_chat_template_kwargs`: a template that doesn't mention a kwarg
    ignores or rejects it. The values in force at share time were never stored, so every combination
    the endpoint can produce is offered and the id alignment decides. The bare `{}` comes first so a
    template's own defaults win ties.
    """
    template = getattr(tokenizer, "chat_template", None) or ""
    variants: list[dict[str, Any]] = [{}]

    def expand(key: str, values: tuple[Any, ...]) -> None:
        nonlocal variants
        if key not in template:
            return
        variants = [
            *variants,
            *({**variant, key: value} for variant in variants for value in values),
        ]

    expand("enable_thinking", (False, True))
    expand("preserve_thinking", (False, True))
    # gpt-oss (harmony) has no thinking switch; the endpoint maps its boolean onto reasoning effort.
    expand("reasoning_effort", ("low", "high"))

    if "strftime_now" in template:
        # Every variant has to be pinned rather than merely offered a pinned alternative: an
        # unpinned render dates itself today, which can only match a share created today.
        day = created_at.date()
        days = [
            day,
            *(day + timedelta(days=offset) for offset in (-DATE_SKEW_DAYS, DATE_SKEW_DAYS)),
        ]
        variants = [{**variant, "strftime_now": FrozenNow(d)} for variant in variants for d in days]
    return variants


def render_shapes(
    messages: list[dict[str, str]],
) -> list[tuple[list[dict[str, str]], bool, bool]]:
    """Candidate ``(messages, add_generation_prompt, continue_final_message)`` triples, best first.

    A stored chat share holds the whole conversation including the assistant turn the model
    produced, but the run that produced it was requested with a *prompt* — so the template render
    that reproduces the stored prompt is one of:

    - everything but the final assistant turn, plus a generation prompt (the run generated it), or
    - the whole conversation with the final turn left open (the run continued a prefill), or
    - the whole conversation, for a share whose tokens were never generated at all.

    The order is a preference and the first shape that aligns wins, even where a later one would
    explain more positions. The chat UI generates that final turn, so the first shape is what
    actually happened, and it is also what the server emits for a fresh run of the same
    conversation -- which keeps a share's default and steered columns split the same way. Ranking by
    length instead picks the prefill render, which covers one more token but attributes the
    template's post-scaffold whitespace to the message body, leaving the assistant bubble opening on
    a blank line in one column and not the other.
    """
    shapes: list[tuple[list[dict[str, str]], bool, bool]] = []
    if messages and messages[-1].get("role") == "assistant":
        shapes.append((messages[:-1], True, False))
        shapes.append((messages, False, True))
    shapes.append((messages, True, False))
    shapes.append((messages, False, False))
    return [shape for shape in shapes if shape[0]]


def align_prompt(
    tokenize: Tokenize,
    tokenizer: Any,
    messages: list[dict[str, str]],
    token_ids: list[int],
    created_at: datetime,
) -> PromptRender | None:
    """Find the chat-template render whose token ids are an exact prefix of ``token_ids``.

    Shapes are tried in preference order (see `render_shapes`) and the first that aligns is
    returned; within one shape the longest match wins, so the template kwargs that explain the most
    of the prompt are the ones chosen, with a template's own defaults winning ties. Returns None
    when nothing aligns, in which case the caller must leave the blob alone rather than attach spans
    that don't describe it.
    """
    for (
        candidate_messages,
        add_generation_prompt,
        continue_final_message,
    ) in render_shapes(messages):
        best: PromptRender | None = None
        for template_kwargs in template_kwarg_variants(tokenizer, created_at):
            try:
                spans = tokenize.message_spans(
                    candidate_messages,
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=continue_final_message,
                    **template_kwargs,
                )
            except Exception:  # noqa: BLE001 - template rendering is tokenizer- and kwarg-dependent
                continue
            rendered = [span.token_id for span in spans]
            if len(rendered) > len(token_ids) or rendered != token_ids[: len(rendered)]:
                continue
            if best is None or len(rendered) > best.prompt_len:
                best = PromptRender(
                    spans=spans,
                    is_prefill=continue_final_message,
                    add_generation_prompt=add_generation_prompt,
                    template_kwargs=template_kwargs,
                    num_messages=len(candidate_messages),
                )
        if best is not None:
            return best
    return None


def annotate_tokens(tokens: list[dict[str, Any]], render: PromptRender, tokenizer: Any) -> None:
    """Write span fields onto ``tokens`` in place, mirroring the lens endpoint's `_span_fields`.

    Prompt positions take the template's spans; the rest are the model's generation, walked in order
    through the same incremental tracker the endpoint streams with (the tracker is stateful, so each
    generated position must be processed exactly once).
    """
    prompt_spans = render.spans
    # The endpoint only carries a message index into the generation when the run continued a
    # prefill; a freshly generated turn was not part of the request's messages.
    gen_message_index = (render.num_messages - 1) if render.is_prefill else None
    tracker = GeneratedTurnSpans.for_prompt(
        tokenizer,
        [span.token_str for span in prompt_spans],
        message_index=gen_message_index,
    )
    for position, token in enumerate(tokens):
        if position < len(prompt_spans):
            span = prompt_spans[position]
        else:
            span = tracker.process(position, int(token["id"]), token["token"])
        token["message_index"] = span.message_index
        token["role"] = span.role
        token["channel"] = span.channel
        token["section"] = span.section


# --------------------------------------------------------------------------- #
# Blob rewriting
# --------------------------------------------------------------------------- #


def token_ids_of(tokens: list[dict[str, Any]]) -> list[int] | None:
    """The stored token id sequence, or None if any token lacks an id (can't be aligned)."""
    ids: list[int] = []
    for token in tokens:
        token_id = token.get("id")
        if not isinstance(token_id, int):
            return None
        ids.append(token_id)
    return ids


def has_spans(tokens: list[dict[str, Any]]) -> bool:
    """Whether the stream already carries spans (making this share nothing to do)."""
    return any(token.get(key) is not None for token in tokens for key in SPAN_KEYS)


def strip_spans(blob: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``blob`` with the span keys removed from every token stream."""
    stripped = copy.deepcopy(blob)
    streams = [stripped.get("tokens")]
    steer = stripped.get("steer")
    if isinstance(steer, dict):
        streams.append(steer.get("tokens"))
    for stream in streams:
        if not isinstance(stream, list):
            continue
        for token in stream:
            for key in SPAN_KEYS:
                token.pop(key, None)
    return stripped


def assert_additive(original: dict[str, Any], updated: dict[str, Any]) -> None:
    """Fail unless the only difference is the span keys we added.

    The deployed webapp reads these blobs too, so anything beyond the additive span fields -- a
    reordered token, a reformatted number, a dropped key -- would be a change to a live page.
    """
    if strip_spans(updated) != original:
        raise ValueError("rewritten blob differs from the original beyond the span keys")


def chat_messages_of(messages: Any) -> list[dict[str, str]]:
    """The `{role, content}` pairs to render, from a blob's stored `messages`."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("no messages to render a chat template from")
    return [{"role": str(m["role"]), "content": str(m["content"])} for m in messages]


def annotate_stream(
    stream: list[dict[str, Any]],
    chat_messages: list[dict[str, str]],
    tokenizer: Any,
    tokenize: Tokenize,
    created_at: datetime,
    label: str,
) -> PromptRender:
    """Align one token stream against the chat template and write its spans in place."""
    token_ids = token_ids_of(stream)
    if token_ids is None:
        raise ValueError(f"{label} stream has tokens without ids, so no render can be verified against it")
    render = align_prompt(tokenize, tokenizer, chat_messages, token_ids, created_at)
    if render is None:
        raise ValueError(f"no chat-template render reproduces the {label} stream's token ids")
    annotate_tokens(stream, render, tokenizer)
    return render


def steer_stream_of(blob: dict[str, Any]) -> list[dict[str, Any]] | None:
    """A blob's steered re-run tokens, if it carries one."""
    steer = blob.get("steer")
    if isinstance(steer, dict) and isinstance(steer.get("tokens"), list) and steer["tokens"]:
        return steer["tokens"]
    return None


def backfill_blob(
    blob: dict[str, Any],
    tokenizer_cache: TokenizerCache,
    created_at: datetime,
) -> tuple[dict[str, Any], PromptRender, PromptRender | None]:
    """Return ``(updated_blob, main_render, steer_render)``, raising ValueError if it can't align."""
    chat_messages = chat_messages_of(blob.get("messages"))
    model = (blob.get("meta") or {}).get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("blob has no meta.model")

    tokens = blob.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("blob has no tokens")

    tokenizer, tokenize = tokenizer_cache.get(model)
    updated = copy.deepcopy(blob)
    render = annotate_stream(updated["tokens"], chat_messages, tokenizer, tokenize, created_at, "main")

    # A share can carry a steered re-run of the same conversation. Its prompt is the same, but its
    # generated tail is not, so it is aligned in its own right rather than assumed to match.
    steer_render: PromptRender | None = None
    steer_tokens = steer_stream_of(updated)
    if steer_tokens is not None:
        steer_render = annotate_stream(steer_tokens, chat_messages, tokenizer, tokenize, created_at, "steered")

    assert_additive(blob, updated)
    return updated, render, steer_render


def backfill_fixture(
    path: Path,
    share_blob: dict[str, Any],
    tokenizer_cache: TokenizerCache,
    created_at: datetime,
) -> tuple[dict[str, Any], PromptRender | None, PromptRender | None]:
    """Backfill a committed export fixture in `public/`, using a share for its messages.

    The J-lens tour's scripted swap is served from `apps/webapp/public/qwen-output.json` rather than
    from inference, and that file is an export from before spans existed -- so it hits the same plain
    render as a legacy share, except worse: the steered column's spanless fallback draws the
    *unsteered* conversation, so the tour shows the answer the swap was supposed to change.

    A fixture carries no `messages` of its own, so they come from the share it was exported from.
    Nothing about that pairing is assumed: the same token-id equality has to hold against the
    fixture's own streams, so a mismatched share fails instead of mislabelling the tokens.
    """
    fixture = json.loads(path.read_text(encoding="utf-8"))
    chat_messages = chat_messages_of(share_blob.get("messages"))
    steer_meta = (fixture.get("steer") or {}).get("meta") or {}
    model = steer_meta.get("model") or (fixture.get("meta") or {}).get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("fixture has no meta.model in either its main or steered run")

    tokenizer, tokenize = tokenizer_cache.get(model)
    updated = copy.deepcopy(fixture)

    main_render: PromptRender | None = None
    tokens = updated.get("tokens")
    if isinstance(tokens, list) and tokens:
        main_render = annotate_stream(tokens, chat_messages, tokenizer, tokenize, created_at, "main")

    steer_render: PromptRender | None = None
    steer_tokens = steer_stream_of(updated)
    if steer_tokens is not None:
        steer_render = annotate_stream(steer_tokens, chat_messages, tokenizer, tokenize, created_at, "steered")

    if main_render is None and steer_render is None:
        raise ValueError("fixture has no token stream to backfill")
    assert_additive(fixture, updated)
    return updated, main_render, steer_render


def upload_blob(bucket: str, region: str, key: str, blob: dict[str, Any]) -> int:
    """Overwrite the share's S3 object with the backfilled blob. Returns the uncompressed size.

    Content type and encoding mirror the share route: gzipped bytes served as plain JSON, so the
    browser's `fetch` still decodes it transparently.
    """
    try:
        import boto3
    except ImportError as error:  # pragma: no cover - environment problem, not logic
        raise SystemExit(
            "boto3 is required to upload. Run this script with:\n"
            "  uv run --with boto3 --with 'psycopg[binary]' python <this script>"
        ) from error

    body = json.dumps(blob, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    client = boto3.client("s3", region_name=region)
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=gzip.compress(body),
        ContentType="application/json",
        ContentEncoding="gzip",
    )
    return len(body)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--database-url",
        default=os.environ.get("POSTGRES_URL_NON_POOLING") or os.environ.get("POSTGRES_PRISMA_URL"),
        help="Postgres connection string (default: $POSTGRES_URL_NON_POOLING). Read-only.",
    )
    parser.add_argument(
        "--share-id",
        action="append",
        default=[],
        help="Only this share id (repeatable).",
    )
    parser.add_argument(
        "--model-id",
        action="append",
        default=[],
        help="Only this Neuronpedia model id (repeatable).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many shares.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for the original and rewritten blobs (originals are kept as the backup). "
        "Required unless --fixture is used, where git holds the original.",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="Backfill a committed export fixture in place (e.g. apps/webapp/public/qwen-output.json) "
        "instead of the S3 shares. Needs --fixture-share-id.",
    )
    parser.add_argument(
        "--fixture-share-id",
        default=None,
        help="The share whose messages the fixture was exported from, used to render its template.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write the JSON report here (default: OUT_DIR).",
    )
    parser.add_argument(
        "--model-map",
        action="append",
        default=[],
        help="Extra meta.model -> HF repo id mapping, as MODEL=REPO (repeatable).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Upload the verified blobs over their existing S3 keys. Without this nothing is uploaded.",
    )
    return parser.parse_args(argv)


def run_fixture(args: argparse.Namespace, tokenizer_cache: TokenizerCache) -> int:
    """Backfill one committed export fixture, reporting to stdout. Writes only under --apply."""
    if not args.fixture_share_id:
        raise SystemExit("--fixture also needs --fixture-share-id: the share it was exported from.")
    shares = list_shares(args.database_url, [], [args.fixture_share_id], None)
    if not shares:
        raise SystemExit(f"No {SHARE_KIND} share {args.fixture_share_id!r} to take messages from.")
    share = shares[0]
    share_blob, _ = fetch_blob(share.url)

    path: Path = args.fixture
    fixture = json.loads(path.read_text(encoding="utf-8"))
    streams = [fixture.get("tokens") or [], steer_stream_of(fixture) or []]
    if any(has_spans(stream) for stream in streams):
        print(f"{path} already has spans; nothing to do")
        return 0

    updated, main_render, steer_render = backfill_fixture(path, share_blob, tokenizer_cache, share.created_at)
    for label, render in (("main", main_render), ("steered", steer_render)):
        if render is not None:
            print(f"{label}: prompt_len={render.prompt_len} template_kwargs={json_safe(render.template_kwargs)}")
    if args.apply:
        # The fixture is committed compact and single-line; keep it that way so the diff is the
        # added span keys rather than a reformat of the whole file.
        path.write_text(
            json.dumps(updated, separators=(",", ":"), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {path}")
    else:
        print(f"verified, not written (pass --apply to rewrite {path})")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        raise SystemExit("No database url: pass --database-url or set POSTGRES_URL_NON_POOLING.")

    model_map = dict(SHORT_MODEL_IDS)
    for entry in args.model_map:
        model, _, repo_id = entry.partition("=")
        if not repo_id:
            raise SystemExit(f"--model-map expects MODEL=REPO, got {entry!r}")
        model_map[model] = repo_id

    if args.fixture:
        return run_fixture(args, TokenizerCache(model_map))

    if not args.out_dir:
        raise SystemExit("--out-dir is required when backfilling shares.")
    out_dir: Path = args.out_dir
    (out_dir / "original").mkdir(parents=True, exist_ok=True)
    (out_dir / "updated").mkdir(parents=True, exist_ok=True)

    shares = list_shares(args.database_url, args.model_id, args.share_id, args.limit)
    print(f"{len(shares)} {SHARE_KIND} share(s) to consider", file=sys.stderr)

    tokenizer_cache = TokenizerCache(model_map)
    results: list[Result] = []

    for index, share in enumerate(shares, start=1):
        print(f"[{index}/{len(shares)}] {share.id} ({share.model_id})", file=sys.stderr)
        try:
            blob, size_before = fetch_blob(share.url)
        except Exception as error:  # noqa: BLE001 - one bad object must not stop the run
            results.append(
                Result(
                    id=share.id,
                    model_id=share.model_id,
                    status="fetch-failed",
                    detail=str(error),
                )
            )
            continue

        tokens = blob.get("tokens") if isinstance(blob.get("tokens"), list) else []
        if has_spans(tokens):
            results.append(
                Result(
                    id=share.id,
                    model_id=share.model_id,
                    status="already-has-spans",
                    num_tokens=len(tokens),
                    bytes_before=size_before,
                )
            )
            continue

        try:
            updated, render, steer_render = backfill_blob(blob, tokenizer_cache, share.created_at)
        except Exception as error:  # noqa: BLE001 - report and move on; the blob is left untouched
            results.append(
                Result(
                    id=share.id,
                    model_id=share.model_id,
                    status="skipped",
                    detail=str(error),
                    num_tokens=len(tokens),
                    bytes_before=size_before,
                )
            )
            continue

        (out_dir / "original" / f"{share.id}.json").write_text(
            json.dumps(blob, separators=(",", ":"), ensure_ascii=False),
            encoding="utf-8",
        )
        updated_json = json.dumps(updated, separators=(",", ":"), ensure_ascii=False)
        (out_dir / "updated" / f"{share.id}.json").write_text(updated_json, encoding="utf-8")

        status = "ready"
        detail = ""
        if args.apply:
            bucket, region, key = parse_s3_url(share.url)
            try:
                upload_blob(bucket, region, key, updated)
                status = "uploaded"
            except Exception as error:  # noqa: BLE001 - report and move on
                status = "upload-failed"
                detail = str(error)

        results.append(
            Result(
                id=share.id,
                model_id=share.model_id,
                status=status,
                detail=detail,
                num_tokens=len(updated["tokens"]),
                prompt_len=render.prompt_len,
                template_kwargs=json_safe(render.template_kwargs),
                is_prefill=render.is_prefill,
                steer_prompt_len=steer_render.prompt_len if steer_render else None,
                bytes_before=size_before,
                bytes_after=len(updated_json.encode("utf-8")),
            )
        )

    report_path: Path = args.report or (out_dir / "report.json")
    report_path.write_text(json.dumps([vars(result) for result in results], indent=2), encoding="utf-8")

    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    print("\n".join(f"{status}: {count}" for status, count in sorted(counts.items())))
    print(f"report: {report_path}")
    # A skipped or failed share is a real finding (it means a share stays plain), so surface it in
    # the exit code rather than only in the report.
    return 1 if any(r.status in {"skipped", "fetch-failed", "upload-failed"} for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
