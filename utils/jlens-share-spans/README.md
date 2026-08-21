# J-lens share span backfill

A one-off migration that adds per-token chat spans (`role` / `section` / `channel` /
`message_index`) to J-lens share blobs stored before those spans existed.

## Why

The webapp used to rebuild chat bubbles from token strings with per-model-family tables (`<|im_start|>`
for ChatML, `<start_of_turn>` for Gemma, harmony markers for gpt-oss). The inference overhaul replaced
that with spans computed server-side from the model's real chat template, so a `chat` share with no
spans now renders as plain, uninteractive text instead of hoverable lens chips — which is every share
created up to that point, including all the curated demos and the tour.

Rather than teach the frontend to guess again, this backfills the spans the server would send today
into the stored blobs.

## What it does, and why it is safe to run before the deploy

For each `chat` share it re-renders the blob's stored `messages` through the model's chat template
with the engine's `Tokenize.message_spans` (the same call `apps/inference/.../lens/prompt.py` makes)
and **requires the rendered token ids to equal the blob's stored ids exactly**. That equality is the
proof of correctness: a blob either aligns, in which case its spans are the server's own, or it is
skipped and reported. Positions past the prompt are the model's generation, which no template can
know, and go through `GeneratedTurnSpans` exactly as the endpoint streams them.

The rewrite is **strictly additive** — the only difference between the old and new blob is those four
keys on each token, enforced by `assert_additive` before anything is uploaded. That is what makes it
safe to run while production is still serving the pre-overhaul commit: that build validates a blob as
"has `kind`, has a `tokens` array" and groups from token strings, so fields it has never heard of are
inert. A backfilled blob renders identically on the deployed site and correctly on the new code.

`is_generated` and the database's `numPromptTokens` are deliberately **not** touched, because the
deployed code does read those (it rebuilds `is_generated` from `numPromptTokens`), so writing them
would change what production renders today. The prompt/generated boundary that alignment discovers is
reported as `prompt_len` instead. Nothing is ever written to the database.

Tokenizers only: no GPU, no model weights, no inference server.

## Running it

Run from a checkout of `interp-engine`, whose environment already has `interp_engine` and `transformers`; `--with`
layers on the two extras this script needs without modifying that environment.

```bash
cd interp-engine

# HF_TOKEN for the gated tokenizers (Gemma, Llama); AWS + Postgres for the shares themselves.
set -a && . ./.env && . ../apps/webapp/.env.prod && set +a

# 1. Dry run: fetch, align, verify, write every rewritten blob locally. Uploads nothing.
uv run --with boto3 --with 'psycopg[binary]' \
  python ../utils/jlens-share-spans/backfill_spans.py --out-dir ~/jlens-span-backfill

# 2. Read the report, then apply. Originals are kept under OUT_DIR/original as the backup.
uv run --with boto3 --with 'psycopg[binary]' \
  python ../utils/jlens-share-spans/backfill_spans.py --out-dir ~/jlens-span-backfill --apply
```

Useful flags: `--share-id` / `--model-id` (repeatable) and `--limit` to scope a run, `--model-map
MODEL=REPO` when a blob's `meta.model` is a short name with no known Hugging Face repo, `--report` to
place the JSON report elsewhere.

The exit code is non-zero if any share was skipped or failed, since a skipped share is one that stays
plain.

## Reading the report

`OUT_DIR/report.json` has one entry per share:

| field | meaning |
| --- | --- |
| `status` | `ready` (dry run, verified), `uploaded`, `already-has-spans`, `skipped`, `fetch-failed`, `upload-failed` |
| `detail` | why it was skipped — usually that no render reproduces the stored ids |
| `prompt_len` | how many leading positions the template explained; the rest were generated |
| `template_kwargs` | the kwargs whose render matched (e.g. `enable_thinking: false`) |
| `is_prefill` | whether the aligning shape was a continued assistant prefill |
| `steer_prompt_len` | same, for the share's steered re-run, when it has one |

## The tour's swap fixture

The guided tour's scripted `spiders → ants` swap does not call inference: `runLensSteer` in
`apps/webapp/components/jlens/jlens-chat.tsx` loads the committed export
`apps/webapp/public/qwen-output.json` instead, so the swap is instant and deterministic. That file is
an export from before spans existed, so it hit the same plain render — except worse than a share
does, because the steered column's spanless fallback drew the *unsteered* `messages`, showing "Eight"
where the swap had produced "Six".

A fixture has no `messages` of its own, so they come from the share it was exported from, and the
same token-id equality has to hold against the fixture's own stream — a mismatched share fails rather
than mislabelling tokens:

```bash
cd interp-engine
set -a && . ./.env && . ../apps/webapp/.env.prod && set +a
uv run --with boto3 --with 'psycopg[binary]' python ../utils/jlens-share-spans/backfill_spans.py \
  --fixture ../apps/webapp/public/qwen-output.json \
  --fixture-share-id cmr2kx72r000mpt2x71l23e0z --apply
```

Without `--apply` it verifies and reports without writing. The file is rewritten in place, staying
compact and single-line so the diff is the added span keys rather than a reformat; git holds the
original, so no `--out-dir` is needed.

## Verifying a backfilled blob

The blobs under `OUT_DIR/updated` can be fed straight through the webapp's real grouping code
(`groupTokensBySpans` in `apps/webapp/components/jlens/jlens-chat-format.ts`) to confirm they produce
the expected user/assistant bubbles before uploading anything. After `--apply`, load the share in the
webapp: `/{modelId}/jlens?shareId={id}`.

## After the deploy

Shares created by the current code carry spans already, so this is a one-time cleanup rather than
something to keep running.
