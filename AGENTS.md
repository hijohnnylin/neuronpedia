# Neuronpedia Development Guide

This file is the single source of truth for agent instructions, and every harness reads it — see
"Agent instruction files" at the bottom before adding rules anywhere else.

## Repository layout

```
apps/
  autointerp/
  graph/
  inference/
  nla/
  sparsity/
  webapp/
    lib/api/        # generated: <app>.d.ts from each server's openapi.json
docs/               # the long-form reasoning this file's rules point at
utils/
webapp-python-client/   # hand-written public SDK
```

Neuronpedia is an interpretability website/platform: a reference for what is inside a neural
network, plus tools for researchers to experiment with those internals — viewing activations for a
given 'neuron' (or feature/latent), steering on them, tracing circuits.

`apps/` holds most of the code. Five of the six are Python FastAPI servers, called by the webapp
rather than by users directly:

- **inference** has the actual model loaded, and serves anything requiring a forward pass. The
  webapp calls it mostly from `apps/webapp/lib/utils/inference.ts`.
- **autointerp** explains and scores neurons/features/latents. Called mostly from
  `apps/webapp/lib/utils/autointerp.ts`.
- **graph**, **nla** and **sparsity** are the attribution-graph, Natural Language Autoencoder
  (activation vector <-> text) and sparse-circuit services.

**webapp** is the sixth: a Next.js app that is both the frontend and the user-facing API under
`apps/webapp/app/api`. Its database schema is `apps/webapp/prisma/schema.prisma`, and it has its
own `AGENTS.md` with frontend conventions.

**Cross-server types are generated from the Python, in one direction.** Each server's pydantic models
are the source of truth; `make openapi` regenerates every spec and every `lib/api/<app>.d.ts` from
them. There is no hand-written spec and no client package in the loop, so adding an endpoint means
writing pydantic models, not editing a schema file. The full workflow, and which servers are
camelCase on the wire versus deliberately snake_case, is under "Cross-server APIs" below.

When updating an existing function, do not change the order of its arguments unless you absolutely
need to — append new ones to the end rather than inserting them, so existing call sites keep
working.

### Committed files that are build outputs

These are checked in, so they look editable and are not. Regenerate them; never hand-edit, and
never resolve a merge conflict in one by hand:

| File | Regenerate with |
| --- | --- |
| `apps/<app>/openapi.json` | `make <app>-openapi` (or `make openapi` for all of them) |
| `apps/webapp/lib/api/*.d.ts` | `make webapp-openapi` (likewise) |
| `apps/webapp/prisma/generated/` | `prisma generate` (see the database section) |

A stale spec or `.d.ts` fails a drift test in the owning app and `openapi-drift.yml`, so getting this
wrong is loud rather than subtle.

### Configuration and `.env` files

(Distinct from "Environment (GPU / network)" below, which is about the sandbox.)

`make init-env` creates the repo-root `.env` holding your personal API keys; it layers *under* each
service's own env file rather than replacing it. The webapp additionally reads
`.env.localhost` / `.env.remote` / `.env.prod`, selected by which npm script runs. Its `dev:*`
scripts go through `scripts/dev-with-env.js`, which restarts the dev server when the selected file
changes — Next only watches the `.env*` names it owns, so an edit to `.env.prod` would otherwise sit
unread until a manual restart.

Inference is configured per model+SAE rather than per environment: the repo root holds
`.env.inference.<model>.<sourceset>` files, chosen with
`make inference-dev MODEL_SOURCESET=gpt2-small.res-jb` and listed by `make inference-list-configs`.
Adding support for a new model or source set means adding one of those files, not editing an
existing one.

### `plans/`

Scratch working notes, gitignored via `*.plan.md`; finished ones move to `plans/done/`. Nothing reads
them. Treat one as a person's thinking at one moment, not a specification, and prefer the code and
this file when they disagree.

## Build and Run Commands

Every service runs directly on the host. Run these from the repo root; `make help` lists them all.

- **Database** (Postgres 16+ with pgvector): `make db-check` for connectivity, `make db-status` for whether it is initialized, then `make db-status && make db-init` to set up an empty one. `make db-reset` is human-only — see "The database" below
- **Webapp**: `make webapp-install`, then `make webapp-dev` for hot reload, or `make webapp-build` and `make webapp-run` for a production build
- **Inference**: `make inference-install`, then `make inference-dev MODEL_SOURCESET=gpt2-small.res-jb` (add `AUTORELOAD=1` to reload on change; `make inference-list-configs` lists the available values)
- **Autointerp**: `make autointerp-install`, then `make autointerp-dev`
- **Graph**: `make graph-install`, then `make graph-dev` (port 5004)
- **NLA**: `make nla-install`, then `make nla-dev` (port 5009)
- **Sparsity**: `make sparsity-install`, then `make sparsity-dev` (port 5005)

Package managers: all five Python services use `uv`; the webapp uses `npm`.

## interp-engine is a dependency, not part of this repo

`interp-engine` is the library that owns the forward pass: hooking a model, capturing activations,
steering, the vLLM backend. Inference, graph and nla all import it as `interp_engine`. It used to
live here at `interp-engine/`; it is now its own repository,
[decoderesearch/interp-engine](https://github.com/decoderesearch/interp-engine), published to PyPI,
with its own tests, docs, lint gates and release process. Its comparison harness against
TransformerLens, nnsight, vLLM and SGLang went with it.

Each of the three apps pins an **exact version** (`interp-engine[...]==1.3.3`) in its
`pyproject.toml`. Exact rather than floored because the engine is first-party and moves in lockstep
with these services: a relock for some unrelated dependency must not silently swap the code that
serves every activation.

**To change the engine and a service that calls it in the same sitting**, use the marker-file
workflow rather than editing dependency files:

```
make engine-link APP=inference      # ENGINE_SRC=../interp-engine by default, i.e. a sibling clone
make engine-status                  # which apps are on a local checkout, and which release
make engine-unlink APP=inference    # back to the pinned release
```

`engine-link` does an editable install into that app's venv and records the path in a gitignored
`apps/<app>/.engine-linked`. Nothing tracked changes, so there is no diff to leak, and the `-dev`
targets print a banner while an app is linked so it is visible rather than something you remember.

**Never hand-edit `[tool.uv.sources]` or let a local path reach `uv.lock`.** A path source resolves
against one machine's home directory, so every other checkout and every CI job dies at install time
with `Distribution not found at: file:///...`, and what ships is whatever sat in that directory
rather than a release anyone can fetch. `.github/scripts/check_no_local_path_deps.py` rejects this
at review time; `make python-lint` runs it too.

Shipping an engine change means releasing it from that repo and bumping the `==` pin here, in a
commit that says why. The two repos share a ruff/pyright block by convention — nothing can enforce
it across the boundary any more — so a rule change in one is worth mirroring in the other.

## The database: never migrate one that already has data

The schema is `apps/webapp/prisma/schema.prisma` and migrations are committed under
`apps/webapp/prisma/migrations/`. Editing the schema is fine. **Applying it to a database that
already exists is not yours to do.**

Two commands are always safe: `prisma generate`, which rewrites the client types and touches no
database (already part of `npm run dev`, so you rarely invoke it), and `make db-status`, which is
read-only and reports whether the schema is already there.

**First-time setup is yours to do, behind that check.** `make db-status && make db-init` applies the
committed migrations, the seed and the pgvector tuning to an *empty* database. `db-status` exits
non-zero as soon as it finds a single user table, so the `&&` is what makes this safe — do not run
`db-init` on its own, because seeding a database that already has rows rewrites their contents.

Never run, and never wrap in a script an agent will run:

- `prisma migrate dev` (and the `migrate:*` npm scripts) — writes a new migration file against one
  developer's local database, which is how the committed history and production drift apart
- `prisma migrate reset` / `make db-reset` — drops the database outright
- `prisma db push` / the `db:push` script — applies the schema with no migration file. It creates
  tables but never runs a migration's data steps, so any backfill is skipped and the new tables
  come up empty
- `prisma db seed` / `make db-init` against a database `db-status` reports as non-empty
- `scripts/baseline-migrations.sh` — rewrites migration history so `migrate deploy` skips work.
  Correct exactly once per database that predates `migrate`, and wrong every other time
- `npm run build` **from inside `apps/webapp`** — its script is
  `prisma generate && prisma migrate deploy && next build`, so it migrates the database as a side
  effect of building. Use `npm run build:simple`, or `make webapp-build`, which runs
  `build:localhost` and touches no database.

So when a schema change is needed on an existing database: edit `schema.prisma`, run
`prisma generate` so the types compile, and stop there. Say that a migration is required and let a
human create and apply it.

## GPU servers are database rows

Inference, graph and NLA hosts are not configured with environment variables. They live in the
`ComputeHost` table and are resolved through `apps/webapp/lib/db/compute-host.ts`. Use
`resolveHost` / `resolveHosts` for a URL, or `computeFetch` when you want failover handled for you.
Do not add a new `USE_LOCALHOST_*` flag or read a host out of `lib/env.ts`; only the shared secrets
live there.

Registration goes through `POST /api/compute-host/register`. `new_pod.py` calls it once the pod
answers, and `make host-register` does it by hand; both need an admin API key. Locally, register a
server you started yourself with `make host-add` (and see it with `make host-list`), which writes to
Postgres directly and needs no key.

Inference pods are launched with whole SAE sets, but the resolver matches a request's individual
`sourceId`, so the route expands each set name into its member sources and links both. This is why
registration cannot be done from the pod: only the webapp can read which sources a set contains.

Registration is declarative. The sources sent replace what the host was recorded as serving, so
re-registering with a shorter list stops traffic for what was dropped.

Nothing maps a HuggingFace id to a Neuronpedia model id -- `Model` stores `tlensId` and
`openRouterId`, not the HuggingFace one, and stripping the org is wrong (`openai-community/gpt2` is
`gpt2-small` here). A pod config therefore has to state its model id under `neuronpedia.model_id` in
`pods.yaml` before it can register itself.

A registering host states which deployment it means to join, and the route refuses it when that
disagrees with `NEURONPEDIA_ENVIRONMENT`. `make host-add` refuses outright to write to a database
that is not on this machine, unless given `--remote`: developing against the production database is
normal here, so the mistake worth catching is a laptop's `127.0.0.1` landing in production's
registry and taking real traffic.

A row means the host is ready to serve. The deploy tool registers once the model is loaded and
deletes the row to stop traffic, so there is no status column -- a host that is not ready should be
absent, not present and skipped.

There is no health state either, and no heartbeats. `computeFetch` shuffles the candidates, fails
over on a 5xx, 408, 429, a timeout or a thrown error, and returns any other 4xx as-is since a
malformed request fails the same way everywhere. Failures are not remembered between requests: a
host that is down refuses connections cheaply, so the memory would only save that cheap retry on a
fraction of traffic.

Every attempt is capped by `ATTEMPT_TIMEOUT_MS[service]`, overridable per call with `timeoutMs`.
The cap is what makes failover reachable at all -- a host that accepts the connection and then
hangs would otherwise hold the serverless function until the platform kills it, and the remaining
hosts would never be tried. An abort from the caller's own signal is not treated as a host fault
and is not retried elsewhere, since the client has already gone.

Note that inference does not go through `computeFetch`. It resolves a URL with `resolveHost` /
`resolveTwoHosts` and fetches directly, because its streaming endpoints need the `ReadableStream`
intact; those calls carry the caller's `AbortSignal` instead.

What a host serves depends on the service, and the schema mirrors that. An inference process loads
a list of SAE sets and a graph process loads a transcoder and a lorsa set, so both link through
join tables. An NLA process bakes its verbalizer, reconstructor and extraction layer into startup
config and no request field can select another, so it serves exactly one `NlaSource` and carries a
`nlaSourceId` column instead. A CHECK constraint requires that column for NLA and forbids it
everywhere else; pointing two NLA sources at one host would return HTTP 200 with numbers from the
wrong model, so let the constraint stop you rather than working around it.

Routing is deliberately forgiving. A host that fails a request is sorted last for 30 seconds rather
than removed, so the last host standing is still tried. There are no heartbeats — liveness is
discovered by making the request.

## Environment (GPU / network)

This machine usually has a CUDA GPU and working internet, but agent sandboxes routinely hide both.
Do **not** conclude "no GPU" / "no CUDA" / "no network" from a sandboxed command alone.

- To check for a GPU, run outside the sandbox: `nvidia-smi`, or
  `python -c "import torch; print(torch.cuda.is_available())"`.
- The same goes for downloads and for `pip` / `uv` / `npm` installs that hit the network, Hugging
  Face included: request permission to leave the sandbox rather than skipping the GPU work or
  inventing a CPU-only plan.
- Prefer actually verifying with an unsandboxed check over assuming the environment is limited.

## Testing

- **Python services**: all five have their own `make test`, so `cd apps/<app> && make test`; run a
  single file with `cd apps/<app> && uv run pytest tests/path/to/test_file.py -v`
- **Webapp Tests**: `cd apps/webapp && npm test -- --reporter=verbose` (vitest, `*.test.ts` next to
  the code under test)

The engine's own suite is not here; it runs in its own repository. A change to how a service *uses*
the engine is tested here, a change to the engine itself is tested there.

**Always run suites verbosely, so results stream in as they finish.** A suite printing one line per
test is one a human can watch; a silent one is indistinguishable from a hang, which matters most for
the slow model-loading tests. Every Python app's `make test` already passes `-v`, so keep that flag
on ad-hoc `pytest` runs. Never pipe a run through `tail`/`head` or redirect it away from the
terminal for the same reason.

**Before running any test suite, make sure `HF_TOKEN` is in the environment.** Gated-model tests
skip themselves when it is absent, so a run without it looks green while having exercised nothing.
If the variable is unset, load it from a gitignored `.env` — the repo-root one that `make init-env`
writes, or `apps/inference/.env` — for example `set -a && . apps/inference/.env && set +a`. If
neither file exists, **tell the user before running the tests** that the gated tests will be
skipped, rather than running the suite and reporting the result as a pass.

Every Python suite runs in CI: inference and autointerp in their own workflows,
graph, nla and sparsity together in `graph-nla-sparsity-tests.yml`. The webapp's unit tests run in
`webapp-lint.yml`.

Two rules for that shared workflow, which its own comments explain at length — read them before
editing it, since its installs are deliberately partial and its dependency layout is fiddly:

- Every command after the install needs `uv run --no-sync`. A bare `uv run` syncs to the lockfile
  first and reinstalls exactly the package the install step skipped.
- When a test would go unnoticed if its dependency were missing, fail hard rather than
  `importorskip`. Graph's `circuit_tracer` extra used to skip, which collected no tests and passed.

`make openapi-check` is the cheap counterpart, reading the committed specs with no service installed
to check what is a property of the artifact: casing per service, no `number`/`integer` unions,
readable operationIds.

## Linting

- **Webapp**: `cd apps/webapp && npm run lint:fix && npm run format:write`
- **Python apps**: `make python-lint` from the repo root runs the exact gate `python-lint.yml` enforces across all five Python apps, and `make python-lint-fix` applies the autofixes. For a single app: `cd apps/<app> && uv run ruff check --fix . && uv run ruff format .`

Every Python project — the five under `apps/` — shares one ruff and pyright config, copied verbatim
into each `pyproject.toml` because neither tool can inherit from outside its own directory. **Change
a rule in all five or in none**; the `config-parity` job in `python-lint.yml` fails if the copies
diverge. Only path-shaped keys may differ — `exclude`, `include`, `extraPaths` and
`reportMissingImports` — each commented in place as a per-app deviation. `interp-engine` keeps a
sixth copy of the same block in its own repo, which nothing here can check; mirror a rule change
there by hand.

`.githooks/pre-commit` runs the fast half of that gate — ruff for the apps a commit touches, eslint
and prettier for the webapp, and the cheap spec/config scripts — over the staged files. It is opt-in
per checkout (`make githooks-install`, and any `npm install` in the webapp), so assume nothing has
run it for you: run the checks above yourself, and never reach for `git commit --no-verify` to get
past it.

**After editing Python anywhere, run `uv run pyright .` in that project** (`make check-type` where
it exists) before you finish. Fix the types properly — `assert`, narrowing, a `Protocol`, a more
precise annotation — rather than reaching for `# type: ignore`, and never loosen the shared config
to silence a single error.

Pyright runs for all five in CI but gates only inference and autointerp, the two whose installs are
complete. Non-blocking is not permission to ignore the others: all five are at zero and the three
ungated ones set `reportMissingImports = "none"`, so an error there is yours.

## Cross-server APIs (webapp <-> the five Python services)

The Python servers define the wire format and the webapp consumes generated types. All five generate
a spec the same way; what differs is only the *casing* on the wire, where `inference` and
`autointerp` alias to camelCase and `graph`, `nla` and `sparsity` stay snake_case on purpose.
Everything else below applies to all five.

What follows is the rules. The reasoning behind them — why those three cannot alias, why streamed
frames need their own base class, what the `/api/lens/share` bug cost — is in
[`docs/CROSS_SERVER_APIS.md`](docs/CROSS_SERVER_APIS.md). Read that when a contract test fails, or
before changing one of these boundaries rather than working within it.

**For inference and autointerp, the wire is camelCase; Python stays snake_case.** Every
request/response model subclasses that app's `BaseSchema` (`neuronpedia_<app>/schemas/common.py`),
which sets `alias_generator=to_camel` with `serialize_by_alias=True` and `populate_by_name=True`. So
you write `max_value` in Python and `maxValue` goes over the wire, in both directions. Never
hand-write a camelCase field name in a Python model.

Because `serialize_by_alias` is on, a bare `model_dump()` already emits the aliases — do not add
`by_alias=True`, and do not bypass the models by hand-building a response dict, which is the one way
to silently emit the wrong casing. Declaring a model in `responses=` is not using it: return it, or
`model_dump()` it if you need `exclude_none`. `/v1/activation/attention` named
`ActivationAttentionResponse` and returned a `JSONResponse(content=<dict>)`, so the spec promised
`seqLen` while the wire sent `seq_len`, and the webapp grew a hand-written type to match.

**Adding an endpoint:**

1. Define request and response models in `apps/<app>/neuronpedia_<app>/schemas/`, subclassing
   `BaseSchema`, and export them from that package's `__init__.py`.
2. Write the handler and document its response with `responses={200: {"model": YourResponse}}`.
   Prefer that over `response_model=`, which re-validates large payloads at runtime.
3. `make openapi` to regenerate `apps/<app>/openapi.json` and every `lib/api/*.d.ts` in one go. It
   skips any service you have not installed and says which, so check that line rather than
   assuming. (The halves are still separate targets — `make <app>-openapi` and `make
   webapp-openapi` — when you want only one.)
4. Add a readable alias in `lib/api/<app>-types.ts` rather than reaching into
   `components['schemas'][...]` at the call site.
5. For inference and autointerp, call it through the typed client and wrap the result in
   `unwrapInferenceResponse` / `unwrapAutointerpResponse`. `openapi-fetch` returns `{ data, error }`
   and does **not** throw, so an unwrapped call turns an upstream 500 into `undefined` and a
   confusing `TypeError`. The other three have no typed client — graph and sparsity use plain
   `fetch`, nla uses `nlaFetch` for its server failover — so there the generated type is applied at
   the `await res.json()` cast, and where the response is validated with yup, the schema is
   annotated with the generated type so a drifting field fails `tsc` rather than at runtime (see
   `SteerResponseSchema` in `lib/utils/graph.ts`).

Enums generate as string unions, so anything needing a runtime value (dropdowns, yup `oneOf`) uses
the hand-written objects in `lib/api/inference-types.ts`, annotated `{ [K in Union]: K }` so a new
spec member breaks the build instead of vanishing from the UI.

**Two places the generated types stop short**, both worked through in
[`docs/CROSS_SERVER_APIS.md`](docs/CROSS_SERVER_APIS.md):

- *Streaming.* SSE and NDJSON frames never reach `openapi.json`. Send them through
  `postInferenceStreaming` in `lib/utils/inference.ts`, which types the body against the spec while
  leaving the `ReadableStream` intact, rather than hand-rolling the `fetch`. Declare frame models on
  `PublicFrameSchema` — `BaseSchema` with the alias generator off, so the names you write are the
  names that ship — and pin them with a contract test like
  `apps/inference/tests/unit/test_lens_frame_contract.py`. Reach for that base only when a payload's
  field names are themselves the contract; leave everything else on `BaseSchema`.
- *The public API.* `/api/*` has real users, and its field names are a contract of their own. Routes
  must map explicitly between the public shape and the inference shape rather than forwarding a body
  through — copy `app/api/steer/route.tsx`. The same goes for anything persisted, such as
  `ExplanationScore.jsonDetails`: an upstream rename must not change stored keys.

### The published SDKs

`neuronpedia-inference-client` and `neuronpedia-autointerp-client` exist on npm and PyPI for callers
outside this repo. Nothing here imports them, and `openapi-publish.yml` builds them from the same
committed `openapi.json` the webapp types come from — so publishing is downstream of "edit the
model, run the make targets", never an input to it. Publishing is **off** until the repository
variable `OPENAPI_PUBLISH_ENABLED` is `"true"`; `make sdk-dry-run SERVICE=inference` runs it locally.

Two things in the servers exist only to keep those SDKs readable, and both are invisible from
inside the repo — the webapp compiles fine either way — so each has a test in the app's
`test_openapi.py` rather than a reviewer to catch it.

- **`generate_unique_id_function=sdk_operation_id`** on the `FastAPI(...)` call. FastAPI's default
  `operation_id` is handler name + path + verb, and openapi-generator turns it straight into the
  client's method name, so the default ships `explanationEndpointV1ExplainDefaultPost()`. The
  helper restores the path-plus-verb naming the hand-written spec gave those clients
  (`explainDefaultPost`). It is duplicated in both apps on purpose; they are separate projects
  with no shared package.
- **Never write `float | int`** (or `StrictFloat | StrictInt`) in a model. It becomes
  `anyOf: [number, integer]`, and the generator materializes one dispatcher class per field —
  `Probability`, `Maxvalue`, `Seed` — so `list[float]` reaches users as `List[ValuesInner]`.
  `StrictFloat` on its own already accepts an int and emits a plain `number`. This is easy to
  reintroduce by copying from generated Python, which emits `Union[StrictFloat, StrictInt]` for
  `type: number`; that is exactly how 39 of them got into these servers in the first place.

The other three servers have no SDK on purpose. They generate a spec and webapp types like
everybody else, but nothing outside the repo calls them.

### Why graph, nla and sparsity are snake_case

Each of the three has a public consumer that already reads its field names, so they generate types
like the other two but do not alias, and each has a `test_field_names_stay_snake_case` asserting no
alias generator is set and no camelCase reaches the spec. Do not "make it match inference" — that
fails the test. nla's SSE frames are declared on `NlaFrameSchema` in `apps/nla/server.py` and pinned
by `apps/nla/tests/test_frame_contract.py`.

Before aliasing any server, or adding a consumer to one of these responses, read
[`docs/CROSS_SERVER_APIS.md`](docs/CROSS_SERVER_APIS.md): it names each existing consumer and the
three questions that decide whether aliasing is safe.

## Style Guidelines

Ruff, eslint and prettier own formatting; run them rather than matching by hand. What they do not
enforce:

- **TS/React**: functional components, `@/` path aliases over relative paths, 120 char line width,
  kebab-case filenames for PascalCase components
- **Python**: absolute imports, docstrings on functions, `UPPER_SNAKE_CASE` for constants
- **No combining Unicode marks anywhere in a `.py` file**, comments and docstrings included. Write
  `J_bar` and `v_hat` rather than `J̄` and `v̂`, and `"\ufe0f"` rather than a literal variation
  selector where the character is real data. Everything else non-ASCII is fine — `ℓ`, `Σ`, `∂`, `‖`,
  `√`, `−` and bare emoji all extract cleanly.

  GitHub code scanning is what makes this a rule rather than a preference. CodeQL's Python extractor
  re-encodes a comment or string holding a combining mark as `\u{304}`, which is Rust's escape
  syntax rather than Python's, then passes the result through `%`-formatting when it reports the
  failure. So a file that also contains a parenthesized `with (...)` — which is what routes it to
  that parser — loses its docstrings from the database, and if any one string holds both a combining
  mark and a `%` specifier the extractor throws and **the whole file goes unanalyzed**. The alert
  says "a parse error occurred" and points at line 1, and `python -m py_compile` passes, so nothing
  about it leads back here.

## Agent instruction files

**Put the rule in the nearest `AGENTS.md`**, so that no rule is visible to one coding agent and
invisible to another. Scope by directory, not by tool: every harness loads a nested `AGENTS.md` when
it works in that subtree, as `apps/webapp/AGENTS.md` does. Each of those needs a one-line
`CLAUDE.md` beside it holding only `@./AGENTS.md`, since that is how Claude Code finds it. `.cursor/rules/*.mdc` can scope by directory too, but only for Cursor, so it is not where a
project rule belongs.

Reserve tool-specific config for what genuinely is tool-specific — hooks, slash commands, model
settings, MCP servers — and write the behavior it enforces here as prose too, because an agent
running without it still has to get it right. Hook scripts live in `utils/agent-hooks/`, shared by
`.cursor/hooks.json` and `.claude/settings.json`, so a new hook belongs there rather than under
either tool's directory. `.cursor/mcp.json` holds one server, Playwright MCP, so that looking at a
rendered page is a capability rather than something each session reinvents with a headless Chrome in
`/tmp`; every harness runs the same `npx @playwright/mcp@latest`, and `apps/webapp/AGENTS.md` has
the workflow. `.github/copilot-instructions.md`, `.gemini/settings.json` and `.aider.conf.yml` are
pointers at this file rather than content.

`make agent-rules-check` (and `agent-rules.yml`) enforces the mechanical half of this: that every
`AGENTS.md` is reachable from every harness and every declared hook is a runnable script. Its script
documents what it deliberately does not check — passing it is not the same as having put a rule in
the right file.

Personal, uncommitted overrides go in `AGENTS.local.md` or `CLAUDE.local.md`, both gitignored.
