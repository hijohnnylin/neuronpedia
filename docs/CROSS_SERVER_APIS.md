# Cross-server APIs: why the rules are the way they are

The rules themselves are in the "Cross-server APIs" section of [`AGENTS.md`](../AGENTS.md), which is
what an agent loads on every turn. This file is the reasoning behind them — read it when one of the
contract tests fails, or before changing a boundary rather than working within one.

## Aliasing: what makes it safe, and where it stops

`inference` and `autointerp` alias their pydantic fields to camelCase on the wire. `graph`, `nla`
and `sparsity` deliberately do not. All five generate a spec and committed TypeScript types the same
way, so the difference is not maturity — generating types and aliasing them are separate decisions,
and these three took the first without the second.

Aliasing assumes the webapp is free to rename fields on the way out. That holds when the webapp owns
the reshaping: it reads a response, builds its own objects, and the upstream names never escape into
anything with other readers. It does not hold for these three:

- **graph** uploads a JSON blob to S3 whose keys come from third-party `circuit_tracer` code and are
  pinned by the published `app/api/graph/graph-schema.json` and the public `/graph/validator` page.
  `/api/graph/tokenize` and `/api/steer-logits` also return graph responses nearly verbatim, the
  latter publishing graph's own *request* field names because it forwards `features` untouched. Its
  `SteerResponse` goes further and aliases *to* SCREAMING_CASE, because that is what the existing
  public response uses.
- **nla** persists upstream records verbatim: `NlaExplainCache.resultJson` stores `ExplainResult`
  objects as they arrive, and those rows back permanent public share URLs. A rename would split that
  column into two casings with no version marker and no migration path.
- **sparsity** forwards `trace_forward` / `trace_backward` into a documented public response from
  `/api/sparsity/connected-neurons`.

Each has a `test_field_names_stay_snake_case` in its own suite, asserting that no alias generator is
set and no camelCase reaches the spec. The reflex to "make it match inference" therefore fails a
test with a pointer back here, rather than shipping.

So before aliasing any server, check the three things that made it safe for the other two: nothing
persists a response with upstream key names, no public route forwards one roughly verbatim, and no
hand-written consumer parses one without a generated type. A "yes" to any of them means pinning that
boundary first, or leaving the server snake_case on purpose — as these three are.

## Streamed frames: why they get their own base class

SSE and NDJSON frames never reach `openapi.json`, so nothing generates or checks a type for them.
The webapp mirrors them as hand-written interfaces (`lib/utils/lens.ts` for lens, four files for
nla), which is unavoidable with no spec to generate from. What keeps a mirror from drifting into a
translation is that frame models declare on `PublicFrameSchema` — `BaseSchema` with the alias
generator switched off — so the names written in Python are the names that ship, and a contract test
asserts the exact key sets.

The reason for that exception is worth understanding before reaching for it. A streamed frame the
webapp forwards *verbatim*, into a public response or into storage, is different from one it reads
and reshapes: those field names already have readers, so aliasing them just means renaming them back
at every consumer.

That arrangement cost a bug. `/api/lens/share` was added as a second consumer of `lensPromptStream`,
did not rename, and wrote share blobs in a casing no viewer reads. Nothing failed to compile, because
both sides were hand-written types that agreed with each other — they were simply both wrong about
what the rest of the system expected.

So reach for `PublicFrameSchema` when a payload's field names are themselves the contract, and leave
everything else on `BaseSchema`.

nla's SSE frames get the same treatment, minus the aliasing question: they are declared on
`NlaFrameSchema` in `apps/nla/server.py` and pinned by `apps/nla/tests/test_frame_contract.py`, which
asserts exact key sets rather than merely that the models parse. Four webapp files read those frames
by hand and one of them writes them to the database, so a rename there compiles everywhere and breaks
in the browser.

## The public API is a separate contract

`/api/*` is the surface with real users. It is camelCase apart from a few older snake_case fields
that predate the convention and cannot move.

Routes therefore map explicitly between the public shape and the inference shape instead of
forwarding a body straight through. `app/api/steer/route.tsx` is the model to copy: it validates the
older snake_case fields (`freq_penalty`, `strength_multiplier`, `steer_method`) with yup and maps
them to their camelCase inference names, so the public names stay put no matter what happens
upstream.

The same applies to anything persisted. `ExplanationScore.jsonDetails` has its own stored key names,
and changing an upstream field must not change them.
