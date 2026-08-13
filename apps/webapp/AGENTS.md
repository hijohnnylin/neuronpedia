# Webapp conventions

Next.js (App Router) with TypeScript, TailwindCSS and shadcn components. It serves both the
frontend and the public API, so see the "Cross-server APIs" section of the root `AGENTS.md` before
touching anything under `app/api` — `/api/*` has real users and its field names are a contract.

## Imports that are easy to get wrong

These have a project-specific source. Reaching for the library default is the common mistake:

- Prisma: `import { prisma } from '@/lib/db'`
- Router, in client components: `import { useRouter } from '@bprogress/next'` — **not**
  `next/navigation`, which skips the progress bar
- Environment variables (`IS_LOCALHOST` and friends): `import { ... } from '@/lib/env'`
- shadcn components: `@/components/shadcn` (e.g. `@/components/shadcn/button`)

## Structure

- API routes are under `app/api`, and should generally be POST. Avoid GET unless the endpoint is
  genuinely a cacheable read.
- Providers live in `@/components/provider`, with `providers.tsx` as the top-level composition.
  Add a nested provider file when you need one rather than reworking `providers.tsx`.
- The database schema is `prisma/schema.prisma`.
- Generated API types are in `lib/api/` and are build outputs — regenerate them with
  `npm run openapi`, never hand-edit.

## Checks

`npm run lint:fix && npm run format:write` before finishing, and `npm test -- --reporter=verbose` if
you touched anything with a `*.test.ts` beside it — verbose so the results stream in one per test
instead of arriving as a silent block.

`lint:fix` is only eslint. The gate `webapp-lint.yml` enforces is `npm run lint`, which is
`eslint --quiet . && tsc --noEmit` — so a type error passes `lint:fix` and fails CI. Run the full
`npm run lint` before finishing.

Current test coverage here is unit-only, pure
functions, no database and no running inference server. However, that could be extended in the future and if needed - ask first before doing so.

## Looking at the UI

Do not hand-roll a headless browser to see what a page renders. The repo-root `.cursor/mcp.json`
configures the Playwright MCP server, which gives screenshots, accessibility snapshots and real
clicking/typing. Claude Code gets the same capability from the same package with
`claude mcp add playwright -- npx -y @playwright/mcp@latest --browser=chrome --headless`. It is
configured with `--browser=chrome` so it drives the system Google Chrome rather than downloading
Playwright's own Chromium, `--isolated` so no profile is kept, and `--output-dir=.playwright-mcp`,
which is gitignored. Pinning to `@latest` is deliberate: the package is still on 0.0.x and its tool
surface moves.

Take screenshots without passing `filename`. `--output-dir` covers the auto-named files, but an
explicit relative `filename` resolves against the working directory instead, so it drops an
untracked PNG in the repo root — and the default path is better anyway, since it returns the image
for the model to look at rather than only a link to it.

**Point it at `npm run dev:localhost`, not `dev:prod`.** `dev:prod` reads the production database
and prod auth, so a fresh browser profile has no session and anything gated renders signed-out,
while the data underneath changes on its own. Because `--isolated` keeps no profile, a flow that
needs a signed-in user needs one created per session, or a `--storage-state` file.

Dev builds compile on first request, so a cold page here can take ten seconds before the first byte
and the homepage is a couple of megabytes of HTML. The 60s navigation default absorbs that, but poll
for the element you care about — a snapshot, or an evaluate loop — rather than sleeping a fixed
interval and screenshotting a skeleton.

There is no browser-level test suite yet, so do not add a `*.spec.ts` as though one exists. The
missing piece is not the runner but deterministic data: a Playwright suite needs seeded fixtures,
and seeding is a migration a human applies (see "The database" in the root `AGENTS.md`).
