/**
 * Dump and restore `Vector` rows as JSON.
 *
 * Nothing else in this repository writes a `Vector`: there is no seed, no admin route and no
 * migration that inserts one. The rows in any live database were put there by hand, so a migration
 * that rewrites the table has no committed way to carry them and no way to put them back. This is
 * that way.
 *
 *   ts-node scripts/vectors.ts dump --out vectors.json
 *   ts-node scripts/vectors.ts load --in vectors.json
 *
 * `20260828230000_replace_persona_axis_with_vector` refers to this as `scripts/axes.ts`, which
 * never existed. The table is `Vector`, so the script is named for it; that migration is already
 * applied everywhere and its checksum is recorded, so the stale name in its header stays.
 *
 * Two things this deliberately does:
 *
 * It preserves `id`. A stored reading records the row that measured it -- `StoredAxis.axis_id` in
 * `lib/utils/steer-wire.ts` -- and nothing re-derives the reading, so a restore that minted fresh
 * cuids would leave every one of those pointing at nothing. `createdAt` is preserved for the same
 * reason: it is when the fit was published, not when this script ran.
 *
 * It refuses to overwrite. A row's numbers are never edited (see the `name` comment on the model):
 * a recalibration is a new row under a new name, because editing one in place silently changes what
 * every reading taken through it meant. So `load` creates what is missing and reports what it
 * skipped. `--overwrite` exists for a restore that died partway and left a row half-written.
 */
import { Prisma, PrismaClient, Vector } from '@prisma/client';
import { readFileSync, writeFileSync } from 'fs';

const prisma = new PrismaClient();

const USAGE = `
Usage: vectors <dump|load> [options]

  dump    [--out <path>]        defaults to stdout; a whole model's directions are megabytes
          [--model <modelId>]   only this model's rows

  load    --in <path>
          [--overwrite]         replace a row that already exists, rather than skipping it
          [--remote]            allow writing to a database that is not on this machine
`;

/**
 * One row as this file writes it: every scalar column, plus the names of the tags it carries.
 *
 * The tags are the one thing about a vector that is not a column of its own, and they are worth
 * carrying: a restored row with none would read identically and list nowhere. Names rather than
 * join rows, because a `VectorTag` is defined once in the target database and a dump has no
 * business restating what `axis` means.
 */
type DumpedVector = Vector & { tags?: string[] };

type Dump = {
  /** When this snapshot was taken, so two files are orderable. Not read on load. */
  dumpedAt: string;
  rows: DumpedVector[];
};

function parseArgs(argv: string[]): Map<string, string> {
  const args = new Map<string, string>();
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i].startsWith('--')) {
      const key = argv[i].slice(2);
      const value = argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[i + 1] : 'true';
      args.set(key, value);
      if (value !== 'true') i += 1;
    }
  }
  return args;
}

function requireArg(args: Map<string, string>, key: string): string {
  const value = args.get(key);
  if (!value || value === 'true') {
    throw new Error(`Missing --${key}\n${USAGE}`);
  }
  return value;
}

/**
 * Refuse to write to anything but a database on this machine, as `compute-host.ts` does.
 *
 * Developing against the production database is normal here, so the mistake worth catching is a
 * restore running with the wrong env file loaded and writing a laptop's snapshot into production.
 */
function assertLocalDatabase(args: Map<string, string>): void {
  if (args.has('remote')) return;

  const url = process.env.POSTGRES_URL_NON_POOLING ?? process.env.POSTGRES_PRISMA_URL ?? '';
  const host = (() => {
    try {
      return new URL(url).hostname;
    } catch {
      return '';
    }
  })();

  if (host === 'localhost' || host === '127.0.0.1' || host === '::1' || host === 'host.docker.internal') return;
  throw new Error(
    `Refusing to write to "${host || 'an unrecognised host'}", which is not a local database. ` +
      `Pass --remote if you truly mean this one.`,
  );
}

/**
 * Every row, with its tag names where there are any to have.
 *
 * The retry is not defensive coding. A dump is wanted most against a database the tag migration has
 * not reached yet -- that is the snapshot the migration tells you to take -- and asking for a
 * relation whose table does not exist is `P2021`, which would fail the one run that matters. So a
 * missing table costs the tags and nothing else, said out loud rather than silently.
 *
 * No `select` either way, so a new scalar column joins the dump without an edit here.
 */
async function dumpedRows(modelId: string | undefined): Promise<DumpedVector[]> {
  const query = {
    where: modelId && modelId !== 'true' ? { modelId } : undefined,
    orderBy: [{ modelId: 'asc' as const }, { name: 'asc' as const }],
  };

  try {
    const rows = await prisma.vector.findMany({
      ...query,
      include: { tags: { select: { tagName: true }, orderBy: { tagName: 'asc' } } },
    });
    return rows.map(({ tags, ...row }) => ({ ...row, tags: tags.map((tag) => tag.tagName) }));
  } catch (e) {
    if (!(e instanceof Prisma.PrismaClientKnownRequestError) || e.code !== 'P2021') throw e;
    console.error('This database has no VectorTag tables, so this dump carries no tags.');
    return prisma.vector.findMany(query);
  }
}

async function dump(args: Map<string, string>) {
  const rows = await dumpedRows(args.get('model'));

  const payload: Dump = { dumpedAt: new Date().toISOString(), rows };
  const serialized = JSON.stringify(payload, null, 2);

  const out = args.get('out');
  if (out && out !== 'true') {
    writeFileSync(out, serialized);
    console.log(`Wrote ${rows.length} row(s) to ${out}`);
  } else {
    console.log(serialized);
  }
}

function parseDump(path: string): DumpedVector[] {
  const parsed: unknown = JSON.parse(readFileSync(path, 'utf8'));
  if (typeof parsed !== 'object' || parsed === null || !Array.isArray((parsed as Dump).rows)) {
    throw new Error(`${path} is not a vectors dump: expected an object with a "rows" array`);
  }
  return (parsed as Dump).rows;
}

/**
 * One parsed row as create input.
 *
 * Two coercions JSON round-tripping forces. `createdAt` came back as a string and the column is a
 * timestamp. `projectionParams` reads as `JsonValue`, which admits `null`, and writes as
 * `InputJsonValue`, which does not -- the column is non-nullable, so a `null` here is a corrupt
 * dump rather than a case to support.
 *
 * A tag the target database has never heard of fails on the foreign key, which is the intended
 * answer: tags are written deliberately, so the fix is to define it there rather than to have a
 * restore coin it from a string in a file.
 */
function toCreateInput(row: DumpedVector): Prisma.VectorUncheckedCreateInput {
  if (row.projectionParams === null) {
    throw new Error(`Vector ${row.modelId}/${row.name}: projectionParams is null, which the column does not allow`);
  }
  const { tags, ...scalars } = row;
  return {
    ...scalars,
    createdAt: new Date(row.createdAt),
    projectionParams: row.projectionParams as Prisma.InputJsonValue,
    tags: { create: (tags ?? []).map((tagName) => ({ tagName })) },
  };
}

async function load(args: Map<string, string>) {
  assertLocalDatabase(args);
  const rows = parseDump(requireArg(args, 'in'));
  const overwrite = args.has('overwrite');

  let created = 0;
  const skipped: string[] = [];
  for (const row of rows) {
    const data = toCreateInput(row);
    const existing = await prisma.vector.findUnique({
      where: { modelId_name: { modelId: row.modelId, name: row.name } },
      select: { id: true },
    });

    if (existing && !overwrite) {
      skipped.push(`${row.modelId}/${row.name}`);
      continue;
    }
    if (existing) {
      // Keyed by the row already there rather than by the dump's id: if a half-written restore
      // minted a new cuid, deleting by the dump's id would miss it and the create would collide.
      await prisma.vector.delete({ where: { id: existing.id } });
    }
    await prisma.vector.create({ data });
    created += 1;
  }

  console.log(`Loaded ${created} row(s)`);
  if (skipped.length > 0) {
    console.log(`Skipped ${skipped.length} that already exist: ${skipped.join(', ')}`);
    console.log('Pass --overwrite to replace them.');
  }
}

async function main() {
  const [command, ...rest] = process.argv.slice(2);
  const args = parseArgs(rest);

  switch (command) {
    case 'dump':
      await dump(args);
      break;
    case 'load':
      await load(args);
      break;
    default:
      console.log(USAGE);
      process.exitCode = 1;
  }
}

main()
  .then(() => prisma.$disconnect())
  .catch(async (e) => {
    console.error(e instanceof Error ? e.message : e);
    await prisma.$disconnect();
    process.exit(1);
  });
