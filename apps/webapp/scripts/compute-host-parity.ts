/**
 * Compare routing before and after the ComputeHost migration.
 *
 * The registry is an expand-only migration: the old tables are still there and
 * still hold the answers the running release serves from. So every routing
 * question they can answer can be replayed against the new resolver, and any
 * disagreement is a regression found before the deploy rather than after it.
 * That is worth doing because this branch gets no preview deploy, and because
 * the failure mode is a 500 rather than something subtle -- the set-name lookup
 * broke exactly this way, and typecheck had nothing to say about it.
 *
 * Strictly read-only: it issues findMany/findFirst and nothing else, so it is
 * safe to point at production. That is in fact the point.
 *
 * Run it with `make host-parity` (add ENV_FILE=... to point somewhere else).
 * It needs tsconfig-paths, since it imports the real resolver rather than a
 * copy of it -- a parity check that reimplements the thing it is checking would
 * agree with itself and prove nothing.
 */
/* eslint-disable no-await-in-loop -- resolved one target at a time on purpose:
   this is pointed at production, and a burst of parallel resolver calls would
   add load to the database it is meant to be quietly inspecting. */
import { ComputeService, PrismaClient } from '@prisma/client';
import { resolveHosts } from '../lib/db/compute-host';

const prisma = new PrismaClient();

type Verdict = 'ok' | 'lost' | 'changed' | 'gained';

type Check = {
  kind: string;
  label: string;
  before: string[];
  after: string[];
  verdict: Verdict;
};

const sortedUnique = (urls: (string | null | undefined)[]): string[] =>
  [...new Set(urls.filter((url): url is string => Boolean(url)))].sort();

function judge(before: string[], after: string[]): Verdict {
  if (before.length === 0 && after.length === 0) {
    return 'ok';
  }
  // The only verdict that means an outage: something used to route and no
  // longer does. A host gained is the wildcard fallback widening, which is
  // safe, and a set that merely differs still serves the request.
  if (before.length > 0 && after.length === 0) {
    return 'lost';
  }
  if (before.length === 0) {
    return 'gained';
  }
  return before.join('|') === after.join('|') ? 'ok' : 'changed';
}

/**
 * An admin id, so the resolver's access check passes for private sources.
 *
 * The old raw lookups did not filter by access -- callers did that separately --
 * so comparing them against an access-filtered result would report every private
 * source as lost. Access control is untouched by this migration, so taking it
 * out of the comparison is what isolates the routing change.
 */
async function adminUser() {
  const admin = await prisma.user.findFirst({ where: { admin: true }, select: { id: true, name: true } });
  if (!admin) {
    throw new Error('No admin user found; cannot bypass the access check for a like-for-like comparison.');
  }
  return { id: admin.id, name: admin.name } as never;
}

/** Name the database, so a run against the wrong one is obvious rather than quietly reassuring. */
function describeDatabase(): string {
  const url = process.env.POSTGRES_URL_NON_POOLING ?? process.env.POSTGRES_PRISMA_URL ?? '';
  try {
    const parsed = new URL(url);
    return `${parsed.hostname}${parsed.pathname}`;
  } catch {
    return 'an unrecognised connection string';
  }
}

async function main() {
  console.log(`Comparing routing against ${describeDatabase()}\n`);
  const user = await adminUser();
  const checks: Check[] = [];

  const record = async (kind: string, label: string, before: string[], target: Parameters<typeof resolveHosts>[0]) => {
    const after = sortedUnique(await resolveHosts(target));
    checks.push({ kind, label, before, after, verdict: judge(before, after) });
  };

  // ---- inference, per source -------------------------------------------
  // Engines are not filtered here, unlike the old lookups which defaulted to
  // TRANSFORMER_LENS. The wider "before" set can only over-report, and the
  // backfill deduplicated on hostUrl across engines, so anything it flags is
  // worth a look either way.
  const sourceLinks = await prisma.inferenceHostSourceOnSource.findMany({
    select: { sourceId: true, sourceModelId: true, inferenceHost: { select: { hostUrl: true } } },
  });

  const bySource = new Map<string, string[]>();
  for (const link of sourceLinks) {
    const key = `${link.sourceModelId}\u0000${link.sourceId}`;
    bySource.set(key, [...(bySource.get(key) ?? []), link.inferenceHost.hostUrl]);
  }

  for (const [key, urls] of bySource) {
    const [modelId, sourceId] = key.split('\u0000');
    await record('inference/source', `${modelId} ${sourceId}`, sortedUnique(urls), {
      service: ComputeService.INFERENCE,
      modelId,
      sourceId,
      user,
    });
  }

  // ---- inference, per source set ---------------------------------------
  // Replays getAllServerHostsForSourceSet: the sources in the set, then every
  // host attached to any of them.
  const linkedSources = await prisma.source.findMany({
    where: { OR: [...bySource.keys()].map((key) => ({ modelId: key.split('\u0000')[0], id: key.split('\u0000')[1] })) },
    select: { id: true, modelId: true, setName: true },
  });

  const bySet = new Map<string, string[]>();
  for (const source of linkedSources) {
    const hosts = bySource.get(`${source.modelId}\u0000${source.id}`) ?? [];
    const key = `${source.modelId}\u0000${source.setName}`;
    bySet.set(key, [...(bySet.get(key) ?? []), ...hosts]);
  }

  for (const [key, urls] of bySet) {
    const [modelId, sourceSetName] = key.split('\u0000');
    await record('inference/sourceSet', `${modelId} ${sourceSetName}`, sortedUnique(urls), {
      service: ComputeService.INFERENCE,
      modelId,
      sourceSetName,
      user,
    });
  }

  // ---- inference, model-wide -------------------------------------------
  // Replays getAllInstanceHostsForModel, which jlens and vector steering use:
  // every instance registered against the model, linked to a source or not.
  const instances = await prisma.inferenceHostSource.findMany({ select: { modelId: true, hostUrl: true } });
  const byModel = new Map<string, string[]>();
  for (const instance of instances) {
    byModel.set(instance.modelId, [...(byModel.get(instance.modelId) ?? []), instance.hostUrl]);
  }
  for (const [modelId, urls] of byModel) {
    await record('inference/model', modelId, sortedUnique(urls), { service: ComputeService.INFERENCE, modelId, user });
  }

  // ---- graph, per source set -------------------------------------------
  const graphLinks = await prisma.graphHostSourceOnSourceSet.findMany({
    select: { sourceSetName: true, sourceSetModelId: true, graphHostSource: { select: { hostUrl: true } } },
  });
  const byGraphSet = new Map<string, string[]>();
  for (const link of graphLinks) {
    const key = `${link.sourceSetModelId}\u0000${link.sourceSetName}`;
    byGraphSet.set(key, [...(byGraphSet.get(key) ?? []), link.graphHostSource.hostUrl ?? '']);
  }
  for (const [key, urls] of byGraphSet) {
    const [modelId, sourceSetName] = key.split('\u0000');
    await record('graph/sourceSet', `${modelId} ${sourceSetName}`, sortedUnique(urls), {
      service: ComputeService.GRAPH,
      modelId,
      sourceSetName,
      user,
    });
  }

  // ---- NLA, per source --------------------------------------------------
  const nlaSources = await prisma.nlaSource.findMany({ select: { id: true, modelId: true, servers: true } });
  for (const nla of nlaSources) {
    await record('nla/source', `${nla.modelId} ${nla.id}`, sortedUnique(nla.servers), {
      service: ComputeService.NLA,
      modelId: nla.modelId,
      nlaSourceId: nla.id,
      user,
    });
  }

  report(checks);
}

function report(checks: Check[]) {
  const counts: Record<Verdict, number> = { ok: 0, lost: 0, changed: 0, gained: 0 };
  for (const check of checks) {
    counts[check.verdict] += 1;
  }

  const show = (verdict: Verdict, heading: string) => {
    const rows = checks.filter((check) => check.verdict === verdict);
    if (rows.length === 0) {
      return;
    }
    console.log(`\n${heading}`);
    for (const row of rows) {
      console.log(`  [${row.kind}] ${row.label}`);
      console.log(`      before: ${row.before.join(', ') || '(none)'}`);
      console.log(`      after:  ${row.after.join(', ') || '(none)'}`);
    }
  };

  show('lost', 'ROUTED BEFORE, ROUTES NOWHERE NOW -- these will 500:');
  show('changed', 'Routes to a different set of hosts (still served, worth a glance):');
  show('gained', 'Routes now but did not before (the wildcard fallback widening):');

  console.log(
    `\n${checks.length} target(s) checked: ` +
      `${counts.ok} unchanged, ${counts.lost} lost, ${counts.changed} changed, ${counts.gained} gained.`,
  );

  // Nothing to compare is not a pass. An empty database answers every routing
  // question with "no hosts" and would otherwise look like a clean bill.
  if (checks.length === 0) {
    console.log(
      '\nNo targets found, so nothing was verified. The old tables are empty here -- ' +
        'point this at the database the migration ran against.',
    );
    process.exitCode = 1;
    return;
  }

  if (counts.lost > 0) {
    console.log('\nDo not deploy until the lost targets are explained.');
  }
  process.exitCode = counts.lost > 0 ? 1 : 0;
}

main()
  .catch((err) => {
    console.error(err);
    process.exitCode = 1;
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
