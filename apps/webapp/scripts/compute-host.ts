/**
 * Register, list and remove compute hosts in the local database.
 *
 * The deploy tool uses POST /api/compute-host/register for real environments.
 * This is the local equivalent: it talks to Postgres directly, so it works
 * before the webapp is running and needs no admin API key.
 *
 *   ts-node scripts/compute-host.ts add \
 *     --service INFERENCE --model gpt2-small --url http://127.0.0.1:5002 \
 *     --sources 6-res-jb,7-res-jb
 *   ts-node scripts/compute-host.ts list
 *   ts-node scripts/compute-host.ts remove --service INFERENCE --url http://127.0.0.1:5002
 */
import { ComputeService, PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

/**
 * Refuse to write to anything but a database on this machine.
 *
 * Developing against the production database is normal here, so the dangerous
 * mistake is running this with the wrong .env loaded and putting a laptop's
 * 127.0.0.1 into production's registry, where it takes real traffic. Checking
 * the connection string tests that directly. `--remote` is the deliberate
 * override for the rare case of curating another environment's registry.
 */
function assertLocalDatabase(args: Map<string, string>): void {
  const url = process.env.POSTGRES_URL_NON_POOLING ?? process.env.POSTGRES_PRISMA_URL ?? '';
  const host = (() => {
    try {
      return new URL(url).hostname;
    } catch {
      return '';
    }
  })();

  const isLocal = host === 'localhost' || host === '127.0.0.1' || host === '::1' || host === 'host.docker.internal';
  if (isLocal || args.has('remote')) {
    return;
  }
  throw new Error(
    `Refusing to write to "${host || 'an unrecognised host'}", which is not a local database. ` +
      `Compute hosts registered by hand are for local development; a real environment registers ` +
      `through POST /api/compute-host/register. Pass --remote if you truly mean this database.`,
  );
}

const USAGE = `
Usage: compute-host <add|list|remove> [options]

  add     --service <${Object.keys(ComputeService).join('|')}>
          --model <modelId>
          --url <baseUrl>
          [--name <label>]              defaults to "<service>-<modelId>-local"
          [--sources a,b]               source ids this host has loaded (INFERENCE)
          [--source-sets a,b]           source set names (GRAPH)
          [--nla-source <id>]           the one NLA source this host serves; required for NLA

  list    [--service <SERVICE>]

  remove  --service <SERVICE> --url <baseUrl>

  --remote  allow add/remove against a database that is not on this machine
`;

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

const list = (args: Map<string, string>, key: string): string[] =>
  (args.get(key) ?? '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);

function requireArg(args: Map<string, string>, key: string): string {
  const value = args.get(key);
  if (!value || value === 'true') {
    throw new Error(`Missing --${key}\n${USAGE}`);
  }
  return value;
}

function parseService(raw: string): ComputeService {
  const service = raw.toUpperCase() as ComputeService;
  if (!(service in ComputeService)) {
    throw new Error(`Unknown service "${raw}". One of: ${Object.keys(ComputeService).join(', ')}`);
  }
  return service;
}

async function add(args: Map<string, string>) {
  assertLocalDatabase(args);
  const service = parseService(requireArg(args, 'service'));
  const modelId = requireArg(args, 'model');
  const hostUrl = requireArg(args, 'url').replace(/\/+$/, '');
  const name = args.get('name') || `${service.toLowerCase()}-${modelId}-local`;

  const model = await prisma.model.findUnique({ where: { id: modelId }, select: { id: true } });
  if (!model) {
    throw new Error(`Unknown model "${modelId}". Import it before registering a host for it.`);
  }

  // An NLA process serves exactly one source, so the column is required for
  // NLA and rejected for everything else. A CHECK constraint enforces the same
  // thing; catching it here gives a readable message.
  const nlaSourceId = args.get('nla-source') || null;
  if ((service === ComputeService.NLA) !== Boolean(nlaSourceId)) {
    throw new Error(
      service === ComputeService.NLA
        ? 'NLA hosts serve exactly one source: pass --nla-source <id>.'
        : `--nla-source only applies to NLA hosts, not ${service}.`,
    );
  }

  const host = await prisma.computeHost.upsert({
    where: { hostUrl_service: { hostUrl, service } },
    update: { name, modelId, nlaSourceId },
    create: { name, hostUrl, service, modelId, nlaSourceId },
  });

  await Promise.all([
    prisma.computeHostOnSource.deleteMany({ where: { computeHostId: host.id } }),
    prisma.computeHostOnSourceSet.deleteMany({ where: { computeHostId: host.id } }),
  ]);

  await Promise.all([
    prisma.computeHostOnSource.createMany({
      data: list(args, 'sources').map((sourceId) => ({ sourceId, sourceModelId: modelId, computeHostId: host.id })),
    }),
    prisma.computeHostOnSourceSet.createMany({
      data: list(args, 'source-sets').map((sourceSetName) => ({
        sourceSetName,
        sourceSetModelId: modelId,
        computeHostId: host.id,
      })),
    }),
  ]);

  console.log(`Registered ${service} host for ${modelId}: ${hostUrl}`);
}

async function show(args: Map<string, string>) {
  const service = args.get('service') ? parseService(args.get('service') as string) : undefined;
  const hosts = await prisma.computeHost.findMany({
    where: service ? { service } : undefined,
    include: { sources: true, sourceSets: true },
    orderBy: [{ service: 'asc' }, { modelId: 'asc' }],
  });

  if (hosts.length === 0) {
    console.log('No compute hosts registered.');
    return;
  }

  hosts.forEach((host) => {
    const links = [
      ...host.sources.map((s) => s.sourceId),
      ...host.sourceSets.map((s) => s.sourceSetName),
      ...(host.nlaSourceId ? [host.nlaSourceId] : []),
    ];
    console.log(
      `${host.service.padEnd(9)} ${host.modelId.padEnd(20)} ${host.hostUrl}` +
        (links.length > 0 ? `  [${links.join(', ')}]` : '  [any]'),
    );
  });
}

async function remove(args: Map<string, string>) {
  assertLocalDatabase(args);
  const service = parseService(requireArg(args, 'service'));
  const hostUrl = requireArg(args, 'url').replace(/\/+$/, '');
  const { count } = await prisma.computeHost.deleteMany({ where: { hostUrl, service } });
  console.log(count === 0 ? `No ${service} host registered at ${hostUrl}` : `Removed ${service} host ${hostUrl}`);
}

async function main() {
  const [command, ...rest] = process.argv.slice(2);
  const args = parseArgs(rest);

  switch (command) {
    case 'add':
      await add(args);
      break;
    case 'list':
      await show(args);
      break;
    case 'remove':
      await remove(args);
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
