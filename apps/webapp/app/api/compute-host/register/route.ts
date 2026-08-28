import { prisma } from '@/lib/db';
import { NEURONPEDIA_ENVIRONMENT } from '@/lib/env';
import { RequestAuthedAdminUser, withAuthedAdminUser } from '@/lib/with-user';
import { ComputeService } from '@prisma/client';
import { NextResponse } from 'next/server';
import { z } from 'zod';

const RegisterSchema = z
  .object({
    // Fleet profile that launched this host, e.g. "inference-gemma-2-2b-a".
    name: z.string().min(1),
    // Trailing slashes are stripped so the same host cannot register twice under
    // two spellings of one URL.
    hostUrl: z
      .string()
      .url()
      .transform((url) => url.replace(/\/+$/, '')),
    service: z.nativeEnum(ComputeService),
    // Which deployment the caller believes it is registering with. Compared
    // against this webapp's own setting and then discarded.
    environment: z.string().min(1),
    modelId: z.string().min(1),
    provider: z.string().nullish(),
    providerRef: z.string().nullish(),
    // The full set this host serves. Registration is declarative: whatever is
    // sent replaces what was there, so a host that drops an SAE stops receiving
    // requests for it.
    sourceIds: z.array(z.string()).default([]),
    sourceSetNames: z.array(z.string()).default([]),
    // Singular, unlike the two above: an NLA process fixes its verbalizer,
    // reconstructor and layer at startup, so it serves one source and no
    // request can ask it for another.
    nlaSourceId: z.string().nullish(),
  })
  .refine((input) => (input.service === ComputeService.NLA) === Boolean(input.nlaSourceId), {
    message: 'nlaSourceId is required for NLA hosts and must be omitted for every other service',
    path: ['nlaSourceId'],
  });

const DeregisterSchema = z.object({
  hostUrl: z.string().transform((url) => url.replace(/\/+$/, '')),
  service: z.nativeEnum(ComputeService),
});

export const POST = withAuthedAdminUser(async (request: RequestAuthedAdminUser) => {
  const parsed = RegisterSchema.safeParse(await request.json());
  if (!parsed.success) {
    return NextResponse.json({ error: 'Invalid request', details: parsed.error.flatten() }, { status: 400 });
  }
  const input = parsed.data;

  // A pod aimed at the wrong deployment otherwise registers silently and starts
  // taking another environment's traffic.
  if (input.environment !== NEURONPEDIA_ENVIRONMENT) {
    return NextResponse.json(
      { error: `Host registered as "${input.environment}" but this is the "${NEURONPEDIA_ENVIRONMENT}" deployment.` },
      { status: 409 },
    );
  }

  const model = await prisma.model.findUnique({ where: { id: input.modelId }, select: { id: true } });
  if (!model) {
    return NextResponse.json({ error: `Unknown model "${input.modelId}"` }, { status: 404 });
  }

  if (input.nlaSourceId) {
    const nlaSource = await prisma.nlaSource.findUnique({
      where: { modelId_id: { modelId: input.modelId, id: input.nlaSourceId } },
      select: { id: true },
    });
    if (!nlaSource) {
      return NextResponse.json(
        { error: `Unknown NLA source "${input.nlaSourceId}" for model "${input.modelId}"` },
        { status: 404 },
      );
    }
  }

  // An inference pod is launched with whole SAE sets (`--sae_sets`), but the
  // resolver matches a request's individual sourceId against ComputeHostOnSource,
  // so a host registered under set names alone would answer nothing. Expand the
  // sets into their members here, where the database is reachable -- the pod
  // itself only knows the set names it loaded. The set links are written too, so
  // a lookup by set name still resolves to the same host.
  const sourceIds = new Set(input.sourceIds);
  if (input.service === ComputeService.INFERENCE && input.sourceSetNames.length > 0) {
    const members = await prisma.source.findMany({
      where: { modelId: input.modelId, setName: { in: input.sourceSetNames } },
      select: { id: true, setName: true },
    });
    const found = new Set(members.map((member) => member.setName));
    const missing = input.sourceSetNames.filter((name) => !found.has(name));
    if (missing.length > 0) {
      return NextResponse.json(
        { error: `Unknown source set(s) for model "${input.modelId}": ${missing.join(', ')}` },
        { status: 404 },
      );
    }
    members.forEach((member) => sourceIds.add(member.id));
  }

  const fields = {
    name: input.name,
    modelId: input.modelId,
    nlaSourceId: input.nlaSourceId ?? null,
    provider: input.provider ?? null,
    providerRef: input.providerRef ?? null,
  };

  const host = await prisma.$transaction(async (tx) => {
    // Upsert rather than create: a pod that restarts re-registers under the
    // same URL and should update its row, not collide with it.
    const row = await tx.computeHost.upsert({
      where: { hostUrl_service: { hostUrl: input.hostUrl, service: input.service } },
      update: fields,
      create: { ...fields, hostUrl: input.hostUrl, service: input.service },
    });

    await Promise.all([
      tx.computeHostOnSource.deleteMany({ where: { computeHostId: row.id } }),
      tx.computeHostOnSourceSet.deleteMany({ where: { computeHostId: row.id } }),
    ]);

    await Promise.all([
      tx.computeHostOnSource.createMany({
        data: [...sourceIds].map((sourceId) => ({
          sourceId,
          sourceModelId: input.modelId,
          computeHostId: row.id,
        })),
      }),
      tx.computeHostOnSourceSet.createMany({
        data: input.sourceSetNames.map((sourceSetName) => ({
          sourceSetName,
          sourceSetModelId: input.modelId,
          computeHostId: row.id,
        })),
      }),
    ]);

    return row;
  });

  // The expanded source list goes back so the caller can confirm what a set name
  // actually resolved to, rather than assuming.
  return NextResponse.json({ ...host, sourceIds: [...sourceIds], sourceSetNames: input.sourceSetNames });
});

export const DELETE = withAuthedAdminUser(async (request: RequestAuthedAdminUser) => {
  const parsed = DeregisterSchema.safeParse(await request.json());
  if (!parsed.success) {
    return NextResponse.json({ error: 'Invalid request', details: parsed.error.flatten() }, { status: 400 });
  }
  const { hostUrl, service } = parsed.data;

  // Join rows cascade with the host.
  const { count } = await prisma.computeHost.deleteMany({ where: { hostUrl, service } });
  if (count === 0) {
    return NextResponse.json({ error: `No ${service} host registered at ${hostUrl}` }, { status: 404 });
  }

  return NextResponse.json({ deleted: count });
});
