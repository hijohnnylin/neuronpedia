import { prisma } from '@/lib/db';
import { NEURONPEDIA_ENVIRONMENT } from '@/lib/env';
import { RequestAuthedAdminUser, withAuthedAdminUser } from '@/lib/with-user';
import { ComputeService } from '@prisma/client';
import { NextResponse } from 'next/server';

/**
 * Every registered compute host, for tools that watch the fleet from outside
 * the webapp. The status page reads this to learn which pods should be up,
 * and points a monitor at each `hostUrl`.
 *
 * Admin only: a host URL is a direct route past the webapp to a GPU server.
 * `?service=INFERENCE` narrows to one service.
 */
export const GET = withAuthedAdminUser(async (request: RequestAuthedAdminUser) => {
  const rawService = request.nextUrl.searchParams.get('service');
  let service: ComputeService | undefined;
  if (rawService) {
    if (!(rawService in ComputeService)) {
      return NextResponse.json(
        { error: `Unknown service "${rawService}". One of: ${Object.keys(ComputeService).join(', ')}` },
        { status: 400 },
      );
    }
    service = rawService as ComputeService;
  }

  const hosts = await prisma.computeHost.findMany({
    where: service ? { service } : undefined,
    include: {
      sources: { select: { sourceId: true } },
      sourceSets: { select: { sourceSetName: true } },
    },
    orderBy: [{ service: 'asc' }, { modelId: 'asc' }, { name: 'asc' }],
  });

  // The environment goes back so a reader pointed at the wrong deployment can
  // tell, the same way registration refuses a mismatched one.
  return NextResponse.json({
    environment: NEURONPEDIA_ENVIRONMENT,
    hosts: hosts.map((host) => ({
      id: host.id,
      name: host.name,
      hostUrl: host.hostUrl,
      service: host.service,
      modelId: host.modelId,
      provider: host.provider,
      providerRef: host.providerRef,
      nlaSourceId: host.nlaSourceId,
      sourceIds: host.sources.map((source) => source.sourceId),
      sourceSetNames: host.sourceSets.map((set) => set.sourceSetName),
      createdAt: host.createdAt,
      updatedAt: host.updatedAt,
    })),
  });
});
