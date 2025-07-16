import { SaveSubgraphRequestSchema } from '@/app/[modelId]/graph/utils';
import { prisma } from '@/lib/db';
import { RequestAuthedUser, withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  try {
    const body = await request.json();
    const validatedData = SaveSubgraphRequestSchema.parse(body);
    const {
      modelId,
      slug,
      displayName,
      pinnedIds,
      supernodes,
      clerps,
      pruningThreshold,
      densityThreshold,
      overwriteId,
    } = validatedData;

    // if overwriteId, check that the logged in user equals the owner of the subgraph
    if (overwriteId) {
      const subgraph = await prisma.graphMetadataSubgraph.findUnique({
        where: { id: overwriteId },
      });
      if (subgraph?.userId !== request.user.id) {
        return NextResponse.json({ error: 'You are not authorized to overwrite this subgraph' }, { status: 403 });
      }
      // user is the owner, update the subgraph
      await prisma.graphMetadataSubgraph.update({
        where: { id: overwriteId },
        data: {
          pinnedIds,
          supernodes,
          clerps,
          pruningThreshold,
          densityThreshold,
        },
      });
      return NextResponse.json({ success: true, subgraphId: overwriteId });
    }

    // save the subgraph
    const subgraph = await prisma.graphMetadataSubgraph.create({
      data: {
        displayName,
        graphMetadata: {
          connect: {
            modelId_slug: {
              modelId,
              slug,
            },
          },
        },
        pinnedIds,
        supernodes,
        clerps,
        pruningThreshold,
        densityThreshold,
        user: {
          connect: {
            id: request.user.id,
          },
        },
      },
    });
    return NextResponse.json({ success: true, subgraphId: subgraph.id });
  } catch (error) {
    console.error('Error saving subgraph:', error);
    return NextResponse.json({ error: 'Failed to save subgraph' }, { status: 500 });
  }
});
