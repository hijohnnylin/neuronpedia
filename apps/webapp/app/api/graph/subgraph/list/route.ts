import { prisma } from '@/lib/db';
import { RequestAuthedUser, withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  try {
    const { modelId, slug } = await request.json();

    // list subgraphs that are owned by the user for this graph
    const subgraphs = await prisma.graphMetadataSubgraph.findMany({
      where: {
        userId: request.user.id,
        graphMetadata: {
          modelId,
          slug,
        },
      },
      include: {
        user: {
          select: {
            name: true,
            id: true,
          },
        },
      },
      orderBy: {
        createdAt: 'desc',
      },
    });

    return NextResponse.json({ success: true, subgraphs });
  } catch (error) {
    console.error('Error saving subgraph:', error);
    return NextResponse.json({ error: 'Failed to save subgraph' }, { status: 500 });
  }
});
