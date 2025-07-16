import { prisma } from '@/lib/db';
import { RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  try {
    const { modelId, slug } = await request.json();

    const subgraphs = await prisma.graphMetadataSubgraph.findMany({
      where: {
        isFeaturedSolution: true,
        graphMetadata: {
          modelId,
          slug,
        },
      },
      include: {
        user: {
          select: {
            name: true,
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
