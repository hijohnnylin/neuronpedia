import { prisma } from '@/lib/db';
import { RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

// Returns the lightweight, always-displayed head metrics for a model. Used by the head finder
// when the user switches models in the selector so the grid reflects the chosen model.
export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  try {
    const body = await request.json();
    const { modelId } = body ?? {};

    if (!modelId) {
      return NextResponse.json({ error: 'Missing required parameters' }, { status: 400 });
    }

    const metrics = await prisma.modelHeadMetrics.findMany({
      where: {
        modelId,
      },
      select: {
        layer: true,
        headIndex: true,
        inductionScore: true,
        prevTokenScore: true,
        patternEntropy: true,
        selfAttentionScore: true,
      },
      orderBy: {
        updatedAt: 'desc',
      },
    });

    return NextResponse.json(metrics);
  } catch (error) {
    console.error('Error fetching head metrics list:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
});
