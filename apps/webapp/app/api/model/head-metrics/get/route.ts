import { prisma } from '@/lib/db';
import { RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  try {
    const body = await request.json();
    const { modelId, layer, headIndex } = body ?? {};

    if (!modelId || layer === undefined || layer === null || headIndex === undefined || headIndex === null) {
      return NextResponse.json({ error: 'Missing required parameters' }, { status: 400 });
    }

    const layerInt = Number(layer);
    const headIndexInt = Number(headIndex);
    if (!Number.isInteger(layerInt) || !Number.isInteger(headIndexInt)) {
      return NextResponse.json({ error: 'Invalid layer or headIndex' }, { status: 400 });
    }

    const detail = await prisma.modelHeadMetrics.findFirst({
      where: {
        modelId,
        layer: layerInt,
        headIndex: headIndexInt,
      },
      select: {
        layer: true,
        headIndex: true,
        selfAttentionScore: true,
        qkDistance: true,
        qkDistanceVariance: true,
        activationHistogram: true,
        qkDistanceHistogram: true,
        topQueryTokens: true,
        topKeyTokens: true,
      },
      orderBy: {
        updatedAt: 'desc',
      },
    });

    if (!detail) {
      return NextResponse.json({ error: 'Head metrics not found' }, { status: 404 });
    }

    return NextResponse.json(detail);
  } catch (error) {
    console.error('Error fetching head metrics detail:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
});
