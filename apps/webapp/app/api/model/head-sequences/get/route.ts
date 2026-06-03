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

    const sequences = await prisma.modelHeadSequence.findMany({
      where: {
        modelId,
        layer: layerInt,
        headIndex: headIndexInt,
      },
      select: {
        id: true,
        layer: true,
        headIndex: true,
        interval: true,
        tokens: true,
        attentionIndices: true,
        attentionValues: true,
        maxActivation: true,
      },
      orderBy: [{ interval: 'desc' }, { maxActivation: 'desc' }],
    });

    return NextResponse.json(sequences);
  } catch (error) {
    console.error('Error fetching head sequences:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
});
