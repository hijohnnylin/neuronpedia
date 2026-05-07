import { prisma } from '@/lib/db';
import { NextResponse } from 'next/server';

export async function GET(_request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const cache = await prisma.nlaExplainCache.findUnique({
    where: { id },
    select: {
      text: true,
      temperature: true,
      modelId: true,
      nlaSourceId: true,
      sortedPositions: true,
      tokens: true,
      resultJson: true,
    },
  });

  if (!cache) {
    return NextResponse.json({ error: 'Not found' }, { status: 404 });
  }

  return NextResponse.json(cache);
}
