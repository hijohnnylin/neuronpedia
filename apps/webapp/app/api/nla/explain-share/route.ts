import { prisma } from '@/lib/db';
import { MAX_COMMENT_LENGTH } from '@/lib/nla-constants';
import { NextResponse } from 'next/server';

type ExplainShareBody = {
  cacheId?: string;
  position?: number | null;
  paragraph?: number | null;
  highlightStart?: number | null;
  highlightEnd?: number | null;
  comment?: string | null;
};

export async function POST(request: Request) {
  let body: ExplainShareBody;
  try {
    body = (await request.json()) as ExplainShareBody;
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }

  const cacheId = typeof body.cacheId === 'string' ? body.cacheId.trim() : '';
  if (!cacheId) {
    return NextResponse.json({ error: 'cacheId is required' }, { status: 400 });
  }

  const cache = await prisma.nlaExplainCache.findUnique({
    where: { id: cacheId },
    select: { id: true },
  });
  if (!cache) {
    return NextResponse.json({ error: 'Unknown cache id' }, { status: 404 });
  }

  const position: number | null =
    typeof body.position === 'number' && Number.isFinite(body.position) && body.position >= 0
      ? Math.floor(body.position)
      : null;

  const rawParagraph = body.paragraph;
  const paragraphOk =
    typeof rawParagraph === 'number' &&
    Number.isFinite(rawParagraph) &&
    Number.isInteger(rawParagraph) &&
    rawParagraph >= 0 &&
    rawParagraph <= 2;

  let paragraph: number | null = paragraphOk ? rawParagraph : null;

  let highlightStart: number | null = null;
  let highlightEnd: number | null = null;
  const hs = body.highlightStart;
  const he = body.highlightEnd;
  if (typeof hs === 'number' && typeof he === 'number' && Number.isFinite(hs) && Number.isFinite(he)) {
    const s = Math.floor(hs);
    const e = Math.floor(he);
    if (s >= 0 && e > s) {
      highlightStart = s;
      highlightEnd = e;
    }
  }

  // Range wins over paragraph (same mutual-exclusion rule as the NLA URL contract).
  if (highlightStart !== null && highlightEnd !== null) {
    paragraph = null;
  } else {
    highlightStart = null;
    highlightEnd = null;
  }

  // Paragraph / range anchoring only makes sense with a locked position.
  if (position === null) {
    paragraph = null;
    highlightStart = null;
    highlightEnd = null;
  }

  let comment = typeof body.comment === 'string' ? body.comment : '';
  if (comment.length > MAX_COMMENT_LENGTH) {
    comment = comment.slice(0, MAX_COMMENT_LENGTH);
  }

  const row = await prisma.nlaExplainShare.create({
    data: {
      cacheId,
      position,
      paragraph,
      highlightStart,
      highlightEnd,
      comment,
    },
    select: { id: true },
  });

  return NextResponse.json({ id: row.id });
}
