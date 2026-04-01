import { getProblemNodes } from '@/lib/db/problem';
import { RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  const body = await request.json();
  const includeUnapproved = body.includeUnapproved === true;

  // Only editors/admins can see unapproved nodes
  let canSeeUnapproved = false;
  if (includeUnapproved && request.user) {
    const { prisma } = await import('@/lib/db');
    const dbUser = await prisma.user.findUnique({
      where: { id: request.user.id },
      select: { admin: true, isProblemEditor: true },
    });
    canSeeUnapproved = dbUser?.admin === true || dbUser?.isProblemEditor === true;
  }

  const nodes = await getProblemNodes(canSeeUnapproved);
  return NextResponse.json(nodes);
});
