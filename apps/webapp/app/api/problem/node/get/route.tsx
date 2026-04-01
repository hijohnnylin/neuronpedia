import { getProblemNode } from '@/lib/db/problem';
import { prisma } from '@/lib/db';
import { RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { ProblemNodeApprovalState } from '@prisma/client';
import { NextResponse } from 'next/server';

export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  const body = await request.json();
  const id = Number(body.id);

  if (!id || Number.isNaN(id)) {
    return NextResponse.json({ error: 'id is required' }, { status: 400 });
  }

  try {
    const node = await getProblemNode(id);

    // Restrict unapproved nodes to creator, editors, and admins
    if (node.approvalState !== ProblemNodeApprovalState.APPROVED) {
      const userId = request.user?.id;
      if (!userId) {
        return NextResponse.json({ error: 'Node not found' }, { status: 404 });
      }
      const isCreator = node.createdById === userId;
      if (!isCreator) {
        const dbUser = await prisma.user.findUniqueOrThrow({
          where: { id: userId },
          select: { admin: true, isProblemEditor: true },
        });
        if (!dbUser.admin && !dbUser.isProblemEditor) {
          return NextResponse.json({ error: 'Node not found' }, { status: 404 });
        }
      }
    }

    return NextResponse.json(node);
  } catch {
    return NextResponse.json({ error: 'Node not found' }, { status: 404 });
  }
});
