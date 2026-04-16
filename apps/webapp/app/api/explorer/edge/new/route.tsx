import { createProblemEdge } from '@/lib/db/problem';
import { RequestAuthedUser, withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const body = await request.json();

  if (!body.sourceNodeId || !body.targetNodeId || !body.type) {
    return NextResponse.json({ error: 'sourceNodeId, targetNodeId, and type are required' }, { status: 400 });
  }

  try {
    const edge = await createProblemEdge(
      {
        sourceNodeId: Number(body.sourceNodeId),
        targetNodeId: Number(body.targetNodeId),
        type: body.type,
      },
      request.user,
    );
    return NextResponse.json(edge);
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : 'Unknown error' }, { status: 500 });
  }
});
