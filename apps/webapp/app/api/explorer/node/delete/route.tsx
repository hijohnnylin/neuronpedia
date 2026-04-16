import { deleteProblemNode } from '@/lib/db/problem';
import { RequestAuthedUser, withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const body = await request.json();

  const id = Number(body.id);
  if (!id || Number.isNaN(id)) {
    return NextResponse.json({ error: 'id is required' }, { status: 400 });
  }

  try {
    await deleteProblemNode(id, request.user);
    return NextResponse.json(true);
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : 'Unknown error' }, { status: 500 });
  }
});
