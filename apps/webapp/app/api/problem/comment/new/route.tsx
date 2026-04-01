import { createProblemComment } from '@/lib/db/problem';
import { RequestAuthedUser, withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const body = await request.json();

  if (!body.problemNodeId || !body.text) {
    return NextResponse.json({ error: 'problemNodeId and text are required' }, { status: 400 });
  }

  const text = (body.text as string).trim();
  if (text.length < 1) {
    return NextResponse.json({ error: 'Comment text must not be empty' }, { status: 400 });
  }
  if (text.length > 2000) {
    return NextResponse.json({ error: 'Comment text must not exceed 2000 characters' }, { status: 400 });
  }

  try {
    const comment = await createProblemComment(
      {
        problemNodeId: Number(body.problemNodeId),
        text,
        parentCommentId: body.parentCommentId,
      },
      request.user,
    );
    return NextResponse.json(comment);
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : 'Unknown error' }, { status: 500 });
  }
});
