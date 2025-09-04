import { createComment } from '@/lib/db/comment';
import { RequestAuthedUser } from '@/lib/types/auth';
import { withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const body = await request.json();
  const { text } = body;
  if (text.trim().length < 5) {
    throw new Error('text too short');
  } else if (text.trim().length > 1024) {
    throw new Error('text too long');
  }
  const comment = await createComment(body.modelId, body.layer, body.index, text, request.user);
  return NextResponse.json(comment);
});
