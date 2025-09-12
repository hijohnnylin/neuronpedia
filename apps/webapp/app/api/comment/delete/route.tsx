import { deleteComment } from '@/lib/db/comment';
import { RequestAuthedUser } from '@/lib/types/auth';
import { withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const body = await request.json();
  await deleteComment(body.id, request.user);
  return NextResponse.json({ message: 'ok' });
});
