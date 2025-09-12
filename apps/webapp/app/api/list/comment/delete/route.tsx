import { deleteListComment } from '@/lib/db/list';
import { RequestAuthedUser } from '@/lib/types/auth';
import { withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const body = await request.json();
  const commentId = body.commentId as string;

  const deleted = await deleteListComment(commentId, request.user);

  return NextResponse.json(deleted);
});
