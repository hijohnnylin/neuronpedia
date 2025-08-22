import { addListComment } from '@/lib/db/list';
import { RequestAuthedUser } from '@/lib/types/auth';
import { withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const body = await request.json();
  const listId = body.listId as string;
  const text = body.text as string;

  if (text.trim().length < 2) {
    throw new Error('Too short');
  }

  const addedComment = await addListComment(listId, text, request.user);

  return NextResponse.json(addedComment);
});
