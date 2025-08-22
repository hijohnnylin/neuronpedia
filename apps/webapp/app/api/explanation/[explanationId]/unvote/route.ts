import { unvote } from '@/lib/db/vote';
import { RequestAuthedUser } from '@/lib/types/auth';
import { withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const POST = withAuthedUser(
  async (request: RequestAuthedUser, { params }: { params: { explanationId: string } }) =>
    unvote(request.user.id, params.explanationId).then((deletedVote) => NextResponse.json(deletedVote)),
);
