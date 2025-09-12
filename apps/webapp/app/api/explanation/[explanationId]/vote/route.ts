import { getExplanationById } from '@/lib/db/explanation';
import { vote } from '@/lib/db/vote';
import { RequestAuthedUser } from '@/lib/types/auth';
import { withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const POST = withAuthedUser(
  async (request: RequestAuthedUser, { params }: { params: { explanationId: string } }) => {
    const explanation = await getExplanationById(params.explanationId, request.user);
    if (!explanation) {
      throw new Error('Explanation not found');
    }

    const result = await vote(request.user.id, explanation.id);

    return NextResponse.json(result);
  },
);
