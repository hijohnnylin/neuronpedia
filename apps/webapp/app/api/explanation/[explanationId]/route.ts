import { getExplanationByIdWithDetails } from '@/lib/db/explanation';
import { RequestOptionalUser } from '@/lib/types/auth';
import { withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const GET = withOptionalUser(
  async (
    request: RequestOptionalUser,
    {
      params,
    }: {
      params: { explanationId: string };
    },
  ) => {
    const explanation = await getExplanationByIdWithDetails(params.explanationId, request.user);
    return NextResponse.json(explanation);
  },
);
