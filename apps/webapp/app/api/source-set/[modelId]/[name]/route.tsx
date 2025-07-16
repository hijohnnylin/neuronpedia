import { getSourceSet } from '@/lib/db/source';
import { RequestOptionalUser, withOptionalUser } from '@/lib/with-user';

import { NextResponse } from 'next/server';

export const GET = withOptionalUser(
  async (
    request: RequestOptionalUser,
    {
      params,
    }: {
      params: { modelId: string; name: string };
    },
  ) => {
    const sourceSet = await getSourceSet(params.modelId, params.name, request.user);
    return NextResponse.json(sourceSet);
  },
);
