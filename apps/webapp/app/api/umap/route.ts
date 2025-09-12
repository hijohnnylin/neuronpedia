import { getUmapExplanations } from '@/lib/db/explanation';
import { RequestOptionalUser } from '@/lib/types/auth';
import { withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  const body = await request.json();
  const { modelId, layers } = body;

  const results = await getUmapExplanations(modelId, layers, request.user);

  return NextResponse.json(results);
});
