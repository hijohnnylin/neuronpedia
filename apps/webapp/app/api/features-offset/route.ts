import { getNeuronsOffset } from '@/lib/db/neuron';
import { RequestOptionalUser } from '@/lib/types/auth';
import { withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const maxDuration = 60;

export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  const body = await request.json();

  const { modelId, layer, offset } = body;

  const neurons = await getNeuronsOffset(modelId, layer, offset, request.user);

  return NextResponse.json(neurons);
});
