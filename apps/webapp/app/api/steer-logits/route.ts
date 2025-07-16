import { SteerLogitsRequestSchema, steerLogits } from '@/lib/utils/graph';
import { RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

// for now this just uses the graph server, but we should merge it with the inference server later on to be consistent

export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  const body = await request.json();

  const validatedBody = SteerLogitsRequestSchema.validateSync(body);

  const {
    modelId,
    prompt,
    features,
    nTokens,
    topK,
    freezeAttention,
    temperature,
    freqPenalty,
    seed,
    steeredOutputOnly,
  } = validatedBody;

  const response = await steerLogits(
    modelId,
    prompt,
    features,
    nTokens,
    topK,
    freezeAttention,
    temperature,
    freqPenalty,
    seed,
    steeredOutputOnly,
  );

  return NextResponse.json(response);
});
