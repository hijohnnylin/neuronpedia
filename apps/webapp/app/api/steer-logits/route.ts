import { getModelById } from '@/lib/db/model';
import { RequestOptionalUser } from '@/lib/types/auth';
import { SteerLogitsRequestSchema, steerLogits } from '@/lib/utils/graph';
import { withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

// for now this just uses the graph server, but we should merge it with the inference server later on to be consistent

export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  const body = await request.json();

  const validatedBody = SteerLogitsRequestSchema.validateSync(body);

  if (!validatedBody.sourceSetName) {
    const model = await getModelById(validatedBody.modelId);
    validatedBody.sourceSetName = model?.defaultGraphSourceSetName;
    if (!validatedBody.sourceSetName) {
      return NextResponse.json(
        {
          error: 'Source Set Missing',
          message: `The model ${validatedBody.modelId} has no default graph source set, so you must provide one in the sourceSetName parameter.`,
        },
        { status: 400 },
      );
    }
  }

  const {
    modelId,
    sourceSetName,
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
    sourceSetName,
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
