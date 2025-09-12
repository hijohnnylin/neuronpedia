import { createActivationsForVector } from '@/lib/db/activation';
import { RequestAuthedUser } from '@/lib/types/auth';
import { withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';
import { array, number, object, string } from 'yup';

const uploadBatchSchema = object({
  modelId: string().required(),
  source: string().required(),
  index: string().required(),
  activations: array()
    .of(
      object({
        tokens: array().of(string()).required().min(1),
        values: array().of(number()).required(),
      }),
    )
    .required()
    .min(1),
});

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const body = await request.json();

  const parsedBody = await uploadBatchSchema.validate(body);
  if (!parsedBody.activations) {
    throw new Error('Activations are required.');
  }

  // ensure activations tokens length matches values length
  // eslint-disable-next-line no-restricted-syntax
  for (const activation of parsedBody.activations) {
    if (activation.tokens.length !== activation.values.length) {
      throw new Error('Activations tokens length does not match values length.');
    }
  }

  // add the activations
  const createdActivations = await createActivationsForVector(
    parsedBody.modelId,
    parsedBody.source,
    parsedBody.index,
    parsedBody.activations as {
      tokens: string[];
      values: number[];
    }[],
    request.user,
  );

  return NextResponse.json({
    message: `${createdActivations.length} activations created.`,
  });
});
