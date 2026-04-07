import { prisma } from '@/lib/db';
import {
  ACTIVATION_RAW_MAX_PROMPT_CHAR_LENGTH,
  ACTIVATION_RAW_MAX_PROMPTS_PER_BATCH,
  getRawActivations,
} from '@/lib/utils/activations-server';
import { RequestAuthedAdminUser, withAuthedAdminUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';
import { array, mixed, object, string } from 'yup';

const activationRawRequestSchema = object({
  modelId: string().required('modelId is required'),
  prompts: array()
    .of(string().required().min(1).max(ACTIVATION_RAW_MAX_PROMPT_CHAR_LENGTH))
    .required()
    .min(1)
    .max(ACTIVATION_RAW_MAX_PROMPTS_PER_BATCH),
  hookPoint: mixed<'residual_stream'>().oneOf(['residual_stream']).default('residual_stream'),
  type: mixed<'final_output_token'>().oneOf(['final_output_token']).default('final_output_token'),
});

export const POST = withAuthedAdminUser(async (request: RequestAuthedAdminUser) => {
  try {
    const body = await request.json();
    const parsedBody = await activationRawRequestSchema.validate(body);

    const activationRawResponse = await getRawActivations({
      model: parsedBody.modelId,
      prompts: parsedBody.prompts,
      hook_point: parsedBody.hookPoint,
      type: parsedBody.type,
    });

    const createdRows = activationRawResponse.results.map((result, promptIndex) => ({
      modelId: parsedBody.modelId,
      prompt: parsedBody.prompts[promptIndex],
      hookPoint: activationRawResponse.hook_point,
      captureType: activationRawResponse.type,
      dtype: activationRawResponse.dtype,
      device: activationRawResponse.device,
      tokenStrings: result.token_strings,
      tokenIds: result.token_ids,
      activations: result.activations,
      promptIndex,
      creatorId: request.user.id,
    }));

    await prisma.activationRaw.createMany({
      data: createdRows,
    });

    return NextResponse.json({
      message: `${createdRows.length} raw activations saved.`,
      count: createdRows.length,
      result: activationRawResponse,
    });
  } catch (error) {
    if (error instanceof Error) {
      return NextResponse.json({ error: error.message }, { status: 400 });
    }
    return NextResponse.json({ error: 'Failed to save raw activations.' }, { status: 500 });
  }
});
