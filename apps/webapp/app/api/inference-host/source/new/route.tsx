import { createInferenceHostSource } from '@/lib/db/inference-host-source';
import { RequestAuthedAdminUser } from '@/lib/types/auth';
import { withAuthedAdminUser } from '@/lib/with-user';
import { InferenceHostSourceSchema } from '@/prisma/generated/zod';
import { NextResponse } from 'next/server';

export const POST = withAuthedAdminUser(async (request: RequestAuthedAdminUser) => {
  const requestJson = await request.json();
  const validatedInput = InferenceHostSourceSchema.parse(requestJson);
  const inferenceHostSource = await createInferenceHostSource({
    ...validatedInput,
  });

  return NextResponse.json(inferenceHostSource);
});
