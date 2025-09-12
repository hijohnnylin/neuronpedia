import { createSource } from '@/lib/db/source';
import { RequestAuthedAdminUser } from '@/lib/types/auth';
import { withAuthedAdminUser } from '@/lib/with-user';
import { SourceSchema } from '@/prisma/generated/zod';
import { JsonObject } from '@prisma/client/runtime/library';
import { NextResponse } from 'next/server';

export const POST = withAuthedAdminUser(async (request: RequestAuthedAdminUser) => {
  const validatedInputSchema = SourceSchema.refine(
    () =>
      // return input.visibility === Visibility.PRIVATE;
      true,
    { message: 'Invalid input' },
  );

  const requestJson = await request.json();
  requestJson.creator = { connect: { id: request.user.id } };
  requestJson.saelensConfig = requestJson.saelensConfig as string;

  const validatedInput = validatedInputSchema.parse(requestJson);

  const source = await createSource(
    {
      ...validatedInput,
      saelensConfig:
        validatedInput.saelensConfig && typeof validatedInput.saelensConfig === 'string'
          ? (JSON.parse(validatedInput.saelensConfig) as JsonObject)
          : null,
    },
    request.user,
  );

  return NextResponse.json(source);
});
