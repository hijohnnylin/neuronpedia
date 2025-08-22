import { createSourceRelease } from '@/lib/db/source';
import { RequestAuthedAdminUser } from '@/lib/types/auth';
import { withAuthedAdminUser } from '@/lib/with-user';
import { SourceReleaseSchema } from '@/prisma/generated/zod';
import { Visibility } from '@prisma/client';
import { NextResponse } from 'next/server';

export const POST = withAuthedAdminUser(async (request: RequestAuthedAdminUser) => {
  const validatedInputSchema = SourceReleaseSchema.refine(
    (input) => input.featured === false && input.isNewUi === false && input.visibility === Visibility.PRIVATE,
    { message: 'Invalid input' },
  );

  const requestJson = await request.json();
  requestJson.creator = { connect: { id: request.user.id } };

  const validatedInput = validatedInputSchema.parse(requestJson);

  const release = await createSourceRelease(validatedInput, request.user);

  return NextResponse.json(release);
});
