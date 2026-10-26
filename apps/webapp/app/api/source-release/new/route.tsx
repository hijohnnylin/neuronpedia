import { createSourceRelease } from '@/lib/db/source';
import { RequestAuthedAdminUser, withAuthedAdminUser } from '@/lib/with-user';
import { SourceReleaseSchema } from '@/prisma/generated/zod';
import { Visibility } from '@prisma/client';
import { NextResponse } from 'next/server';
import { z } from 'zod';

export const POST = withAuthedAdminUser(async (request: RequestAuthedAdminUser) => {
  const validatedInputSchema = SourceReleaseSchema.extend({
    urls: z.string().array().max(100),
    defaultUmapSourceIds: z.string().array().max(100),
  }).refine((input) => input.featured === false && input.isNewUi === false && input.visibility === Visibility.PRIVATE, {
    message: 'Invalid input',
  });

  const requestJson = await request.json();
  requestJson.creator = { connect: { id: request.user.id } };

  const validatedInput = validatedInputSchema.parse(requestJson);

  const release = await createSourceRelease(validatedInput, request.user);

  return NextResponse.json(release);
});
