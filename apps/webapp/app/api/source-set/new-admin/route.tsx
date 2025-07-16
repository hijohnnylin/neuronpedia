import { withAuthedAdminUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const POST = withAuthedAdminUser(async () =>
  // const validatedInputSchema = SourceSetSchema.refine((input) => input.visibility === Visibility.PRIVATE, {
  //   message: 'Invalid input',
  // });

  // const requestJson = await request.json();
  // requestJson.creator = { connect: { id: request.user.id } };

  // const validatedInput = validatedInputSchema.parse(requestJson);

  // const set = await createSourceSet(validatedInput, request.user);

  NextResponse.json('unused'),
);
