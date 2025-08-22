import { updateUserAccount } from '@/lib/db/user';
import { RequestAuthedUser } from '@/lib/types/auth';
import { withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';
import { z } from 'zod';

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const body = await request.json();

  const name = z
    .string()
    .regex(/^[a-zA-Z0-9-]+$/, 'Name should contain only numbers, letters, and dashes')
    .min(1)
    .max(39)
    .transform((val: string) => val.toLowerCase())
    .parse(body.name);

  if (!name) {
    throw new Error('Invalid name.');
  }

  const updatedUser = await updateUserAccount(
    request.user.id,
    name,
    body.newsletterNotifyEmail,
    body.unsubscribeAllEmail,
  );

  return NextResponse.json(updatedUser);
});
