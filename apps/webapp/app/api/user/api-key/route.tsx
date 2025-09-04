import { getUserSecret } from '@/lib/db/userSecret';
import { RequestAuthedUser } from '@/lib/types/auth';
import { withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const body = await request.json();
  const { type } = body;
  const secret = await getUserSecret(request.user.name, type);
  if (!secret) {
    return NextResponse.json({ message: 'Secret not found' }, { status: 404 });
  }
  return NextResponse.json(secret);
});
