import { getProblemLogs } from '@/lib/db/problem';
import { RequestAuthedAdminUser, withAuthedAdminUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const POST = withAuthedAdminUser(async (request: RequestAuthedAdminUser) => {
  const body = await request.json();
  const limit = typeof body.limit === 'number' ? Math.min(body.limit, 200) : 50;

  const logs = await getProblemLogs(limit);
  return NextResponse.json(logs);
});
