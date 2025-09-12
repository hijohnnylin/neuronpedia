import { importConfigFromS3 } from '@/lib/db/import';
import { env } from '@/lib/env';
import { RequestAuthedAdminUser, RequestOptionalUser } from '@/lib/types/auth';
import { getAuthedAdminUser, withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const GET = withOptionalUser(async (request: RequestOptionalUser) => {
  if (!env.IS_LOCALHOST && request.user && !(await getAuthedAdminUser(request as RequestAuthedAdminUser))) {
    return NextResponse.json({ error: 'This route is only available on localhost or to admin users' }, { status: 400 });
  }

  await importConfigFromS3();
  return NextResponse.json({ message: 'Config synced' }, { status: 200 });
});
