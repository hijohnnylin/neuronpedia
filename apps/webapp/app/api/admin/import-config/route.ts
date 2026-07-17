import { importConfigFromS3 } from '@/lib/db/import';
import { IS_LOCALHOST } from '@/lib/env';
import { getAuthedAdminUser, RequestAuthedAdminUser, RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const GET = withOptionalUser(async (request: RequestOptionalUser) => {
  if (!IS_LOCALHOST && (!request.user || !(await getAuthedAdminUser(request as RequestAuthedAdminUser)))) {
    return NextResponse.json(
      { error: 'This route is only available on localhost or to admin users' },
      { status: 403 },
    );
  }

  await importConfigFromS3();
  return NextResponse.json({ message: 'Config synced' }, { status: 200 });
});
