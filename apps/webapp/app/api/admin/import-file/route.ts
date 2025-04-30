/* eslint-disable no-restricted-syntax */
/* eslint-disable no-await-in-loop */

import { importJsonlString } from '@/lib/db/import';
import { IS_LOCALHOST } from '@/lib/env';
import { downloadAndDecompressFile } from '@/lib/utils/s3';
import { getAuthedAdminUser, RequestAuthedAdminUser, RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

// Hobby plans don't support > 60 seconds
// export const maxDuration = 300;

export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  const body = await request.json();
  const { path, tableName } = body;

  if (!path || !tableName) {
    return NextResponse.json({ error: 'path and tableName are required' }, { status: 400 });
  }

  if (!IS_LOCALHOST && request.user && !(await getAuthedAdminUser(request as RequestAuthedAdminUser))) {
    return NextResponse.json({ error: 'This route is only available on localhost or to admin users' }, { status: 400 });
  }
  const linesJsonlString = await downloadAndDecompressFile(path);
  await importJsonlString(tableName, linesJsonlString);

  return NextResponse.json({ success: true });
});
