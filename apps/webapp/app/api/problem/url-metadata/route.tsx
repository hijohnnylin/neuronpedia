import { fetchUrlMetadata } from '@/lib/problem-utils';
import { RequestAuthedUser, withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const body = await request.json();
  const { url } = body;

  if (!url || typeof url !== 'string') {
    return NextResponse.json({ error: 'url is required' }, { status: 400 });
  }

  if (!/^https?:\/\/.+/i.test(url)) {
    return NextResponse.json({ error: 'Invalid URL. Must start with http:// or https://' }, { status: 400 });
  }

  try {
    const meta = await fetchUrlMetadata(url);
    return NextResponse.json(meta);
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    console.error('[url-metadata] Failed to fetch:', url, message);
    return NextResponse.json({ error: message }, { status: 502 });
  }
});
