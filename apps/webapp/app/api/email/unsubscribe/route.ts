import { setUserEmailUnsubscribe } from '@/lib/db/user';
import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';

const querySchema = z.object({
  code: z.string().min(1),
  list: z.enum(['newsletter', 'all']),
});

// One-click unsubscribe (RFC 8058). Mail clients POST to the List-Unsubscribe URL, so the
// code and list are in the query string, not the body. The unsubscribe pages also call this.
export async function POST(request: NextRequest) {
  const parsed = querySchema.safeParse(Object.fromEntries(request.nextUrl.searchParams));
  if (!parsed.success) {
    return NextResponse.json({ error: 'Invalid unsubscribe link' }, { status: 400 });
  }
  const unsubscribed = await setUserEmailUnsubscribe(parsed.data.code, parsed.data.list);
  if (!unsubscribed) {
    return NextResponse.json({ error: 'Invalid unsubscribe link' }, { status: 404 });
  }
  return NextResponse.json({ unsubscribed: true });
}
