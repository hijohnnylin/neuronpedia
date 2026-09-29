import { CLOUDFLARE_TURNSTILE_SECRETKEY, NEXTAUTH_URL, TURNSTILE_ENABLED } from '@/lib/env';
import { authErrorUrl, SIGN_IN_ERROR_TURNSTILE } from '@/lib/utils/sign-in-email';
import { TURNSTILE_SIGN_IN_ACTION, verifyTurnstileToken } from '@/lib/utils/turnstile';
import { ipAddress } from '@vercel/functions';
import NextAuth from 'next-auth';
import { NextRequest, NextResponse } from 'next/server';
import { authOptions } from './authOptions';

const handler = NextAuth(authOptions);

type RouteContext = { params: Promise<{ nextauth: string[] }> };

// The client sends the Turnstile token in the query string of the email sign-in POST.
async function POST(req: NextRequest, context: RouteContext) {
  const { nextauth } = await context.params;
  if (TURNSTILE_ENABLED && nextauth?.[0] === 'signin' && nextauth?.[1] === 'email') {
    const result = await verifyTurnstileToken({
      secret: CLOUDFLARE_TURNSTILE_SECRETKEY,
      token: req.nextUrl.searchParams.get('turnstile'),
      expectedAction: TURNSTILE_SIGN_IN_ACTION,
      remoteIp: ipAddress(req),
    });
    if (!result.ok) {
      console.warn(`Email sign-in blocked by Turnstile: ${result.reason}`);
      const url = authErrorUrl(NEXTAUTH_URL || req.nextUrl.origin, SIGN_IN_ERROR_TURNSTILE);
      const body = new URLSearchParams(await req.text());
      if (body.get('json') === 'true') {
        return NextResponse.json({ url }, { status: 403 });
      }
      return NextResponse.redirect(url, 302);
    }
  }
  return handler(req, context);
}

export { handler as GET, POST };
