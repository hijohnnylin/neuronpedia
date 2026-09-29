import { NextRequest } from 'next/server';

// The confirm page posts here. Link scanners open links but do not send POST forms, so they cannot sign in.
export async function POST(req: NextRequest) {
  const form = await req.formData();
  const params = new URLSearchParams();
  for (const name of ['token', 'email', 'callbackUrl']) {
    const value = form.get(name);
    if (typeof value === 'string') {
      params.set(name, value);
    }
  }
  return new Response(null, {
    status: 303,
    headers: { Location: `/api/auth/callback/email?${params}` },
  });
}
