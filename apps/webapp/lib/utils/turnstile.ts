// Cloudflare Turnstile token check. See https://developers.cloudflare.com/turnstile/get-started/server-side-validation/

export const TURNSTILE_SITEVERIFY_URL = 'https://challenges.cloudflare.com/turnstile/v0/siteverify';
export const TURNSTILE_SIGN_IN_ACTION = 'sign-in';

// Cloudflare does not accept tokens longer than this.
const MAX_TOKEN_LENGTH = 2048;
const VERIFY_TIMEOUT_MS = 10000;

export type TurnstileVerifyResult = { ok: true } | { ok: false; reason: string };

export async function verifyTurnstileToken({
  secret,
  token,
  expectedAction,
  remoteIp,
  fetchFn = fetch,
}: {
  secret: string;
  token: string | null | undefined;
  expectedAction: string;
  remoteIp?: string | null;
  fetchFn?: typeof fetch;
}): Promise<TurnstileVerifyResult> {
  if (!token) {
    return { ok: false, reason: 'missing-token' };
  }
  if (token.length > MAX_TOKEN_LENGTH) {
    return { ok: false, reason: 'token-too-long' };
  }

  const body = new URLSearchParams({ secret, response: token });
  if (remoteIp) {
    body.set('remoteip', remoteIp);
  }

  let data: {
    success?: boolean;
    action?: string;
    'error-codes'?: string[];
    metadata?: { result_with_testing_key?: boolean };
  };
  try {
    const res = await fetchFn(TURNSTILE_SITEVERIFY_URL, {
      method: 'POST',
      body,
      signal: AbortSignal.timeout(VERIFY_TIMEOUT_MS),
    });
    if (!res.ok) {
      return { ok: false, reason: `http-${res.status}` };
    }
    data = await res.json();
  } catch (error) {
    return { ok: false, reason: `request-failed: ${error instanceof Error ? error.message : String(error)}` };
  }

  if (data.success !== true) {
    return { ok: false, reason: (data['error-codes'] || []).join(',') || 'not-success' };
  }
  // Cloudflare test secret keys give no action. Only a test secret key can give a test result.
  const isTestResult = data.metadata?.result_with_testing_key === true && !data.action;
  if (data.action !== expectedAction && !isTestResult) {
    return { ok: false, reason: `wrong-action: ${data.action}` };
  }
  return { ok: true };
}
