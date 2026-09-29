import { describe, expect, it, vi } from 'vitest';
import { TURNSTILE_SIGN_IN_ACTION, TURNSTILE_SITEVERIFY_URL, verifyTurnstileToken } from './turnstile';

function mockFetch(body: unknown, status = 200) {
  return vi.fn(async () => new Response(JSON.stringify(body), { status })) as unknown as typeof fetch & {
    mock: { calls: [string, RequestInit][] };
  };
}

const args = { secret: 'secret-key', token: 'token-1', expectedAction: TURNSTILE_SIGN_IN_ACTION };

describe('verifyTurnstileToken', () => {
  it('accepts a good token with the correct action', async () => {
    const fetchFn = mockFetch({ success: true, action: TURNSTILE_SIGN_IN_ACTION });
    expect(await verifyTurnstileToken({ ...args, remoteIp: '1.2.3.4', fetchFn })).toEqual({ ok: true });

    const [url, init] = fetchFn.mock.calls[0];
    expect(url).toBe(TURNSTILE_SITEVERIFY_URL);
    expect(init.method).toBe('POST');
    const sent = init.body as URLSearchParams;
    expect(sent.get('secret')).toBe('secret-key');
    expect(sent.get('response')).toBe('token-1');
    expect(sent.get('remoteip')).toBe('1.2.3.4');
  });

  it('does not send remoteip when there is no IP', async () => {
    const fetchFn = mockFetch({ success: true, action: TURNSTILE_SIGN_IN_ACTION });
    await verifyTurnstileToken({ ...args, fetchFn });
    expect((fetchFn.mock.calls[0][1].body as URLSearchParams).has('remoteip')).toBe(false);
  });

  it('rejects a missing or too long token without calling Cloudflare', async () => {
    const fetchFn = mockFetch({ success: true, action: TURNSTILE_SIGN_IN_ACTION });
    expect(await verifyTurnstileToken({ ...args, token: '', fetchFn })).toEqual({
      ok: false,
      reason: 'missing-token',
    });
    expect(await verifyTurnstileToken({ ...args, token: null, fetchFn })).toMatchObject({ ok: false });
    expect(await verifyTurnstileToken({ ...args, token: 'x'.repeat(2049), fetchFn })).toMatchObject({ ok: false });
    expect(fetchFn).not.toHaveBeenCalled();
  });

  it('rejects when Cloudflare says the token is not good', async () => {
    const fetchFn = mockFetch({ success: false, 'error-codes': ['timeout-or-duplicate'] });
    expect(await verifyTurnstileToken({ ...args, fetchFn })).toEqual({
      ok: false,
      reason: 'timeout-or-duplicate',
    });
  });

  it('rejects a token made for a different action', async () => {
    const fetchFn = mockFetch({ success: true, action: 'other' });
    expect(await verifyTurnstileToken({ ...args, fetchFn })).toMatchObject({ ok: false });
  });

  it('rejects a token with no action', async () => {
    const fetchFn = mockFetch({ success: true });
    expect(await verifyTurnstileToken({ ...args, fetchFn })).toMatchObject({ ok: false });
  });

  it('accepts a result from a Cloudflare test secret key, which has no action', async () => {
    const fetchFn = mockFetch({ success: true, metadata: { result_with_testing_key: true } });
    expect(await verifyTurnstileToken({ ...args, fetchFn })).toEqual({ ok: true });
  });

  it('rejects a test key result with a different action', async () => {
    const fetchFn = mockFetch({ success: true, action: 'other', metadata: { result_with_testing_key: true } });
    expect(await verifyTurnstileToken({ ...args, fetchFn })).toMatchObject({ ok: false });
  });

  it('rejects when success is not exactly true', async () => {
    const fetchFn = mockFetch({ success: 'true', action: TURNSTILE_SIGN_IN_ACTION });
    expect(await verifyTurnstileToken({ ...args, fetchFn })).toMatchObject({ ok: false });
  });

  it('rejects on an HTTP error', async () => {
    const fetchFn = mockFetch({}, 500);
    expect(await verifyTurnstileToken({ ...args, fetchFn })).toEqual({ ok: false, reason: 'http-500' });
  });

  it('rejects when the request fails', async () => {
    const fetchFn = vi.fn(async () => {
      throw new Error('network down');
    }) as unknown as typeof fetch;
    expect(await verifyTurnstileToken({ ...args, fetchFn })).toMatchObject({ ok: false });
  });
});
