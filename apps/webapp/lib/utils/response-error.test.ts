import { describe, expect, it } from 'vitest';
import { responseErrorMessage } from './response-error';

describe('responseErrorMessage', () => {
  it('uses the message field of a JSON error body', async () => {
    const res = new Response(JSON.stringify({ message: 'Feature not found' }), { status: 404 });
    expect(await responseErrorMessage(res)).toBe('Feature not found');
  });

  it('uses the error field when there is no message', async () => {
    const res = new Response(JSON.stringify({ error: 'Rate limited' }), { status: 429 });
    expect(await responseErrorMessage(res)).toBe('Rate limited');
  });

  it('falls back to the status for a non-JSON body such as a proxy error page', async () => {
    const res = new Response('<!DOCTYPE html><html><body>504 Gateway Time-out</body></html>', {
      status: 504,
      headers: { 'Content-Type': 'text/html' },
    });
    expect(await responseErrorMessage(res)).toBe('Error: 504');
  });

  it('falls back to the status for a JSON body without message or error', async () => {
    const res = new Response(JSON.stringify({}), { status: 500 });
    expect(await responseErrorMessage(res)).toBe('Error: 500');
  });
});
