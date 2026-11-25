import { describe, expect, it, vi } from 'vitest';

vi.mock('@/lib/with-user', () => ({
  withOptionalUser: (handler: (r: Request) => Promise<Response>) => handler,
}));

// eslint-disable-next-line import/first
import { POST } from './route';

describe('POST /api/graph/generate', () => {
  it('returns 400, not 500, for a body that is not valid JSON', async () => {
    const res = await POST(
      new Request('http://localhost/api/graph/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{"prompt": ',
      }) as never,
    );
    expect(res.status).toBe(400);
    // `message` keeps the value the old 500 response carried, so clients reading it see no change.
    expect(await res.json()).toEqual({ error: 'Invalid JSON', message: 'Invalid JSON body' });
  });
});
