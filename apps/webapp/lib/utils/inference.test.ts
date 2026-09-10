import { beforeEach, describe, expect, it, vi } from 'vitest';

// vi.mock is hoisted above the imports, so its factory can only close over
// values created by vi.hoisted.
const { resolveHosts, getTransformerLensModelIdIfExists } = vi.hoisted(() => ({
  resolveHosts: vi.fn(),
  getTransformerLensModelIdIfExists: vi.fn(),
}));

vi.mock('../db/compute-host', () => ({
  resolveHosts,
  resolveHost: vi.fn(),
  NoComputeHostError: class NoComputeHostError extends Error {},
}));
vi.mock('@/lib/db/model', () => ({ getTransformerLensModelIdIfExists }));
vi.mock('@/lib/db/neuron', () => ({ getNeuronOnly: vi.fn() }));
vi.mock('../env', () => ({ INFERENCE_SERVER_SECRET: 'test-secret' }));

import { SteerOutputType } from '@prisma/client';
import { lensPromptStream, steerCompletion } from './inference';

const HEADERS_TIMEOUT_MS = 8_000;

type Attempt = { url: string; body: Record<string, unknown>; signal?: AbortSignal | null };

let attempts: Attempt[] = [];

/** A response that never arrives, and rejects the way undici does when the signal fires. */
const silent = (signal?: AbortSignal | null) =>
  new Promise<Response>((_, reject) => {
    if (signal?.aborted) {
      reject(signal.reason);
      return;
    }
    signal?.addEventListener('abort', () => reject(signal.reason));
  });

/** Install a fetch that answers each successive attempt from `answers`. */
const respondWith = (answers: ((attempt: Attempt, signal?: AbortSignal | null) => Promise<Response>)[]) => {
  const fetchMock = vi.fn((url: string, init: RequestInit) => {
    const attempt = { url, body: JSON.parse(String(init.body)), signal: init.signal };
    attempts.push(attempt);
    const answer = answers[attempts.length - 1] ?? answers[answers.length - 1];
    return answer(attempt, init.signal);
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
};

const runLens = (signal?: AbortSignal) =>
  lensPromptStream('gpt2-small', { prompt: 'hello' } as never, signal) as Promise<Response>;

beforeEach(() => {
  attempts = [];
  vi.useFakeTimers();
  resolveHosts.mockReset().mockResolvedValue(['https://a', 'https://b']);
  getTransformerLensModelIdIfExists.mockReset().mockResolvedValue('gpt2');
});

describe('lensPromptStream host selection', () => {
  it('uses the first host that accepts the request', async () => {
    respondWith([async () => new Response('{}', { status: 200 })]);

    const response = await runLens();

    expect(response.status).toBe(200);
    expect(attempts).toHaveLength(1);
    expect(attempts[0].url).toBe('https://a/v1/lens/prompt');
  });

  it('moves past a host that reports itself busy', async () => {
    respondWith([
      async () => new Response('{"busy": true}', { status: 429 }),
      async () => new Response('{}', { status: 200 }),
    ]);

    const response = await runLens();

    expect(response.status).toBe(200);
    expect(attempts.map((attempt) => attempt.url)).toEqual(['https://a/v1/lens/prompt', 'https://b/v1/lens/prompt']);
  });

  it('returns a deterministic client error without trying anyone else', async () => {
    respondWith([async () => new Response('{"error": "prompt too long"}', { status: 400 })]);

    expect((await runLens()).status).toBe(400);
    expect(attempts).toHaveLength(1);
  });
});

describe('lensPromptStream headers deadline', () => {
  it('gives up on a silent host and tries the next one', async () => {
    respondWith([async (_attempt, signal) => silent(signal), async () => new Response('{}', { status: 200 })]);

    const pending = runLens();
    await vi.advanceTimersByTimeAsync(HEADERS_TIMEOUT_MS);

    expect((await pending).status).toBe(200);
    expect(attempts.map((attempt) => attempt.url)).toEqual(['https://a/v1/lens/prompt', 'https://b/v1/lens/prompt']);
  });

  it('stops counting once the headers arrive, so a long stream is never cut', async () => {
    let push: (chunk: string) => void = () => {};
    let close: () => void = () => {};
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        push = (chunk) => controller.enqueue(new TextEncoder().encode(chunk));
        close = () => controller.close();
      },
    });
    respondWith([async () => new Response(body, { status: 200 })]);

    const response = await runLens();
    // Far longer than the deadline: a run is minutes of generation, and only the wait for
    // headers was ever bounded.
    await vi.advanceTimersByTimeAsync(HEADERS_TIMEOUT_MS * 10);
    push('{"kind":"done"}\n');
    close();

    await expect(new Response(response.body).text()).resolves.toBe('{"kind":"done"}\n');
  });

  it('queues on a silent host rather than failing when every host went quiet', async () => {
    respondWith([
      async (_attempt, signal) => silent(signal),
      async (_attempt, signal) => silent(signal),
      async () => new Response('{}', { status: 200 }),
    ]);

    const pending = runLens();
    await vi.advanceTimersByTimeAsync(HEADERS_TIMEOUT_MS * 2);

    expect((await pending).status).toBe(200);
    // Pass 2 goes back to the first host that was reachable-but-silent, and asks it to wait
    // its turn instead of failing fast.
    expect(attempts).toHaveLength(3);
    expect(attempts[2].url).toBe('https://a/v1/lens/prompt');
    expect(attempts[2].body.failIfBusy).toBe(false);
  });

  it('prefers a host that answered busy over one that went quiet, for the queue', async () => {
    respondWith([
      async (_attempt, signal) => silent(signal),
      async () => new Response('{"busy": true}', { status: 429 }),
      async () => new Response('{}', { status: 200 }),
    ]);

    const pending = runLens();
    await vi.advanceTimersByTimeAsync(HEADERS_TIMEOUT_MS);

    expect((await pending).status).toBe(200);
    expect(attempts[2].url).toBe('https://b/v1/lens/prompt');
  });

  it('does not bound the queueing attempt, which is meant to wait', async () => {
    respondWith([
      async () => new Response('{"busy": true}', { status: 429 }),
      async () => new Response('{"busy": true}', { status: 429 }),
      async (_attempt, signal) => silent(signal),
    ]);

    const pending = runLens();
    // Long past the deadline. Nothing should abort the third attempt.
    await vi.advanceTimersByTimeAsync(HEADERS_TIMEOUT_MS * 10);

    expect(attempts).toHaveLength(3);
    expect(attempts[2].body.failIfBusy).toBe(false);
    await expect(Promise.race([pending, Promise.resolve('still waiting')])).resolves.toBe('still waiting');
  });

  it('does not try another host when the caller is the one who aborted', async () => {
    const caller = new AbortController();
    respondWith([async (_attempt, signal) => silent(signal)]);

    const pending = runLens(caller.signal);
    caller.abort(new DOMException('client went away', 'AbortError'));

    await expect(pending).rejects.toThrow(/client went away/);
    expect(attempts).toHaveLength(1);
  });
});

describe('steerCompletion host selection', () => {
  const feature = {
    modelId: 'gpt2-small',
    layer: '0-res-jb',
    index: 1,
    strength: 1,
    neuron: { vector: [0.1, 0.2], hookName: 'blocks.0.hook_resid_post' },
  };

  const runSteer = () =>
    steerCompletion('gpt2-small', [SteerOutputType.DEFAULT], 'hello', 1, 4, 0, 0, 1, [feature] as never, true, null);

  it('moves to another host when one is unreachable', async () => {
    respondWith([
      async () => {
        throw new TypeError('fetch failed');
      },
      async () => new Response('data: {}\n\n', { status: 200 }),
    ]);

    await runSteer();

    expect(attempts.map((attempt) => attempt.url)).toEqual([
      'https://a/v1/steer/completion',
      'https://b/v1/steer/completion',
    ]);
  });

  it('moves to another host when a dead pod answers through its gateway', async () => {
    respondWith([
      async () => new Response('Not Found', { status: 404 }),
      async () => new Response('data: {}\n\n', { status: 200 }),
    ]);

    await runSteer();

    expect(attempts).toHaveLength(2);
  });

  it('waits on a host that is merely slow, rather than hopping off it', async () => {
    // Steer has no fail-fast, so silence means "queued behind a generation" far more often
    // than "wedged". Hopping would abandon a pod that was about to answer.
    let respond: (() => void) | undefined;
    respondWith([
      async () =>
        new Promise<Response>((resolve) => {
          respond = () => resolve(new Response('data: {}\n\n', { status: 200 }));
        }),
    ]);

    const pending = runSteer();
    await vi.advanceTimersByTimeAsync(HEADERS_TIMEOUT_MS * 20);

    expect(attempts).toHaveLength(1);
    respond?.();
    await expect(pending).resolves.toBeDefined();
  });
});
