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
import { INFERENCE_REQUEST_TIMEOUT_MS, lensPromptStream, steerCompletion } from './inference';

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

  it('asks the first pass to refuse rather than queue', async () => {
    respondWith([async () => new Response('data: {}\n\n', { status: 200 })]);

    await runSteer();

    expect(attempts[0].body.failIfBusy).toBe(true);
  });

  it('waits on a silent host instead of hopping, until every pod honours the flag', async () => {
    // A pod predating `fail_if_busy` on steer queues silently rather than refusing, and that
    // looks identical to wedged from here. Hopping would abandon a generation on each host it
    // passed. Flip STEER_HEADERS_TIMEOUT_MS on once the pods are rolled, and change this test.
    let respond: ((response: Response) => void) | undefined;
    respondWith([
      async (_attempt, signal) =>
        new Promise<Response>((resolve, reject) => {
          signal?.addEventListener('abort', () => reject(signal.reason));
          respond = resolve;
        }),
    ]);

    const pending = runSteer();
    await vi.advanceTimersByTimeAsync(HEADERS_TIMEOUT_MS * 5);

    expect(attempts).toHaveLength(1);
    respond?.(new Response('data: {}\n\n', { status: 200 }));
    await expect(pending).resolves.toBeDefined();
  });

  it('still leaves a busy pod immediately, which is what the flag is for', async () => {
    // The 429 does the real work: an upgraded pod refuses in milliseconds, so failover does not
    // depend on the headers deadline at all.
    respondWith([
      async () => new Response('{"busy": true}', { status: 429 }),
      async () => new Response('data: {}\n\n', { status: 200 }),
    ]);

    await runSteer();

    expect(attempts.map((attempt) => attempt.url)).toEqual([
      'https://a/v1/steer/completion',
      'https://b/v1/steer/completion',
    ]);
  });

  it('queues on a busy host rather than failing when every host declined', async () => {
    // Every pod refusing means the fleet is occupied, not broken. Steer generations are long
    // enough that queueing is normal, so this pass is what keeps a busy fleet working.
    respondWith([
      async () => new Response('{"busy": true}', { status: 429 }),
      async () => new Response('{"busy": true}', { status: 429 }),
      async () => new Response('data: {}\n\n', { status: 200 }),
    ]);

    await runSteer();

    expect(attempts).toHaveLength(3);
    expect(attempts[2].url).toBe('https://a/v1/steer/completion');
    expect(attempts[2].body.failIfBusy).toBe(false);
  });

  it('does not bound the queueing attempt, which is meant to wait', async () => {
    respondWith([
      async () => new Response('{"busy": true}', { status: 429 }),
      async () => new Response('{"busy": true}', { status: 429 }),
      async (_attempt, signal) => silent(signal),
    ]);

    const pending = runSteer();
    pending.catch(() => {}); // nothing should reject; asserted below
    await vi.advanceTimersByTimeAsync(HEADERS_TIMEOUT_MS * 20);

    expect(attempts).toHaveLength(3);
    // No signal at all, with no caller signal to combine: nothing can cut this attempt short.
    expect(attempts[2].signal ?? null).toBeNull();
  });
});

describe('steerCompletion without streaming', () => {
  const feature = {
    modelId: 'gpt2-small',
    layer: '0-res-jb',
    index: 1,
    strength: 1,
    neuron: { vector: [0.1, 0.2], hookName: 'blocks.0.hook_resid_post' },
  };

  /** `/api/steer` collects a whole completion, so nothing arrives until generation ends. */
  const runNonStreaming = () =>
    steerCompletion(
      'gpt2-small',
      [SteerOutputType.DEFAULT],
      'hello',
      1,
      4,
      0,
      0,
      1,
      [feature] as never,
      true,
      null,
      undefined,
      false,
    );

  it('does not abandon a completion that takes longer than the headers deadline', async () => {
    // The reply is one JSON body, so its headers arrive only when the work is finished. A
    // headers deadline here would abort every completion slower than 8s.
    let respond: ((response: Response) => void) | undefined;
    respondWith([
      async (_attempt, signal) =>
        new Promise<Response>((resolve, reject) => {
          signal?.addEventListener('abort', () => reject(signal.reason));
          respond = resolve;
        }),
    ]);

    const pending = runNonStreaming();
    await vi.advanceTimersByTimeAsync(HEADERS_TIMEOUT_MS * 3);

    expect(attempts).toHaveLength(1);
    respond?.(new Response(JSON.stringify({ outputs: [] }), { status: 200 }));
    await expect(pending).resolves.toBeDefined();
  });

  it('still asks the first pass to refuse rather than queue', async () => {
    respondWith([async () => new Response(JSON.stringify({ outputs: [] }), { status: 200 })]);

    await runNonStreaming();

    expect(attempts[0].body.failIfBusy).toBe(true);
    expect(attempts[0].body.stream).toBe(false);
  });

  it('bounds the call as a whole, so a wedged host cannot hang it forever', async () => {
    // `AbortSignal.timeout` runs on a native timer that fake timers do not reach, so stand in
    // a controller driven by `setTimeout`, which they do. That also pins the duration asked for.
    const timeoutSpy = vi.spyOn(AbortSignal, 'timeout').mockImplementation((ms) => {
      const controller = new AbortController();
      setTimeout(() => controller.abort(new DOMException(`timed out after ${ms}ms`, 'TimeoutError')), Number(ms));
      return controller.signal;
    });
    respondWith([async (_attempt, signal) => silent(signal)]);

    const pending = runNonStreaming();
    pending.catch(() => {}); // asserted below
    await vi.advanceTimersByTimeAsync(INFERENCE_REQUEST_TIMEOUT_MS + 1);

    await expect(pending).rejects.toThrow(/timed out/);
    expect(timeoutSpy).toHaveBeenCalledWith(INFERENCE_REQUEST_TIMEOUT_MS);
    // The deadline is the caller's own, so it stops the call rather than moving to the next host.
    expect(attempts).toHaveLength(1);
    timeoutSpy.mockRestore();
  });
});
