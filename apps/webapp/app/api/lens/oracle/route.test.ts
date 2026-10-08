import { oraclePrefixDigest, verifyOracleRead } from '@/lib/utils/oracle-signature';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({ lensOracleStream: vi.fn() }));
vi.mock('@/lib/utils/inference', () => ({ lensOracleStream: mocks.lensOracleStream }));

import { POST } from './route';

const MODEL = 'oracle-route-test';
const TOKEN_IDS = [7, 8, 9, 10];

function oracleRun(): Response {
  const lines = [
    { kind: 'meta', model: 'M', adapter: 'a:b', position: 2, token: ' x', layers: [20, 24], max_bullets: 2 },
    { kind: 'read', layer: 24, bullets: ['One'], text: '- One\n', finish: 'bullets', cached: false },
    { kind: 'read', layer: 20, bullets: ['Two'], text: '- Two\n', finish: 'eos', cached: false },
    { kind: 'done', elapsed_ms: 9, cached_layers: 0 },
  ];
  return new Response(lines.map((m) => `${JSON.stringify(m)}\n`).join(''));
}

async function readsOf(stream: boolean, position: number) {
  const res = await POST(
    new Request('http://localhost/api/lens/oracle', {
      method: 'POST',
      body: JSON.stringify({ modelId: MODEL, tokenIds: TOKEN_IDS, position, maxBullets: 2, stream }),
    }),
  );
  expect(res.status).toBe(200);
  if (!stream) {
    return (await res.json()).reads;
  }
  return (await res.text())
    .trim()
    .split('\n')
    .map((l) => JSON.parse(l))
    .filter((m) => m.kind === 'read');
}

function valid(
  read: { layer: number; bullets: string[]; text: string; finish: string; sig: string },
  position: number,
) {
  return verifyOracleRead(
    {
      modelId: MODEL,
      adapter: 'a:b',
      maxBullets: 1,
      position,
      prefixDigest: oraclePrefixDigest(TOKEN_IDS.slice(0, position + 1)),
      layer: read.layer,
      bullets: read.bullets,
      text: read.text,
      finish: read.finish,
    },
    read.sig,
  );
}

describe('POST /api/lens/oracle signatures', () => {
  const saved = process.env.NEXTAUTH_SECRET;
  beforeEach(() => {
    process.env.NEXTAUTH_SECRET = 'test-secret';
    mocks.lensOracleStream.mockReset().mockImplementation(async () => oracleRun());
  });
  afterEach(() => {
    process.env.NEXTAUTH_SECRET = saved;
  });

  it('signs streamed reads, then the same reads from the cache', async () => {
    const streamed = await readsOf(true, 2);
    expect(streamed).toHaveLength(2);
    expect(streamed.every((r: never) => valid(r, 2))).toBe(true);

    const cached = await readsOf(true, 2);
    expect(mocks.lensOracleStream).toHaveBeenCalledTimes(1);
    expect(cached.map((r: { cached: boolean }) => r.cached)).toEqual([true, true]);
    expect(cached.every((r: never) => valid(r, 2))).toBe(true);
  });

  it('signs buffered reads', async () => {
    const reads = await readsOf(false, 1);
    expect(reads.every((r: never) => valid(r, 1))).toBe(true);
    // A signature is for one position only.
    expect(reads.some((r: never) => valid(r, 2))).toBe(false);
  });

  it('streams partials unsigned, asks for them only when streaming, and does not cache them', async () => {
    mocks.lensOracleStream.mockImplementation(async () => {
      const text = await oracleRun().text();
      const partial = JSON.stringify({ kind: 'partial', layer: 24, text: '- On' });
      const [meta, ...rest] = text.trim().split('\n');
      return new Response([meta, partial, ...rest].map((l) => `${l}\n`).join(''));
    });
    const body = (stream: boolean) =>
      JSON.stringify({ modelId: `${MODEL}-partial`, tokenIds: TOKEN_IDS, position: 3, maxBullets: 2, stream });
    const post = (stream: boolean) =>
      POST(new Request('http://localhost/api/lens/oracle', { method: 'POST', body: body(stream) }));

    const messages = (await (await post(true)).text())
      .trim()
      .split('\n')
      .map((l) => JSON.parse(l));
    expect(mocks.lensOracleStream.mock.calls[0][1].partial).toBe(true);
    expect(messages.map((m) => m.kind)).toEqual(['meta', 'partial', 'read', 'read', 'done']);
    expect(messages[1]).toEqual({ kind: 'partial', layer: 24, text: '- On' });

    const cached = (await (await post(true)).text()).trim().split('\n');
    expect(mocks.lensOracleStream).toHaveBeenCalledTimes(1);
    expect(cached.map((l) => JSON.parse(l).kind)).toEqual(['meta', 'read', 'read', 'done']);

    await POST(
      new Request('http://localhost/api/lens/oracle', {
        method: 'POST',
        body: JSON.stringify({ modelId: `${MODEL}-buffered`, tokenIds: TOKEN_IDS, position: 3, stream: false }),
      }),
    );
    expect(mocks.lensOracleStream.mock.calls[1][1].partial).toBe(false);
  });

  it('reads more positions in one request, and signs and caches each by its position', async () => {
    mocks.lensOracleStream.mockImplementation(async () => {
      const lines = [
        {
          kind: 'meta',
          model: 'M',
          adapter: 'a:b',
          position: 1,
          token: ' y',
          positions: [1, 3],
          tokens: [' y', ' z'],
          layers: [20],
          max_bullets: 2,
        },
        { kind: 'read', position: 3, layer: 20, bullets: ['Z'], text: '- Z\n', finish: 'bullets', cached: false },
        { kind: 'read', position: 1, layer: 20, bullets: ['Y'], text: '- Y\n', finish: 'bullets', cached: false },
        { kind: 'done', elapsed_ms: 9, cached_layers: 0 },
      ];
      return new Response(lines.map((m) => `${JSON.stringify(m)}\n`).join(''));
    });
    const post = async (body: object) =>
      (
        await (
          await POST(
            new Request('http://localhost/api/lens/oracle', {
              method: 'POST',
              body: JSON.stringify({ modelId: MODEL, tokenIds: TOKEN_IDS, maxBullets: 2, maxTokens: 64, ...body }),
            }),
          )
        ).text()
      )
        .trim()
        .split('\n')
        .map((l) => JSON.parse(l));

    const streamed = await post({ position: 1, positions: [3, 1], layers: [20] });
    expect(mocks.lensOracleStream.mock.calls[0][1]).toMatchObject({
      position: 1,
      positions: [3],
      maxBullets: 1,
      maxTokens: 32,
    });
    const reads = streamed.filter((m) => m.kind === 'read');
    expect(reads.map((r) => r.position)).toEqual([3, 1]);
    expect(reads.every((r) => valid(r, r.position))).toBe(true);

    const one = await post({ position: 3, layers: [20] });
    expect(mocks.lensOracleStream).toHaveBeenCalledTimes(1);
    expect(one[0]).toMatchObject({ kind: 'meta', position: 3, token: ' z', positions: [3], tokens: [' z'] });
    expect(one[1]).toMatchObject({ kind: 'read', position: 3, text: '- Z\n', cached: true });
    expect(valid(one[1], 3)).toBe(true);

    // The caps are fixed, so other caps from a client hit the same cache.
    await post({ position: 3, layers: [20], maxBullets: 3, maxTokens: 96 });
    expect(mocks.lensOracleStream).toHaveBeenCalledTimes(1);
  });

  it('rejects a position outside tokenIds', async () => {
    const res = await POST(
      new Request('http://localhost/api/lens/oracle', {
        method: 'POST',
        body: JSON.stringify({ modelId: MODEL, tokenIds: TOKEN_IDS, position: 0, positions: [4] }),
      }),
    );
    expect(res.status).toBe(400);
  });
});
