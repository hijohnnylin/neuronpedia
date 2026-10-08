import { describe, expect, it } from 'vitest';
import { LensOracleMetaMessage, LensOracleReadMessage } from './lens';
import { cachedOracleMessages, getOracleEntry, mapNdjson, oracleCacheKey, recordOracleMessage } from './oracle-cache';

function streamOf(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      chunks.forEach((c) => controller.enqueue(encoder.encode(c)));
      controller.close();
    },
  });
}

async function textOf(body: ReadableStream<Uint8Array>): Promise<string> {
  return new Response(body).text();
}

const META: LensOracleMetaMessage = {
  kind: 'meta',
  model: 'Qwen/Qwen3.6-27B',
  adapter: 'a:b',
  position: 1,
  token: ' x',
  layers: [20, 24],
  max_bullets: 2,
};
const READ: LensOracleReadMessage = {
  kind: 'read',
  layer: 24,
  bullets: ['One'],
  text: '- One\n',
  finish: 'bullets',
  cached: false,
};

describe('mapNdjson', () => {
  it('maps each message once, when lines split across chunks', async () => {
    const lines = `${JSON.stringify(META)}\n${JSON.stringify(READ)}\n{"kind":"done","elapsed_ms":5,"cached_layers":0}`;
    const seen: string[] = [];
    const out = await textOf(
      mapNdjson(streamOf([lines.slice(0, 17), lines.slice(17, 140), lines.slice(140)]), (m) => {
        seen.push(m.kind);
        return m.kind === 'read' ? { ...m, sig: 'v1.x' } : m;
      }),
    );
    expect(seen).toEqual(['meta', 'read', 'done']);
    const messages = out
      .trim()
      .split('\n')
      .map((l) => JSON.parse(l));
    expect(messages[1]).toEqual({ ...READ, sig: 'v1.x' });
    expect(messages[2].kind).toBe('done');
  });

  it('passes a line that is not JSON through unchanged', async () => {
    const out = await textOf(mapNdjson(streamOf(['not json\n', `${JSON.stringify(READ)}\n`]), (m) => m));
    expect(out).toBe(`not json\n${JSON.stringify(READ)}\n`);
  });
});

describe('recordOracleMessage', () => {
  it('keeps a read without its signature', () => {
    const key = oracleCacheKey('test-model', 2, 96, [1, 2]);
    recordOracleMessage(key, true, META);
    recordOracleMessage(key, true, { ...READ, layer: 20, sig: 'v1.a' });
    recordOracleMessage(key, true, { ...READ, sig: 'v1.b' });
    const hit = cachedOracleMessages([{ position: 1, entry: getOracleEntry(key) }], []);
    expect(hit?.filter((m) => m.kind === 'read').map((m) => 'sig' in m)).toEqual([false, false]);
  });
});

describe('cachedOracleMessages', () => {
  it('answers only when every position has every layer', () => {
    const a = oracleCacheKey('test-model-many', 2, 96, [1]);
    const b = oracleCacheKey('test-model-many', 2, 96, [1, 2]);
    recordOracleMessage(a, false, { ...META, position: 0, token: ' a' });
    recordOracleMessage(a, false, { ...READ, layer: 20 });
    recordOracleMessage(b, false, { ...META, position: 1, token: ' b' });
    const hits = () => [
      { position: 0, entry: getOracleEntry(a) },
      { position: 1, entry: getOracleEntry(b) },
    ];
    expect(cachedOracleMessages(hits(), [20])).toBeNull();

    recordOracleMessage(b, false, { ...READ, layer: 20 });
    const hit = cachedOracleMessages(hits(), [20])!;
    expect(hit[0]).toMatchObject({ kind: 'meta', position: 0, positions: [0, 1], tokens: [' a', ' b'], layers: [20] });
    expect(hit.filter((m) => m.kind === 'read').map((m) => ('position' in m ? m.position : null))).toEqual([0, 1]);
    expect(hit[hit.length - 1]).toMatchObject({ kind: 'done', cached_layers: 2 });
  });
});
