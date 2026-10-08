// Oracle reads this server process has served, so a repeat does not reach the inference server.
// A read is greedy: it depends only on the model, the bullet and token caps and the token prefix.
// The adapter is the model's, and changes only with a deployment, which starts a new process.

import { createHash } from 'crypto';
import { LensOracleMetaMessage, LensOracleReadMessage, LensOracleStreamMessage } from './lens';

// Positions kept (a page reads up to 1024 per run); one read per layer, ~1 KB each.
const MAX_ENTRIES = 8192;

export type OracleCacheEntry = {
  meta: LensOracleMetaMessage;
  reads: Map<number, LensOracleReadMessage>;
  // The server's oracle layers, once a request for its default layers got a meta.
  allLayers: number[] | null;
};

const entries = new Map<string, OracleCacheEntry>();

export function oracleCacheKey(modelId: string, maxBullets: number, maxTokens: number, prefix: number[]): string {
  const digest = createHash('sha1').update(prefix.join(',')).digest('hex');
  return `${modelId}|${maxBullets}|${maxTokens}|${prefix.length}|${digest}`;
}

export function getOracleEntry(key: string): OracleCacheEntry | null {
  const entry = entries.get(key);
  if (!entry) {
    return null;
  }
  entries.delete(key);
  entries.set(key, entry);
  return entry;
}

// The messages that answer a request for `layers` (empty: the server's default layers) at
// each position, or null when a position misses a layer.
export function cachedOracleMessages(
  hits: { position: number; entry: OracleCacheEntry | null }[],
  layers: number[],
): LensOracleStreamMessage[] | null {
  const first = hits[0]?.entry;
  const wanted = layers.length > 0 ? layers : first?.allLayers;
  if (!first || !wanted || !hits.every(({ entry }) => entry && wanted.every((layer) => entry.reads.has(layer)))) {
    return null;
  }
  return [
    {
      ...first.meta,
      position: hits[0].position,
      positions: hits.map((h) => h.position),
      tokens: hits.map((h) => h.entry!.meta.token),
      layers: wanted,
    },
    ...hits.flatMap(({ position, entry }) =>
      wanted.map((layer) => ({ ...entry!.reads.get(layer)!, position, cached: true })),
    ),
    { kind: 'done', elapsed_ms: 0, cached_layers: wanted.length * hits.length },
  ];
}

// Keep what one message of an inference response says. `defaultLayers`: the request asked for
// the server's default layers.
export function recordOracleMessage(key: string, defaultLayers: boolean, message: LensOracleStreamMessage) {
  if (message.kind === 'meta') {
    const entry = entries.get(key);
    entries.delete(key);
    entries.set(key, {
      meta: message,
      reads: entry?.reads ?? new Map(),
      allLayers: defaultLayers ? message.layers : (entry?.allLayers ?? null),
    });
    while (entries.size > MAX_ENTRIES) {
      const oldest = entries.keys().next().value;
      if (oldest === undefined) {
        break;
      }
      entries.delete(oldest);
    }
  } else if (message.kind === 'read') {
    const { sig: _sig, ...read } = message;
    entries.get(key)?.reads.set(message.layer, { ...read, cached: false });
  }
}

// `body` with each NDJSON message it carries replaced by `onMessage(message)`, one
// message per line. A line that is not JSON goes through unchanged.
export function mapNdjson(
  body: ReadableStream<Uint8Array>,
  onMessage: (message: LensOracleStreamMessage) => LensOracleStreamMessage,
): ReadableStream<Uint8Array> {
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let pending = '';
  const take = (text: string, controller: TransformStreamDefaultController<Uint8Array>) => {
    pending += text;
    const lines = pending.split('\n');
    pending = lines.pop() ?? '';
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) {
        continue;
      }
      let message: LensOracleStreamMessage;
      try {
        message = JSON.parse(trimmed);
      } catch {
        controller.enqueue(encoder.encode(`${line}\n`));
        continue;
      }
      controller.enqueue(encoder.encode(`${JSON.stringify(onMessage(message))}\n`));
    }
  };
  return body.pipeThrough(
    new TransformStream<Uint8Array, Uint8Array>({
      transform(chunk, controller) {
        take(decoder.decode(chunk, { stream: true }), controller);
      },
      flush(controller) {
        take(`${decoder.decode()}\n`, controller);
      },
    }),
  );
}
