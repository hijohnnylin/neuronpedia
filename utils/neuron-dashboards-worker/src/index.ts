// Serves neuron dashboard files from the R2 bucket to browsers, at
// https://dashboards.neuronpedia.org/v2/<model>/... (R2 key: neuron-dashboards/v2/<model>/...).

const KEY_PREFIX = 'neuron-dashboards/';

// Only files the webapp reads. raw/, docs.parquet and percentile tables stay private.
const PUBLIC_PATH =
  /^v\d+\/(corpus\/(docs\.txt|docs_offsets\.npy)|[a-z0-9][a-z0-9.-]*\/(meta\.json|corpus\/(vocab\.json|tokens\.npy|offsets\.npy)|neurons\/layer_\d{2,3}\.(idx|json\.gz)))$/;

// Large files are served only in small ranges: a shard (~800 MB) one neuron (~32 KB) at a time, and the
// corpus files one document (a few KB) at a time.
const MAX_RANGE_BYTES: [RegExp, number][] = [
  [/\.json\.gz$/, 4 * 1024 * 1024],
  [/\/corpus\/(docs\.txt|docs_offsets\.npy|tokens\.npy|offsets\.npy)$/, 64 * 1024],
];

export function maxRangeBytes(path: string): number | null {
  const rule = MAX_RANGE_BYTES.find(([pattern]) => pattern.test(path));
  return rule ? rule[1] : null;
}

const CACHE_CONTROL = 'public, max-age=86400';

const CORS_HEADERS: Record<string, string> = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
  'Access-Control-Allow-Headers': 'Range',
  'Access-Control-Expose-Headers': 'Content-Range, Content-Length, Accept-Ranges, ETag',
  'Access-Control-Max-Age': '86400',
};

type ByteRange = { offset: number; end: number };

function contentType(path: string) {
  return path.endsWith('.json') ? 'application/json; charset=utf-8' : 'application/octet-stream';
}

function textResponse(status: number, message: string) {
  return new Response(`${message}\n`, {
    status,
    headers: { ...CORS_HEADERS, 'Content-Type': 'text/plain; charset=utf-8', 'Cache-Control': 'no-store' },
  });
}

// Accepts one "bytes=a-b" range. Returns null when there is no header, 'invalid' for any other form.
export function parseRange(header: string | null): ByteRange | null | 'invalid' {
  if (header === null) {
    return null;
  }
  const match = /^bytes=(\d+)-(\d+)$/.exec(header.trim());
  if (!match) {
    return 'invalid';
  }
  const offset = Number(match[1]);
  const end = Number(match[2]);
  if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(end) || end < offset) {
    return 'invalid';
  }
  return { offset, end };
}

export function isPublicPath(path: string) {
  return PUBLIC_PATH.test(path);
}

async function fetchFromBucket(env: Env, path: string, range: ByteRange | null) {
  const key = KEY_PREFIX + path;
  if (range === null) {
    const object = await env.BUCKET.get(key);
    if (!object) {
      return null;
    }
    return { body: object.body, size: object.size, etag: object.httpEtag, offset: 0, length: object.size };
  }
  const head = await env.BUCKET.head(key);
  if (!head) {
    return null;
  }
  if (range.offset >= head.size) {
    return { body: null, size: head.size, etag: head.httpEtag, offset: range.offset, length: 0 };
  }
  const length = Math.min(range.end, head.size - 1) - range.offset + 1;
  const object = await env.BUCKET.get(key, { range: { offset: range.offset, length } });
  if (!object) {
    return null;
  }
  return { body: object.body, size: object.size, etag: object.httpEtag, offset: range.offset, length };
}

export default {
  async fetch(request, env, ctx): Promise<Response> {
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }
    if (request.method !== 'GET' && request.method !== 'HEAD') {
      return textResponse(405, 'method not allowed');
    }

    const url = new URL(request.url);
    const path = url.pathname.slice(1);
    if (!isPublicPath(path)) {
      return textResponse(404, 'not found');
    }

    const maxRange = maxRangeBytes(path);
    let range = parseRange(request.headers.get('Range'));
    if (range === 'invalid') {
      if (maxRange !== null) {
        return textResponse(416, 'use a single "bytes=a-b" range');
      }
      range = null;
    }
    if (maxRange !== null && (range === null || range.end - range.offset + 1 > maxRange)) {
      return textResponse(416, `this file is served only in ranges of at most ${maxRange} bytes`);
    }

    // The Cache API refuses 206 responses, so a range is cached as a 200 under its own key.
    const cache = caches.default;
    const cacheKey = new Request(`${url.origin}/${path}${range ? `?bytes=${range.offset}-${range.end}` : ''}`, {
      method: 'GET',
    });
    let cached = await cache.match(cacheKey);
    if (!cached) {
      const result = await fetchFromBucket(env, path, range);
      if (!result) {
        return textResponse(404, 'not found');
      }
      if (result.length === 0) {
        return new Response(null, {
          status: 416,
          headers: { ...CORS_HEADERS, 'Content-Range': `bytes */${result.size}` },
        });
      }
      cached = new Response(result.body, {
        status: 200,
        headers: {
          'Content-Type': contentType(path),
          'Content-Length': String(result.length),
          'Cache-Control': CACHE_CONTROL,
          ETag: result.etag,
          'X-Object-Size': String(result.size),
          'X-Range-Offset': String(result.offset),
        },
      });
      ctx.waitUntil(cache.put(cacheKey, cached.clone()));
    }

    const headers = new Headers(cached.headers);
    for (const [name, value] of Object.entries(CORS_HEADERS)) {
      headers.set(name, value);
    }
    headers.set('Accept-Ranges', 'bytes');
    let status = 200;
    if (range) {
      const offset = Number(headers.get('X-Range-Offset'));
      const length = Number(headers.get('Content-Length'));
      headers.set('Content-Range', `bytes ${offset}-${offset + length - 1}/${headers.get('X-Object-Size')}`);
      status = 206;
    }
    headers.delete('X-Object-Size');
    headers.delete('X-Range-Offset');
    return new Response(request.method === 'HEAD' ? null : cached.body, { status, headers });
  },
} satisfies ExportedHandler<Env>;
