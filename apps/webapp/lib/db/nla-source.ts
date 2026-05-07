import { prisma } from '../db';
import { NLA_SERVER_SECRET, USE_LOCALHOST_NLA } from '../env';

const LOCALHOST_NLA_URL = 'http://localhost:5009';

/**
 * All configured NLA inference servers for the given source. Returns the
 * row's `servers[]` (in DB order) when a `(modelId, nlaSourceId)` pair
 * resolves; otherwise falls back to the single-server `NLA_SERVER` env.
 *
 * Caller is expected to shuffle / pick / failover. For one-shot use cases
 * that don't care about failover, use `getNlaServerUrl` (returns one
 * random pick); for actual upstream calls use `nlaFetch` which handles
 * shuffle + failover automatically.
 */
export async function getNlaServerUrls(modelId?: string, nlaSourceId?: string): Promise<string[]> {
  if (USE_LOCALHOST_NLA) {
    return [LOCALHOST_NLA_URL];
  }

  if (modelId && nlaSourceId) {
    const source = await prisma.nlaSource.findUnique({
      where: { modelId_id: { modelId, id: nlaSourceId } },
    });
    if (source && source.servers.length > 0) {
      return [...source.servers];
    }
  }

  console.warn(`[getNlaServerUrls] No NLA servers configured for modelId: ${modelId}, nlaSourceId: ${nlaSourceId}`);
  throw new Error(`No NLA servers configured for modelId: ${modelId}, nlaSourceId: ${nlaSourceId}`);
}

/**
 * Pick one server at random for the given source. Kept for callers that
 * issue a single fetch and don't need failover (e.g. the autointerp
 * scorer). New call sites should prefer `nlaFetch`.
 */
export async function getNlaServerUrl(modelId?: string, nlaSourceId?: string): Promise<string> {
  const servers = await getNlaServerUrls(modelId, nlaSourceId);
  if (servers.length === 0) {
    throw new Error('No NLA servers configured');
  }
  return servers[Math.floor(Math.random() * servers.length)];
}

export function getNlaHeaders(): HeadersInit {
  const headers: HeadersInit = { 'Content-Type': 'application/json' };
  if (NLA_SERVER_SECRET) {
    headers['X-SECRET-KEY'] = NLA_SERVER_SECRET;
  }
  return headers;
}

/**
 * Fisher–Yates shuffle (in place). Used by `nlaFetch` so each call hits
 * the configured servers in a fresh random order — distributes load and
 * randomises which server is tried first when the primary is busy.
 */
function shuffleInPlace<T>(arr: T[]): T[] {
  for (let i = arr.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

/**
 * Fetch from an NLA inference server with shuffle + failover across the
 * source's `servers[]`.
 *
 *   1. Resolves the candidate servers via `getNlaServerUrls`.
 *   2. Shuffles them so each call randomises which is tried first.
 *   3. Tries each in turn:
 *        - 2xx response   → return immediately.
 *        - non-2xx         → discard body, remember as "lastResponse",
 *                            try the next server.
 *        - network throw   → remember as "lastError", try the next server.
 *   4. If every server errored, return the last non-2xx response so the
 *      caller can forward its status/body to the client (preserves
 *      semantics like 429 / 503 from upstream). If all attempts threw
 *      (no server even responded), throws the last error.
 *
 * `init.body` MUST be reusable across attempts (e.g. a string or
 * `Uint8Array` — NOT a `ReadableStream`, which is consumed on first use).
 * NLA auth headers (`getNlaHeaders`) are merged in automatically; pass
 * extras via `init.headers`.
 */
export async function nlaFetch(
  modelId: string | undefined,
  nlaSourceId: string | undefined,
  path: string,
  init: Omit<RequestInit, 'headers'> & { headers?: HeadersInit } = {},
): Promise<Response> {
  const servers = shuffleInPlace(await getNlaServerUrls(modelId, nlaSourceId));
  if (servers.length === 0) {
    throw new Error('No NLA servers configured');
  }

  const mergedHeaders = new Headers(getNlaHeaders());
  if (init.headers) {
    new Headers(init.headers).forEach((value, key) => {
      mergedHeaders.set(key, value);
    });
  }
  const fetchInit: RequestInit = { ...init, headers: mergedHeaders };

  let lastResponse: Response | null = null;
  let lastError: unknown = null;

  for (let i = 0; i < servers.length; i += 1) {
    const baseUrl = servers[i];
    const url = `${baseUrl}${path}`;
    try {
      // eslint-disable-next-line no-await-in-loop
      const res = await fetch(url, fetchInit);
      if (res.ok) {
        // Drain the previous attempt's body so the upstream connection
        // can be released. Best-effort; ignore errors.
        if (lastResponse?.body) {
          // eslint-disable-next-line no-await-in-loop
          await lastResponse.body.cancel().catch(() => undefined);
        }
        return res;
      }
      // Replacing the previous failed response — release its body too.
      if (lastResponse?.body) {
        // eslint-disable-next-line no-await-in-loop
        await lastResponse.body.cancel().catch(() => undefined);
      }
      lastResponse = res;
      console.warn(`[nlaFetch] ${url} → ${res.status}; ${servers.length - i - 1} server(s) remaining`);
    } catch (err) {
      lastError = err;
      console.warn(`[nlaFetch] ${url} threw:`, err);
    }
  }

  if (lastResponse) {
    return lastResponse;
  }
  throw lastError instanceof Error ? lastError : new Error('All NLA servers unreachable');
}
