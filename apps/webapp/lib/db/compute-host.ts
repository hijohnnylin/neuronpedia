import { prisma } from '@/lib/db';
import { ComputeService, Prisma } from '@prisma/client';
import {
  AUTOINTERP_SERVER_SECRET,
  GRAPH_SERVER_SECRET,
  INFERENCE_SERVER_SECRET,
  NLA_SERVER_SECRET,
  SPARSITY_SERVER_SECRET,
} from '../env';
import { getSourceSetNameFromSource } from '../utils/source';
import { AuthenticatedUser } from '../with-user';
import { userCanAccessModelAndSourceSet } from './userCanAccess';

/**
 * Ceiling on one attempt, per service, so a host that accepts the connection
 * and then hangs is abandoned while there is still budget to try another.
 * Without this the request occupies the serverless function until the platform
 * kills it, and the remaining hosts are never reached.
 *
 * Sized to the work, not to taste: building an attribution graph is minutes of
 * GPU time, an NLA pass is seconds. A caller with a tighter or looser bound
 * should pass `timeoutMs` rather than moving these.
 */
const ATTEMPT_TIMEOUT_MS: Record<ComputeService, number> = {
  [ComputeService.GRAPH]: 300_000,
  [ComputeService.AUTOINTERP]: 120_000,
  [ComputeService.NLA]: 90_000,
  [ComputeService.INFERENCE]: 60_000,
  [ComputeService.SPARSITY]: 60_000,
};

// Resolution is a join across three or four tables and the answer rarely
// changes, so hold it briefly per serverless instance.
const CACHE_TTL_MS = 30_000;

export class NoComputeHostError extends Error {
  constructor(target: ResolveTarget) {
    super(
      `No ${target.service} host available for model "${target.modelId}"` +
        (target.sourceId ? ` source "${target.sourceId}"` : '') +
        (target.sourceSetName ? ` source set "${target.sourceSetName}"` : '') +
        (target.nlaSourceId ? ` NLA source "${target.nlaSourceId}"` : ''),
    );
    this.name = 'NoComputeHostError';
  }
}

/**
 * What the caller needs a host for.
 *
 * For INFERENCE, `sourceId` and `sourceSetName` narrow to hosts that have the
 * matching SAE loaded. Passing neither asks for any host serving the model,
 * which is what jlens, steering and the assistant axis want.
 */
export type ResolveTarget = {
  service: ComputeService;
  modelId: string;
  sourceId?: string;
  sourceSetName?: string;
  nlaSourceId?: string;
  user?: AuthenticatedUser | null;
};

type CachedHost = { hostUrl: string };

const cache = new Map<string, { hosts: CachedHost[]; expiresAt: number }>();

const cacheKey = (target: ResolveTarget) =>
  [target.service, target.modelId, target.sourceId ?? '', target.sourceSetName ?? '', target.nlaSourceId ?? ''].join(
    '\u0000',
  );

const SERVICE_SECRETS: Record<ComputeService, string> = {
  INFERENCE: INFERENCE_SERVER_SECRET,
  GRAPH: GRAPH_SERVER_SECRET,
  NLA: NLA_SERVER_SECRET,
  AUTOINTERP: AUTOINTERP_SERVER_SECRET,
  SPARSITY: SPARSITY_SERVER_SECRET,
};

export const computeHeaders = (service: ComputeService, extra?: HeadersInit): Headers => {
  const headers = new Headers({ 'Content-Type': 'application/json' });
  const secret = SERVICE_SECRETS[service];
  if (secret) {
    headers.set('X-SECRET-KEY', secret);
  }
  if (extra) {
    new Headers(extra).forEach((value, key) => headers.set(key, value));
  }
  return headers;
};

function shuffleInPlace<T>(arr: T[]): T[] {
  for (let i = arr.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

/**
 * Candidate URLs in random order, so load spreads instead of piling onto
 * whichever row the database returned first.
 *
 * There is no health ranking. `computeFetch` fails over within a request, and
 * a host that is down refuses connections cheaply, so remembering the failure
 * between requests would only save that cheap retry on a fraction of traffic.
 */
function orderHosts(hosts: CachedHost[]): string[] {
  return [...new Set(shuffleInPlace([...hosts]).map((host) => host.hostUrl))];
}

function buildWhere(target: ResolveTarget): Prisma.ComputeHostWhereInput {
  const base: Prisma.ComputeHostWhereInput = {
    service: target.service,
    modelId: target.modelId,
  };

  if (target.nlaSourceId) {
    return { ...base, nlaSourceId: target.nlaSourceId };
  }
  if (target.sourceSetName) {
    const setLink = {
      sourceSets: { some: { sourceSetName: target.sourceSetName, sourceSetModelId: target.modelId } },
    };
    if (target.service !== ComputeService.INFERENCE) {
      return { ...base, ...setLink };
    }
    // An inference host is linked to the individual sources it loaded, not to
    // the sets they belong to, so a set matches when the host serves any source
    // in it. Registration does also write a direct set link, so accept either:
    // the rows carried over from the pre-registry tables have only the source
    // side, and matching on the set link alone would miss them.
    return {
      ...base,
      OR: [
        setLink,
        { sources: { some: { sourceModelId: target.modelId, source: { setName: target.sourceSetName } } } },
      ],
    };
  }
  if (target.sourceId) {
    return { ...base, sources: { some: { sourceId: target.sourceId, sourceModelId: target.modelId } } };
  }
  return base;
}

async function queryHosts(target: ResolveTarget): Promise<CachedHost[]> {
  const select = { hostUrl: true };
  const hosts = await prisma.computeHost.findMany({ where: buildWhere(target), select });

  // An inference host with no source links serves anything the model can do, so
  // a source-specific lookup that found nothing falls back to those. Graph and
  // NLA have no such notion: an unlinked host there is a misconfiguration, not
  // a wildcard.
  if (hosts.length === 0 && target.service === ComputeService.INFERENCE && (target.sourceId || target.sourceSetName)) {
    return prisma.computeHost.findMany({
      where: {
        service: ComputeService.INFERENCE,
        modelId: target.modelId,
        sources: { none: {} },
      },
      select,
    });
  }

  return hosts;
}

async function assertCanAccess(target: ResolveTarget): Promise<boolean> {
  const sourceSetName = target.sourceSetName ?? (target.sourceId ? getSourceSetNameFromSource(target.sourceId) : null);
  if (!sourceSetName) {
    return true;
  }
  return userCanAccessModelAndSourceSet(target.modelId, sourceSetName, target.user ?? null, true);
}

/**
 * Base URLs for the given target, best candidate first. Empty when the caller
 * cannot see the source set, or when nothing is registered.
 */
export async function resolveHosts(target: ResolveTarget): Promise<string[]> {
  if (!(await assertCanAccess(target))) {
    return [];
  }

  const key = cacheKey(target);
  const cached = cache.get(key);
  if (cached && cached.expiresAt > Date.now()) {
    return orderHosts(cached.hosts);
  }

  const hosts = await queryHosts(target);
  cache.set(key, { hosts, expiresAt: Date.now() + CACHE_TTL_MS });
  return orderHosts(hosts);
}

/** The single best host. Throws when there is none. */
export async function resolveHost(target: ResolveTarget): Promise<string> {
  const hosts = await resolveHosts(target);
  if (hosts.length === 0) {
    throw new NoComputeHostError(target);
  }
  return hosts[0];
}

/**
 * Two distinct hosts, for callers that run a pair of requests concurrently
 * (steering runs a default and a steered completion side by side). Falls back
 * to the same host twice when only one is registered.
 */
export async function resolveTwoHosts(target: ResolveTarget): Promise<[string, string]> {
  const hosts = await resolveHosts(target);
  if (hosts.length === 0) {
    throw new NoComputeHostError(target);
  }
  return hosts.length === 1 ? [hosts[0], hosts[0]] : [hosts[0], hosts[1]];
}

/**
 * Whether a failed request says anything about the host.
 *
 * A 422 for a malformed body means the same on every host, so retrying is both
 * pointless and a good way to mark a whole healthy fleet as sick. Only statuses
 * that describe the server's own condition count.
 */
const isHostFault = (status: number) => status >= 500 || status === 408 || status === 429;

/**
 * Fetch from a compute host, failing over across every candidate.
 *
 * A 2xx returns immediately. A 5xx, 408, 429 or a thrown error moves on to the
 * next host. Any other 4xx is the caller's fault and is returned as-is, since
 * no other host would answer differently.
 *
 * When every host has been tried, the last response is returned so the caller
 * can forward upstream semantics such as 429 or 503; if no host responded at
 * all, the last error is thrown.
 *
 * Each attempt is capped by `timeoutMs`, defaulting to the service's entry in
 * ATTEMPT_TIMEOUT_MS. An attempt that times out counts as a host fault. An
 * abort from the caller's own signal does not: the client has gone, so another
 * host would only burn more GPU time on an answer nobody will read.
 *
 * `init.body` must be replayable across attempts, so a string or a Uint8Array
 * rather than a ReadableStream.
 */
export async function computeFetch(
  target: ResolveTarget,
  path: string,
  init: Omit<RequestInit, 'headers'> & { headers?: HeadersInit; timeoutMs?: number } = {},
): Promise<Response> {
  const hosts = await resolveHosts(target);
  if (hosts.length === 0) {
    throw new NoComputeHostError(target);
  }

  const { timeoutMs = ATTEMPT_TIMEOUT_MS[target.service], signal: callerSignal, ...rest } = init;
  const fetchInit: RequestInit = { ...rest, headers: computeHeaders(target.service, init.headers) };

  let lastResponse: Response | null = null;
  let lastError: unknown = null;

  const releaseLast = async () => {
    if (lastResponse?.body) {
      await lastResponse.body.cancel().catch(() => undefined);
    }
  };

  for (let i = 0; i < hosts.length; i += 1) {
    const hostUrl = hosts[i];
    const url = `${hostUrl}${path}`;
    const attemptTimeout = AbortSignal.timeout(timeoutMs);
    try {
      // eslint-disable-next-line no-await-in-loop
      const res = await fetch(url, {
        ...fetchInit,
        signal: callerSignal ? AbortSignal.any([callerSignal, attemptTimeout]) : attemptTimeout,
      });
      if (res.ok || !isHostFault(res.status)) {
        // eslint-disable-next-line no-await-in-loop
        await releaseLast();
        return res;
      }
      // eslint-disable-next-line no-await-in-loop
      await releaseLast();
      lastResponse = res;
      console.warn(`[computeFetch] ${url} -> ${res.status}; ${hosts.length - i - 1} host(s) remaining`);
    } catch (err) {
      if (callerSignal?.aborted) {
        throw err;
      }
      lastError = err;
      const why = attemptTimeout.aborted ? `timed out after ${timeoutMs}ms` : `threw: ${err}`;
      console.warn(`[computeFetch] ${url} ${why}; ${hosts.length - i - 1} host(s) remaining`);
    }
  }

  if (lastResponse) {
    return lastResponse;
  }
  throw lastError instanceof Error ? lastError : new Error(`All ${target.service} hosts unreachable`);
}
