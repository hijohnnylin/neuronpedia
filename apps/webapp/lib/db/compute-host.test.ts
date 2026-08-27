import { ComputeService } from '@prisma/client';
import { beforeEach, describe, expect, it, vi } from 'vitest';

// vi.mock is hoisted above the imports, so its factory can only close over
// values created by vi.hoisted.
const { findMany, userCanAccessModelAndSourceSet } = vi.hoisted(() => ({
  findMany: vi.fn(),
  userCanAccessModelAndSourceSet: vi.fn(),
}));

vi.mock('@/lib/db', () => ({ prisma: { computeHost: { findMany } } }));
vi.mock('./userCanAccess', () => ({ userCanAccessModelAndSourceSet }));
// Pinned so the secret assertion does not depend on the runner's environment.
vi.mock('../env', () => ({
  INFERENCE_SERVER_SECRET: 'test-secret',
  GRAPH_SERVER_SECRET: '',
  NLA_SERVER_SECRET: '',
  AUTOINTERP_SERVER_SECRET: '',
  SPARSITY_SERVER_SECRET: '',
}));

// eslint-disable-next-line import/first
import { computeFetch, resolveHosts, resolveTwoHosts } from './compute-host';

const host = (hostUrl: string) => ({ hostUrl });

// Each test uses its own model id: the resolver caches by target, so sharing one
// would leak results between tests.
let modelCounter = 0;
const freshModel = () => {
  modelCounter += 1;
  return `model-${modelCounter}`;
};

beforeEach(() => {
  findMany.mockReset();
  userCanAccessModelAndSourceSet.mockReset().mockResolvedValue(true);
});

describe('resolveHosts', () => {
  it('offers every registered host, deduplicated', async () => {
    findMany.mockResolvedValue([host('https://a'), host('https://b'), host('https://a')]);

    const hosts = await resolveHosts({ service: ComputeService.INFERENCE, modelId: freshModel() });
    expect(hosts.slice().sort()).toEqual(['https://a', 'https://b']);
  });

  it('matches an NLA host on its single source column', async () => {
    findMany.mockResolvedValue([]);
    const modelId = freshModel();
    await resolveHosts({ service: ComputeService.NLA, modelId, nlaSourceId: 'nla-a' });

    // An NLA process is bound to one source at startup, so this is a column
    // match, not a join.
    expect(findMany.mock.calls[0][0].where).toMatchObject({
      service: ComputeService.NLA,
      modelId,
      nlaSourceId: 'nla-a',
    });
  });

  it('matches a graph host on its source set', async () => {
    findMany.mockResolvedValue([]);
    const modelId = freshModel();
    await resolveHosts({ service: ComputeService.GRAPH, modelId, sourceSetName: 'set' });

    expect(findMany.mock.calls[0][0].where).toMatchObject({
      service: ComputeService.GRAPH,
      modelId,
      sourceSets: { some: { sourceSetName: 'set', sourceSetModelId: modelId } },
    });
  });

  it('matches an inference host on a source set through the sources it serves', async () => {
    findMany.mockResolvedValue([]);
    const modelId = freshModel();
    await resolveHosts({ service: ComputeService.INFERENCE, modelId, sourceSetName: 'gemmascope-res-16k' });

    // Inference hosts are linked per source, so asking only for a set link finds
    // nothing -- which is how every set-name lookup broke against backfilled rows.
    expect(findMany.mock.calls[0][0].where).toMatchObject({
      service: ComputeService.INFERENCE,
      modelId,
      OR: [
        { sourceSets: { some: { sourceSetName: 'gemmascope-res-16k', sourceSetModelId: modelId } } },
        { sources: { some: { sourceModelId: modelId, source: { setName: 'gemmascope-res-16k' } } } },
      ],
    });
  });

  it('falls back to model-wide inference hosts when a source has none of its own', async () => {
    findMany.mockResolvedValueOnce([]).mockResolvedValueOnce([host('https://wildcard')]);

    const hosts = await resolveHosts({
      service: ComputeService.INFERENCE,
      modelId: freshModel(),
      sourceId: '6-res-jb',
    });

    expect(hosts).toEqual(['https://wildcard']);
    // The fallback asks only for hosts with no source links, which is what marks
    // a host as serving anything the model can do.
    expect(findMany.mock.calls[1][0].where.sources).toEqual({ none: {} });
  });

  it('does not fall back for graph, where an unlinked host is a misconfiguration', async () => {
    findMany.mockResolvedValue([]);

    const hosts = await resolveHosts({
      service: ComputeService.GRAPH,
      modelId: freshModel(),
      sourceSetName: 'set',
    });

    expect(hosts).toEqual([]);
    expect(findMany).toHaveBeenCalledTimes(1);
  });

  it('returns nothing when the user cannot see the source set', async () => {
    userCanAccessModelAndSourceSet.mockResolvedValue(false);

    const hosts = await resolveHosts({
      service: ComputeService.INFERENCE,
      modelId: freshModel(),
      sourceSetName: 'private-set',
    });

    expect(hosts).toEqual([]);
    expect(findMany).not.toHaveBeenCalled();
  });

  it('serves a repeat lookup from cache', async () => {
    findMany.mockResolvedValue([host('https://a')]);
    const modelId = freshModel();

    await resolveHosts({ service: ComputeService.INFERENCE, modelId });
    await resolveHosts({ service: ComputeService.INFERENCE, modelId });

    expect(findMany).toHaveBeenCalledTimes(1);
  });
});

describe('resolveTwoHosts', () => {
  it('returns two distinct hosts when they exist', async () => {
    findMany.mockResolvedValue([host('https://a'), host('https://b')]);

    const [first, second] = await resolveTwoHosts({ service: ComputeService.INFERENCE, modelId: freshModel() });
    expect(first).not.toEqual(second);
  });

  it('repeats the only host rather than failing', async () => {
    findMany.mockResolvedValue([host('https://a')]);

    expect(await resolveTwoHosts({ service: ComputeService.INFERENCE, modelId: freshModel() })).toEqual([
      'https://a',
      'https://a',
    ]);
  });

  it('throws when nothing is registered', async () => {
    findMany.mockResolvedValue([]);

    await expect(resolveTwoHosts({ service: ComputeService.INFERENCE, modelId: freshModel() })).rejects.toThrow(
      /No INFERENCE host available/,
    );
  });
});

describe('computeFetch', () => {
  const ok = () => new Response('{}', { status: 200 });
  const boom = () => new Response('nope', { status: 503 });

  it('moves to the next host after a failure', async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(boom()).mockResolvedValueOnce(ok());
    vi.stubGlobal('fetch', fetchMock);
    findMany.mockResolvedValue([host('https://a'), host('https://b')]);

    const res = await computeFetch({ service: ComputeService.INFERENCE, modelId: freshModel() }, '/v1/x', {
      method: 'POST',
      body: '{}',
    });

    expect(res.status).toBe(200);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('returns the last upstream response when every host fails', async () => {
    findMany.mockResolvedValue([host('https://a'), host('https://b')]);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(boom()));

    const res = await computeFetch({ service: ComputeService.INFERENCE, modelId: freshModel() }, '/v1/x');

    // 503 rather than a generic 500, so the caller can tell "busy" from "broken".
    expect(res.status).toBe(503);
  });

  it('throws when no host even responded', async () => {
    findMany.mockResolvedValue([host('https://a')]);
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('ECONNREFUSED')));

    await expect(computeFetch({ service: ComputeService.INFERENCE, modelId: freshModel() }, '/v1/x')).rejects.toThrow(
      'ECONNREFUSED',
    );
  });

  it('returns a client error from the first host without trying the rest', async () => {
    findMany.mockResolvedValue([host('https://a'), host('https://b')]);
    const fetchMock = vi.fn().mockResolvedValue(new Response('bad body', { status: 422 }));
    vi.stubGlobal('fetch', fetchMock);

    const res = await computeFetch({ service: ComputeService.INFERENCE, modelId: freshModel() }, '/v1/x');

    expect(res.status).toBe(422);
    // A malformed request fails the same way everywhere, so retrying it would
    // only mark a healthy fleet as sick.
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('abandons a host that hangs and fails over to the next', async () => {
    findMany.mockResolvedValue([host('https://a'), host('https://b')]);
    const fetchMock = vi
      .fn()
      .mockImplementationOnce(
        (_url: string, init: RequestInit) =>
          new Promise((_resolve, reject) => {
            init.signal?.addEventListener('abort', () => reject((init.signal as AbortSignal).reason));
          }),
      )
      .mockResolvedValueOnce(ok());
    vi.stubGlobal('fetch', fetchMock);

    const res = await computeFetch({ service: ComputeService.INFERENCE, modelId: freshModel() }, '/v1/x', {
      timeoutMs: 20,
    });

    expect(res.status).toBe(200);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('does not try another host when the caller itself aborts', async () => {
    findMany.mockResolvedValue([host('https://a'), host('https://b')]);
    const controller = new AbortController();
    controller.abort();
    const fetchMock = vi.fn().mockRejectedValue(new DOMException('aborted', 'AbortError'));
    vi.stubGlobal('fetch', fetchMock);

    // The client has gone, so a second host would only burn GPU time on an
    // answer nobody will read.
    await expect(
      computeFetch({ service: ComputeService.INFERENCE, modelId: freshModel() }, '/v1/x', {
        signal: controller.signal,
      }),
    ).rejects.toThrow();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('retries a host that failed on an earlier request', async () => {
    findMany.mockResolvedValue([host('https://flaky')]);
    const fetchMock = vi.fn().mockResolvedValueOnce(boom()).mockResolvedValueOnce(ok());
    vi.stubGlobal('fetch', fetchMock);
    const target = { service: ComputeService.INFERENCE, modelId: freshModel() };

    expect((await computeFetch(target, '/v1/x')).status).toBe(503);
    // No memory of the failure between requests, so the host is tried again
    // straight away rather than sitting out a backoff window.
    expect((await computeFetch(target, '/v1/x')).status).toBe(200);
  });

  it('sends the service secret', async () => {
    findMany.mockResolvedValue([host('https://a')]);
    const fetchMock = vi.fn().mockResolvedValue(ok());
    vi.stubGlobal('fetch', fetchMock);

    await computeFetch({ service: ComputeService.INFERENCE, modelId: freshModel() }, '/v1/x');

    const headers = fetchMock.mock.calls[0][1].headers as Headers;
    expect(headers.get('Content-Type')).toBe('application/json');
    expect(headers.get('X-SECRET-KEY')).toBe('test-secret');
  });
});
