import { oraclePrefixDigest, signOracleRead } from '@/lib/utils/oracle-signature';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { gunzipSync } from 'zlib';

const mocks = vi.hoisted(() => ({
  shareCreate: vi.fn(),
  s3Send: vi.fn(),
  lensPromptStream: vi.fn(),
}));

vi.mock('@/app/[modelId]/graph/utils', () => ({ NP_GRAPH_BUCKET: 'test-bucket' }));
vi.mock('@/lib/db', () => ({
  prisma: {
    jlensShare: { create: mocks.shareCreate },
    jlensSharePutRequest: { count: vi.fn(async () => 0), create: vi.fn() },
  },
}));
vi.mock('@/lib/utils/inference', () => ({ lensPromptStream: mocks.lensPromptStream }));
vi.mock('@/lib/with-user', () => ({
  withOptionalUser: (handler: (r: Request) => Promise<Response>) => handler,
}));
vi.mock('next/headers', () => ({ headers: async () => new Headers({ 'x-forwarded-for': '10.0.0.1' }) }));
vi.mock('@aws-sdk/client-s3', () => ({
  S3Client: class {
    send = mocks.s3Send;
  },
  PutObjectCommand: class {
    input: unknown;

    constructor(input: unknown) {
      this.input = input;
    }
  },
}));

// eslint-disable-next-line import/first
import { POST } from './route';

const MODEL = 'qwen3.6-27b';
const TOKEN_IDS = [101, 102, 103];

function lensRun(): Response {
  const meta = { kind: 'meta', model: 'Qwen/Qwen3.6-27B', types: [], layers_by_type: {}, oracle_layers: [20, 24] };
  const tokens = TOKEN_IDS.map((id, position) => ({ kind: 'token', position, token: `t${id}`, id, results: [] }));
  return new Response([meta, ...tokens].map((m) => `${JSON.stringify(m)}\n`).join(''));
}

function signedRead(position: number, layer: number, text: string) {
  const read = { position, layer, adapter: 'a:b', bullets: [text], text: `- ${text}\n`, finish: 'bullets' };
  const sig = signOracleRead({
    modelId: MODEL,
    maxBullets: 2,
    prefixDigest: oraclePrefixDigest(TOKEN_IDS.slice(0, position + 1)),
    ...read,
  })!;
  return { ...read, sig };
}

function shareBody(extra: Record<string, unknown>) {
  return {
    modelId: MODEL,
    kind: 'chat',
    inputTokenIds: TOKEN_IDS,
    messages: [{ role: 'user', content: 'hi' }],
    topN: 8,
    temperature: 0,
    numCompletionTokens: 0,
    hideNonWordTokens: true,
    ...extra,
  };
}

async function post(body: unknown): Promise<Response> {
  const request = new Request('http://localhost/api/lens/share', { method: 'POST', body: JSON.stringify(body) });
  return (POST as unknown as (r: Request) => Promise<Response>)(request);
}

async function postForm(share: string): Promise<Response> {
  const request = new Request('http://localhost/api/lens/share', {
    method: 'POST',
    body: new URLSearchParams({ share }),
  });
  return (POST as unknown as (r: Request) => Promise<Response>)(request);
}

function storedBlob() {
  const { input } = mocks.s3Send.mock.calls[0][0] as { input: { Body: Buffer } };
  return JSON.parse(gunzipSync(input.Body).toString('utf8'));
}

describe('POST /api/lens/share', () => {
  const saved = process.env.NEXTAUTH_SECRET;
  beforeEach(() => {
    process.env.NEXTAUTH_SECRET = 'test-secret';
    mocks.shareCreate.mockReset();
    mocks.s3Send.mockReset();
    mocks.lensPromptStream.mockReset().mockImplementation(async () => lensRun());
  });
  afterEach(() => {
    process.env.NEXTAUTH_SECRET = saved;
  });

  it('stores lensColumns and the legacy tab, and the signed oracle reads in S3', async () => {
    const reads = [signedRead(2, 24, 'Two'), signedRead(0, 20, 'Zero')];
    const res = await post(
      shareBody({ lensColumns: ['LOGIT_LENS', 'ORACLE_LENS', 'JACOBIAN_LENS'], oracle: { maxBullets: 2, reads } }),
    );
    expect(res.status).toBe(200);
    const { data } = mocks.shareCreate.mock.calls[0][0];
    expect(data.lensColumns).toEqual(['JACOBIAN_LENS', 'ORACLE_LENS', 'LOGIT_LENS']);
    expect(data.activeLensModeTab).toBe('DIFF');
    expect(storedBlob().oracle).toEqual({ maxBullets: 2, reads: [reads[1], reads[0]] });
  });

  it('maps an older activeLensModeTab to lensColumns', async () => {
    const res = await post(shareBody({ activeLensModeTab: 'DIFF' }));
    expect(res.status).toBe(200);
    const { data } = mocks.shareCreate.mock.calls[0][0];
    expect(data.lensColumns).toEqual(['JACOBIAN_LENS', 'LOGIT_LENS']);
    expect(data.activeLensModeTab).toBe('DIFF');
    expect(storedBlob().oracle).toBeUndefined();
  });

  it('needs lensColumns or activeLensModeTab', async () => {
    const res = await post(shareBody({}));
    expect(res.status).toBe(400);
    expect(mocks.shareCreate).not.toHaveBeenCalled();
  });

  it('drops an oracle read with a changed text and saves the share with the other reads', async () => {
    const kept = signedRead(0, 20, 'Zero');
    const tampered = { ...signedRead(1, 24, 'One'), text: '- Something the oracle did not say\n' };
    const res = await post(
      shareBody({ lensColumns: ['JACOBIAN_LENS'], oracle: { maxBullets: 2, reads: [tampered, kept] } }),
    );
    expect(res.status).toBe(200);
    expect(storedBlob().oracle).toEqual({ maxBullets: 2, reads: [kept] });
  });

  it('saves the share with no oracle block when no read is valid', async () => {
    const tampered = { ...signedRead(1, 24, 'One'), text: '- Something the oracle did not say\n' };
    const res = await post(shareBody({ lensColumns: ['JACOBIAN_LENS'], oracle: { maxBullets: 2, reads: [tampered] } }));
    expect(res.status).toBe(200);
    expect(storedBlob().oracle).toBeUndefined();
  });

  it('stores the layer range in S3, and stores none when it is not sent', async () => {
    const res = await post(shareBody({ lensColumns: ['JACOBIAN_LENS'], layerRange: [0, 21] }));
    expect(res.status).toBe(200);
    expect(storedBlob().layerRange).toEqual([0, 21]);

    mocks.s3Send.mockClear();
    const res2 = await post(shareBody({ lensColumns: ['JACOBIAN_LENS'] }));
    expect(res2.status).toBe(200);
    expect(storedBlob()).not.toHaveProperty('layerRange');
  });

  it('rejects a layer range that is not [first, last]', async () => {
    for (const layerRange of [[21, 0], [0], [0, 1, 2], [-1, 5]]) {
      const res = await post(shareBody({ lensColumns: ['JACOBIAN_LENS'], layerRange }));
      expect(res.status).toBe(400);
    }
    expect(mocks.shareCreate).not.toHaveBeenCalled();
  });

  it('rejects a column that does not exist', async () => {
    const res = await post(shareBody({ lensColumns: ['DIFF'] }));
    expect(res.status).toBe(400);
  });

  it('answers a form post with a redirect to the share page, and makes the share anonymous', async () => {
    const res = await postForm(JSON.stringify(shareBody({ activeLensModeTab: 'JACOBIAN_LENS' })));
    expect(res.status).toBe(303);
    const { data } = mocks.shareCreate.mock.calls[0][0];
    expect(res.headers.get('Location')).toBe(`/jlens/${data.id}`);
    expect(data.userId).toBeNull();
  });

  it('answers a failed form post with an HTML error page', async () => {
    const res = await postForm(JSON.stringify(shareBody({})));
    expect(res.status).toBe(400);
    expect(res.headers.get('Content-Type')).toMatch(/^text\/html/);
    expect(await res.text()).toContain('Provide &#39;lensColumns&#39;');
    expect(mocks.shareCreate).not.toHaveBeenCalled();
  });

  it('refuses a form with no JSON in its share field', async () => {
    const res = await postForm('not json');
    expect(res.status).toBe(400);
    expect(await res.text()).toContain('no valid JSON');
    expect(mocks.lensPromptStream).not.toHaveBeenCalled();
  });
});
