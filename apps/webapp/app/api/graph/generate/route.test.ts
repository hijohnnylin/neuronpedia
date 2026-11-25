import { describe, expect, it, vi } from 'vitest';

vi.mock('@/app/[modelId]/graph/utils', () => ({
  ATTRIBUTION_GRAPH_SCHEMA: {},
  makeGraphPublicAccessGraphUrl: vi.fn(),
  NP_GRAPH_BUCKET: 'test-bucket',
}));
vi.mock('@/lib/db', () => ({ prisma: {} }));
vi.mock('@/lib/db/model', () => ({ getModelById: vi.fn() }));
vi.mock('@/lib/utils/graph', () => ({
  generateGraphAndUploadToS3: vi.fn(),
  getGraphTokenize: vi.fn(),
  GRAPH_ANONYMOUS_USER_ID: 'anonymous',
  GRAPH_MAX_TOKENS: 64,
  GRAPH_S3_USER_GRAPHS_DIR: 'user-graphs',
  GRAPH_SLUG_MIN: 3,
  graphGenerateSchemaClient: { validate: vi.fn() },
  LORSA_MAX_TOKENS: 64,
  LORSA_MODELS: [],
}));
vi.mock('@/lib/with-user', () => ({
  withOptionalUser: (handler: (r: Request) => Promise<Response>) => handler,
}));
vi.mock('@aws-sdk/client-s3', () => ({ S3Client: class {}, PutObjectCommand: class {} }));
vi.mock('@aws-sdk/s3-request-presigner', () => ({ getSignedUrl: vi.fn() }));

// eslint-disable-next-line import/first
import { POST } from './route';

describe('POST /api/graph/generate', () => {
  it('returns 400, not 500, when the request body is not valid JSON', async () => {
    const request = new Request('http://localhost/api/graph/generate', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: '{"prompt": ',
    });

    const response = await (POST as unknown as (r: Request) => Promise<Response>)(request);

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toMatchObject({ error: 'Invalid JSON body' });
  });
});
