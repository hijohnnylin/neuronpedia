import { NextRequest } from 'next/server';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  neuronFindUnique: vi.fn(),
  sourceFindUnique: vi.fn(),
  getActivationForFeature: vi.fn(),
}));

vi.mock('@/lib/db', () => ({
  prisma: {
    neuron: { findUnique: mocks.neuronFindUnique },
    source: { findUnique: mocks.sourceFindUnique },
  },
}));
vi.mock('@/lib/db/user', () => ({
  getAuthenticatedUserFromApiKey: vi.fn(),
  getUserById: vi.fn(),
  makeAuthedUserFromSessionOrReturnNull: vi.fn(async () => null),
}));
vi.mock('@/lib/db/userCanAccess', () => ({ assertUserCanAccessModelAndSourceSet: vi.fn() }));
vi.mock('@/lib/utils/inference', () => ({ getActivationForFeature: mocks.getActivationForFeature }));

import { POST } from './route';

function post(index: string) {
  return POST(
    new NextRequest('http://localhost/api/activation/new', {
      method: 'POST',
      body: JSON.stringify({ feature: { modelId: 'qwen3-32b', source: '2', index }, customText: 'hello' }),
    }),
  );
}

function fileSource(inferenceEnabled: boolean) {
  return {
    inferenceEnabled,
    set: { externalDashboardsUrl: 'https://dashboards.example/v1/qwen3-32b', model: { neuronsPerLayer: 25600 } },
  };
}

describe('POST /api/activation/new for a neuron with no Neuron row', () => {
  beforeEach(() => {
    mocks.neuronFindUnique.mockReset().mockResolvedValue(null);
    mocks.sourceFindUnique.mockReset();
    mocks.getActivationForFeature.mockReset().mockResolvedValue({ values: [1, 2] });
  });

  it('runs inference when the source has dashboard files and inference is on', async () => {
    mocks.sourceFindUnique.mockResolvedValue(fileSource(true));
    const res = await post('25599');
    expect(res.status).toBe(200);
    expect(mocks.getActivationForFeature).toHaveBeenCalledOnce();
  });

  it('refuses when inference is off for the source', async () => {
    mocks.sourceFindUnique.mockResolvedValue(fileSource(false));
    expect((await post('3')).status).toBe(400);
    expect(mocks.getActivationForFeature).not.toHaveBeenCalled();
  });

  it('returns 404 for an index past the layer, or a source with no dashboard files', async () => {
    mocks.sourceFindUnique.mockResolvedValue(fileSource(true));
    expect((await post('25600')).status).toBe(404);
    expect((await post('3.5')).status).toBe(404);
    mocks.sourceFindUnique.mockResolvedValue({ ...fileSource(true), set: { externalDashboardsUrl: null } });
    expect((await post('3')).status).toBe(404);
    expect(mocks.getActivationForFeature).not.toHaveBeenCalled();
  });
});
