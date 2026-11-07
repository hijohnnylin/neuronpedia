import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '../api-error';
import { uploadExplanations, uploadExplanationsSchema } from './explanation-upload';

// vi.mock is hoisted above the imports, so its factory can only close over
// values created by vi.hoisted.
const mocks = vi.hoisted(() => ({
  userFindUnique: vi.fn(),
  typeFindMany: vi.fn(),
  modelTypeFindMany: vi.fn(),
  neuronFindMany: vi.fn(),
  explanationFindMany: vi.fn(),
  createManyAndReturn: vi.fn(),
  canAccess: vi.fn(),
  ensureNeuronRow: vi.fn(),
}));

vi.mock('@/lib/db', () => ({
  prisma: {
    user: { findUnique: mocks.userFindUnique },
    explanationType: { findMany: mocks.typeFindMany },
    explanationModelType: { findMany: mocks.modelTypeFindMany },
    neuron: { findMany: mocks.neuronFindMany },
    explanation: { findMany: mocks.explanationFindMany, createManyAndReturn: mocks.createManyAndReturn },
  },
}));
vi.mock('@/lib/db/userCanAccess', () => ({ userCanAccessModelAndSourceSet: mocks.canAccess }));
vi.mock('@/lib/db/neuron-dashboard', () => ({ ensureNeuronRowForFeature: mocks.ensureNeuronRow }));

const user = { id: 'uploader', name: 'uploader' };
const item = {
  modelId: 'qwen3-8b',
  layer: '9',
  index: '10358',
  description: 'wine',
  typeName: 'my-site',
};

async function errorOf(promise: Promise<unknown>) {
  try {
    await promise;
  } catch (e) {
    return e as ApiError;
  }
  throw new Error('expected a rejection');
}

beforeEach(() => {
  vi.clearAllMocks();
  mocks.userFindUnique.mockResolvedValue({ admin: false, canUploadExplanations: true });
  mocks.typeFindMany.mockResolvedValue([{ name: 'my-site', creatorId: 'uploader' }]);
  mocks.modelTypeFindMany.mockResolvedValue([]);
  mocks.canAccess.mockResolvedValue(true);
  mocks.neuronFindMany.mockResolvedValue([{ modelId: 'qwen3-8b', layer: '9', index: '10358' }]);
  mocks.explanationFindMany.mockResolvedValue([]);
  mocks.createManyAndReturn.mockImplementation(async ({ data }) =>
    data.map((d: typeof item, i: number) => ({ id: `e${i}`, ...d })),
  );
});

describe('uploadExplanationsSchema', () => {
  it('turns a number index into a string', async () => {
    const body = await uploadExplanationsSchema.validate({ explanations: [{ ...item, index: 10358 }] });
    expect(body.explanations[0].index).toBe('10358');
  });

  it('refuses links that are not http or https', async () => {
    await expect(
      uploadExplanationsSchema.validate({ explanations: [{ ...item, sourceUrl: 'javascript:alert(1)' }] }),
    ).rejects.toThrow(/http or https/);
  });

  it('refuses a strength on an explanation that is not about steering', async () => {
    await expect(uploadExplanationsSchema.validate({ explanations: [{ ...item, steerStrength: 1 }] })).rejects.toThrow(
      /STEERING/,
    );
  });

  it('refuses metadata that is not an object', async () => {
    await expect(uploadExplanationsSchema.validate({ explanations: [{ ...item, metadata: [1, 2] }] })).rejects.toThrow(
      /JSON object/,
    );
  });

  it('refuses an empty batch', async () => {
    await expect(uploadExplanationsSchema.validate({ explanations: [] })).rejects.toThrow();
  });
});

describe('uploadExplanations', () => {
  it('needs the upload flag', async () => {
    mocks.userFindUnique.mockResolvedValue({ admin: false, canUploadExplanations: false });
    expect((await errorOf(uploadExplanations([item], user))).status).toBe(403);
    expect(mocks.createManyAndReturn).not.toHaveBeenCalled();
  });

  it('refuses a type that the uploader does not own', async () => {
    mocks.typeFindMany.mockResolvedValue([{ name: 'my-site', creatorId: 'someone-else' }]);
    expect((await errorOf(uploadExplanations([item], user))).status).toBe(403);
  });

  it('lets an admin upload to any type', async () => {
    mocks.userFindUnique.mockResolvedValue({ admin: true, canUploadExplanations: false });
    mocks.typeFindMany.mockResolvedValue([{ name: 'my-site', creatorId: 'someone-else' }]);
    await expect(uploadExplanations([item], user)).resolves.toMatchObject({ skipped: 0 });
  });

  it('refuses an unknown type', async () => {
    mocks.typeFindMany.mockResolvedValue([]);
    expect((await errorOf(uploadExplanations([item], user))).status).toBe(400);
  });

  it('refuses a source the uploader cannot see', async () => {
    mocks.canAccess.mockResolvedValue(false);
    expect((await errorOf(uploadExplanations([item], user))).status).toBe(404);
  });

  it('refuses the whole batch when one feature is missing', async () => {
    const err = await errorOf(uploadExplanations([item, { ...item, index: '99' }], user));
    expect(err.status).toBe(404);
    expect(err.message).toContain('qwen3-8b@9:99');
    expect(mocks.createManyAndReturn).not.toHaveBeenCalled();
  });

  it('writes the fields as the uploader and returns a link', async () => {
    const result = await uploadExplanations(
      [{ ...item, pole: 'NEGATIVE', sourceUrl: 'https://example.com/n', metadata: { title: 'Wine Bottle' } }],
      user,
    );
    const row = mocks.createManyAndReturn.mock.calls[0][0].data[0];
    expect(row).toMatchObject({
      authorId: 'uploader',
      pole: 'NEGATIVE',
      aspect: null,
      sourceUrl: 'https://example.com/n',
      metadata: { title: 'Wine Bottle' },
    });
    expect(result.created[0].url).toMatch(/\/qwen3-8b\/9\/10358$/);
  });

  it('skips explanations that are already there, and repeats in the batch', async () => {
    mocks.explanationFindMany.mockResolvedValue([{ ...item, pole: null, aspect: null, steerStrength: null }]);
    const result = await uploadExplanations(
      [item, { ...item, description: 'red wine' }, { ...item, description: 'red wine' }],
      user,
    );
    expect(mocks.createManyAndReturn.mock.calls[0][0].data).toHaveLength(1);
    expect(result).toMatchObject({ skipped: 2 });
  });

  it('keeps the same text at another pole as a separate explanation', async () => {
    mocks.explanationFindMany.mockResolvedValue([{ ...item, pole: null, aspect: null, steerStrength: null }]);
    const result = await uploadExplanations([{ ...item, pole: 'NEGATIVE' }], user);
    expect(result.created).toHaveLength(1);
  });
});
