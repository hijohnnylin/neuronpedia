import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '../api-error';
import fixture from '../utils/__fixtures__/neuron-dashboard-qwen3-32b-31-12345.json';
import { NeuronDashboardError, NeuronDashboardRecord, recordToNeuronDashboard } from '../utils/neuron-dashboards';
import {
  ensureNeuronRowForDashboard,
  ensureNeuronRowForFeature,
  getActivationsForFeature,
  markNearZeroAsNonActivating,
  NEAR_ZERO_FRACTION,
} from './neuron-dashboard';

// vi.mock is hoisted above the imports, so its factory can only close over
// values created by vi.hoisted.
const { sourceSetFindUnique, activationFindMany, neuronUpsert, neuronFindUnique, fetchNeuronDashboard } = vi.hoisted(
  () => ({
    sourceSetFindUnique: vi.fn(),
    activationFindMany: vi.fn(),
    neuronUpsert: vi.fn(),
    neuronFindUnique: vi.fn(),
    fetchNeuronDashboard: vi.fn(),
  }),
);

vi.mock('@/lib/db', () => ({
  prisma: {
    sourceSet: { findUnique: sourceSetFindUnique },
    activation: { findMany: activationFindMany },
    neuron: { upsert: neuronUpsert, findUnique: neuronFindUnique },
  },
}));
vi.mock('@/lib/utils/neuron-dashboards', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  fetchNeuronDashboard,
}));

const ids = { modelId: 'qwen3-32b', layer: '31', index: '12345' };
const vocab: string[] = [];
Object.entries(fixture.vocab).forEach(([id, token]) => {
  vocab[Number(id)] = token;
});
const dashboard = recordToNeuronDashboard(fixture.record as NeuronDashboardRecord, vocab, ids);

beforeEach(() => {
  vi.clearAllMocks();
});

describe('getActivationsForFeature', () => {
  it('reads from the dashboard files when the source set has them', async () => {
    sourceSetFindUnique.mockResolvedValue({ externalDashboardsUrl: 'https://dashboards.example/v1/qwen3-32b' });
    fetchNeuronDashboard.mockResolvedValue(dashboard);

    const { activations, dashboard: loaded } = await getActivationsForFeature(ids);

    expect(sourceSetFindUnique.mock.calls[0][0].where).toEqual({
      modelId_name: { modelId: 'qwen3-32b', name: 'neurons' },
    });
    expect(fetchNeuronDashboard).toHaveBeenCalledWith('https://dashboards.example/v1/qwen3-32b', ids);
    expect(activationFindMany).not.toHaveBeenCalled();
    expect(loaded).toBe(dashboard);
    expect(activations).toHaveLength(dashboard.activations.length);
    // the explainers read tokens and values; the route ranks by maxValue
    const top = activations.toSorted((a, b) => b.maxValue - a.maxValue)[0];
    expect(top.maxValue).toBe(fixture.record.stats.max);
    expect(top.tokens.length).toBe(top.values.length);
    expect(top).toMatchObject({ modelId: 'qwen3-32b', layer: '31', index: '12345' });
  });

  it('reads Activation rows when the source set has no dashboard files', async () => {
    sourceSetFindUnique.mockResolvedValue({ externalDashboardsUrl: null });
    activationFindMany.mockResolvedValue([{ id: 'a' }]);

    const { activations, dashboard: loaded } = await getActivationsForFeature({ ...ids, layer: '6-res-jb' });

    expect(sourceSetFindUnique.mock.calls[0][0].where.modelId_name.name).toBe('res-jb');
    expect(fetchNeuronDashboard).not.toHaveBeenCalled();
    expect(loaded).toBeNull();
    expect(activations).toEqual([{ id: 'a' }]);
  });
});

describe('getActivationsForFeature errors', () => {
  it('turns a missing dashboard file into a 404', async () => {
    sourceSetFindUnique.mockResolvedValue({ externalDashboardsUrl: 'https://dashboards.example/v1/qwen3-32b' });
    fetchNeuronDashboard.mockRejectedValue(new NeuronDashboardError('no dashboard'));

    const error = await getActivationsForFeature(ids).catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(404);
  });
});

describe('markNearZeroAsNonActivating', () => {
  it('zeroes only rows whose every value is near zero', async () => {
    sourceSetFindUnique.mockResolvedValue({ externalDashboardsUrl: 'https://dashboards.example/v1/qwen3-32b' });
    fetchNeuronDashboard.mockResolvedValue(dashboard);
    const { activations } = await getActivationsForFeature(ids);
    const { max, min } = fixture.record.stats;
    const limit = NEAR_ZERO_FRACTION * Math.max(Math.abs(max), Math.abs(min));

    const marked = markNearZeroAsNonActivating(activations, dashboard);

    expect(marked).toHaveLength(activations.length);
    marked.forEach((row, i) => {
      const quiet = activations[i].values.every((v) => Math.abs(v) < limit);
      if (quiet) {
        expect(row.maxValue).toBe(0);
        expect(row.values.every((v) => v === 0)).toBe(true);
      } else {
        expect(row).toBe(activations[i]);
      }
    });
    // the top examples stay as they are
    expect(marked.filter((row) => row.maxValue === fixture.record.stats.max).length).toBeGreaterThan(0);
  });
});

describe('ensureNeuronRowForFeature', () => {
  it('does nothing when the Neuron row exists', async () => {
    neuronFindUnique.mockResolvedValue({ index: '12345' });
    await ensureNeuronRowForFeature(ids);
    expect(sourceSetFindUnique).not.toHaveBeenCalled();
    expect(neuronUpsert).not.toHaveBeenCalled();
  });

  it('creates the row for a file dashboard, using only the id fields as the key', async () => {
    neuronFindUnique.mockResolvedValue(null);
    sourceSetFindUnique.mockResolvedValue({ externalDashboardsUrl: 'https://dashboards.example/v1/qwen3-32b' });
    fetchNeuronDashboard.mockResolvedValue(dashboard);
    neuronUpsert.mockResolvedValue({ ...ids });

    await ensureNeuronRowForFeature({ ...ids, description: 'list note' } as typeof ids);

    expect(neuronFindUnique.mock.calls[0][0].where).toEqual({ modelId_layer_index: ids });
    expect(neuronUpsert.mock.calls[0][0].where).toEqual({ modelId_layer_index: ids });
  });

  it('creates nothing for a database source set', async () => {
    neuronFindUnique.mockResolvedValue(null);
    sourceSetFindUnique.mockResolvedValue({ externalDashboardsUrl: null });
    await ensureNeuronRowForFeature({ ...ids, layer: '6-res-jb' });
    expect(neuronUpsert).not.toHaveBeenCalled();
  });
});

describe('ensureNeuronRowForDashboard', () => {
  it('creates a bare row only if missing, and returns it with the dashboard stats and logits', async () => {
    neuronUpsert.mockResolvedValue({ ...ids, sourceSetName: 'neurons', maxActApprox: 0, pos_str: [] });

    const feature = await ensureNeuronRowForDashboard(ids, dashboard);

    const args = neuronUpsert.mock.calls[0][0];
    expect(args.where).toEqual({ modelId_layer_index: ids });
    expect(args.update).toEqual({});
    expect(args.create).toEqual({ ...ids, sourceSetName: 'neurons', maxActApprox: fixture.record.stats.max });
    expect(feature.pos_str).toEqual(dashboard.neuron.pos_str);
    expect(feature.maxActApprox).toBe(fixture.record.stats.max);
  });
});
