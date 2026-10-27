import { describe, expect, it } from 'vitest';
import { isSteerableNeuronLayer } from './source';

const DASHBOARD_NEURONS = { name: 'neurons', externalDashboardsUrl: 'https://example.com/neuron-dashboards/v2' };

describe('isSteerableNeuronLayer', () => {
  it('is true for a neuron layer with file dashboards and inference', () => {
    expect(isSteerableNeuronLayer(DASHBOARD_NEURONS, { id: '40', inferenceEnabled: true })).toBe(true);
  });

  it('is false without inference on the layer', () => {
    expect(isSteerableNeuronLayer(DASHBOARD_NEURONS, { id: '40', inferenceEnabled: false })).toBe(false);
  });

  it('is false for a neuron set that keeps its dashboards in the database', () => {
    expect(
      isSteerableNeuronLayer({ name: 'neurons', externalDashboardsUrl: null }, { id: '7', inferenceEnabled: true }),
    ).toBe(false);
  });

  it('is false for an SAE source', () => {
    expect(
      isSteerableNeuronLayer(
        { name: 'res-jb', externalDashboardsUrl: 'https://example.com' },
        { id: '7-res-jb', inferenceEnabled: true },
      ),
    ).toBe(false);
  });

  it('is false when the source set or source is not known', () => {
    expect(isSteerableNeuronLayer(undefined, { id: '40', inferenceEnabled: true })).toBe(false);
    expect(isSteerableNeuronLayer(DASHBOARD_NEURONS, undefined)).toBe(false);
  });
});
