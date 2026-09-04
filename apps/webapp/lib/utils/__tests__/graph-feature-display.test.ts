import { describe, expect, test } from 'vitest';

import type { CLTGraph, CLTGraphNode } from '@/app/[modelId]/graph/graph-types';
import { getGraphFeatureDisplayLocation } from '../graph-feature-display';

describe('getGraphFeatureDisplayLocation', () => {
  test('prefers fetched Neuronpedia feature details for display labels', () => {
    const node = {
      feature: 0,
      featureDetailNP: {
        layer: '12-gemmascope-2-transcoder-16k',
        index: '3456',
      },
    } as CLTGraphNode;

    const graph = {
      metadata: {
        scan: 'gemma-3-1b-it',
      },
    } as CLTGraph;

    expect(getGraphFeatureDisplayLocation('gemma-3-1b-it', node, graph)).toEqual({
      layer: 12,
      index: 3456,
    });
  });

  test('falls back to legacy feature decoding when Neuronpedia details are unavailable', () => {
    const node = {
      feature: 1267,
    } as CLTGraphNode;

    const graph = {
      metadata: {
        scan: 'gemma-3-27b-it',
      },
    } as CLTGraph;

    expect(getGraphFeatureDisplayLocation('gemma-3-27b-it', node, graph)).toEqual({
      layer: 7,
      index: 42,
    });
  });
});