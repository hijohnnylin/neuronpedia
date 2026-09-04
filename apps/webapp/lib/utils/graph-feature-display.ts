import type { CLTGraph, CLTGraphNode } from '@/app/[modelId]/graph/graph-types';
import { getIndexFromFeatureAndGraph, getLayerFromFeatureAndGraph } from '@/app/[modelId]/graph/utils';

type GraphFeatureDisplayLocation = {
  layer: number;
  index: number;
};

function parseLeadingInteger(value?: string): number | null {
  if (!value) {
    return null;
  }

  const match = value.match(/^\d+/);
  if (!match) {
    return null;
  }

  return Number.parseInt(match[0], 10);
}

export function getGraphFeatureDisplayLocation(
  modelId: string,
  node: CLTGraphNode,
  selectedGraph: CLTGraph | null,
): GraphFeatureDisplayLocation {
  const layer = parseLeadingInteger(node.featureDetailNP?.layer);
  const index = parseLeadingInteger(node.featureDetailNP?.index);

  if (layer !== null && index !== null) {
    return { layer, index };
  }

  return {
    layer: getLayerFromFeatureAndGraph(modelId, node, selectedGraph),
    index: getIndexFromFeatureAndGraph(modelId, node, selectedGraph),
  };
}