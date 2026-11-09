import { getFeaturePath } from '@/lib/utils/neuron-identifier';

// A feature dashboard with `?compare=<modelId>/<layer>/<index>` shows that second feature in a
// column to the right. Without the param, the page is the single dashboard.
export const COMPARE_PARAM = 'compare';

export type FeatureId = { modelId: string; layer: string; index: string };

export type FeatureHrefBuilder = (modelId: string, layer: string | undefined, index: string | number) => string;

// Links for one column of the split view.
export type FeatureCompareLinks = {
  // Where a feature picked in this column goes. The other column stays the same.
  hrefFor: FeatureHrefBuilder;
  // The single dashboard of the other column's feature.
  closeHref: string;
};

export function getCompareValue(feature: FeatureId) {
  return [feature.modelId, feature.layer, feature.index].map((part) => encodeURIComponent(part)).join('/');
}

// Returns null for a missing or malformed value, so a bad link falls back to the single dashboard.
export function parseCompareValue(value: string | string[] | undefined): FeatureId | null {
  if (typeof value !== 'string') {
    return null;
  }
  const parts = value.split('/');
  if (parts.length !== 3 || parts.some((part) => part.trim().length === 0)) {
    return null;
  }
  const [modelId, layer, index] = parts;
  return { modelId, layer, index };
}

export function getComparePath(left: FeatureId, right: FeatureId) {
  return `${getFeaturePath(left.modelId, left.layer, left.index)}?${COMPARE_PARAM}=${getCompareValue(right)}`;
}

export function getCompareLinks(
  left: FeatureId,
  right: FeatureId,
): { left: FeatureCompareLinks; right: FeatureCompareLinks } {
  return {
    left: {
      hrefFor: (modelId, layer, index) => getComparePath({ modelId, layer: layer || '', index: String(index) }, right),
      closeHref: getFeaturePath(right.modelId, right.layer, right.index),
    },
    right: {
      hrefFor: (modelId, layer, index) => getComparePath(left, { modelId, layer: layer || '', index: String(index) }),
      closeHref: getFeaturePath(left.modelId, left.layer, left.index),
    },
  };
}
