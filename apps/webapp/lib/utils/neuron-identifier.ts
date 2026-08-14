import { ExplanationWithPartialRelations } from '@/prisma/generated/zod';

export class NeuronIdentifier {
  modelId: string;

  layer: string;

  index: string;

  constructor(modelId: string = '', layer: string = '', index: string = '') {
    this.modelId = modelId;
    this.layer = layer;
    this.index = index;
  }

  equals(other: NeuronIdentifier) {
    return this.modelId === other.modelId && this.layer === other.layer && this.index === other.index;
  }

  toString() {
    return `${this.modelId}@${this.layer}:${this.index}`;
  }
}

export function getExplanationNeuronIdentifier(exp: ExplanationWithPartialRelations) {
  return new NeuronIdentifier(exp.modelId, exp.layer, exp.index);
}

// Path to a feature dashboard. Each segment is encoded: some of these ids come from
// user-supplied JSON (the manual dashboard) or from an API response, and an unencoded
// segment could otherwise carry its own path, query string or `javascript:` scheme.
export function getFeaturePath(
  modelId: string | undefined,
  layer: string | undefined,
  index: string | number | undefined,
) {
  return `/${encodeURIComponent(modelId ?? '')}/${encodeURIComponent(layer ?? '')}/${encodeURIComponent(index ?? '')}`;
}
