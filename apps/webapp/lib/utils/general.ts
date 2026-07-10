import { ExplanationWithPartialRelations, Model, ModelWithPartialRelations } from 'prisma/generated/zod';

export const UNNAMED_AUTHOR_NAME = 'Unnamed';

export enum SearchExplanationsType {
  BY_ALL = 'byAll',
  BY_RELEASE = 'byRelease',
  BY_SOURCE = 'bySource',
  BY_MODEL = 'byModel',
}

export type SearchExplanationsResponse = {
  request: {
    modelId: string;
    layers: string[];
    query: string;
    offset: number;
  };
  results: ExplanationWithPartialRelations[];
  resultsCount: number;
  hasMore: boolean;
  nextOffset: number;
};

// Matches a model-size token like "-1b", "-12b", "-1.5b", "-350m" that sits at
// the end of the id or is immediately followed by another "-segment" (e.g.
// "-2b-it"). Used to sort models by size rather than lexicographically (which
// would otherwise place "gemma-3-12b" before "gemma-3-1b").
const MODEL_SIZE_REGEX = /-(\d+(?:\.\d+)?)([bm])(?=$|-)/i;

// Returns a lexicographically-sortable key for a model id: the size token is
// replaced by its zero-padded parameter count (normalized to millions, B →
// *1000). This keeps the original sort intact while ordering same-family models
// by numeric size ("gemma-3-1b" before "gemma-3-12b") and keeping suffixed
// variants like "-it" immediately after the matching base size
// ("gemma-2-2b" then "gemma-2-2b-it" then "gemma-2-9b").
export function getModelIdSortKey(modelId: string): string {
  return modelId.replace(MODEL_SIZE_REGEX, (_match, num: string, unit: string) => {
    const value = parseFloat(num) * (unit.toLowerCase() === 'b' ? 1000 : 1);
    return `-${Math.round(value).toString().padStart(12, '0')}`;
  });
}

// Two-tier comparator: the original lexicographic sort, but with model size
// compared numerically rather than as text.
export function compareModelIdsBySize(a: string, b: string): number {
  return getModelIdSortKey(a).localeCompare(getModelIdSortKey(b));
}

export function formatToGlobalModels(models: ModelWithPartialRelations[]) {
  const modelsFormatted: {
    [key: string]: Model;
  } = {};
  models.forEach((m) => {
    modelsFormatted[m.id] = m;
  });
  return modelsFormatted;
}
