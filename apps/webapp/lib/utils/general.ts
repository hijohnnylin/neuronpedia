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

export function formatToGlobalModels(models: ModelWithPartialRelations[]) {
  const modelsFormatted: {
    [key: string]: Model;
  } = {};
  models.forEach((m) => {
    modelsFormatted[m.id] = m;
  });
  return modelsFormatted;
}
