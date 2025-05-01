/* eslint-disable no-restricted-syntax */

import { prisma } from '@/lib/db';
import { getGlobalModels } from '@/lib/db/model';
import { Explanation, Prisma } from '@prisma/client';
import { ExplanationPartial, ExplanationPartialWithRelations, NeuronWithPartialRelations } from 'prisma/generated/zod';
import { PUBLIC_ACTIVATIONS_USER_IDS } from '../env';
import { getSourceSetNameFromSource } from '../utils/source';
import { AuthenticatedUser } from '../with-user';
import {
  assertUserCanAccessModelAndSource,
  assertUserCanAccessModelAndSourceSet,
  assertUserCanAccessRelease,
} from './userCanAccess';

export const MAX_EXPLANATION_SEARCH_RESULTS = 20;

export type ExplanationSearchResult = {
  modelId: string;
  layer: string;
  index: string;
  description: string;
  explanationModelName: string;
  typeName: string;
  maxActApprox: number;
  frac_nonzero: number;
  neuron: NeuronWithPartialRelations | undefined;
};

export const searchExplanationsAllVec = async (
  queryEmbedding: string,
  offset: number,
  user: AuthenticatedUser | null = null,
) => {
  // get all sourceSets that user can access
  const accessibleModels = await getGlobalModels(user);
  const sourceIds: string[] = [];
  for (const model of accessibleModels) {
    for (const sourceSet of model.sourceSets) {
      for (const source of sourceSet.sources) {
        sourceIds.push(source.id);
      }
    }
  }

  let results = (await prisma.$queryRaw`
    SELECT "modelId", "layer", "index", description, "explanationModelName", "typeName",
        1 - (embedding <=> ${queryEmbedding}::vector) AS cosine_similarity
    FROM "Explanation"
    ORDER BY (embedding <=> ${queryEmbedding}::vector)
    LIMIT ${MAX_EXPLANATION_SEARCH_RESULTS}
    OFFSET ${offset || 0};
  `) as ExplanationPartialWithRelations[];

  // if the layer is not in sourceIds, then remove it - user's not supposed to see it
  results = results.filter((result) => {
    if (result.layer && sourceIds.includes(result.layer)) {
      return true;
    }
    return false;
  });
  return results;
};

export const searchExplanationsVec = async (
  modelId: string,
  layers: string[],
  queryEmbedding: string,
  offset: number,
  user: AuthenticatedUser | null = null,
) => {
  // check if all layers are the same sourceSet, if so the check is much faster
  let isAllSameSourceSet = true;
  for (const layer of layers) {
    const sourceSet = getSourceSetNameFromSource(layer);
    if (sourceSet !== getSourceSetNameFromSource(layers[0])) {
      isAllSameSourceSet = false;
    }
  }
  if (isAllSameSourceSet) {
    await assertUserCanAccessModelAndSourceSet(modelId, getSourceSetNameFromSource(layers[0]), user);
  } else {
    for (const layer of layers) {
      // eslint-disable-next-line no-await-in-loop
      await assertUserCanAccessModelAndSource(modelId, layer, user);
    }
  }
  const results = await prisma.$queryRaw`
    SELECT "modelId", "layer", "index", description, "explanationModelName", "typeName",  
        1 - (embedding <=> ${queryEmbedding}::vector) AS cosine_similarity
    FROM "Explanation"
    WHERE "modelId" = ${modelId}
      AND layer IN (${Prisma.join(layers)})
    ORDER BY (embedding <=> ${queryEmbedding}::vector)
    LIMIT ${MAX_EXPLANATION_SEARCH_RESULTS}
    OFFSET ${offset || 0};
  `;
  return results as ExplanationPartialWithRelations[];
};

export const searchExplanationsReleaseVec = async (
  releaseName: string,
  queryEmbedding: string,
  offset: number,
  user: AuthenticatedUser | null = null,
) => {
  await assertUserCanAccessRelease(releaseName, user);

  const results = await prisma.$queryRaw`
    SELECT "modelId", "layer", "index", description, "explanationModelName", "typeName",
        1 - (embedding <=> ${queryEmbedding}::vector) AS cosine_similarity
    FROM "Explanation" e
    WHERE (e."modelId", e."layer") IN (
      SELECT
        "Source"."modelId",
        "Source"."id"
      FROM
        "SourceSet"
        JOIN "Source" ON "setName" = "SourceSet".name
      WHERE
        "releaseName" = ${releaseName}
        and "Source"."hasDashboards" = true
        and "SourceSet"."hasDashboards" = true
    )
    ORDER BY (embedding <=> ${queryEmbedding}::vector)
    LIMIT ${MAX_EXPLANATION_SEARCH_RESULTS}
    OFFSET ${offset || 0};
  `;

  return results as ExplanationPartialWithRelations[];
};

export const searchExplanationsModelVec = async (
  modelName: string,
  queryEmbedding: string,
  offset: number,
  user: AuthenticatedUser | null = null,
) => {
  // get all sourceSets that user can access
  const accessibleModels = await getGlobalModels(user);
  const sourceIds: string[] = [];
  for (const model of accessibleModels) {
    for (const sourceSet of model.sourceSets) {
      for (const source of sourceSet.sources) {
        sourceIds.push(source.id);
      }
    }
  }

  let results = (await prisma.$queryRaw`
    SELECT "modelId", "layer", "index", description, "explanationModelName", "typeName",
        1 - (embedding <=> ${queryEmbedding}::vector) AS cosine_similarity
    FROM "Explanation"
    WHERE "modelId" = ${modelName}
    ORDER BY (embedding <=> ${queryEmbedding}::vector)
    LIMIT ${MAX_EXPLANATION_SEARCH_RESULTS}
    OFFSET ${offset || 0};
  `) as ExplanationPartialWithRelations[];

  // if the layer is not in sourceIds, then remove it - user's not supposed to see it
  results = results.filter((result) => {
    if (result.layer && sourceIds.includes(result.layer)) {
      return true;
    }
    return false;
  });

  return results as ExplanationPartialWithRelations[];
};

export const exportExplanations = async (
  modelId: string,
  sourceId: string,
  maxResults: number | null = null,
  user: AuthenticatedUser | null = null,
) => {
  await assertUserCanAccessModelAndSource(modelId, sourceId, user);

  return prisma.explanation.findMany({
    where: {
      modelId,
      layer: sourceId,
    },
    select: {
      id: true,
      modelId: true,
      layer: true,
      index: true,
      description: true,
      explanationModelName: true,
      typeName: true,
    },
    take: maxResults || undefined,
  });
};

export const deleteExplanationById = async (id: string, user: AuthenticatedUser) => {
  const explanation = await prisma.explanation.findUnique({
    where: {
      id,
    },
  });
  if (explanation) {
    await assertUserCanAccessModelAndSource(explanation.modelId, explanation.layer, user);
    if (explanation.authorId !== user.id) {
      throw new Error('You can only delete explanations you created.');
    } else {
      return prisma.explanation.delete({
        where: {
          id,
        },
      });
    }
  } else {
    return null;
  }
};

export const deleteExplanationScoreById = async (id: string, user: AuthenticatedUser) => {
  const explanationScore = await prisma.explanationScore.findUnique({
    where: {
      id,
    },
  });
  if (explanationScore) {
    if (explanationScore.initiatedByUserId !== user.id) {
      throw new Error('You can only delete explanations you created.');
    } else {
      return prisma.explanationScore.delete({
        where: {
          id,
        },
      });
    }
  } else {
    return null;
  }
};

export const getExplanationById = async (id: string, user: AuthenticatedUser | null = null) => {
  const explanation = await prisma.explanation.findUnique({
    where: {
      id,
    },
  });
  if (explanation) {
    await assertUserCanAccessModelAndSource(explanation.modelId, explanation.layer, user);
    return explanation;
  }
  return null;
};

export const getExplanationByIdWithDetails = async (id: string, user: AuthenticatedUser | null = null) => {
  const explanation = await prisma.explanation.findUniqueOrThrow({
    where: {
      id,
    },
    include: {
      author: {
        select: {
          name: true,
        },
      },
      neuron: {
        include: {
          activations: {
            where: {
              creatorId: {
                in: PUBLIC_ACTIVATIONS_USER_IDS,
              },
            },
          },
        },
      },
      scores: true,
      activationsV1: {
        include: {
          activation: true,
        },
        orderBy: {
          score: 'desc',
        },
      },
      model: true,
      votes: true,
    },
  });
  if (explanation) {
    await assertUserCanAccessModelAndSource(explanation.modelId, explanation.layer, user);
    return explanation;
  }
  return null;
};

export const getUmapExplanations = async (modelId: string, layers: string[], user: AuthenticatedUser | null = null) => {
  for (const layer of layers) {
    // eslint-disable-next-line no-await-in-loop
    await assertUserCanAccessModelAndSource(modelId, layer, user);
  }

  const results: { [layer: string]: ExplanationPartial[] } = {};
  const promises = [];
  for (const layer of layers) {
    promises.push(
      prisma.explanation.findMany({
        where: {
          authorId: {
            in: PUBLIC_ACTIVATIONS_USER_IDS,
          },
          modelId,
          layer,
          umap_x: {
            not: { equals: 0 },
          },
          umap_y: {
            not: { equals: 0 },
          },
        },
        select: {
          id: true,
          layer: true,
          index: true,
          description: true,
          umap_cluster: true,
          umap_x: true,
          umap_y: true,
          umap_log_feature_sparsity: true,
        },
      }),
    );
  }
  const resultsArray = await Promise.all(promises);
  // eslint-disable-next-line
  for (let i = 0; i < layers.length; i++) {
    results[layers[i]] = resultsArray[i];
  }
  return results;
};

export const createNewExplanationWithoutScore = async (
  modelId: string,
  layer: string,
  index: string,
  explanationText: string,
  userId: string,
  user: AuthenticatedUser | null = null,
): Promise<Explanation> => {
  await assertUserCanAccessModelAndSource(modelId, layer, user);
  const newExplanation = await prisma.explanation.create({
    data: {
      modelId,
      layer,
      index,
      description: explanationText,
      authorId: userId,
    },
    include: {
      author: {
        select: {
          name: true,
        },
      },
    },
  });
  console.log('score - new explanation created');
  return newExplanation;
};
