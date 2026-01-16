import { prisma } from '@/lib/db';
import { Neuron } from '@prisma/client';
import { ActivationAllPost200Response, ActivationTopkByTokenPost200Response } from 'neuronpedia-inference-client';
import { NeuronWithPartialRelations } from 'prisma/generated/zod';
import { PUBLIC_ACTIVATIONS_USER_IDS } from '../env';
import { EXPLANATIONTYPE_HUMAN } from '../utils/autointerp';
import { NeuronIdentifier } from '../utils/neuron-identifier';
import { getSourceSetNameFromSource } from '../utils/source';
import { AuthenticatedUser } from '../with-user';
import {
  assertUserCanAccessModel,
  assertUserCanAccessModelAndSource,
  assertUserCanAccessModelAndSourceSet,
  assertUserCanWriteModelAndSource,
  userCanAccessModelAndSourceSet,
} from './userCanAccess';

const NEURONS_TO_LOAD_PER_REQUEST = 25;

// more efficent checking
export const filterToUserCanAccessNeurons = async (
  features: NeuronIdentifier[],
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  user: AuthenticatedUser | null = null,
) =>
  // const alreadyCheckedAndAllowed: { modelId: string; sourceSet: string }[] = [];
  // // filter features to unique modelId/sourceSet
  // const uniqueFeatures = features.filter(
  //   (feature, index, self) =>
  //     index === self.findIndex((t) => t.modelId === feature.modelId && t.layer === feature.layer),
  // );
  // console.log('length of unique features', uniqueFeatures.length);
  // check each of these feature in parallel with await Promise.all userCanAccessModelAndSourceSet
  // const startTime = Date.now();
  // await Promise.all(
  //   uniqueFeatures.map(async (feature) => {
  //     const sourceSetName = isNeuronLayerSource(feature.layer)
  //       ? NEURONS_SOURCESET
  //       : getSourceSetNameFromSource(feature.layer);
  //     const canAccess = await userCanAccessModelAndSourceSet(feature.modelId, sourceSetName, user, true);
  //     if (canAccess) {
  //       alreadyCheckedAndAllowed.push({ modelId: feature.modelId, sourceSet: sourceSetName });
  //     }
  //   }),
  // );
  // const endTime = Date.now();
  // console.log(`Time taken to check access for ${uniqueFeatures.length} unique features: ${endTime - startTime}ms`);
  // now filter features to only the ones that are allowed
  // const allowedFeatures = features.filter((feature) => {
  //   // check if each feature has layer and index that are in a pair in alreadyCheckedAndAllowed
  //   const sourceSetName = isNeuronLayerSource(feature.layer)
  //     ? NEURONS_SOURCESET
  //     : getSourceSetNameFromSource(feature.layer);
  //   return alreadyCheckedAndAllowed.some((n) => n.modelId === feature.modelId && n.sourceSet === sourceSetName);
  // });
  // return allowedFeatures;
  features;

export const upsertVector = async (
  modelId: string,
  sourceId: string,
  vector: number[],
  vectorLabel: string,
  hookName: string,
  vectorDefaultSteerStrength: number,
  user: AuthenticatedUser,
) => {
  await assertUserCanAccessModel(modelId, user);
  // make a random number from 10000000 to 99999999 to use as the index
  const randomNumber = Math.floor(Math.random() * 90000000) + 90000000;
  return prisma.neuron.create({
    data: {
      modelId,
      layer: sourceId,
      index: randomNumber.toString(),
      creatorId: user.id,
      maxActApprox: 0,
      hasVector: true,
      vector,
      vectorLabel,
      hookName,
      vectorDefaultSteerStrength,
    },
  });
};

export const getVectorFromDb = async (modelId: string, source: string, index: string) => {
  const neuron = prisma.neuron.findUnique({
    where: {
      modelId_layer_index: {
        modelId,
        layer: source,
        index,
      },
    },
    select: {
      modelId: true,
      layer: true,
      index: true,
      vectorLabel: true,
      createdAt: true,
      creatorId: true,
      vectorDefaultSteerStrength: true,
      hookName: true,
      vector: true,
    },
  });
  return neuron;
};

export const getVectorsForUser = async (userId: string) => {
  console.log('starting', userId);
  const neurons = await prisma.$queryRaw`
  SELECT 
    "modelId",
    "layer",
    "index",
    "vectorLabel",
    "createdAt",
    "creatorId",
    "vectorDefaultSteerStrength",
    "hookName",
    "vector"
  FROM "Neuron"
  WHERE "creatorId" = ${userId} 
    AND "hasVector" = true;
  `;
  console.log('completed');
  return neurons as Neuron[];
};

export const getVectorsForModelAndUser = async (modelId: string, userId: string) =>
  prisma.neuron.findMany({
    where: { modelId, creatorId: userId, hasVector: true },
  });

export const neuronExistsAndUserHasAccess = async (
  modelId: string,
  layer: string,
  index: string,
  user: AuthenticatedUser | null = null,
) => {
  const canAccess = await userCanAccessModelAndSourceSet(modelId, getSourceSetNameFromSource(layer), user, true);
  if (!canAccess) {
    return null;
  }
  return prisma.neuron.findUnique({
    where: {
      modelId_layer_index: {
        modelId,
        layer,
        index: index.toString(),
      },
    },
    select: {
      modelId: true,
      layer: true,
      index: true,
      vector: true,
      vectorLabel: true,
      hookName: true,
    },
  });
};

export const getNeurons = async (
  features: NeuronIdentifier[],
  user: AuthenticatedUser | null = null,
  actsToReturn?: number,
) => {
  const allowedNeurons = await filterToUserCanAccessNeurons(features, user);
  return prisma.neuron.findMany({
    where: {
      OR: allowedNeurons,
    },
    include: {
      explanations: {
        select: {
          description: true,
          explanationModelName: true,
          typeName: true,
        },
      },
      activations: {
        where: {
          creatorId: {
            in: PUBLIC_ACTIVATIONS_USER_IDS,
          },
        },
        ...(actsToReturn ? { take: actsToReturn, orderBy: { maxValue: 'desc' } } : {}),
      },
    },
  });
};

export const getNumFeaturesForSae = async (modelId: string, saeId: string, user: AuthenticatedUser | null = null) => {
  await assertUserCanAccessModelAndSource(modelId, saeId, user);
  return prisma.neuron.count({
    where: {
      modelId,
      layer: saeId,
    },
  });
};

export const getNeuronOnly = async (modelId: string, layer: string, index: string) =>
  prisma.neuron.findUnique({
    where: { modelId_layer_index: { modelId, layer, index } },
  });

export const getNeuronOptimized = async (
  modelId: string,
  layer: string,
  index: string,
  user: AuthenticatedUser | null = null,
) => {
  await assertUserCanAccessModelAndSource(modelId, layer, user);

  // get the neuron. it will tell us if it has a vector, if it has a vector then include all activations
  const neuronResult = await prisma.neuron.findUnique({
    where: {
      modelId_layer_index: {
        modelId,
        layer,
        index,
      },
    },
    include: {
      model: true,
      lists: {
        include: {
          list: true,
        },
      },
      creator: {
        select: {
          name: true,
        },
      },
      source: true,
      sourceSet: true,
      comments: {
        include: {
          user: {
            select: {
              name: true,
            },
          },
        },
      },
    },
  });

  // separate parallel queries instead of one query because Prisma creates insane requests to DB otherwise
  const activations = prisma.activation.findMany({
    where: {
      modelId,
      layer,
      index,
      creatorId: {
        in:
          neuronResult?.hasVector && neuronResult?.creatorId
            ? [neuronResult.creatorId].concat(PUBLIC_ACTIVATIONS_USER_IDS)
            : PUBLIC_ACTIVATIONS_USER_IDS,
      },
    },
    orderBy: {
      maxValue: 'desc',
    },
  });
  const explanations = prisma.explanation.findMany({
    where: { modelId, layer, index },
    select: {
      id: true,
      description: true,
      explanationModelName: true,
      typeName: true,
      scores: true,
      triggeredByUser: {
        select: {
          id: true,
          name: true,
        },
      },
    },
    orderBy: [{ scoreV2: 'desc' }, { scoreV1: 'desc' }],
  });
  const res = await Promise.all([activations, explanations]); // , comments]);
  if (!res || !neuronResult) {
    return null;
  }
  const neuron = neuronResult as NeuronWithPartialRelations;
  // this does not work correctly, so always set it to blank
  neuron.correlated_features_indices = [];
  neuron.correlated_features_l1 = [];
  neuron.correlated_features_pearson = [];

  const [acts, exps] = res;
  neuron.activations = acts;
  neuron.explanations = exps;

  return neuron;
};

export const createNeurons = async (
  modelId: string,
  layer: string,
  featuresToCreate: any[],
  user: AuthenticatedUser,
) => {
  await assertUserCanWriteModelAndSource(modelId, layer, user);
  await prisma.neuron.createMany({
    data: featuresToCreate,
    skipDuplicates: true,
  });
};

export const deleteNeuron = async (modelId: string, layer: string, index: string, user: AuthenticatedUser) => {
  // await assertUserCanWriteModelAndSource(modelId, layer, user);

  // find the neuron
  const neuron = await getNeuronOptimized(modelId, layer, index);
  if (!neuron) {
    return null;
  }

  // ensure neuron is owned by the user
  if (neuron.creatorId !== user.id) {
    return null;
  }

  // delete the neuron
  return prisma.neuron.delete({
    where: {
      modelId_layer_index: {
        modelId,
        layer,
        index,
      },
    },
  });
};

export const getNeuronsForSearcher = async (
  modelId: string,
  sourceSet: string,
  result: ActivationAllPost200Response,
  user: AuthenticatedUser | null = null,
) => {
  // we already checked at /search-all, but check here again anyway in case someone tries to use this method directly
  await assertUserCanAccessModelAndSourceSet(modelId, sourceSet, user);

  // get the neuron info (w/ explanations)
  return prisma.neuron.findMany({
    where: {
      modelId,
      OR: result.activations.map((activation) => ({
        layer: activation.source,
        index: activation.index.toString(),
      })),
    },
    include: {
      activations: {
        orderBy: {
          maxValue: 'desc',
        },
        take: 1,
        where: {
          creatorId: {
            in: PUBLIC_ACTIVATIONS_USER_IDS,
          },
        },
      },
      explanations: {
        include: {
          author: {
            select: {
              name: true,
            },
          },
          votes: true,
        },
        orderBy: [{ scoreV2: 'desc' }, { scoreV1: 'desc' }],
      },
      model: true,
    },
  });
};

export const getNeuronsForTopkSearcherExplanationOnly = async (
  modelId: string,
  source: string,
  result: ActivationTopkByTokenPost200Response,
  user: AuthenticatedUser | null = null,
) => {
  // we already checked at /search-all, but check here again anyway in case someone tries to use this method directly
  await assertUserCanAccessModelAndSource(modelId, source, user);

  const featsToGet: NeuronIdentifier[] = [];
  result.results.forEach((r) => {
    r.topFeatures.forEach((f) => {
      featsToGet.push(new NeuronIdentifier(modelId, source, f.featureIndex.toString()));
    });
  });

  // get the neuron info (w/ explanations)
  return prisma.neuron.findMany({
    where: {
      modelId,
      OR: featsToGet.map((feature) => ({
        layer: source,
        index: feature.index,
      })),
    },
    include: {
      explanations: {
        where: {
          typeName: { not: EXPLANATIONTYPE_HUMAN },
        },
        orderBy: { createdAt: 'desc' },
        take: 1,
      },
    },
  });
};

export const getNeuronsOffset = async (
  modelId: string,
  layer: string,
  offset: number,
  user: AuthenticatedUser | null = null,
) => {
  await assertUserCanAccessModelAndSource(modelId, layer, user);

  // TODO: make index column an integer
  const naturalSortNeuronsIndices: { index: string }[] = await prisma.$queryRaw`
    select "index" from "Neuron"
    where "modelId" = ${modelId}
    and "layer" = ${layer}
    and "maxActApprox" > 0
    order by 
      (substring("index", '^[0-9]+'))::int, 
      substring("index", '[^0-9_].*$')
    limit ${NEURONS_TO_LOAD_PER_REQUEST}
    offset ${offset || 0}
  `;

  const indices = naturalSortNeuronsIndices.map((neuron) => neuron.index);

  return prisma.neuron.findMany({
    where: {
      modelId,
      layer,
      index: { in: indices },
    },
    take: NEURONS_TO_LOAD_PER_REQUEST,
    include: {
      sourceSet: true,
      activations: {
        orderBy: {
          maxValue: 'desc',
        },
        where: {
          creatorId: {
            in: PUBLIC_ACTIVATIONS_USER_IDS,
          },
        },
      },
      explanations: {
        include: {
          author: {
            select: {
              name: true,
            },
          },
          votes: true,
        },
        orderBy: [{ scoreV2: 'desc' }, { scoreV1: 'desc' }],
      },
      model: true,
    },
  });
};

export const getNeuronsForQuickList = async (features: NeuronIdentifier[], user: AuthenticatedUser | null = null) => {
  const allowedNeurons = await filterToUserCanAccessNeurons(features, user);

  return prisma.neuron.findMany({
    where: {
      OR: allowedNeurons,
    },
    include: {
      activations: {
        where: {
          creatorId: {
            in: PUBLIC_ACTIVATIONS_USER_IDS,
          },
        },
        include: {
          creator: {
            select: {
              name: true,
            },
          },
        },
        orderBy: {
          maxValue: 'desc',
        },
        take: 3,
      },
      explanations: {
        include: {
          author: {
            select: {
              name: true,
            },
          },
          votes: true,
        },
        orderBy: [{ scoreV2: 'desc' }, { scoreV1: 'desc' }],
      },
      model: true,
    },
  });
};

export const dangerousGetNeuronRangeInternalUseOnly = async (
  modelId: string,
  layer: string,
  beginIndex: number,
  endIndex: number,
) => {
  // make a string array of indexes from beginIndex to endIndex
  const indexes = [];
  // eslint-disable-next-line no-plusplus
  for (let i = beginIndex; i <= endIndex; i++) {
    indexes.push(i.toString());
  }

  const neurons = prisma.neuron.findMany({
    where: {
      modelId,
      layer,
      index: { in: indexes },
    },
    select: {
      modelId: true,
      layer: true,
      index: true,
      maxActApprox: true,
      frac_nonzero: true,
      freq_hist_data_bar_heights: true,
      freq_hist_data_bar_values: true,
      logits_hist_data_bar_heights: true,
      logits_hist_data_bar_values: true,
      neg_str: true,
      neg_values: true,
      pos_str: true,
      pos_values: true,
      // correlated_neurons_indices: true,
      // correlated_neurons_l1: true,
      // correlated_features_pearson: true,
      neuron_alignment_indices: true,
      neuron_alignment_l1: true,
      neuron_alignment_values: true,
      topkCosSimIndices: true,
      topkCosSimValues: true,
      activations: {
        where: {
          creatorId: {
            in: PUBLIC_ACTIVATIONS_USER_IDS,
          },
        },
        select: {
          // id: true,
          tokens: true,
          maxValue: true,
          minValue: true,
          values: true,
          maxValueTokenIndex: true,
          dfaValues: true,
          dfaMaxValue: true,
          dfaTargetIndex: true,
          // logitContributions: false,
          // lossValues: true,
        },
      },
      explanations: {
        select: {
          description: true,
          typeName: true,
          explanationModelName: true,
        },
      },
    },
  });
  return neurons;
};
