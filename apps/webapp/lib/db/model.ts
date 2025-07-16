import { prisma } from '@/lib/db';
import { AuthenticatedUser } from '@/lib/with-user';
import { Model, Visibility } from '@prisma/client';
import { AllowUnlistedFor, userCanAccessClause } from './userCanAccess';

// for globalModels, return unlisted to everyone, but filter them out in the UI
export const getGlobalModels = async (user?: AuthenticatedUser | null) => {
  const query = {
    where: userCanAccessClause(user, AllowUnlistedFor.EVERYONE),
    include: {
      sourceSets: {
        where: {
          ...userCanAccessClause(user, AllowUnlistedFor.EVERYONE),
          hasDashboards: true,
        },
        include: {
          sources: {
            where: {
              ...userCanAccessClause(user, AllowUnlistedFor.EVERYONE),
              hasDashboards: true,
            },
            select: {
              id: true,
              modelId: true,
              visibility: true,
              setName: true,
              hasUmap: true,
              hasUmapLogSparsity: true,
              hasUmapClusters: true,
              num_prompts: true,
              num_tokens_in_prompt: true,
              dataset: true,
              inferenceEnabled: true,
              hasDashboards: true,
              notes: true,
              createdAt: true,
              cosSimMatchSourceId: true,
              cosSimMatchModelId: true,
            },
          },
        },
      },
    },
  };
  return prisma.model.findMany(query);
};

export const getModelById = async (modelId: string, user?: AuthenticatedUser | null) => {
  const model = await prisma.model.findFirst({
    where: {
      id: modelId,
      ...userCanAccessClause(user, AllowUnlistedFor.EVERYONE),
    },
  });
  return model;
};

// sometimes transformerlens model IDs do not match our model IDs, so we need to replace them
export const getTransformerLensModelIdIfExists = async (modelId: string) => {
  const model = await getModelById(modelId);
  if (model?.tlensId) {
    return model.tlensId;
  }
  return modelId;
};

export const getModelByIdWithSourceSets = async (modelId: string, user?: AuthenticatedUser | null) =>
  prisma.model.findUnique({
    where: {
      id: modelId,
      ...userCanAccessClause(user, AllowUnlistedFor.EVERYONE),
    },
    include: {
      sourceSets: {
        where: userCanAccessClause(user, AllowUnlistedFor.EVERYONE),
        orderBy: {
          name: 'asc',
        },
        include: {
          sources: true,
        },
      },
    },
  });

export const createModel = async (model: Model, user: AuthenticatedUser) => {
  // eslint-disable-next-line no-param-reassign
  model.creatorId = user.id;

  const existingModel = await prisma.model.findUnique({
    where: {
      id: model.id,
    },
  });

  if (existingModel) {
    throw new Error('Model already exists.');
  } else {
    return prisma.model.create({
      data: model,
    });
  }
};

export const createModelAdmin = async (
  modelId: string,
  displayName: string,
  layers: number,
  owner: string,
  user: AuthenticatedUser,
) =>
  prisma.model.create({
    data: {
      id: modelId,
      displayName,
      displayNameShort: displayName,
      creatorId: user?.id,
      visibility: Visibility.PRIVATE,
      layers,
      neuronsPerLayer: 0,
      owner,
    },
  });
