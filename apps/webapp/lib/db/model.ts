import { prisma } from '@/lib/db';
import { AuthenticatedUser } from '@/lib/with-user';
import { Model, Visibility } from '@prisma/client';
import { AllowUnlistedFor, userCanAccessClause } from './userCanAccess';

export const REPLACE_MODEL_ID_MAP_FOR_LW_TEMPORARY_REDIRECT = {
  'llama31-8b-it': 'llama3.1-8b-it',
  'qwen25-7b-it': 'qwen2.5-7b-it',
};

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

// The reverse of the model id in the URL: given what HuggingFace calls a model, find ours.
//
// `findFirst` rather than `findUnique` because the access clause is an extra condition, which
// `findUnique` does not accept. `hfRepoId` is unique in the schema, so this is still at most one
// row -- a caller who cannot see it gets null, the same as an unknown repo.
export const getModelByHfRepoId = async (hfRepoId: string, user?: AuthenticatedUser | null) =>
  prisma.model.findFirst({
    where: {
      hfRepoId,
      ...userCanAccessClause(user, AllowUnlistedFor.EVERYONE),
    },
  });

// Whichever model claims this repo, visibility ignored. For deciding whether a write would collide
// with the unique index, never for serving to a user: a PRIVATE model still occupies the repo id,
// and `getModelByHfRepoId` would not see it.
export const getModelByHfRepoIdIgnoringAccess = async (hfRepoId: string) =>
  prisma.model.findUnique({ where: { hfRepoId } });

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
  hfRepoId?: string | null,
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
      hfRepoId: hfRepoId ?? null,
    },
  });
