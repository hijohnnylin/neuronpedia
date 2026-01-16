import { prisma } from '@/lib/db';
import { getSourceSetNameFromSource } from '@/lib/utils/source';
import { AuthenticatedUser } from '@/lib/with-user';
import { InferenceEngine, Source, SourceRelease, SourceSet, Visibility } from '@prisma/client';
import { DEFAULT_CREATOR_USER_ID } from '../env';

import {
  AllowUnlistedFor,
  assertUserCanAccessModel,
  userCanAccessClause,
  userCanAccessModelAndSourceSet,
} from './userCanAccess';

export const getSourceRelease = async (name: string, user: AuthenticatedUser | null = null) =>
  prisma.sourceRelease.findUnique({
    where: {
      name,
      ...userCanAccessClause(user, AllowUnlistedFor.EVERYONE),
    },
    include: {
      sourceSets: {
        where: userCanAccessClause(user, AllowUnlistedFor.CREATOR_AND_ALLOWED_USERS),
        include: {
          sources: {
            where: userCanAccessClause(user, AllowUnlistedFor.CREATOR_AND_ALLOWED_USERS),
            orderBy: {
              id: 'asc',
            },
          },
        },
        orderBy: {
          name: 'asc',
        },
      },
    },
  });

export const getSourceSet = async (modelId: string, sourceSetName: string, user: AuthenticatedUser | null = null) => {
  const query = {
    where: {
      modelId_name: {
        modelId,
        name: sourceSetName,
      },
      ...userCanAccessClause(user, AllowUnlistedFor.EVERYONE),
    },
    include: {
      sources: {
        where: userCanAccessClause(user, AllowUnlistedFor.CREATOR_AND_ALLOWED_USERS),
      },
      releases: {
        where: userCanAccessClause(user, AllowUnlistedFor.EVERYONE),
      },
      model: true,
    },
  };
  return prisma.sourceSet.findUnique(query);
};

export const getSourceInferenceHosts = async (
  modelId: string,
  sourceId: string,
  user: AuthenticatedUser | null = null,
  engine: InferenceEngine = InferenceEngine.TRANSFORMER_LENS,
) => {
  const canAccess = await userCanAccessModelAndSourceSet(modelId, getSourceSetNameFromSource(sourceId), user, true);
  if (!canAccess) {
    return null;
  }

  return prisma.inferenceHostSourceOnSource.findMany({
    where: {
      sourceModelId: modelId,
      sourceId,
      inferenceHost: {
        engine,
      },
    },
    include: {
      inferenceHost: true,
    },
  });
};

export const getSource = async (modelId: string, sourceId: string, user: AuthenticatedUser | null = null) => {
  // ensure we can access the sourceSet
  const canAccess = await userCanAccessModelAndSourceSet(modelId, getSourceSetNameFromSource(sourceId), user, true);
  if (!canAccess) {
    return null;
  }
  return prisma.source.findUnique({
    where: {
      modelId_id: {
        modelId,
        id: sourceId,
      },
      ...userCanAccessClause(user, AllowUnlistedFor.EVERYONE),
    },
    include: {
      evals: {
        include: {
          type: true,
        },
      },
      set: {
        include: {
          releases: {
            where: userCanAccessClause(user, AllowUnlistedFor.EVERYONE),
          },
          model: true,
        },
      },
    },
  });
};

export const createSource = async (input: Source, user: AuthenticatedUser) => {
  // eslint-disable-next-line no-param-reassign
  input.creatorId = user.id;
  // First check if source exists
  const existingSource = await prisma.source.findUnique({
    where: {
      modelId_id: {
        modelId: input.modelId,
        id: input.id,
      },
    },
  });

  // Only create if it doesn't exist
  if (!existingSource) {
    return prisma.source.create({
      data: {
        ...input,
        saelensConfig: input.saelensConfig as any,
      },
    });
  }

  // Return existing source if already exists
  return existingSource;
};

export const upsertSourceForVector = async (
  modelId: string,
  layerNumber: number,
  sourceSetName: string,
  user: AuthenticatedUser,
) => {
  await assertUserCanAccessModel(modelId, user);
  const existingSourceSet = await prisma.sourceSet.findUnique({
    where: {
      modelId_name: {
        modelId,
        name: sourceSetName,
      },
    },
  });
  if (!existingSourceSet) {
    await prisma.sourceSet.create({
      data: {
        modelId,
        name: sourceSetName,
        description: 'Default',
        creatorId: DEFAULT_CREATOR_USER_ID,
        creatorName: 'Neuronpedia',
        visibility: Visibility.UNLISTED,
      },
    });
  }
  const existingSource = await prisma.source.findUnique({
    where: {
      modelId_id: {
        modelId,
        id: `${layerNumber}-${sourceSetName}`,
      },
    },
  });
  if (existingSource) {
    return existingSource;
  }

  return prisma.source.create({
    data: {
      modelId,
      id: `${layerNumber}-${sourceSetName}`,
      setName: sourceSetName,
      creatorId: DEFAULT_CREATOR_USER_ID,
      visibility: Visibility.UNLISTED,
    },
  });
};

export const createSourceSetWithSources = async (input: SourceSet, layers: number, user: AuthenticatedUser) => {
  // eslint-disable-next-line no-param-reassign
  input.creatorId = user.id;

  const existingSourceSet = await prisma.sourceSet.findUnique({
    where: {
      modelId_name: {
        modelId: input.modelId,
        name: input.name,
      },
    },
  });

  if (existingSourceSet) {
    throw new Error('Source set name already exists for this model.');
  } else {
    // make sources to create
    const sourcesToCreate = [];
    for (let i = 0; i < layers; i += 1) {
      sourcesToCreate.push({
        // modelId: input.modelId,
        id: `${i}-${input.name}`,
        // setName: input.name,
        creatorId: user.id,
        visibility: Visibility.UNLISTED,
        inferenceEnabled: false,
        hasDashboards: true,
      });
    }

    return prisma.sourceSet.create({
      data: {
        ...input,
        sources: {
          createMany: {
            data: sourcesToCreate,
          },
        },
      },
    });
  }
};

export const createSourceRelease = async (input: SourceRelease, user: AuthenticatedUser) => {
  // eslint-disable-next-line no-param-reassign
  input.creatorId = user.id;
  // First check if source release exists
  const existingSourceRelease = await prisma.sourceRelease.findUnique({
    where: {
      name: input.name,
    },
  });

  // Only create if it doesn't exist
  if (!existingSourceRelease) {
    return prisma.sourceRelease.create({
      data: input,
    });
  }

  // Return existing source release if already exists
  return existingSourceRelease;
};

export const getGlobalSourceReleases = async (user?: AuthenticatedUser | null) => {
  const query = {
    where: userCanAccessClause(user, AllowUnlistedFor.CREATOR_AND_ALLOWED_USERS),
    include: {
      sourceSets: {
        where: {
          ...userCanAccessClause(user, AllowUnlistedFor.CREATOR_AND_ALLOWED_USERS),
          hasDashboards: true,
        },
        include: {
          sources: {
            where: {
              ...userCanAccessClause(user, AllowUnlistedFor.CREATOR_AND_ALLOWED_USERS),
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
  const releases = await prisma.sourceRelease.findMany(query);

  return releases;
};
