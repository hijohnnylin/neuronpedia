import { prisma } from '@/lib/db';
import { AuthenticatedUser } from '@/lib/types/auth';
import { getSourceSetNameFromSource } from '@/lib/utils/source';
import { Visibility } from '@prisma/client';
import { env } from '../env';

export const ERROR_NOT_FOUND_MESSAGE = `The model, source, or feature you specified is not available. Check available public models/sources (including which ones have inference enabled) at ${env.NEXT_PUBLIC_URL}/available-resources`;

// for the most part, allowUnlistedFor is EVERYONE (since they can all technically access it), except global
export enum AllowUnlistedFor {
  EVERYONE,
  CREATOR_AND_ALLOWED_USERS,
}

export function userCanAccessClause(
  user?: AuthenticatedUser | null,
  allowUnlisted: AllowUnlistedFor = AllowUnlistedFor.CREATOR_AND_ALLOWED_USERS,
) {
  const toReturn = {
    OR: [
      // users in general can only see PUBLIC models
      ...[{ visibility: Visibility.PUBLIC }],
      // set returning unlisted behavior based on allowUnlisted
      ...(allowUnlisted === AllowUnlistedFor.EVERYONE
        ? [{ visibility: Visibility.UNLISTED }]
        : // DO NOT REMOVE USER CHECK HERE - otherwise creatorId and allowedUsers will just allow everyone
          allowUnlisted === AllowUnlistedFor.CREATOR_AND_ALLOWED_USERS && user
          ? [
              {
                AND: [
                  {
                    visibility: Visibility.UNLISTED,
                    OR: [{ creatorId: user?.id }],
                  },
                ],
              },
            ]
          : []),
      // if it's UNLISTED or PRIVATE
      // creator of model can access
      // allowedUsers can access
      ...(user ? [{ creatorId: user.id }] : []),
    ],
  };
  // debug log
  // if (allowUnlisted === AllowUnlistedFor.CREATOR_AND_ALLOWED_USERS) {
  //   console.log(JSON.stringify(toReturn, null, 2));
  // }
  return toReturn;
}

export const userCanAccessRelease = async (releaseName: string, user: AuthenticatedUser | null = null) => {
  // user is admin
  if (user) {
    const userDb = await prisma.user.findUnique({
      where: {
        id: user?.id,
      },
      select: {
        admin: true,
      },
    });
    if (userDb?.admin) {
      return true;
    }
  }
  // user can access release
  const release = await prisma.sourceRelease.findUnique({
    where: {
      name: releaseName,
      ...userCanAccessClause(user, AllowUnlistedFor.EVERYONE),
    },
  });
  if (!release) {
    return false;
  }
  return true;
};

export const userCanAccessModel = async (
  modelId: string,
  user: AuthenticatedUser | null = null,
  // TODO implement
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  allowUnlisted = false,
) => {
  // user is admin
  if (user) {
    const userDb = await prisma.user.findUnique({
      where: {
        id: user?.id,
      },
      select: {
        admin: true,
      },
    });
    if (userDb?.admin) {
      return true;
    }
  }
  // user can access model
  const model = await prisma.model.findUnique({
    where: {
      id: modelId,
      ...userCanAccessClause(user, AllowUnlistedFor.EVERYONE),
    },
  });
  if (!model) {
    return false;
  }
  return true;
};

export const userCanAccessModelAndSourceSet = async (
  modelId: string,
  sourceSetName: string,
  user: AuthenticatedUser | null = null,
  // TODO implement
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  allowUnlisted = false,
) => {
  // user is admin
  if (user) {
    const userDb = await prisma.user.findUnique({
      where: {
        id: user?.id,
      },
      select: {
        admin: true,
      },
    });
    if (userDb?.admin) {
      return true;
    }
  }
  // user can access model
  const model = await prisma.model.findUnique({
    where: {
      id: modelId,
      ...userCanAccessClause(user, AllowUnlistedFor.EVERYONE),
    },
  });
  if (!model) {
    return false;
  }
  // can access layer / sourceSet
  // look up the sourceSet
  const sourceSet = await prisma.sourceSet.findUnique({
    where: {
      modelId_name: {
        modelId,
        name: sourceSetName,
      },
      ...userCanAccessClause(user, AllowUnlistedFor.EVERYONE),
    },
  });
  if (!sourceSet) {
    return false;
  }
  return true;
};

export const assertUserCanAccessRelease = async (releaseName: string, user: AuthenticatedUser | null = null) => {
  const permitted = await userCanAccessRelease(releaseName, user);
  if (!permitted) {
    throw new Error(ERROR_NOT_FOUND_MESSAGE);
  }
  return true;
};

export const assertUserCanAccessModelAndSource = async (
  modelId: string,
  layer: string,
  user: AuthenticatedUser | null = null,
) => {
  const permitted = await userCanAccessModelAndSourceSet(modelId, getSourceSetNameFromSource(layer), user, true);
  if (!permitted) {
    throw new Error(ERROR_NOT_FOUND_MESSAGE);
  }
  return true;
};

export const assertUserCanAccessModel = async (modelId: string, user: AuthenticatedUser | null = null) => {
  const permitted = await userCanAccessModel(modelId, user, true);
  if (!permitted) {
    throw new Error(ERROR_NOT_FOUND_MESSAGE);
  }
  return true;
};

export const assertUserCanAccessModelAndSourceSet = async (
  modelId: string,
  sourceSet: string,
  user: AuthenticatedUser | null = null,
) => {
  const permitted = await userCanAccessModelAndSourceSet(modelId, sourceSet, user, true);
  if (!permitted) {
    throw new Error(ERROR_NOT_FOUND_MESSAGE);
  }
  return true;
};

// ========== MARK: Write access to model/sourceSet (for creating/uploading features)

export const userCanWriteModelAndSourceSet = async (
  modelId: string,
  sourceSetName: string,
  user: AuthenticatedUser,
) => {
  // user is admin
  if (user) {
    const userDb = await prisma.user.findUnique({
      where: {
        id: user?.id,
      },
      select: {
        admin: true,
      },
    });
    if (userDb?.admin) {
      return true;
    }
  }
  // user can write model - only creator can do this
  const model = await prisma.model.findUnique({
    where: {
      id: modelId,
      creatorId: user.id,
    },
  });
  if (!model) {
    return false;
  }
  // user can write source/sourceSet - only creator can
  const sourceSet = await prisma.sourceSet.findUnique({
    where: {
      modelId_name: {
        modelId,
        name: sourceSetName,
      },
      creatorId: user.id,
    },
  });
  if (!sourceSet) {
    return false;
  }
  return true;
};

export const assertUserCanWriteModelAndSource = async (modelId: string, layer: string, user: AuthenticatedUser) => {
  const permitted = await userCanWriteModelAndSourceSet(modelId, getSourceSetNameFromSource(layer), user);
  if (!permitted) {
    throw new Error(ERROR_NOT_FOUND_MESSAGE);
  }
  return true;
};
