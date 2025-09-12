import { prisma } from '@/lib/db';
import { UserSecretType } from '@prisma/client';
import { env } from '../env';
import { AuthenticatedUser } from '../types/auth';
import { getUserByName } from './user';

export const getUserSecret = async (username: string, type: UserSecretType) =>
  prisma.userSecret.findUnique({
    where: {
      username_type: {
        username,
        type,
      },
    },
  });

export const updateUserSecret = async (username: string, type: UserSecretType, value: string) =>
  prisma.userSecret.upsert({
    where: {
      username_type: {
        username,
        type,
      },
    },
    update: {
      value,
    },
    create: {
      username,
      type,
      value,
    },
  });

export const removeUserSecret = async (username: string, type: UserSecretType) => {
  if (type === UserSecretType.NEURONPEDIA) {
    throw new Error('Cannot remove neuronpedia secret');
  }
  return prisma.userSecret.delete({
    where: {
      username_type: {
        username,
        type,
      },
    },
  });
};

export const getAutoInterpKeyToUse = async (type: UserSecretType, user: AuthenticatedUser) => {
  // get user, if it's admin, use the stored secret
  const authedUser = await getUserByName(user.name);
  if (authedUser.canTriggerExplanations) {
    if (type === UserSecretType.OPENAI) {
      return env.OPENAI_API_KEY;
    }
    if (type === UserSecretType.GOOGLE) {
      return env.GEMINI_API_KEY;
    }
    if (type === UserSecretType.ANTHROPIC) {
      return env.ANTHROPIC_API_KEY;
    }
    return env.OPENROUTER_API_KEY;
  }
  const secret = await getUserSecret(user.name, type);
  return secret?.value;
};
