import { prisma } from '@/lib/db';
import { AuthenticatedUser } from '@/lib/types/auth';
import { getSourceSetNameFromSource } from '../utils/source';
import { getVector } from './neuron';
import { assertUserCanAccessModelAndSource, assertUserCanAccessModelAndSourceSet } from './userCanAccess';

export const getActivationById = async (activationId: string, user: AuthenticatedUser | null = null) => {
  const activation = await prisma.activation.findUnique({
    where: {
      id: activationId,
    },
  });
  if (activation) {
    await assertUserCanAccessModelAndSource(activation.modelId, activation.layer, user);
    return activation;
  }
  return null;
};

export const createActivationsForVector = async (
  modelId: string,
  source: string,
  index: string,
  activations: {
    tokens: string[];
    values: number[];
  }[],
  user: AuthenticatedUser,
) => {
  // ensure user is the creator of the vector
  const vector = await getVector(modelId, source, index);
  if (!vector) {
    throw new Error('Vector not found.');
  }
  if (vector.creatorId !== user.id) {
    throw new Error('User is not the creator of the vector.');
  }

  // create the activations
  return prisma.activation.createManyAndReturn({
    data: activations.map((activation) => ({
      ...activation,
      modelId,
      layer: source,
      index,
      creatorId: user.id,
      maxValue: Math.max(...activation.values),
      maxValueTokenIndex: activation.values.indexOf(Math.max(...activation.values)),
      minValue: Math.min(...activation.values),
    })),
    skipDuplicates: true,
  });
};

export const createInferenceActivationsAndReturn = async (
  modelId: string,
  sourceSet: string,
  activations: any[],
  user: AuthenticatedUser | null = null,
) => {
  // ensure that the creatorId is either the authenticated user or the bot
  // eslint-disable-next-line no-restricted-syntax
  for (const activation of activations) {
    if (!activation.creatorId) {
      throw new Error('Missing creatorId');
    }
    if (activation.modelId !== modelId || !modelId || !activation.modelId) {
      throw new Error(`Invalid modelId: ${activation.modelId}`);
    }
    if (!activation.layer || !sourceSet || getSourceSetNameFromSource(activation.layer) !== sourceSet) {
      throw new Error(`Invalid sourceSet: ${activation.sourceSet}`);
    }
  }
  // ensure the model and source are visible to user
  await assertUserCanAccessModelAndSourceSet(modelId, sourceSet, user);

  return prisma.activation.createManyAndReturn({
    data: activations,
    skipDuplicates: true,
  });
};
