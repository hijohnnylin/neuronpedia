import { prisma } from '@/lib/db';
import { AuthenticatedUser } from '../types/auth';
import { assertUserCanAccessModelAndSource } from './userCanAccess';

export const getCommentsByUser = async (user: AuthenticatedUser) =>
  prisma.comment.findMany({
    where: {
      userId: user.id,
    },
    include: {
      neuron: true,
    },
    orderBy: { createdAt: 'desc' },
  });

export const createComment = async (
  modelId: string,
  layer: string,
  index: string,
  text: string,
  user: AuthenticatedUser,
) => {
  await assertUserCanAccessModelAndSource(modelId, layer, user);
  return prisma.comment.create({
    data: {
      text: text.trim(),
      modelId,
      layer,
      index,
      userId: user.id,
    },
  });
};

export const deleteComment = async (id: string, user: AuthenticatedUser) =>
  // *important* we use delete many so we can specify username to ensure nobody deletes other peoples' comments
  prisma.comment.deleteMany({
    where: {
      id,
      userId: user.id,
    },
  });
