import { prisma } from '@/lib/db';
import { AuthenticatedUser } from '@/lib/types/auth';
import { assertUserCanAccessModelAndSource } from './userCanAccess';

export const createBookmark = async (modelId: string, layer: string, index: string, user: AuthenticatedUser) => {
  await assertUserCanAccessModelAndSource(modelId, layer, user);
  return prisma.bookmark.create({
    data: {
      userId: user.id,
      modelId,
      layer,
      index,
    },
  });
};

export const getBookmarksByUserWithExplanations = async (user: AuthenticatedUser) =>
  prisma.bookmark.findMany({
    where: {
      userId: user.id,
    },
    include: {
      neuron: {
        include: {
          explanations: {
            include: {
              author: {
                select: {
                  name: true,
                },
              },
            },
            orderBy: [{ scoreV2: 'desc' }, { scoreV1: 'desc' }],
          },
        },
      },
    },
    orderBy: { createdAt: 'desc' },
  });

export const getBookmarksByUser = async (user: AuthenticatedUser) =>
  prisma.bookmark.findMany({
    where: {
      userId: user.id,
    },
    orderBy: { createdAt: 'desc' },
  });

export const deleteBookmark = async (modelId: string, layer: string, index: string, user: AuthenticatedUser) =>
  // *important* we use delete many so we can specify username to ensure nobody deletes other peoples' bookmarks
  prisma.bookmark.deleteMany({
    where: {
      userId: user.id,
      modelId,
      layer,
      index,
    },
  });
