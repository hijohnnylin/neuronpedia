import { prisma } from '@/lib/db';
import { RequestAuthedUser, withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

/**
 * @swagger
 * /api/graph/list:
 *   get:
 *     summary: List User's Graphs
 *     description: Retrieves a list of all graph metadata for the authenticated user
 *     tags:
 *       - Attribution Graphs
 *     security:
 *       - apiKey: []
 *     responses:
 *       200:
 *         description: Successfully retrieved graph list
 */

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  try {
    const userId = request.user.id;

    const graphMetadatas = await prisma.graphMetadata.findMany({
      where: {
        userId,
      },
      orderBy: {
        updatedAt: 'desc',
      },
    });

    return NextResponse.json(graphMetadatas);
  } catch (error) {
    console.error('Error fetching graph list:', error);
    return NextResponse.json({ error: 'Failed to fetch graph list' }, { status: 500 });
  }
});
