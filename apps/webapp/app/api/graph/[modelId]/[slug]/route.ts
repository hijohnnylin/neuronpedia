import { prisma } from '@/lib/db';
import { RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

/**
 * @swagger
 * /api/graph/{modelId}/{slug}:
 *   get:
 *     summary: Get Graph Metadata
 *     description: Retrieves metadata for a specific graph by model ID and slug
 *     tags:
 *       - Attribution Graphs
 *     parameters:
 *       - name: modelId
 *         in: path
 *         description: Model ID
 *         required: true
 *         schema:
 *           type: string
 *       - name: slug
 *         in: path
 *         description: Graph slug
 *         required: true
 *         schema:
 *           type: string
 *     responses:
 *       200:
 *         description: Successfully retrieved graph metadata
 *       404:
 *         description: Graph not found
 *       500:
 *         description: Server error
 */

export const GET = withOptionalUser(
  async (request: RequestOptionalUser, { params }: { params: { modelId: string; slug: string } }) => {
    try {
      const { modelId, slug } = params;

      const graphMetadata = await prisma.graphMetadata.findUnique({
        where: {
          modelId_slug: {
            modelId,
            slug,
          },
        },
        include: {
          user: {
            select: {
              name: true,
              id: true,
            },
          },
        },
      });

      if (!graphMetadata) {
        return NextResponse.json({ error: 'Graph not found' }, { status: 404 });
      }

      return NextResponse.json(graphMetadata);
    } catch (error) {
      console.error('Error fetching graph metadata:', error);
      return NextResponse.json({ error: 'Failed to fetch graph metadata' }, { status: 500 });
    }
  },
);
