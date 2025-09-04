import { getVectorsForUser } from '@/lib/db/neuron';
import { RequestAuthedUser } from '@/lib/types/auth';
import { withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

/**
 * @swagger
 * /api/vector/list-owned:
 *   post:
 *     summary: List User's Vectors
 *     description: Returns all vectors owned by the authenticated user
 *     tags:
 *       - Vectors
 *     security:
 *       - apiKey: []
 *     responses:
 *       200:
 *         description: List of user's vectors
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 vectors:
 *                   type: array
 *                   items:
 *                     type: object
 *                     properties:
 *                       modelId:
 *                         type: string
 *                       layer:
 *                         type: string
 *                       index:
 *                         type: string
 *                       vectorLabel:
 *                         type: string
 *                       createdAt:
 *                         type: string
 *                         format: date-time
 *                       vectorDefaultSteerStrength:
 *                         type: number
 *                       hookName:
 *                         type: string
 *                       vector:
 *                         type: array
 *                         items:
 *                           type: number
 */

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const vectors = await getVectorsForUser(request.user.id);

  return NextResponse.json({ vectors });
});
