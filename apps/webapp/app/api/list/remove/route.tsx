import { removeNeuronFromList } from '@/lib/db/list';
import { RequestAuthedUser } from '@/lib/types/auth';
import { withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

/**
 * @swagger
 * /api/list/remove:
 *   post:
 *     summary: Remove Feature from List
 *     description: Removes a specified feature from a list.
 *     tags:
 *       - Lists
 *     security:
 *       - apiKey: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - listId
 *               - modelId
 *               - layer
 *               - index
 *             properties:
 *               listId:
 *                 type: string
 *                 description: The ID of the list from which to remove the feature
 *               modelId:
 *                 type: string
 *                 description: The ID of the model the feature belongs to
 *               layer:
 *                 type: string
 *                 description: The layer name or SAE ID of the feature
 *               index:
 *                 type: string
 *                 description: The index of the feature
 *     responses:
 *       200:
 *         description: Successfully removed feature from list
 */

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const body = await request.json();
  const listId = body.listId as string;
  const modelId = body.modelId as string;
  const layer = body.layer as string;
  const index = body.index as string;

  const removed = await removeNeuronFromList(listId, modelId, layer, index, request.user);

  return NextResponse.json(removed);
});
