import { updateListNeuronDescription } from '@/lib/db/list';
import { RequestAuthedUser } from '@/lib/types/auth';
import { withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

/**
 * @swagger
 * /api/list/edit-feature:
 *   post:
 *     summary: Edit List Feature
 *     description: Updates the description of a specific feature in a list.
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
 *               - description
 *             properties:
 *               listId:
 *                 type: string
 *                 description: The ID of the list containing the feature
 *               modelId:
 *                 type: string
 *                 description: The ID of the model the feature belongs to
 *               layer:
 *                 type: string
 *                 description: The layer of the feature
 *               index:
 *                 type: string
 *                 description: The index of the feature
 *               description:
 *                 type: string
 *                 description: The new description for the feature
 *     responses:
 *       200:
 *         description: Successfully updated feature description
 */

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const body = await request.json();
  const listId = body.listId as string;
  const modelId = body.modelId as string;
  const layer = body.layer as string;
  const index = body.index as string;
  const description = body.description as string;

  const updatedLoN = await updateListNeuronDescription(listId, modelId, layer, index, description, request.user);

  return NextResponse.json(updatedLoN);
});
