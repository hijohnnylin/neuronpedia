import { addNeuronsToList } from '@/lib/db/list';
import { RequestAuthedUser } from '@/lib/types/auth';
import { ListNeuronToAdd } from '@/lib/utils/list';
import { withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const maxDuration = 15;

/**
 * @swagger
 * /api/list/add-features:
 *   post:
 *     summary: Add Features to List
 *     description: Adds one or more features to an existing list.
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
 *               - featuresToAdd
 *             properties:
 *               listId:
 *                 type: string
 *                 description: The ID of the list to add features to
 *               featuresToAdd:
 *                 description: The features to add to the list. Takes a max of 100 features at a time.
 *                 type: array
 *                 items:
 *                   type: object
 *                   required:
 *                     - modelId
 *                     - layer
 *                     - index
 *                   properties:
 *                     modelId:
 *                       type: string
 *                       description: The ID of the model the feature belongs to
 *                     layer:
 *                       type: string
 *                       description: The layer of the feature
 *                     index:
 *                       type: string
 *                       description: The index of the feature
 *                     description:
 *                       type: string
 *                       description: Optional description for the feature
 *     responses:
 *       200:
 *         description: Successfully added features to the list
 */

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const body = await request.json();
  const listId = body.listId as string;
  const featuresToAdd = body.featuresToAdd as ListNeuronToAdd[];

  if (featuresToAdd.length > 100) {
    return NextResponse.json({ error: 'Too many features to add. Max is 100 at a time.' }, { status: 400 });
  }

  const addedNeurons = await addNeuronsToList(listId, featuresToAdd, request.user);

  return NextResponse.json(addedNeurons);
});
