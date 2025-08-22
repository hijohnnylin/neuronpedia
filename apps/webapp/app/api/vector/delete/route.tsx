import { deleteNeuron } from '@/lib/db/neuron';
import { RequestAuthedUser } from '@/lib/types/auth';
import { withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';
import { object, string, ValidationError } from 'yup';

const deleteVectorSchema = object({
  modelId: string().required(),
  source: string().required(),
  index: string().required(),
});

/**
 * @swagger
 * /api/vector/delete:
 *   post:
 *     summary: Delete Vector
 *     description: Deletes an existing vector on Neuronpedia. You can only delete vectors you created.
 *     tags:
 *       - Vectors
 *     security:
 *       - apiKey: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - modelId
 *               - source
 *               - index
 *             properties:
 *               modelId:
 *                 type: string
 *                 description: ID of the model
 *               source:
 *                 type: string
 *                 description: Source identifier for the vector
 *               index:
 *                 type: string
 *                 description: Index of the vector to delete
 *     responses:
 *       200:
 *         description: Vector deleted successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 message:
 *                   type: string
 */

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const bodyJson = await request.json();

  try {
    const body = await deleteVectorSchema.validate(bodyJson);

    await deleteNeuron(body.modelId, body.source, body.index, request.user);

    return NextResponse.json({ message: 'Success' });
  } catch (error) {
    if (error instanceof ValidationError) {
      return NextResponse.json({ message: error.message }, { status: 400 });
    }
    return NextResponse.json({ message: 'Unknown Error' }, { status: 500 });
  }
});
