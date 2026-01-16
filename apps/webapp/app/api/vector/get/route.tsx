import { getVectorFromDb } from '@/lib/db/neuron';
import { getVectorFromInstance } from '@/lib/utils/inference';
import { RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';
import { object, string } from 'yup';

const getVectorSchema = object({
  modelId: string().required(),
  source: string().required(),
  index: string().required(),
});

/**
 * @swagger
 * /api/vector/get:
 *   post:
 *     summary: Get Vector Details
 *     description: Returns details for a specific vector by model ID, source and index
 *     tags:
 *       - Vectors
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
 *                 description: Source/layer identifier
 *               index:
 *                 type: string
 *                 description: Vector index
 *     responses:
 *       200:
 *         description: Vector details
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 vector:
 *                   type: object
 *                   properties:
 *                     modelId:
 *                       type: string
 *                     layer:
 *                       type: string
 *                     index:
 *                       type: string
 *                     vectorLabel:
 *                       type: string
 *                     vectorDefaultSteerStrength:
 *                       type: number
 *                     hookName:
 *                       type: string
 *                     vector:
 *                       type: array
 *                       items:
 *                         type: number
 *                     createdAt:
 *                       type: string
 *                       format: date-time
 */

export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  const body = await getVectorSchema.validate(await request.json());
  const vector = await getVectorFromDb(body.modelId, body.source, body.index);

  if (!vector || !vector?.vector || vector?.vector?.length === 0) {
    const vectorFromInstance = await getVectorFromInstance(body.modelId, body.source, body.index);
    if (!vectorFromInstance) {
      return NextResponse.json({ message: 'Vector not found' }, { status: 404 });
    }
    return NextResponse.json({
      vector: {
        modelId: body.modelId,
        layer: body.source,
        index: body.index,
        vectorLabel: '',
        hookName: vectorFromInstance.hookName,
        vector: vectorFromInstance.vector,
        vectorDefaultSteerStrength: 10,
      },
    });
  }
  return NextResponse.json({ vector });
});
