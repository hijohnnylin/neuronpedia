import { getModelById } from '@/lib/db/model';
import { upsertVector } from '@/lib/db/neuron';
import { upsertSourceForVector } from '@/lib/db/source';
import { env } from '@/lib/env';
import { RequestAuthedUser } from '@/lib/types/auth';
import { VECTOR_VALID_HOOK_TYPE_TO_DETAILS } from '@/lib/utils/neuron-vector';
import { STEER_STRENGTH_MAX, STEER_STRENGTH_MIN } from '@/lib/utils/steer';
import { withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';
import { array, number, object, string, ValidationError } from 'yup';

const MAX_VECTOR_LENGTH = 24576;
const VECTOR_SOURCESET_MIDDLE = 'neuronpedia';
const newVectorSchema = object({
  modelId: string().required(),
  layerNumber: number().required(),
  hookType: string().required().oneOf(Object.keys(VECTOR_VALID_HOOK_TYPE_TO_DETAILS)),
  vector: array().max(MAX_VECTOR_LENGTH).of(number().required()).required(),
  vectorDefaultSteerStrength: number().required().min(STEER_STRENGTH_MIN).max(STEER_STRENGTH_MAX),
  vectorLabel: string().required(),
});

/**
 * @swagger
 * /api/vector/new:
 *   post:
 *     summary: Create Vector
 *     description: Creates a new vector to store on Neuronpedia
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
 *               - layerNumber
 *               - hookType
 *               - vector
 *               - vectorDefaultSteerStrength
 *               - vectorLabel
 *             properties:
 *               modelId:
 *                 type: string
 *                 description: ID of the model
 *               layerNumber:
 *                 type: number
 *                 description: Layer number in the model (0-based)
 *               hookType:
 *                 type: string
 *                 description: Hook type. Currently only "resid-pre" is supported.
 *                 enum:
 *                   - resid-pre
 *               vector:
 *                 type: array
 *                 items:
 *                   type: number
 *                 maxItems: 24576
 *                 description: The vector's values. Must match the model's dimension.
 *               vectorDefaultSteerStrength:
 *                 type: number
 *                 minimum: -100
 *                 maximum: 100
 *                 description: Default steering strength (-100 to 100).
 *               vectorLabel:
 *                 type: string
 *                 description: Label/name for the vector
 *     responses:
 *       200:
 *         description: Vector created successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 message:
 *                   type: string
 *                 url:
 *                   type: string
 *                 id:
 *                   type: object
 *                   properties:
 *                     modelId:
 *                       type: string
 *                     source:
 *                       type: string
 *                     index:
 *                       type: string
 */

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const bodyJson = await request.json();

  try {
    const body = await newVectorSchema.validate(bodyJson);

    // ensure model exists
    const model = await getModelById(body.modelId);
    if (!model) {
      return NextResponse.json({ message: 'Model not found' }, { status: 400 });
    }

    // ensure layer number is less than model.numLayers
    if (body.layerNumber >= model.layers) {
      return NextResponse.json({ message: "Layer number is greater than model's numLayers" }, { status: 400 });
    }

    // ensure dimension of vector matches model.dimension
    if (body.vector.length !== model.dimension) {
      return NextResponse.json({ message: 'Vector dimension does not match model dimension' }, { status: 400 });
    }

    // make a source for this model and layer if it doesn't exist
    const sourceSetName = `${VECTOR_SOURCESET_MIDDLE}-${
      VECTOR_VALID_HOOK_TYPE_TO_DETAILS[body.hookType].sourceSetSuffix
    }`;
    const source = await upsertSourceForVector(body.modelId, body.layerNumber, sourceSetName, request.user);

    const hookName = `blocks.${body.layerNumber}.${body.hookType}`;
    // make the vector
    const vector = await upsertVector(
      body.modelId,
      source.id,
      body.vector,
      body.vectorLabel,
      hookName,
      body.vectorDefaultSteerStrength,
      request.user,
    );

    return NextResponse.json({
      message: 'Success',
      url: `${env.NEXT_PUBLIC_URL}/${vector.modelId}/${vector.layer}/${vector.index}`,
      vector: {
        modelId: vector.modelId,
        source: vector.layer,
        index: vector.index,
        label: vector.vectorLabel,
        createdAt: vector.createdAt,
        defaultSteerStrength: vector.vectorDefaultSteerStrength,
        hookName: vector.hookName,
        values: vector.vector,
      },
    });
  } catch (error) {
    if (error instanceof ValidationError) {
      console.log('error', error);
      return NextResponse.json({ message: error.message }, { status: 400 });
    }
    return NextResponse.json({ message: 'Unknown Error' }, { status: 500 });
  }
});
