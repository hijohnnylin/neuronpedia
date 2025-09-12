import { assertUserCanAccessModelAndSourceSet } from '@/lib/db/userCanAccess';
import { RequestOptionalUser } from '@/lib/types/auth';
import { getActivationForFeature } from '@/lib/utils/inference';
import { getSourceSetNameFromSource } from '@/lib/utils/source';
import { withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

/**
 * @swagger
 * /api/activation/new:
 *   post:
 *     summary: Feature Activation for Text
 *     description: Gets activation values for a given feature when processing custom input text. Equivalent to going to a feature page and entering custom text. The example shows this feature https://neuronpedia.org/gpt2-small/9-res-jb/200
 *     tags:
 *       - Activations
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - feature
 *               - customText
 *             properties:
 *               feature:
 *                 type: object
 *                 properties:
 *                   modelId:
 *                     description: The model the feature is in.
 *                     type: string
 *                     required: true
 *                     default: gpt2-small
 *                   source:
 *                     description: The SAE ID the feature is in.
 *                     type: string
 *                     required: true
 *                     default: 9-res-jb
 *                   index:
 *                     description: The index of the feature.
 *                     type: string
 *                     required: true
 *                     default: "200"
 *               customText:
 *                 type: string
 *                 description: The custom text to process
 *                 required: true
 *                 default: And you
 *     responses:
 *       200:
 *         description: Successful response, with activation data
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/InferenceActivationResultSingle'
 *       400:
 *         description: Bad request, missing or invalid input
 *       401:
 *         description: Unauthorized, user doesn't have access to the model or SAE set
 *       500:
 *         description: Internal server error
 *
 * components:
 *   schemas:
 *     NeuronSchema:
 *       type: object
 *       required:
 *         - modelId
 *         - source
 *         - index
 *       properties:
 *         modelId:
 *           type: string
 *         source:
 *           type: string
 *         index:
 *           type: string
 *         sourceSetName:
 *           type: string
 *     InferenceActivationResultSingle:
 *       type: object
 *       properties:
 *         tokens:
 *           type: array
 *           items:
 *             type: string
 *         activations:
 *           type: object
 *           properties:
 *             layer:
 *               type: string
 *             index:
 *               type: number
 *             values:
 *               type: array
 *               items:
 *                 type: number
 *             maxValue:
 *               type: number
 *             maxValueIndex:
 *               type: number
 *             dfaValues:
 *               type: array
 *               items:
 *                 type: number
 *             dfaTargetIndex:
 *               type: number
 *             dfaMaxValue:
 *               type: number
 *         error:
 *           type: string
 */

export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  const body = await request.json();

  let { neuron } = body;
  if (!neuron) {
    neuron = body.feature;
  }
  if (!neuron.layer) {
    neuron.layer = neuron.source;
  }
  if (!neuron.modelId || (!neuron.layer && !neuron.source) || !neuron.index) {
    throw new Error('Invalid feature.');
  }
  if (!body.customText || body.customText.trim().length === 0) {
    throw new Error('Missing custom text.');
  }

  const sourceSetName = getSourceSetNameFromSource(neuron.layer);
  try {
    await assertUserCanAccessModelAndSourceSet(neuron.modelId, sourceSetName, request.user);
  } catch (error) {
    return NextResponse.json({ message: error instanceof Error ? error.message : 'Unknown Error' }, { status: 500 });
  }
  const activation = await getActivationForFeature(neuron, body.customText, request.user);

  return NextResponse.json(activation);
});
