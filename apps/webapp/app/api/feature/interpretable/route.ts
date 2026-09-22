import { badRequest, toErrorResponse } from '@/lib/api-error';
import { prisma } from '@/lib/db';
import { checkInterpretability } from '@/lib/external/interpretability-jev';
import { NextRequest, NextResponse } from 'next/server';
import { object, string } from 'yup';

const NUM_ACTIVATIONS_TO_LOAD = 20;
const NUM_ZERO_ACTIVATIONS_TO_LOAD_MAX = 5;

const bodySchema = object({
  modelId: string().required(),
  layer: string().required(),
  index: string().required(),
});

/**
 * @swagger
 * /api/feature/interpretable:
 *   post:
 *     summary: Check If Interpretable
 *     description: Intruder detection with Jev, with no explanation involved. Builds up to five groups of four activating texts plus one non-activating intruder and asks which text is the odd one out. Returns the share of groups answered correctly. The result is computed on demand and not stored.
 *     tags:
 *       - Explanations
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required: [modelId, layer, index]
 *             properties:
 *               modelId: { type: string }
 *               layer: { type: string }
 *               index: { type: string }
 *     responses:
 *       200:
 *         description: Score, per-group picks and the cropped texts shown to the model.
 */
export async function POST(request: NextRequest) {
  try {
    const body = await bodySchema.validate(await request.json());

    let activations = await prisma.activation.findMany({
      where: { neuron: { modelId: body.modelId, layer: body.layer, index: body.index } },
      select: { tokens: true, values: true, maxValue: true },
    });
    if (activations.length === 0) {
      throw badRequest('Feature not found, or it has no activations stored.');
    }
    // Same preparation as the explanation scorers: dedupe by text, strongest first.
    activations = [...new Map(activations.map((item) => [item.tokens.join(''), item])).values()];
    activations.sort((a, b) => b.maxValue - a.maxValue);
    const zeroActivations = activations.filter((a) => a.maxValue === 0).slice(0, NUM_ZERO_ACTIVATIONS_TO_LOAD_MAX);
    const topActivations = activations.slice(0, NUM_ACTIVATIONS_TO_LOAD).filter((a) => a.maxValue > 0);

    const result = await checkInterpretability(
      topActivations,
      zeroActivations,
      `${body.modelId}/${body.layer}/${body.index}`,
    );
    return NextResponse.json(result);
  } catch (error) {
    return toErrorResponse(error, request);
  }
}
