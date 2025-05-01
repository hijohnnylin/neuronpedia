import { getNeuronsForTopkSearcherExplanationOnly } from '@/lib/db/neuron';
import { assertUserCanAccessModelAndSource } from '@/lib/db/userCanAccess';
import { NEXT_PUBLIC_SEARCH_TOPK_MAX_CHAR_LENGTH } from '@/lib/env';
import { getActivationsTopKByToken, SearchTopKResult } from '@/lib/utils/inference';
import { RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { ActivationTopkByTokenPost200Response } from 'neuronpedia-inference-client';
import { NextResponse } from 'next/server';
import { boolean, number, object, string, ValidationError } from 'yup';

export const maxDuration = 30;

const NUMBER_TOPK_RESULTS = 10;
const DEFAULT_DENSITY_THRESHOLD = 0.01;

const searchWithTopKRequestSchema = object({
  modelId: string().required().max(50),
  source: string().required().max(50),
  text: string().required().max(NEXT_PUBLIC_SEARCH_TOPK_MAX_CHAR_LENGTH),
  numResults: number().optional().min(1).max(20).default(NUMBER_TOPK_RESULTS),
  ignoreBos: boolean().optional().default(true),
  densityThreshold: number().optional().default(DEFAULT_DENSITY_THRESHOLD),
});

/**
 * @swagger
 * /api/search-topk-by-token:
 *   post:
 *     summary: Top Features for Text by Token
 *     tags:
 *       - Search TopK by Token
 *     description: Returns the top-k activating features for each token in the provided text
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - modelId
 *               - source
 *               - text
 *             properties:
 *               modelId:
 *                 type: string
 *                 description: ID of the model to search
 *                 maxLength: 50
 *                 example: "gpt2-small"
 *               source:
 *                 type: string
 *                 description: Source name to search
 *                 maxLength: 50
 *                 example: "6-res-jb"
 *               text:
 *                 type: string
 *                 description: Text to analyze
 *                 maxLength: 1000
 *                 example: "The quick brown fox jumps over the lazy dog"
 *               numResults:
 *                 type: number
 *                 description: Number of top features to return per token
 *                 minimum: 1
 *                 maximum: 20
 *                 default: 10
 *                 example: 10
 *               ignoreBos:
 *                 type: boolean
 *                 description: Whether to ignore beginning-of-sequence token
 *                 default: true
 *                 example: true
 *               densityThreshold:
 *                 type: number
 *                 description: Filter features by density threshold (0-1)
 *                 minimum: 0
 *                 maximum: 1
 *                 default: 0.01
 *                 example: 0.01
 *     responses:
 *       200:
 *         description: Successful response with top-k activations by token
 *       400:
 *         description: Bad request, validation error
 */

export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  try {
    const body = await searchWithTopKRequestSchema.validate(await request.json());

    const { modelId, source, text, numResults, ignoreBos } = body;

    const densityThreshold = body.densityThreshold || -1;
    if (densityThreshold !== -1 && (densityThreshold <= 0 || densityThreshold >= 1)) {
      throw new Error('densityThreshold must be between 0 and 1.');
    }

    await assertUserCanAccessModelAndSource(modelId, source, request.user);

    const result: ActivationTopkByTokenPost200Response = await getActivationsTopKByToken(
      modelId,
      source,
      text,
      numResults,
      ignoreBos,
      request.user,
    );

    const neuronData = await getNeuronsForTopkSearcherExplanationOnly(modelId, source, result, request.user);

    // make a toReturn object
    const toReturn: SearchTopKResult = {
      source,
      results: result.results.map((r) => ({
        position: r.tokenPosition,
        token: r.token,
        topFeatures: r.topFeatures.map((f) => ({
          activationValue: f.activationValue,
          featureIndex: f.featureIndex,
          feature: neuronData.find((n) => n.index === f.featureIndex.toString()),
        })),
      })),
    };

    // remove all that are below the density threshold
    if (densityThreshold !== -1) {
      toReturn.results = toReturn.results.map((r) => {
        // eslint-disable-next-line no-param-reassign
        r.topFeatures = r.topFeatures.filter(
          (f) => f.feature?.frac_nonzero && f.feature?.frac_nonzero <= densityThreshold,
        );
        return r;
      });
    }

    return NextResponse.json(toReturn);
  } catch (error) {
    console.error('error', error);
    if (error instanceof ValidationError) {
      return NextResponse.json({ message: error.message }, { status: 400 });
    }
    return NextResponse.json({ message: 'Unknown Error' }, { status: 500 });
  }
});
