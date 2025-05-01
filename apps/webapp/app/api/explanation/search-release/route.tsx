import { MAX_EXPLANATION_SEARCH_RESULTS, searchExplanationsReleaseVec } from '@/lib/db/explanation';
import { getNeurons } from '@/lib/db/neuron';
import { getExplanationEmbeddingSql } from '@/lib/external/embedding';
import { NeuronIdentifier } from '@/lib/utils/neuron-identifier';
import { RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';
import { number, object, string, ValidationError } from 'yup';

export const maxDuration = 20;

const explanationSearchReleaseRequestSchema = object({
  releaseName: string().required(),
  query: string().required().min(3),
  offset: number().integer(),
});

const ACTS_TO_RETURN = 3;
/**
 * @swagger
 * /api/explanation/search-release:
 *   post:
 *     summary: Search by Release
 *     tags:
 *       - Explanations
 *     security:
 *       - apiKey: []
 *     description: Search for explanations within a specific release. Takes a query and returns up to 20 results at a time.
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - releaseName
 *               - query
 *             properties:
 *               releaseName:
 *                 type: string
 *                 description: The name of the release to search within.
 *               query:
 *                 type: string
 *                 description: The search query (minimum 3 characters).
 *               offset:
 *                 type: number
 *                 description: Optional. The offset for pagination.
 *     responses:
 *       200:
 *         description: Successful response
 */

export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  const bodyJson = await request.json();
  if (!bodyJson) {
    return NextResponse.json({ message: 'Invalid JSON body' }, { status: 400 });
  }

  try {
    const body = await explanationSearchReleaseRequestSchema.validate(bodyJson);

    const queryEmbeddingStr = await getExplanationEmbeddingSql(body.query);
    const result = await searchExplanationsReleaseVec(
      body.releaseName,
      queryEmbeddingStr,
      body.offset || 0,
      request.user,
    );

    // get the neurons
    const neurons = await getNeurons(
      result.map((r) => new NeuronIdentifier(r.modelId, r.layer, r.index)),
      request.user,
      ACTS_TO_RETURN,
    );

    // add the neurons to the result
    result.forEach((r) => {
      // eslint-disable-next-line no-param-reassign
      r.neuron = neurons.find((n) => n.modelId === r.modelId && n.layer === r.layer && n.index === r.index);
    });

    const hasMore = result.length === MAX_EXPLANATION_SEARCH_RESULTS;

    return NextResponse.json({
      request: body,
      results: result,
      resultsCount: result.length,
      hasMore,
      nextOffset: hasMore ? (body.offset || 0) + result.length : undefined,
    });
  } catch (error) {
    if (error instanceof ValidationError) {
      return NextResponse.json({ message: error.message }, { status: 400 });
    }
    return NextResponse.json({ message: 'Unknown Error' }, { status: 500 });
  }
});
