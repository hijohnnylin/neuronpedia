import { getModelById } from '@/lib/db/model';
import {
  getGraphTokenize,
  GRAPH_DESIREDLOGITPROB_DEFAULT,
  GRAPH_DESIREDLOGITPROB_MAX,
  GRAPH_DESIREDLOGITPROB_MIN,
  GRAPH_GENERATION_ENABLED_MODELS,
  GRAPH_MAXNLOGITS_DEFAULT,
  GRAPH_MAXNLOGITS_MAX,
  GRAPH_MAXNLOGITS_MIN,
} from '@/lib/utils/graph';
import { NextResponse } from 'next/server';
import * as yup from 'yup';

const MAX_TOKENIZE_CHARS = 10000;

const tokenizeRequestSchema = yup.object({
  prompt: yup.string().max(MAX_TOKENIZE_CHARS).min(1).required(),
  modelId: yup.string().min(1).required().oneOf(GRAPH_GENERATION_ENABLED_MODELS),
  sourceSetName: yup.string().nullable(),
  maxNLogits: yup
    .number()
    .integer('Must be an integer.')
    .min(GRAPH_MAXNLOGITS_MIN, `Must be at least ${GRAPH_MAXNLOGITS_MIN}.`)
    .max(GRAPH_MAXNLOGITS_MAX, `Must be at most ${GRAPH_MAXNLOGITS_MAX}.`)
    .default(GRAPH_MAXNLOGITS_DEFAULT)
    .required('This field is required.'),
  desiredLogitProb: yup
    .number()
    .min(GRAPH_DESIREDLOGITPROB_MIN, `Must be at least ${GRAPH_DESIREDLOGITPROB_MIN}.`)
    .max(GRAPH_DESIREDLOGITPROB_MAX, `Must be at most ${GRAPH_DESIREDLOGITPROB_MAX}.`)
    .default(GRAPH_DESIREDLOGITPROB_DEFAULT)
    .required('This field is required.'),
});

/**
 * @swagger
 * /api/graph/tokenize:
 *   post:
 *     summary: Steer Graph Features (1/2) - Tokenize
 *     description: |
 *       Tokenizes a prompt using the graph server for the requested model/source set and returns token positions plus salient next-token logits.
 *
 *       Use this before `/api/steer-logits` to identify the token positions you want to steer. The zero-based index of a token in `input_tokens` is the position to use for `token_active_position` and, when steering a specific prompt token, `steer_position`.
 *     tags:
 *       - Attribution Graphs
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - modelId
 *               - prompt
 *               - maxNLogits
 *               - desiredLogitProb
 *             properties:
 *               modelId:
 *                 type: string
 *                 enum: [gemma-2-2b, gemma-3-4b-it, qwen3-4b, qwen3-1.7b]
 *                 description: The model id to tokenize with.
 *                 example: gemma-2-2b
 *               sourceSetName:
 *                 type: string
 *                 nullable: true
 *                 description: Optional source set name. If omitted or null, the model's default graph source set is used.
 *                 example: gemmascope-transcoder-16k
 *               prompt:
 *                 type: string
 *                 description: Prompt text to tokenize.
 *                 maxLength: 10000
 *                 example: "Fact: The capital of the state containing Dallas is"
 *               maxNLogits:
 *                 type: integer
 *                 description: Maximum number of salient logits to return.
 *                 minimum: 5
 *                 maximum: 15
 *                 default: 10
 *                 example: 10
 *               desiredLogitProb:
 *                 type: number
 *                 description: Desired cumulative probability threshold for salient logits.
 *                 minimum: 0.6
 *                 maximum: 0.99
 *                 default: 0.95
 *                 example: 0.95
 *           example:
 *             modelId: gemma-2-2b
 *             sourceSetName: gemmascope-transcoder-16k
 *             prompt: "Fact: The capital of the state containing Dallas is"
 *             maxNLogits: 10
 *             desiredLogitProb: 0.95
 *     responses:
 *       200:
 *         description: Tokenized prompt and salient next-token logits.
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 prompt:
 *                   type: string
 *                   description: The prompt that was tokenized.
 *                 input_tokens:
 *                   type: array
 *                   description: Prompt tokens in order. Use each token's zero-based array index as its position for steering.
 *                   items:
 *                     type: string
 *                 salient_logits:
 *                   type: array
 *                   description: Salient next-token logits from the forward pass.
 *                   items:
 *                     type: object
 *                     properties:
 *                       token:
 *                         type: string
 *                         description: Candidate next token.
 *                       token_id:
 *                         type: integer
 *                         description: Candidate token id.
 *                       probability:
 *                         type: number
 *                         description: Candidate token probability.
 *                 total_salient_tokens:
 *                   type: integer
 *                   description: Number of salient logits returned.
 *                 cumulative_probability:
 *                   type: number
 *                   description: Cumulative probability covered by the salient logits.
 *       400:
 *         description: Invalid request, invalid JSON, validation error, or missing source set.
 *       502:
 *         description: Graph server error.
 *       500:
 *         description: Failed to tokenize text.
 */
export async function POST(request: Request) {
  try {
    let body;
    try {
      body = await request.json();
    } catch (error) {
      return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
    }

    const validatedData = await tokenizeRequestSchema.validate(body);

    if (!validatedData.sourceSetName) {
      const model = await getModelById(validatedData.modelId);
      validatedData.sourceSetName = model?.defaultGraphSourceSetName;
      if (!validatedData.sourceSetName) {
        return NextResponse.json(
          {
            error: 'Source Set Missing',
            message: `The model ${validatedData.modelId} has no default graph source set, so you must provide one in the sourceSetName parameter.`,
          },
          { status: 400 },
        );
      }
    }

    const tokenizedResponse = await getGraphTokenize(
      validatedData.prompt,
      validatedData.maxNLogits,
      validatedData.desiredLogitProb,
      validatedData.modelId,
      validatedData.sourceSetName,
    );

    // console.log('tokenizedResponse', tokenizedResponse);
    return NextResponse.json(tokenizedResponse, { status: 200 });
  } catch (error) {
    console.error('Error in tokenize route:', error);
    if (error instanceof yup.ValidationError) {
      return NextResponse.json({ error: 'Validation error', details: error.errors }, { status: 400 });
    }
    const errorMessage = error instanceof Error ? error.message : String(error);
    const isGraphServerError = errorMessage.includes('Graph server') || errorMessage.includes('External API');
    return NextResponse.json(
      {
        error: isGraphServerError ? 'Graph server error' : 'Failed to tokenize text',
        message: errorMessage,
      },
      { status: isGraphServerError ? 502 : 500 },
    );
  }
}
