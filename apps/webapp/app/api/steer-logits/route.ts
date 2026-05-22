import { getModelById } from '@/lib/db/model';
import { SteerLogitsRequestSchema, steerLogits } from '@/lib/utils/graph';
import { RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

// for now this just uses the graph server, but we should merge it with the inference server later on to be consistent

/**
 * @swagger
 * /api/steer-logits:
 *   post:
 *     summary: Steer Graph Features (2/2) - Steer + Logits
 *     description: |
 *       Generates default and steered completions from an attribution graph prompt, returning top logits for each generated token.
 *       This is equivalent to the steer button in the /circuit-tracer subgraph (bottom left). In our example, we negatively steer on a "Texas" feature to have the model generate "Albany" instead of "Austin".
 *       To get the token positions to steer on, you first tokenize the prompt using the `/api/graph/tokenize` endpoint.
 *       This endpoint currently supports only `gemma-2-2b` and `qwen3-4b`.
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
 *               - features
 *             properties:
 *               modelId:
 *                 type: string
 *                 enum: [gemma-2-2b, qwen3-4b]
 *                 description: The model id to steer. Only `gemma-2-2b` and `qwen3-4b` are supported currently. Email us if you need other models supported.
 *                 example: gemma-2-2b
 *               sourceSetName:
 *                 type: string
 *                 nullable: true
 *                 description: Optional source set name. If omitted or null, the model's default graph source set is used.
 *                 example: gemmascope-transcoder-16k
 *               prompt:
 *                 type: string
 *                 description: Prompt text to generate from.
 *                 example: "Fact: The capital of the state containing Dallas is"
 *               features:
 *                 type: array
 *                 description: Features to steer or ablate while generating.
 *                 items:
 *                   type: object
 *                   required:
 *                     - layer
 *                     - index
 *                     - token_active_position
 *                     - steer_generated_tokens
 *                     - ablate
 *                   properties:
 *                     layer:
 *                       type: integer
 *                       description: Feature layer.
 *                       example: 14
 *                     index:
 *                       type: integer
 *                       description: Feature index.
 *                       example: 2268
 *                     token_active_position:
 *                       type: integer
 *                       description: Prompt token position where the feature is active.
 *                       example: 9
 *                     steer_position:
 *                       type: integer
 *                       nullable: true
 *                       description: Token position to steer. Use null when steer_generated_tokens is true - if you want to steer both specific steer positions and generated tokens, you must use two feature entries.
 *                       example: 9
 *                     steer_generated_tokens:
 *                       type: boolean
 *                       description: Whether to apply steering to generated tokens.
 *                       example: false
 *                     delta:
 *                       type: number
 *                       nullable: true
 *                       description: Steering delta. Use null when ablating. In the UI, we take the top known activation value for the feature and multiply it by the multiplier on the slider to get the delta. For our example here, we are doing -1.0x steering on the feature, which has a top known activation value of about 200, so the delta is -200.
 *                       example: -200
 *                     ablate:
 *                       type: boolean
 *                       description: Whether to ablate this feature instead of applying `delta`.
 *                       example: false
 *               nTokens:
 *                 type: integer
 *                 description: Number of completion tokens to generate.
 *                 minimum: 1
 *                 maximum: 256
 *                 default: 256
 *                 example: 2
 *               topK:
 *                 type: integer
 *                 description: Number of top logits to return per generated token.
 *                 minimum: 0
 *                 maximum: 10
 *                 default: 5
 *                 example: 1
 *               freezeAttention:
 *                 type: boolean
 *                 description: Whether to freeze attention during steering.
 *                 default: true
 *                 example: true
 *               temperature:
 *                 type: number
 *                 description: Sampling temperature.
 *                 minimum: 0
 *                 maximum: 2
 *                 default: 0.5
 *                 example: 0
 *               freqPenalty:
 *                 type: number
 *                 description: Frequency penalty.
 *                 minimum: -2
 *                 maximum: 2
 *                 default: 1
 *                 example: 0
 *               seed:
 *                 type: integer
 *                 nullable: true
 *                 description: Random seed. Use null for no fixed seed.
 *                 default: 16
 *           example:
 *             modelId: gemma-2-2b
 *             sourceSetName: gemmascope-transcoder-16k
 *             prompt: "Fact: The capital of the state containing Dallas is"
 *             features:
 *               - layer: 14
 *                 index: 2268
 *                 token_active_position: 9
 *                 steer_position: 9
 *                 steer_generated_tokens: false
 *                 delta: -200
 *                 ablate: false
 *             nTokens: 2
 *             topK: 1
 *             freezeAttention: true
 *             temperature: 0
 *             freqPenalty: 0
 *             seed: 16
 *     responses:
 *       200:
 *         description: Default and steered generations with top logits by token.
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 DEFAULT_GENERATION:
 *                   type: string
 *                   description: Unsteered generated text.
 *                 STEERED_GENERATION:
 *                   type: string
 *                   description: Steered generated text.
 *                 DEFAULT_LOGITS_BY_TOKEN:
 *                   type: array
 *                   description: Top logits for each unsteered generated token.
 *                   items:
 *                     type: object
 *                     properties:
 *                       token:
 *                         type: string
 *                         description: Generated token.
 *                       top_logits:
 *                         type: array
 *                         items:
 *                           type: object
 *                           properties:
 *                             prob:
 *                               type: number
 *                               description: Token probability.
 *                             token:
 *                               type: string
 *                               description: Candidate token.
 *                 STEERED_LOGITS_BY_TOKEN:
 *                   type: array
 *                   description: Top logits for each steered generated token.
 *                   items:
 *                     type: object
 *                     properties:
 *                       token:
 *                         type: string
 *                         description: Generated token.
 *                       top_logits:
 *                         type: array
 *                         items:
 *                           type: object
 *                           properties:
 *                             prob:
 *                               type: number
 *                               description: Token probability.
 *                             token:
 *                               type: string
 *                               description: Candidate token.
 *       400:
 *         description: Invalid request or missing source set.
 */
export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  const body = await request.json();

  const validatedBody = SteerLogitsRequestSchema.validateSync(body);

  if (!validatedBody.sourceSetName) {
    const model = await getModelById(validatedBody.modelId);
    validatedBody.sourceSetName = model?.defaultGraphSourceSetName;
    if (!validatedBody.sourceSetName) {
      return NextResponse.json(
        {
          error: 'Source Set Missing',
          message: `The model ${validatedBody.modelId} has no default graph source set, so you must provide one in the sourceSetName parameter.`,
        },
        { status: 400 },
      );
    }
  }

  const {
    modelId,
    sourceSetName,
    prompt,
    features,
    nTokens,
    topK,
    freezeAttention,
    temperature,
    freqPenalty,
    seed,
    steeredOutputOnly,
  } = validatedBody;

  const response = await steerLogits(
    modelId,
    sourceSetName,
    prompt,
    features,
    nTokens,
    topK,
    freezeAttention,
    temperature,
    freqPenalty,
    seed,
    steeredOutputOnly,
  );

  return NextResponse.json(response);
});
