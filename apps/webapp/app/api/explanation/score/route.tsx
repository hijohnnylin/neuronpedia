import { prisma } from '@/lib/db';
import { getAutoInterpKeyToUse } from '@/lib/db/userSecret';
import { generateScoreEleuther } from '@/lib/external/autointerp-scorer-eleuther';
import { generateScoreRecallAlt } from '@/lib/external/autointerp-scorer-recall-json';
import {
  ERROR_RECALL_ALT_FAILED,
  ERROR_REQUIRES_OPENROUTER,
  requiresOpenRouterForExplanationScoreType,
} from '@/lib/utils/autointerp';
import { RequestAuthedUser, withAuthedUser } from '@/lib/with-user';
import { UserSecretType } from '@prisma/client';
import { NextResponse } from 'next/server';
import { object, string, ValidationError } from 'yup';
import { getUserByName } from '../../../../lib/db/user';

// Hobby plans don't support > 60 seconds
// export const maxDuration = 120;

/**
 * @swagger
 * /api/explanation/score:
 *   post:
 *     summary: Auto-Interp - Score
 *     description: Scores an explanation using a specified scoring model and method. IMPORTANT- Your account needs to have an active OpenRouter and/or OpenAI/Gemini/Anthropic key to do this, which you can set up at `https://www.neuronpedia.org/account` once logged in.
 *     tags:
 *       - Explanations
 *     security:
 *       - apiKey: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - explanationId
 *               - scorerModel
 *               - scorerType
 *             properties:
 *               explanationId:
 *                 type: string
 *                 description: The ID of the explanation to score. You can get the explanationId from the Get Feature endpoint `/api/feature/{modelId}/{layer}/{index}` - it's in the `explanations` array in the response.
 *               scorerModel:
 *                 type: string
 *                 description: The model to use for scoring (e.g., `gpt-4o-mini`, `gemini-1.5-flash`, etc). You can see the full list of supported models on any feature dashboard page in the auto-interp dropdown.
 *                 example: "gpt-4o-mini"
 *               scorerType:
 *                 type: string
 *                 description: The type of scoring method to use (recall_alt, eleuther_fuzz, eleuther_recall, or eleuther_embedding)
 *                 example: "recall_alt"
 *     responses:
 *       200:
 *         description: Successfully scored the explanation
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 score:
 *                   type: object
 *                   properties:
 *                     id:
 *                       type: string
 *                       description: Unique identifier for the score
 *                     initiatedByUserId:
 *                       type: string
 *                       description: ID of the user who initiated the scoring
 *                     value:
 *                       type: number
 *                       description: The calculated score value, from 0 to 1.
 *                     explanationId:
 *                       type: string
 *                       description: ID of the explanation that was scored
 *                     explanationScoreTypeName:
 *                       type: string
 *                       description: The scoring method used (e.g., recall_alt)
 *                     explanationScoreModelName:
 *                       type: string
 *                       description: The model used for scoring (e.g., gpt-4o-mini)
 *                     jsonDetails:
 *                       type: string
 *                       description: The scoring details, as a JSON string. This is a different format depending on the explanationScoreType used.
 *                     createdAt:
 *                       type: string
 *                       format: date-time
 *                       description: Timestamp when the score was created
 *       400:
 *         description: Bad request (invalid input, duplicate score, or missing OpenRouter key)
 *       500:
 *         description: Internal server error
 */

const scoreRequestSchema = object({
  explanationId: string().required(),
  scorerModel: string().required(),
  scorerType: string().required(),
});

const NUM_ACTIVATIONS_TO_LOAD = 20;
const NUM_ZERO_ACTIVATIONS_TO_LOAD_MAX = 5;

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const bodyJson = await request.json();
  const user = await getUserByName(request.user.name);

  if (!bodyJson) {
    return NextResponse.json({ message: 'Invalid JSON body' }, { status: 400 });
  }

  try {
    const body = await scoreRequestSchema.validate(bodyJson);

    // get the scorer type
    const explanationScoreType = await prisma.explanationScoreType.findUnique({
      where: {
        name: body.scorerType,
      },
    });
    if (!explanationScoreType) {
      return NextResponse.json({ message: 'Unsupported explanation score type' }, { status: 400 });
    }

    // get the scorer model type
    const explanationScoreModel = await prisma.explanationScoreModel.findUnique({
      where: {
        name: body.scorerModel,
      },
    });
    if (!explanationScoreModel) {
      return NextResponse.json({ message: 'Unsupported explanation score model' }, { status: 400 });
    }
    const explanationScoreModelOpenRouterId = explanationScoreModel.openRouterModelId;

    // enforce using OpenRouter
    const scorerKeyType = UserSecretType.OPENROUTER;
    let scorerKey = await getAutoInterpKeyToUse(UserSecretType.OPENROUTER, user);
    if (!scorerKey && requiresOpenRouterForExplanationScoreType(explanationScoreType.name)) {
      console.log('no openrouter key found and needed it');
      return NextResponse.json({ message: ERROR_REQUIRES_OPENROUTER }, { status: 400 });
    }
    // silence ts warnings later
    scorerKey = scorerKey || '';

    // if the explanation score type and model already exist for this explanationId, return error
    const existingExplanationScore = await prisma.explanationScore.findFirst({
      where: {
        explanationId: body.explanationId,
        explanationScoreTypeName: explanationScoreType.name,
        explanationScoreModelName: explanationScoreModel.name,
      },
    });
    if (existingExplanationScore) {
      return NextResponse.json(
        {
          message: 'Explanation score using this same method and model already exists',
        },
        { status: 400 },
      );
    }

    // get the explanation
    const explanation = await prisma.explanation.findUnique({
      where: {
        id: body.explanationId,
      },
    });
    if (!explanation) {
      return NextResponse.json({ message: `Explanation not found: ${body.explanationId}` }, { status: 404 });
    }

    // get the neuron and all its activations
    let activations = await prisma.activation.findMany({
      where: {
        neuron: {
          modelId: explanation.modelId,
          layer: explanation.layer,
          index: explanation.index,
        },
      },
      // orderBy: {
      //   maxValue: "desc",
      // },
    });

    // remove duplicates in activations where the tokens.join("") is the same
    activations = [...new Map(activations.map((item) => [item.tokens.join(''), item])).values()];

    // re-sort by maxvalue
    activations.sort((a, b) => b.maxValue - a.maxValue);

    // get all activations where maxValue is 0
    const zeroActivations = activations
      .filter((activation) => activation.maxValue === 0)
      .slice(0, NUM_ZERO_ACTIVATIONS_TO_LOAD_MAX);

    // take the top 10
    activations = activations.slice(0, NUM_ACTIVATIONS_TO_LOAD);

    // remove where maxValue is 0 ( in case we had too many duplicates )
    activations = activations.filter((activation) => activation.maxValue > 0);

    if (explanationScoreType.name === 'recall_alt') {
      try {
        const score = await generateScoreRecallAlt(
          activations,
          zeroActivations,
          explanation,
          explanationScoreModel,
          explanationScoreModelOpenRouterId,
          request.user,
          scorerKeyType,
          scorerKey,
        );

        return NextResponse.json({ score });
      } catch (error) {
        if (error instanceof Error && error.message === ERROR_RECALL_ALT_FAILED) {
          return NextResponse.json({ message: ERROR_RECALL_ALT_FAILED }, { status: 400 });
        }
        throw error;
      }
    }
    if (explanationScoreType.name === 'eleuther_fuzz') {
      const score = await generateScoreEleuther(
        'fuzz',
        activations,
        zeroActivations,
        explanation,
        explanationScoreModel,
        request.user,
        scorerKey,
      );
      return NextResponse.json({ score });
    }
    if (explanationScoreType.name === 'eleuther_recall') {
      const score = await generateScoreEleuther(
        'detection',
        activations,
        zeroActivations,
        explanation,
        explanationScoreModel,
        request.user,
        scorerKey,
      );
      return NextResponse.json({ score });
    }
    if (explanationScoreType.name === 'eleuther_embedding') {
      const score = await generateScoreEleuther(
        'embedding',
        activations,
        zeroActivations,
        explanation,
        explanationScoreModel,
        request.user,
        scorerKey,
      );
      return NextResponse.json({ score });
    }
    return NextResponse.json({ message: 'Unsupported explanation score type' }, { status: 400 });
  } catch (error) {
    if (error instanceof ValidationError) {
      return NextResponse.json({ message: error.message }, { status: 400 });
    }
    return NextResponse.json({ message: 'Unknown Error' }, { status: 500 });
  }
});
