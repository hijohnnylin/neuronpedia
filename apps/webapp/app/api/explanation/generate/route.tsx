import { prisma } from '@/lib/db';
import { getUserByName } from '@/lib/db/user';
import { getAutoInterpKeyToUse } from '@/lib/db/userSecret';
import { generateExplanationOaiTokenActivationPair } from '@/lib/external/autointerp-explainer';
import { generateExplanationOaiAttentionHead } from '@/lib/external/autointerp-explainer-att';
import { generateExplanationEleutherActsTop20 } from '@/lib/external/autointerp-explainer-eleuther';
import { generateExplanationNpMaxActLogits } from '@/lib/external/autointerp-explainer-logits';
import { getExplanationEmbeddingSql } from '@/lib/external/embedding';
import {
  ERROR_NO_AUTOINTERP_KEY,
  ERROR_REQUIRES_OPENROUTER,
  getAutoInterpModelTypeFromModelId,
  getKeyTypeForAutoInterpModelType,
  requiresOpenRouterForExplanationType,
} from '@/lib/utils/autointerp';
import { RequestAuthedUser, withAuthedUser } from '@/lib/with-user';
import { UserSecretType } from '@prisma/client';
import { NextResponse } from 'next/server';
import { boolean, number, object, string, ValidationError } from 'yup';

// Hobby plans don't support > 60 seconds
// export const maxDuration = 120;

const explainRequestSchema = object({
  modelId: string().required(),
  layer: string().required(),
  index: number().integer().required(),
  explanationType: string().required(),
  explanationModelName: string().required(),
  useOpenRouter: boolean().optional().default(false),
});

const ACTIVATIONS_TO_LOAD = 10;

/**
 * @swagger
 * /api/explanation/generate:
 *   post:
 *     summary: Auto-Interp - Generate
 *     description: Generates an explanation for a specific feature using the specified explanation type and model. IMPORTANT- this requires you to set your OpenAI/Anthropic/Google auto-interp API keys under Neuronpedia Settings.
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
 *               - modelId
 *               - layer
 *               - index
 *               - explanationType
 *               - explanationModelName
 *             properties:
 *               modelId:
 *                 type: string
 *                 description: The ID of the model the feature belongs to
 *               layer:
 *                 type: string
 *                 description: The layer of the feature
 *               index:
 *                 type: number
 *                 description: The index of the feature
 *               explanationType:
 *                 type: string
 *                 description: The type of explanation to generate. Currently supported types are "oai_attention-head" and "oai_token-act-pair".
 *               explanationModelName:
 *                 type: string
 *                 description: "The name of the model to use for generating the explanation. See full list of supported models in the auto-interpdropdown on a Neuronpedia dashboard: https://neuronpedia.org/gpt2-small/8-res-jb/55"
 *     responses:
 *       200:
 *         description: Successfully generated explanation
 */

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const bodyJson = await request.json();
  if (!bodyJson) {
    return NextResponse.json({ message: 'Invalid JSON body' }, { status: 400 });
  }

  const user = await getUserByName(request.user.name);

  try {
    const body = await explainRequestSchema.validate(bodyJson);

    // get the explanation type
    const explanationType = await prisma.explanationType.findUnique({
      where: {
        name: body.explanationType,
      },
    });
    if (!explanationType) {
      return NextResponse.json({ message: 'Unsupported explanation type' }, { status: 400 });
    }

    // ensure the explanation model name is supported
    const explanationModel = await prisma.explanationModelType.findUnique({
      where: {
        name: body.explanationModelName,
      },
    });
    if (!explanationModel) {
      return NextResponse.json({ message: 'Unsupported explanation model' }, { status: 400 });
    }
    const explanationModelOpenRouterId = explanationModel.openRouterModelId;

    // get explanations for this neuron and if the explanation type and model already exist, return error
    const existingExplanation = await prisma.explanation.findFirst({
      where: {
        modelId: body.modelId,
        layer: body.layer,
        index: body.index.toString(),
        typeName: explanationType.name,
        explanationModelName: explanationModel.name,
      },
    });
    if (existingExplanation) {
      return NextResponse.json(
        {
          message: 'An auto-interp with this explanation type and model already exists.',
        },
        { status: 400 },
      );
    }

    // ensure that the scorer model type is supported and user has a key for it
    // if modelType is unknown (eg llama) we use openrouter
    const explainerModelType = getAutoInterpModelTypeFromModelId(explanationModel.name || '');
    let explainerKey: string | null = null;
    let explainerKeyType: UserSecretType = getKeyTypeForAutoInterpModelType(explainerModelType);
    let key = await getAutoInterpKeyToUse(explainerKeyType, user);
    console.log(`explainerKeyType: ${explainerKeyType}`);
    if (!key) {
      console.log('no key found, falling back to openrouter');
      // fall back to look for openrouter key
      explainerKeyType = UserSecretType.OPENROUTER;
      key = await getAutoInterpKeyToUse(explainerKeyType, user);

      if (!key) {
        if (requiresOpenRouterForExplanationType(explanationType.name)) {
          return NextResponse.json({ message: ERROR_REQUIRES_OPENROUTER }, { status: 400 });
        }
        console.log('no openrouter key found either');
        return NextResponse.json({ message: ERROR_NO_AUTOINTERP_KEY }, { status: 400 });
      }
    }
    explainerKey = key;

    if (explanationType.name === 'eleuther_acts_top20') {
      // this requires openrouter key
      explainerKeyType = UserSecretType.OPENROUTER;
      explainerKey = (await getAutoInterpKeyToUse(explainerKeyType, user)) || null;
      if (!explainerKey) {
        return NextResponse.json({ message: ERROR_REQUIRES_OPENROUTER }, { status: 400 });
      }

      let activations = await prisma.activation.findMany({
        where: {
          neuron: {
            modelId: body.modelId,
            layer: body.layer,
            index: body.index.toString(),
          },
        },
      });

      // remove duplicates in activations where the tokens.join("") is the same
      activations = [...new Map(activations.map((item) => [item.tokens.join(''), item])).values()];

      // re-sort by maxvalue
      activations.sort((a, b) => b.maxValue - a.maxValue);

      // take the top 20
      activations = activations.slice(0, 20);

      if (activations.length === 0) {
        return NextResponse.json({ message: 'No activations found for this neuron' }, { status: 400 });
      }

      const explainResult = await generateExplanationEleutherActsTop20(activations, explanationModel, explainerKey);

      if (!explainResult) {
        throw new Error('Error getting result');
      }
      const exp = await prisma.explanation.create({
        data: {
          modelId: body.modelId,
          layer: body.layer,
          index: body.index.toString(),
          description: explainResult,
          authorId: user.id,
          triggeredByUserId: user.id,
          typeName: explanationType.name,
          explanationModelName: explanationModel.name,
        },
      });

      const embeddingStr = await getExplanationEmbeddingSql(explainResult);
      await prisma.$queryRaw`
      UPDATE "Explanation"
      SET embedding = ${embeddingStr}::vector
      WHERE id = ${exp.id}
    `;

      return NextResponse.json({ explanation: exp });
    }
    if (explanationType.name === 'oai_attention-head') {
      let activations = await prisma.activation.findMany({
        where: {
          neuron: {
            modelId: body.modelId,
            layer: body.layer,
            index: body.index.toString(),
          },
        },
      });

      // remove duplicates in activations where the tokens.join("") is the same
      activations = [...new Map(activations.map((item) => [item.tokens.join(''), item])).values()];

      // re-sort by maxvalue
      activations.sort((a, b) => b.maxValue - a.maxValue);

      // take the top 10
      activations = activations.slice(0, ACTIVATIONS_TO_LOAD);

      if (activations.length === 0) {
        return NextResponse.json({ message: 'No activations found for this neuron' }, { status: 400 });
      }

      const explainResult = await generateExplanationOaiAttentionHead(
        activations,
        explanationModel,
        explanationModelOpenRouterId,
        explainerModelType,
        explainerKeyType,
        explainerKey,
      );

      if (!explainResult) {
        throw new Error('Error getting result');
      }

      const exp = await prisma.explanation.create({
        data: {
          modelId: body.modelId,
          layer: body.layer,
          index: body.index.toString(),
          description: explainResult,
          authorId: user.id,
          triggeredByUserId: user.id,
          typeName: explanationType.name,
          explanationModelName: explanationModel.name,
        },
      });

      const embeddingStr = await getExplanationEmbeddingSql(explainResult);
      await prisma.$queryRaw`
      UPDATE "Explanation"
      SET embedding = ${embeddingStr}::vector
      WHERE id = ${exp.id}
    `;

      return NextResponse.json({ explanation: exp });
    }
    if (explanationType.name === 'np_max-act-logits') {
      let activations = await prisma.activation.findMany({
        where: {
          neuron: {
            modelId: body.modelId,
            layer: body.layer,
            index: body.index.toString(),
          },
        },
      });

      // remove duplicates in activations where the tokens.join("") is the same
      activations = [...new Map(activations.map((item) => [item.tokens.join(''), item])).values()];

      // re-sort by maxvalue
      activations.sort((a, b) => b.maxValue - a.maxValue);

      // take the top 10
      activations = activations.slice(0, ACTIVATIONS_TO_LOAD);
      if (activations.length === 0) {
        return NextResponse.json({ message: 'No activations found for this neuron' }, { status: 400 });
      }
      // get the feature itself
      const feature = await prisma.neuron.findUnique({
        where: {
          modelId_layer_index: {
            modelId: body.modelId,
            layer: body.layer,
            index: body.index.toString(),
          },
        },
      });
      if (!feature) {
        return NextResponse.json({ message: 'Feature not found' }, { status: 400 });
      }
      const explainResult = await generateExplanationNpMaxActLogits(
        activations,
        feature,
        explanationModel,
        explanationModelOpenRouterId,
        explainerModelType,
        explainerKeyType,
        explainerKey,
      );
      if (!explainResult) {
        throw new Error('Error getting result');
      }
      const exp = await prisma.explanation.create({
        data: {
          modelId: body.modelId,
          layer: body.layer,
          index: body.index.toString(),
          description: explainResult,
          authorId: user.id,
          triggeredByUserId: user.id,
          typeName: explanationType.name,
          explanationModelName: explanationModel.name,
        },
      });

      const embeddingStr = await getExplanationEmbeddingSql(explainResult);
      await prisma.$queryRaw`
      UPDATE "Explanation"
      SET embedding = ${embeddingStr}::vector
      WHERE id = ${exp.id}
    `;

      return NextResponse.json({ explanation: exp });
    }
    if (explanationType.name === 'oai_token-act-pair') {
      let activations = await prisma.activation.findMany({
        where: {
          neuron: {
            modelId: body.modelId,
            layer: body.layer,
            index: body.index.toString(),
          },
        },
      });

      // remove duplicates in activations where the tokens.join("") is the same
      activations = [...new Map(activations.map((item) => [item.tokens.join(''), item])).values()];

      // re-sort by maxvalue
      activations.sort((a, b) => b.maxValue - a.maxValue);

      // take the top 10
      activations = activations.slice(0, ACTIVATIONS_TO_LOAD);
      if (activations.length === 0) {
        return NextResponse.json({ message: 'No activations found for this neuron' }, { status: 400 });
      }

      const explainResult = await generateExplanationOaiTokenActivationPair(
        activations,
        explanationModel,
        explanationModelOpenRouterId,
        explainerModelType,
        explainerKeyType,
        explainerKey,
      );

      if (!explainResult) {
        throw new Error('Error getting result');
      }

      const exp = await prisma.explanation.create({
        data: {
          modelId: body.modelId,
          layer: body.layer,
          index: body.index.toString(),
          description: explainResult,
          authorId: user.id,
          triggeredByUserId: user.id,
          typeName: explanationType.name,
          explanationModelName: explanationModel.name,
        },
      });

      const embeddingStr = await getExplanationEmbeddingSql(explainResult);
      await prisma.$queryRaw`
      UPDATE "Explanation"
      SET embedding = ${embeddingStr}::vector
      WHERE id = ${exp.id}
    `;

      return NextResponse.json({ explanation: exp });
    }
    return NextResponse.json({ message: 'Unsupported explanation type' }, { status: 400 });
  } catch (error) {
    if (error instanceof ValidationError) {
      return NextResponse.json({ message: error.message }, { status: 400 });
    }
    return NextResponse.json({ message: 'Unknown Error' }, { status: 500 });
  }
});
