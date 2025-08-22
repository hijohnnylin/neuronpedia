import { prisma } from '@/lib/db';
import { getSourceSet } from '@/lib/db/source';
import { getUserById } from '@/lib/db/user';
import { env } from '@/lib/env';
import { getOAIEmbedding } from '@/lib/external/embedding';
import { RequestAuthedUser } from '@/lib/types/auth';
import { getSourceSetNameFromSource } from '@/lib/utils/source';
import { withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';
import pgvector from 'pgvector';
import * as yup from 'yup';
import { array, number, object, string } from 'yup';

const uploadBatchSchema = object({
  modelId: string().required().lowercase(),
  source: string().required().lowercase(),
  features: array()
    .of(
      object({
        index: number().integer().required().min(0).max(10_000_000),
        density: number().optional().min(0).max(1),
        topLogits: array()
          .of(
            object({
              token: string().required(),
              value: number().required(),
            }),
          )
          .optional()
          .nullable()
          .max(20),
        bottomLogits: array()
          .of(
            object({
              token: string().required(),
              value: number().required(),
            }),
          )
          .optional()
          .nullable()
          .max(20),
        explanations: array()
          .of(
            object({
              text: string().required().min(1).max(512),
              methodName: string().optional().nullable().lowercase(),
              modelName: string().optional().nullable().lowercase(),
            }),
          )
          .optional(),
        activations: array()
          .of(
            object({
              tokens: array().of(string().required()).required().min(1).max(1024),
              values: array().of(number().required()).required().min(1).max(1024),
              quantileMax: number().optional().nullable().max(99999999),
              quantileMin: number().optional().nullable().max(99999999),
              quantileFraction: number().optional().nullable().min(0).max(1),
            }),
          )
          .required()
          .min(1),
      }),
    )
    .required()
    .min(1)
    .max(128),
});

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const body = await request.json();
  let parsedBody;
  try {
    parsedBody = await uploadBatchSchema.validate(body);
  } catch (error) {
    if (error instanceof yup.ValidationError) {
      return NextResponse.json({ error: error.message, path: error.path }, { status: 400 });
    }
    throw error;
  }

  // get the user
  const user = await getUserById(request.user.id);
  if (!user) {
    throw new Error('User not found.');
  }

  // get the sourceSet from sourceId
  const sourceSetName = getSourceSetNameFromSource(parsedBody.source);
  // look up the sourceSet
  const sourceSet = await getSourceSet(parsedBody.modelId, sourceSetName, request.user);
  if (!sourceSet) {
    throw new Error('Source set not found.');
  }

  // ensure the user is the creator of the sourceSet
  if (sourceSet.creatorId !== request.user.id) {
    throw new Error('User is not the creator of this source set.');
  }

  // check that the source exists
  const source = sourceSet.sources.find((src) => src.id === parsedBody.source);
  if (!source) {
    throw new Error('Source not found.');
  }

  // ensure activations tokens length matches values length
  // iterate through the features
  for (const feature of parsedBody.features) {
    // ensure activations tokens length matches values length
    for (const activation of feature.activations) {
      if (activation.tokens.length !== activation.values.length) {
        throw new Error(
          `One of the activations tokens lengths does not match its values length. Activation text: ${activation.tokens.join(', ')}`,
        );
      }
    }
  }

  // ensure all indexes are unique
  const uniqueIndexes = new Set(parsedBody.features.map((f) => f.index));
  if (uniqueIndexes.size !== parsedBody.features.length) {
    throw new Error('All indexes must be unique.');
  }

  // if we have get all explanation methodNames as a set
  const explanationMethodNames = new Set(
    parsedBody.features.flatMap((f) => f.explanations?.map((e) => e.methodName) || []).filter((m) => m !== undefined),
  );

  // validate that all method names exist as ExplanationType
  const explanationTypes = await prisma.explanationType.findMany();
  const explanationTypeNames = new Set(explanationTypes.map((t) => t.name));

  // Check that every method name is a known explanation type
  const missingMethodNames = Array.from(explanationMethodNames).filter(
    (m) => m !== null && !explanationTypeNames.has(m),
  );
  if (missingMethodNames.length > 0) {
    throw new Error(
      `Invalid Explanation Method Names: ${missingMethodNames.join(', ')}. Valid methods are: ${Array.from(explanationTypeNames).join(', ')}.\n\nIf you want to add your explanation method type, contact support@neuronpedia.org.`,
    );
  }

  // get all explanation modelNames as a set
  const explanationModelNames = new Set(
    parsedBody.features.flatMap((f) => f.explanations?.map((e) => e.modelName) || []).filter((m) => m !== undefined),
  );

  // validate that all model names exist as ExplanationModel
  const explanationModelTypes = await prisma.explanationModelType.findMany();
  const explanationModelTypeNames = new Set(explanationModelTypes.map((t) => t.name));

  // Check that every model name is a known explanation model type
  const missingModelNames = Array.from(explanationModelNames).filter(
    (m) => m !== null && !explanationModelTypeNames.has(m),
  );
  if (missingModelNames.length > 0) {
    throw new Error(
      `Invalid Explanation Model Names: ${missingModelNames.join(', ')}. Valid models are: ${Array.from(explanationModelTypeNames).join(', ')}.\n\nIf you want to add your explanation model type, contact support@neuronpedia.org.`,
    );
  }

  // if we have explanations, batch get the embeddings
  // first get all the explanations into an array, then get all the embeddings
  const explanationTexts: string[] = parsedBody.features.flatMap((f) => f.explanations?.map((e) => e.text) || []);
  const explanationEmbeddings = await getOAIEmbedding('text-embedding-3-large', 256, explanationTexts);
  // ensure we got an array of arrays
  if (!Array.isArray(explanationEmbeddings) || !explanationEmbeddings.every((embedding) => Array.isArray(embedding))) {
    throw new Error('Failed to get valid embeddings for explanations - expected array of arrays.');
  }

  if (explanationEmbeddings.length !== explanationTexts.length) {
    throw new Error('Failed to get all embeddings for explanations.');
  }
  // map the explanations to the embeddings
  const explanationEmbeddingsMap = new Map(explanationTexts.map((e, i) => [e, explanationEmbeddings[i]]));

  // convert into neurons with explanation and activations
  const neuronsToCreate: {
    modelId: string;
    layer: string;
    index: string;
    creatorId: string;
    sourceSetName: string;
    frac_nonzero: number;
    hasVector: boolean;
    maxActApprox: number;
  }[] = [];
  const explanationsToCreate: {
    description: string;
    authorId: string;
    index: string;
    layer: string;
    modelId: string;
    typeName?: string;
    modelName?: string;
  }[] = [];
  const activationsToCreate: {
    tokens: string[];
    values: number[];
    index: string;
    layer: string;
    modelId: string;
    creatorId: string;
    maxValueTokenIndex: number;
    maxValue: number;
    minValue: number;
    dataSource: string;
    dataIndex: string;
  }[] = [];

  // if we have topLogits and bottomLogits, parse them as:

  for (const feature of parsedBody.features) {
    const neuron = {
      modelId: parsedBody.modelId,
      layer: parsedBody.source,
      index: feature.index.toString(),
      creatorId: request.user.id,
      sourceSetName,
      frac_nonzero: feature.density ?? -1,
      maxActApprox: 0,
      hasVector: false,
      pos_str: [] as string[],
      pos_values: [] as number[],
      neg_str: [] as string[],
      neg_values: [] as number[],
    };
    // add topLogits and bottomLogits if they exist
    if (feature.topLogits) {
      neuron.pos_str = feature.topLogits.map((logit) => logit.token);
      neuron.pos_values = feature.topLogits.map((logit) => logit.value);
    }
    if (feature.bottomLogits) {
      neuron.neg_str = feature.bottomLogits.map((logit) => logit.token);
      neuron.neg_values = feature.bottomLogits.map((logit) => logit.value);
    }

    // add explanations if it exists
    if (feature.explanations) {
      for (const explanationItem of feature.explanations) {
        const explanation = {
          description: explanationItem.text,
          authorId: request.user.id,
          index: feature.index.toString(),
          layer: parsedBody.source,
          modelId: parsedBody.modelId,
          typeName: explanationItem.methodName ?? undefined,
          modelName: explanationItem.modelName ?? undefined,
        };
        explanationsToCreate.push(explanation);
      }
    }

    // add all activations
    let maxActFound = 0;
    for (const activation of feature.activations) {
      const act = {
        tokens: activation.tokens,
        values: activation.values,
        index: feature.index.toString(),
        layer: parsedBody.source,
        modelId: parsedBody.modelId,
        creatorId: env.PUBLIC_ACTIVATIONS_USER_IDS[0],
        maxValueTokenIndex: activation.values.indexOf(Math.max(...activation.values)),
        maxValue: Math.max(...activation.values),
        minValue: Math.min(...activation.values),
        dataSource: `User ${user.name}`,
        dataIndex: user.id,
        binMin: activation.quantileMin,
        binMax: activation.quantileMax,
        binContains: activation.quantileFraction,
      };
      maxActFound = Math.max(maxActFound, act.maxValue);
      activationsToCreate.push(act);
    }
    neuron.maxActApprox = maxActFound;

    neuronsToCreate.push(neuron);
  }

  try {
    await prisma.$transaction(async (tx) => {
      await tx.neuron.createMany({
        data: neuronsToCreate,
        skipDuplicates: true,
      });
      const created = await tx.explanation.createManyAndReturn({
        data: explanationsToCreate,
        skipDuplicates: true,
      });
      // Update embeddings for all created explanations in one SQL query
      if (created.length > 0) {
        const embeddingUpdates = await Promise.all(
          created.map(async (explanation) => {
            const embedding = explanationEmbeddingsMap.get(explanation.description);
            if (!embedding) {
              throw new Error(`Could not find embedding for explanation: ${explanation.description}`);
            }
            return {
              id: explanation.id,
              embedding: pgvector.toSql(embedding),
            };
          }),
        );

        const updateQuery = `
          UPDATE "Explanation" 
          SET embedding = CASE id
            ${embeddingUpdates.map((update) => `WHEN '${update.id}' THEN '${update.embedding}'::vector`).join(' ')}
          END
          WHERE id IN ('${embeddingUpdates.map((update) => update.id).join("','")}')
        `;

        await tx.$executeRawUnsafe(updateQuery);
      }

      await tx.activation.createMany({
        data: activationsToCreate,
        skipDuplicates: true,
      });
    });
  } catch (error) {
    console.error('Database transaction failed:', error);
    return NextResponse.json(
      { error: 'Failed to create features. Please try again or contact support@neuronpedia.org.' },
      { status: 500 },
    );
  }

  return NextResponse.json({
    message: `${neuronsToCreate.length} features created.`,
  });
});
