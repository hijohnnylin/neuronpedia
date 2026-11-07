import { badRequest, forbidden, notFound } from '@/lib/api-error';
import { prisma } from '@/lib/db';
import { ensureNeuronRowForFeature } from '@/lib/db/neuron-dashboard';
import { userCanAccessModelAndSourceSet } from '@/lib/db/userCanAccess';
import { NEXT_PUBLIC_URL } from '@/lib/env';
import { getSourceSetNameFromSource } from '@/lib/utils/source';
import { AuthenticatedUser } from '@/lib/with-user';
import { ExplanationAspect, ExplanationPole, Prisma } from '@prisma/client';
import { array, InferType, mixed, number, object, string } from 'yup';

export const UPLOAD_EXPLANATIONS_MAX = 500;
const MAX_DESCRIPTION_LENGTH = 2000;
const MAX_METADATA_BYTES = 16_000;
const NEURON_LOOKUP_CONCURRENCY = 16;

const HTTP_URL = /^https?:\/\/\S+$/;

const httpUrl = () =>
  string()
    .trim()
    .max(2000)
    .matches(HTTP_URL, { message: '${path} must be an http or https URL', excludeEmptyString: true });

export const uploadedExplanationSchema = object({
  modelId: string().trim().required(),
  layer: string().trim().required(),
  index: mixed<string | number>()
    .required()
    .transform((v) => (typeof v === 'number' ? String(v) : v))
    .test('index', '${path} must be a non-negative integer', (v) => typeof v === 'string' && /^\d+$/.test(v)),
  description: string().trim().required().max(MAX_DESCRIPTION_LENGTH),
  typeName: string().trim().required(),
  explanationModelName: string().trim().optional(),
  pole: mixed<ExplanationPole>().oneOf(Object.values(ExplanationPole)).optional(),
  aspect: mixed<ExplanationAspect>().oneOf(Object.values(ExplanationAspect)).optional(),
  steerStrength: number()
    .optional()
    .test(
      'steering-only',
      '${path} needs aspect STEERING',
      (v, ctx) => v === undefined || ctx.parent.aspect === ExplanationAspect.STEERING,
    ),
  sourceUrl: httpUrl().optional(),
  contributorName: string().trim().max(200).optional(),
  contributorUrl: httpUrl().optional(),
  metadata: mixed<Record<string, unknown>>()
    .optional()
    .test(
      'object',
      '${path} must be a JSON object',
      (v) => v === undefined || (typeof v === 'object' && v !== null && !Array.isArray(v)),
    )
    .test(
      'size',
      `\${path} must be at most ${MAX_METADATA_BYTES} bytes as JSON`,
      (v) => v === undefined || JSON.stringify(v).length <= MAX_METADATA_BYTES,
    ),
});

export const uploadExplanationsSchema = object({
  explanations: array().of(uploadedExplanationSchema.required()).required().min(1).max(UPLOAD_EXPLANATIONS_MAX),
});

export type UploadedExplanation = InferType<typeof uploadedExplanationSchema>;

export type UploadExplanationsResult = {
  created: { id: string; url: string }[];
  skipped: number;
};

const featureKey = (e: { modelId: string; layer: string; index: string }) => `${e.modelId}@${e.layer}:${e.index}`;

// Same unit, type, text, sign and strength: uploading it again changes nothing.
const sameKey = (e: {
  modelId: string;
  layer: string;
  index: string;
  typeName: string | null;
  description: string;
  pole?: ExplanationPole | null;
  aspect?: ExplanationAspect | null;
  steerStrength?: number | null;
}) =>
  JSON.stringify([featureKey(e), e.typeName, e.description, e.pole ?? null, e.aspect ?? null, e.steerStrength ?? null]);

const unique = <T>(values: T[]) => Array.from(new Set(values));

/**
 * Adds explanations as the uploader. All checks run before any write, so a batch is all or nothing.
 * The uploader needs canUploadExplanations, must own each explanation type, and must be able to see each source.
 */
export async function uploadExplanations(
  items: UploadedExplanation[],
  user: AuthenticatedUser,
): Promise<UploadExplanationsResult> {
  const userDb = await prisma.user.findUnique({
    where: { id: user.id },
    select: { admin: true, canUploadExplanations: true },
  });
  if (!userDb?.admin && !userDb?.canUploadExplanations) {
    throw forbidden('Your account cannot upload explanations. Contact us to request access.');
  }

  const typeNames = unique(items.map((e) => e.typeName));
  const types = await prisma.explanationType.findMany({
    where: { name: { in: typeNames } },
    select: { name: true, creatorId: true },
  });
  const missingTypes = typeNames.filter((n) => !types.some((t) => t.name === n));
  if (missingTypes.length > 0) {
    throw badRequest(`Unknown explanation type: ${missingTypes.join(', ')}. Ask us to create your type first.`);
  }
  const notOwned = types.filter((t) => t.creatorId !== user.id).map((t) => t.name);
  if (!userDb.admin && notOwned.length > 0) {
    throw forbidden(`You can only upload to explanation types that you own. Not yours: ${notOwned.join(', ')}`);
  }

  const modelNames = unique(items.map((e) => e.explanationModelName).filter((n): n is string => !!n));
  if (modelNames.length > 0) {
    const models = await prisma.explanationModelType.findMany({
      where: { name: { in: modelNames } },
      select: { name: true },
    });
    const missingModels = modelNames.filter((n) => !models.some((m) => m.name === n));
    if (missingModels.length > 0) {
      throw badRequest(`Unknown explanationModelName: ${missingModels.join(', ')}`);
    }
  }

  const sources = unique(items.map((e) => `${e.modelId}@${e.layer}`));
  const sourceAccess = await Promise.all(
    sources.map(async (key) => {
      const [modelId, layer] = key.split('@');
      return userCanAccessModelAndSourceSet(modelId, getSourceSetNameFromSource(layer), user, true);
    }),
  );
  const deniedSources = sources.filter((_, i) => !sourceAccess[i]);
  if (deniedSources.length > 0) {
    throw notFound(`Model or source not found: ${deniedSources.join(', ')}`);
  }

  const features = unique(items.map((e) => featureKey({ ...e, index: String(e.index) }))).map((key) => {
    const item = items.find((e) => featureKey({ ...e, index: String(e.index) }) === key)!;
    return { modelId: item.modelId, layer: item.layer, index: String(item.index) };
  });
  for (let i = 0; i < features.length; i += NEURON_LOOKUP_CONCURRENCY) {
    // eslint-disable-next-line no-await-in-loop
    await Promise.all(features.slice(i, i + NEURON_LOOKUP_CONCURRENCY).map((f) => ensureNeuronRowForFeature(f)));
  }
  const neuronWhere = { OR: features.map((f) => ({ modelId: f.modelId, layer: f.layer, index: f.index })) };
  const neurons = await prisma.neuron.findMany({
    where: neuronWhere,
    select: { modelId: true, layer: true, index: true },
  });
  const found = new Set(neurons.map(featureKey));
  const missingFeatures = features.map(featureKey).filter((k) => !found.has(k));
  if (missingFeatures.length > 0) {
    throw notFound(`Feature not found: ${missingFeatures.slice(0, 20).join(', ')}`);
  }

  const existing = await prisma.explanation.findMany({
    where: { ...neuronWhere, typeName: { in: typeNames } },
    select: {
      modelId: true,
      layer: true,
      index: true,
      typeName: true,
      description: true,
      pole: true,
      aspect: true,
      steerStrength: true,
    },
  });
  const seen = new Set(existing.map(sameKey));
  const toCreate: Prisma.ExplanationCreateManyInput[] = [];
  items.forEach((e) => {
    const row = {
      modelId: e.modelId,
      layer: e.layer,
      index: String(e.index),
      description: e.description,
      typeName: e.typeName,
      explanationModelName: e.explanationModelName ?? null,
      pole: e.pole ?? null,
      aspect: e.aspect ?? null,
      steerStrength: e.steerStrength ?? null,
    };
    const key = sameKey(row);
    if (seen.has(key)) {
      return;
    }
    seen.add(key);
    toCreate.push({
      ...row,
      sourceUrl: e.sourceUrl || null,
      contributorName: e.contributorName || null,
      contributorUrl: e.contributorUrl || null,
      metadata: (e.metadata as Prisma.InputJsonObject | undefined) ?? Prisma.DbNull,
      authorId: user.id,
    });
  });

  const created =
    toCreate.length > 0
      ? await prisma.explanation.createManyAndReturn({
          data: toCreate,
          select: { id: true, modelId: true, layer: true, index: true },
        })
      : [];
  return {
    created: created.map((c) => ({ id: c.id, url: `${NEXT_PUBLIC_URL}/${c.modelId}/${c.layer}/${c.index}` })),
    skipped: items.length - created.length,
  };
}
