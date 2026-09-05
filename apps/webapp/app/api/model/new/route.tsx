import { createModel, getModelByHfRepoIdIgnoringAccess, getModelById } from '@/lib/db/model';
import { HF_REPO_ID_ERROR_MESSAGE, MAX_HF_REPO_ID_CHARS, isValidHfRepoId } from '@/lib/utils/model';
import { RequestAuthedUser, withAuthedUser } from '@/lib/with-user';
import { Visibility } from '@prisma/client';
import { NextResponse } from 'next/server';

import * as yup from 'yup';

/**
 * @swagger
 * /api/model/new:
 *   post:
 *     summary: Create a new model
 *     description: Creates a new model with the specified parameters. Requires authentication.
 *     tags:
 *       - Models
 *     security:
 *       - apiKey: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - id
 *               - layers
 *             properties:
 *               id:
 *                 type: string
 *                 minLength: 2
 *                 maxLength: 32
 *                 pattern: '^[a-z][a-z0-9.-]*[a-z0-9]$|^[a-z]$'
 *                 description: Unique model identifier. Must start with lowercase letter, contain only lowercase letters, digits, hyphens, and periods, and not end with hyphen or period.
 *                 example: "gpt2-new"
 *               layers:
 *                 type: integer
 *                 minimum: 1
 *                 maximum: 127
 *                 description: Number of layers in the model
 *                 example: 12
 *               displayName:
 *                 type: string
 *                 minLength: 1
 *                 maxLength: 64
 *                 description: Human-readable display name for the model
 *                 example: "GPT-2 New"
 *               url:
 *                 type: string
 *                 format: uri
 *                 nullable: true
 *                 description: Optional URL for the model's website or documentation
 *                 example: "https://huggingface.co/gpt2"
 *               hfRepoId:
 *                 type: string
 *                 nullable: true
 *                 pattern: '^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$'
 *                 description: The HuggingFace repo this model corresponds to, as `namespace/name`. Case-sensitive, and unique across models. Omit it for a model that is not on the Hub.
 *                 example: "openai-community/gpt2"
 *     responses:
 *       200:
 *         description: Model created successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 id:
 *                   type: string
 *                   description: The created model's ID
 *                 message:
 *                   type: string
 *                   description: Success message
 *       400:
 *         description: Bad request - validation error or model already exists
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 error:
 *                   type: string
 *                   description: Error message
 *                 path:
 *                   type: string
 *                   description: Field path that caused the validation error (if applicable)
 *       401:
 *         description: Unauthorized - authentication required
 *       409:
 *         description: Conflict - the given hfRepoId is already mapped to another model
 *       500:
 *         description: Internal server error
 */

const NewModelRequestSchema = yup.object({
  id: yup
    .string()
    .required()
    .min(2)
    .max(32, 'Model ID must be less than 32 characters')
    .matches(
      /^[a-z][a-z0-9.-]*[a-z0-9]$|^[a-z]$/,
      'Name must contain only lowercase letters, digits, hyphens, and periods, start with a lowercase letter, not end with a hyphen or period, and not contain double hyphens',
    )
    .test('no-double-hyphens', 'Name cannot contain double hyphens', (value) => !value || !value.includes('--'))
    .test('no-double-periods', 'Name cannot contain double periods', (value) => !value || !value.includes('..')),
  layers: yup.number().required().integer().min(1).max(127, 'Layers must be an integer less than 128'),
  displayName: yup.string().optional().min(1).max(64, 'Display name must be less than 64 characters'),
  url: yup.string().optional().nullable().url('Must be a valid URL'),
  // Not lowercased, unlike `id`: repo ids are case-sensitive on the Hub (`google/gemma-4-E2B`).
  hfRepoId: yup
    .string()
    .optional()
    .nullable()
    .max(MAX_HF_REPO_ID_CHARS)
    .test('hf-repo-id', HF_REPO_ID_ERROR_MESSAGE, (value) => !value || isValidHfRepoId(value)),
});

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const requestJson = await request.json();

  let validatedRequest;
  try {
    validatedRequest = await NewModelRequestSchema.validate(requestJson);
  } catch (error) {
    if (error instanceof yup.ValidationError) {
      return NextResponse.json({ error: error.message, path: error.path }, { status: 400 });
    }
    throw error;
  }

  // check if model already exists
  const existingModel = await getModelById(validatedRequest.id);
  if (existingModel) {
    return NextResponse.json({ error: 'Model already exists' }, { status: 400 });
  }

  const hfRepoId = validatedRequest.hfRepoId?.trim() || null;
  if (hfRepoId) {
    // Checked here so a duplicate reads as a conflict rather than surfacing as a 500 from the
    // unique index. Racy by nature, which is why the index exists as well.
    const modelWithRepo = await getModelByHfRepoIdIgnoringAccess(hfRepoId);
    if (modelWithRepo) {
      return NextResponse.json(
        { error: `HuggingFace repo ${hfRepoId} is already mapped to model ${modelWithRepo.id}` },
        { status: 409 },
      );
    }
  }

  const model = {
    id: validatedRequest.id,
    creatorId: request.user.id,
    layers: validatedRequest.layers,
    displayName: validatedRequest.displayName ?? validatedRequest.id,
    displayNameShort: validatedRequest.displayName ?? validatedRequest.id,
    website: validatedRequest.url ?? null,
    visibility: Visibility.UNLISTED,
    instruct: false,
    tlensId: null,
    openRouterId: null,
    hfRepoId,
    inferenceEnabled: false,
    dimension: null,
    defaultSourceId: null,
    defaultSourceSetName: null,
    defaultGraphSourceSetName: null,
    thinking: false,

    neuronsPerLayer: 0,
    createdAt: new Date(),
    updatedAt: new Date(),
    owner: 'custom',
  };

  const newModel = await createModel(model, request.user);

  return NextResponse.json(newModel);
});
