import { getModelByHfRepoId } from '@/lib/db/model';
import { HF_REPO_ID_ERROR_MESSAGE, MAX_HF_REPO_ID_CHARS, isValidHfRepoId } from '@/lib/utils/model';
import { RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';
import * as yup from 'yup';

/**
 * @swagger
 * /api/model/lookup:
 *   post:
 *     summary: Find the Neuronpedia model for a HuggingFace repo
 *     description: |
 *       Neuronpedia model ids are slash-free, so they cannot be a HuggingFace repo id and are not
 *       derived from one by stripping the namespace (`openai-community/gpt2` is `gpt2-small` here).
 *       This resolves one to the other. A repo maps to at most one model.
 *     tags:
 *       - Models
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - hfRepoId
 *             properties:
 *               hfRepoId:
 *                 type: string
 *                 description: The HuggingFace repo id, as `namespace/name`. Case-sensitive.
 *                 example: "openai-community/gpt2"
 *     responses:
 *       200:
 *         description: The model serving that repo
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 id:
 *                   type: string
 *                   description: The Neuronpedia model id, which is also its URL segment
 *                   example: "gpt2-small"
 *                 hfRepoId:
 *                   type: string
 *                 displayName:
 *                   type: string
 *                 displayNameShort:
 *                   type: string
 *                 layers:
 *                   type: integer
 *                 neuronsPerLayer:
 *                   type: integer
 *                 dimension:
 *                   type: integer
 *                   nullable: true
 *                 instruct:
 *                   type: boolean
 *                 thinking:
 *                   type: boolean
 *                 inferenceEnabled:
 *                   type: boolean
 *                 website:
 *                   type: string
 *                   nullable: true
 *       400:
 *         description: Bad request - hfRepoId missing or malformed
 *       404:
 *         description: No model on Neuronpedia corresponds to that repo
 */

const LookupRequestSchema = yup.object({
  hfRepoId: yup
    .string()
    .required()
    .max(MAX_HF_REPO_ID_CHARS)
    .test('hf-repo-id', HF_REPO_ID_ERROR_MESSAGE, (value) => !value || isValidHfRepoId(value)),
});

export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  let validatedRequest;
  try {
    validatedRequest = await LookupRequestSchema.validate(await request.json());
  } catch (error) {
    if (error instanceof yup.ValidationError) {
      return NextResponse.json({ error: error.message, path: error.path }, { status: 400 });
    }
    throw error;
  }

  const model = await getModelByHfRepoId(validatedRequest.hfRepoId, request.user);
  if (!model) {
    return NextResponse.json(
      { error: `No Neuronpedia model for HuggingFace repo ${validatedRequest.hfRepoId}` },
      { status: 404 },
    );
  }

  // Mapped field by field rather than returned whole. This is a public route, so its shape is a
  // contract: a column added to `Model` must not appear here by accident. `tlensId` and
  // `openRouterId` are withheld deliberately -- the first is an internal spelling being retired,
  // the second an upstream provider id, as `/api/nla/sources` also withholds it.
  return NextResponse.json({
    id: model.id,
    hfRepoId: model.hfRepoId,
    displayName: model.displayName,
    displayNameShort: model.displayNameShort,
    layers: model.layers,
    neuronsPerLayer: model.neuronsPerLayer,
    dimension: model.dimension,
    instruct: model.instruct,
    thinking: model.thinking,
    inferenceEnabled: model.inferenceEnabled,
    website: model.website,
  });
});
