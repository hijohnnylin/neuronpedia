import { assertUserCanAccessModelAndSourceSet } from '@/lib/db/userCanAccess';
import { runInferenceActivationSource } from '@/lib/utils/inference';
import { getSourceSetNameFromSource } from '@/lib/utils/source';
import { RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';
import * as yup from 'yup';

const activationSourceSchema = yup.object({
  modelId: yup.string().required('modelId is required'),
  source: yup.string().required('source is required'),
  customText: yup.lazy((value) =>
    Array.isArray(value) ? yup.array().of(yup.string().required()).min(1).max(4).required() : yup.string().required(),
  ),
});

/**
 * @swagger
 * /api/activation/source:
 *   post:
 *     summary: All Feature Activations in a Source/SAE
 *     description: Gets activation values for all features in a source/SAE when processing custom input text(s). It returns the activations and tokenized prompts only, without any of the dashboards or explanations.
 *     tags:
 *       - Activations
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - modelId
 *               - source
 *               - customText
 *             properties:
 *               modelId:
 *                 description: The model the source is in.
 *                 type: string
 *                 required: true
 *                 default: gpt2-small
 *               source:
 *                 description: The source/SAE ID.
 *                 type: string
 *                 required: true
 *                 default: 9-res-jb
 *               customText:
 *                 oneOf:
 *                   - type: string
 *                   - type: array
 *                     items:
 *                       type: string
 *                 description: The custom text to process. Either a single string or an array of strings. If it's a single string, the max is a 1024 token text. If it's an array, the max is 4 strings of 256 tokens max each.
 *                 required: true
 *                 default: Hello world
 *     responses:
 *       200:
 *         description: Successful response, with activation data
 *       400:
 *         description: Bad request, missing or invalid input
 *       401:
 *         description: Unauthorized, user doesn't have access to the model or SAE set
 *       500:
 *         description: Internal server error
 */

export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  const body = await request.json();

  let validatedBody;
  try {
    validatedBody = await activationSourceSchema.validate(body);
  } catch (error) {
    return NextResponse.json(
      { message: error instanceof Error ? error.message : 'Invalid request body' },
      { status: 400 },
    );
  }

  const { modelId, source, customText } = validatedBody;

  const sourceSetName = getSourceSetNameFromSource(source);
  try {
    await assertUserCanAccessModelAndSourceSet(modelId, sourceSetName, request.user);
  } catch (error) {
    return NextResponse.json({ message: error instanceof Error ? error.message : 'Unknown Error' }, { status: 500 });
  }
  const activation = await runInferenceActivationSource(
    modelId,
    source,
    customText instanceof Array ? customText : [customText],
    request.user,
  );

  return NextResponse.json(activation);
});
