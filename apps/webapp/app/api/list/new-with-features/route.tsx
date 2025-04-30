import { addNeuronsToList, newList, updateListMetadata } from '@/lib/db/list';
import { neuronExistsAndUserHasAccess } from '@/lib/db/neuron';
import { NEXT_PUBLIC_URL } from '@/lib/env';
import { ListWithPartialRelationsAndUrl } from '@/lib/utils/list';
import { RequestAuthedUser, withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';
import { array, number, object, string, ValidationError } from 'yup';

// Hobby plans don't support > 60 seconds
// export const maxDuration = 120;

const newWithFeaturesSchema = object({
  name: string().required(),
  description: string().default(''),
  testText: string().optional().min(1, 'Test text must not be empty'),
  features: array().of(
    object({
      modelId: string().required(),
      layer: string().required(),
      index: number().required().integer(),
      description: string().default(''),
    }),
  ),
});

/**
 * @swagger
 * /api/list/new-with-features:
 *   post:
 *     summary: Create List with Features
 *     description: Creates a new list and adds specified features to it.
 *     tags:
 *       - Lists
 *     security:
 *       - apiKey: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - name
 *               - features
 *             properties:
 *               name:
 *                 type: string
 *                 description: The name of the new list
 *                 default: Twitter features
 *               description:
 *                 type: string
 *                 description: Optional description for the list
 *                 default: This is a twitter list
 *               testText:
 *                 type: string
 *                 description: Optional test text for the list, which will show the activations for the text for all the features in the list.
 *               features:
 *                 type: array
 *                 description: The features to add to the list
 *                 default: [{"modelId": "gpt2-small", "layer": "9-res-jb", "index": 1124}, {"modelId": "gpt2-small", "layer": "2-att-kk", "index": 10116}]
 *                 items:
 *                   type: object
 *                   required:
 *                     - modelId
 *                     - layer
 *                     - index
 *                   properties:
 *                     modelId:
 *                       type: string
 *                       description: The ID of the model
 *                     layer:
 *                       type: string
 *                       description: The SAE ID / layer name of the feature
 *                     index:
 *                       type: integer
 *                       description: The index of the feature
 *                     description:
 *                       type: string
 *                       description: Optional description for the feature
 *
 *     responses:
 *       200:
 *         description: Successfully created list
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 url:
 *                   type: string
 *                   description: The URL of the newly created list
 *                 name:
 *                   type: string
 *                   description: The name of the created list
 *                 description:
 *                   type: string
 *                   description: The description of the created list
 */

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const bodyJson = await request.json();

  try {
    const body = await newWithFeaturesSchema.validate(bodyJson);

    const name = body.name as string;
    const description = body.description as string;
    if (!name || name.trim().length === 0) {
      return NextResponse.json({ message: 'List name is required' }, { status: 400 });
    }

    // validate each feature exists
    const { features } = body;
    const featuresToCreate = [];
    if (features) {
      try {
        // eslint-disable-next-line no-restricted-syntax
        for await (const feat of features) {
          const feature = await neuronExistsAndUserHasAccess(
            feat.modelId,
            feat.layer,
            feat.index.toString(),
            request.user,
          );
          if (!feature) {
            console.log("Feature not found or user doesn't have access, skipping: ", feat);
          } else {
            featuresToCreate.push({
              modelId: feat.modelId,
              layer: feat.layer,
              index: feat.index.toString(),
              description: feat.description,
              userId: request.user.id,
            });
          }
        }
      } catch (error) {
        return NextResponse.json({ message: `Feature not found - ${(error as Error).message}` }, { status: 400 });
      }
    }

    const list = await newList(name, description, request.user);
    if (body.testText) {
      await updateListMetadata(list.id, list.name, list.description, request.user, body.testText);
    }
    await addNeuronsToList(list.id, featuresToCreate, request.user);

    const listWithUrl: ListWithPartialRelationsAndUrl = {
      ...list,
      url: `${NEXT_PUBLIC_URL}/list/${list.id}`,
    };

    return NextResponse.json(listWithUrl);
  } catch (error) {
    if (error instanceof ValidationError) {
      return NextResponse.json({ message: error.message }, { status: 400 });
    }
    return NextResponse.json({ message: 'Unknown Error' }, { status: 500 });
  }
});
