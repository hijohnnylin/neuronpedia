import { newList, updateListMetadata } from '@/lib/db/list';
import { env } from '@/lib/env';
import { RequestAuthedUser } from '@/lib/types/auth';
import { ListWithPartialRelationsAndUrl } from '@/lib/utils/list';
import { withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';
import { object, string, ValidationError } from 'yup';

const newListSchema = object({
  name: string().required().trim().min(1, 'Name must not be empty'),
  description: string(),
  testText: string().optional().nullable().min(1, 'Test text must not be empty'),
});

/**
 * @swagger
 * /api/list/new:
 *   post:
 *     summary: Create List
 *     description: Creates an empty new list for the authenticated user.
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
 *             properties:
 *               name:
 *                 type: string
 *                 description: The name of the new list
 *                 default: Test List
 *               description:
 *                 type: string
 *                 description: Optional description for the list
 *                 default: This is a test list
 *               testText:
 *                 type: string
 *                 description: Optional test text for the list, which will show the activations for the text for all the features in the list.
 *                 default: null
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
  let body = await request.json();

  try {
    body = await newListSchema.validate(body);

    const list = await newList(body.name, body.description, request.user);

    if (body.testText) {
      await updateListMetadata(list.id, list.name, list.description, request.user, body.testText);
    }

    const listWithUrl: ListWithPartialRelationsAndUrl = {
      ...list,
      url: `${env.NEXT_PUBLIC_URL}/list/${list.id}`,
    };

    return NextResponse.json(listWithUrl);
  } catch (error) {
    if (error instanceof ValidationError) {
      return NextResponse.json({ error: error.message }, { status: 400 });
    }
    return NextResponse.json({ error: 'Unknown Error' }, { status: 500 });
  }
});
