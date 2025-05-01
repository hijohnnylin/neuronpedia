import { updateListMetadata } from '@/lib/db/list';
import { RequestAuthedUser, withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

// Hobby plans don't support > 60 seconds
// export const maxDuration = 120;

/**
 * @swagger
 * /api/list/update:
 *   post:
 *     summary: Update List Metadata
 *     description: Updates an existing list's metadata.
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
 *               - listId
 *               - name
 *             properties:
 *               listId:
 *                 type: string
 *                 description: The ID of the list to update
 *               name:
 *                 type: string
 *                 description: The new name for the list
 *               description:
 *                 type: string
 *                 description: The new description for the list
 *               defaultTestText:
 *                 type: string
 *                 description: Optional new default test text for the list
 *     responses:
 *       200:
 *         description: Successfully updated list
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
 *                   description: The updated name of the list
 *                 description:
 *                   type: string
 *                   description: The updated description of the list
 */

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const body = await request.json();
  const listId = body.listId as string;
  const name = body.name as string;
  const description = body.description as string;
  const defaultTestText = body.defaultTestText ? (body.defaultTestText as string) : null;
  if (!name || name.trim().length === 0) {
    throw new Error('Name length invalid');
  }

  const updatedList = await updateListMetadata(listId, name, description, request.user, defaultTestText);

  return NextResponse.json(updatedList);
});
