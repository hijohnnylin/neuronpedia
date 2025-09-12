import { deleteList } from '@/lib/db/list';
import { RequestAuthedUser } from '@/lib/types/auth';
import { withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';
/**
 * @swagger
 * /api/list/delete:
 *   post:
 *     summary: Delete List
 *     description: Deletes an existing list that you own.
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
 *             properties:
 *               listId:
 *                 type: string
 *                 description: The ID of the list to delete
 *     responses:
 *       200:
 *         description: Successfully deleted list
 *         content:
 *           application/json:
 *             schema:
 *               type: boolean
 *               description: Returns true if the list was successfully deleted
 */

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const body = await request.json();
  const listId = body.listId as string;

  await deleteList(listId, request.user);

  return NextResponse.json(true);
});
