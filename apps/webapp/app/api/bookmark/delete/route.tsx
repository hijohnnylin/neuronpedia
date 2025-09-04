import { deleteBookmark } from '@/lib/db/bookmark';
import { RequestAuthedUser } from '@/lib/types/auth';
import { withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

/**
 * @swagger
 * /api/bookmark/delete:
 *   post:
 *     summary: Delete Bookmark
 *     description: Deletes a bookmark for a specific feature from the authenticated user's bookmarks.
 *     tags:
 *       - Bookmarks
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
 *             properties:
 *               modelId:
 *                 type: string
 *                 description: The ID of the model the feature belongs to
 *               layer:
 *                 type: string
 *                 description: The layer of the feature
 *               index:
 *                 type: string
 *                 description: The index of the feature
 *     responses:
 *       200:
 *         description: Successfully deleted bookmark
 */

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const body = await request.json();
  const deleted = await deleteBookmark(body.modelId, body.layer, body.index, request.user);
  return NextResponse.json(deleted);
});
