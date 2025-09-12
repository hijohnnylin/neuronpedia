import { getListWithDetails } from '@/lib/db/list';
import { RequestOptionalUser } from '@/lib/types/auth';
import { withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

/**
 * @swagger
 * /api/list/get:
 *   post:
 *     summary: Get List Details
 *     description: Retrieves detailed information about a specific list.
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
 *                 description: The ID of the list to retrieve
 *     responses:
 *       200:
 *         description: Successfully retrieved list details
 */

export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  const body = await request.json();
  const listId = body.listId as string;

  const list = await getListWithDetails(listId, request.user);

  return NextResponse.json(list);
});
