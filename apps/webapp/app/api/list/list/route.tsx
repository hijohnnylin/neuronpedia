import { getUserListsSimple } from '@/lib/db/list';
import { RequestAuthedUser } from '@/lib/types/auth';
import { withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

/**
 * @swagger
 * /api/list/list:
 *   post:
 *     summary: Get Lists
 *     description: Retrieves all lists created by the authenticated user.
 *     tags:
 *       - Lists
 *     security:
 *       - apiKey: []
 *     responses:
 *       200:
 *         description: Successfully retrieved user lists
 */

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const lists = await getUserListsSimple(request.user.id);

  return NextResponse.json(lists);
});
