import { deleteExplanationById } from '@/lib/db/explanation';
import { RequestAuthedUser, withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

/**
 * @swagger
 * /api/explanation/{explanationId}/delete:
 *   post:
 *     summary: Auto-Interp - Delete
 *     description: Deletes an explanation by its ID. Only the user who created the explanation can delete it.
 *     tags:
 *       - Explanations
 *     security:
 *       - apiKey: []
 *     parameters:
 *       - in: path
 *         name: explanationId
 *         required: true
 *         schema:
 *           type: string
 *         description: The ID of the explanation to delete
 *     responses:
 *       200:
 *         description: Explanation successfully deleted
 */

export const POST = withAuthedUser(
  async (
    request: RequestAuthedUser,
    {
      params,
    }: {
      params: { explanationId: string };
    },
  ) => {
    await deleteExplanationById(params.explanationId, request.user);

    return NextResponse.json({ message: 'Explanation deleted' });
  },
);
