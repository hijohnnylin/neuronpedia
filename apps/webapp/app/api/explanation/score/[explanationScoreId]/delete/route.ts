import { deleteExplanationScoreById } from '@/lib/db/explanation';
import { RequestAuthedUser, withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

/**
 * @swagger
 * /api/explanation/score/{explanationScoreId}/delete:
 *   post:
 *     summary: Auto-Interp - Score Delete
 *     description: Deletes an explanation score by its ID. Only the user who created the score can delete it.
 *     tags:
 *       - Explanations
 *     security:
 *       - apiKey: []
 *     parameters:
 *       - name: explanationScoreId
 *         in: path
 *         required: true
 *         description: ID of the explanation score to delete
 *         schema:
 *           type: string
 *     responses:
 *       200:
 *         description: Explanation score successfully deleted
 */

export const POST = withAuthedUser(
  async (
    request: RequestAuthedUser,
    {
      params,
    }: {
      params: { explanationScoreId: string };
    },
  ) => {
    await deleteExplanationScoreById(params.explanationScoreId, request.user);

    return NextResponse.json({ message: 'Explanation score deleted' });
  },
);
