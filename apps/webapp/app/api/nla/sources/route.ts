import { prisma } from '@/lib/db';
import { NextResponse } from 'next/server';

/**
 * @swagger
 * /api/nla/sources:
 *   get:
 *     summary: List Available NLA Sources
 *     tags:
 *       - NLA
 *     security:
 *       - apiKey: []
 *       - {}
 *     description: |
 *       Returns every `(modelId, nlaSourceId)` pair available through
 *       `/api/nla/explain` and `/api/nla/completion`, with public metadata
 *       (display name, author, layer, AR/AV ids). Use the `modelId` and
 *       `id` fields from this response to populate the corresponding
 *       request fields of `/api/nla/explain` and `/api/nla/completion`.
 *
 *       `model.openRouterAvailable` indicates whether the model has an
 *       OpenRouter mapping (required for `/api/nla/completion`'s
 *       generation path). Sources without it can still be used for
 *       `/api/nla/explain` if you supply your own `text`.
 *     responses:
 *       200:
 *         description: Available sources.
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 sources:
 *                   type: array
 *                   items:
 *                     type: object
 *                     properties:
 *                       modelId: { type: string }
 *                       id: { type: string, description: "Pass as `nlaSourceId` to explain/completion." }
 *                       displayName: { type: string }
 *                       description: { type: string }
 *                       url: { type: string }
 *                       author: { type: string }
 *                       av: { type: string }
 *                       ar: { type: string }
 *                       layerNum: { type: integer }
 *                       norm: { type: number }
 *                       model:
 *                         type: object
 *                         properties:
 *                           id: { type: string }
 *                           displayName: { type: string }
 *                           openRouterAvailable: { type: boolean }
 */
export async function GET() {
  const rows = await prisma.nlaSource.findMany({
    select: {
      id: true,
      modelId: true,
      displayName: true,
      description: true,
      url: true,
      author: true,
      av: true,
      ar: true,
      layerNum: true,
      norm: true,
      createdAt: true,
      model: {
        select: {
          id: true,
          displayName: true,
          openRouterId: true,
        },
      },
    },
    orderBy: [{ modelId: 'asc' }, { layerNum: 'asc' }, { id: 'asc' }],
  });

  // Strip the literal `openRouterId` (an upstream provider identifier we
  // don't need to publish) and surface a boolean instead so researchers
  // can tell which sources support the `/api/nla/completion` generation
  // path without exposing the mapping itself.
  const sources = rows.map((r) => ({
    modelId: r.modelId,
    id: r.id,
    displayName: r.displayName,
    description: r.description,
    url: r.url,
    author: r.author,
    av: r.av,
    ar: r.ar,
    layerNum: r.layerNum,
    norm: r.norm,
    createdAt: r.createdAt,
    model: {
      id: r.model.id,
      displayName: r.model.displayName,
      openRouterAvailable: Boolean(r.model.openRouterId),
    },
  }));

  return NextResponse.json({ sources });
}
