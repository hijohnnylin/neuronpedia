import { ERROR_NOT_FOUND_MESSAGE } from '@/lib/db/userCanAccess';
import { env } from '@/lib/env';
import { RequestOptionalUser } from '@/lib/types/auth';
import { withOptionalUser } from '@/lib/with-user';
import { PrismaClient } from '@prisma/client';
import { NextResponse } from 'next/server';

const prisma = new PrismaClient();

/**
@swagger
{
  "/api/activation/get": {
    "post": {
      "tags": [
        "Activations"
      ],
      "summary": "Get Activations",
      "description": "Returns all activations for a given model, source, and index.",
      "requestBody": {
        "required": true,
        "content": {
          "application/json": {
            "schema": {
              "type": "object",
              "properties": {
                "modelId": {
                  "type": "string",
                  "description": "Model ID"
                },
                "source": {
                  "type": "string",
                  "description": "Source or layer"
                },
                "index": {
                  "type": "string",
                  "description": "Index of the feature/latent"
                }
              },
              "required": ["modelId", "source", "index"]
            }
          }
        }
      },
      "responses": {
        "200": {
          "description": "Activation data"
        },
      }
    }
  }
}
 */

export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  try {
    const body = await request.json();
    const { modelId, source, index } = body;

    if (!modelId || !source || !index) {
      return NextResponse.json({ error: 'Missing required parameters' }, { status: 400 });
    }

    const activations = await prisma.activation.findMany({
      where: {
        modelId,
        layer: source,
        index,
        creatorId: {
          in: env.PUBLIC_ACTIVATIONS_USER_IDS,
        },
      },
      orderBy: {
        maxValue: 'desc',
      },
    });

    if (!activations || activations.length === 0) {
      return NextResponse.json({ error: ERROR_NOT_FOUND_MESSAGE }, { status: 404 });
    }

    return NextResponse.json(activations);
  } catch (error) {
    console.error('Error fetching activations:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
});
