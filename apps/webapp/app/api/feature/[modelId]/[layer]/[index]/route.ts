import { getNeuronOptimized } from '@/lib/db/neuron';
import { ERROR_NOT_FOUND_MESSAGE } from '@/lib/db/userCanAccess';
import { RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

/**
@swagger
{
  "/api/feature/{modelId}/{layer}/{index}": {
    "get": {
      "tags": [
        "Features"
      ],
      "summary": "Get Feature",
      "security": [{
          "apiKey": []
      }],
      "description": "Returns a feature, including its activations and explanations.",
      "parameters": [
        {
          "name": "modelId",
          "in": "path",
          "description": "Model ID",
          "required": true,
          "schema": {
            "type": "string",
            "default": "gpt2-small"
          }
        },
        {
          "name": "layer",
          "in": "path",
          "description": "SAE ID or Layer",
          "required": true,
          "schema": {
            "type": "string",
            "default": "0-res-jb"
          }
        },
        {
          "name": "index",
          "in": "path",
          "description": "Index",
          "required": true,
          "schema": {
            "type": "number",
            "default": 14057
          }
        }
      ],
      "responses": {
        "200": {
          "description": null
        }
      }
    }
  }
}
 */

export const GET = withOptionalUser(
  async (
    request: RequestOptionalUser,
    {
      params,
    }: {
      params: Promise<{ modelId: string; layer: string; index: string }>;
    },
  ) => {
    const { modelId, layer, index } = await params;
    const neuron = await getNeuronOptimized(modelId, layer, index, request.user);
    if (!neuron) {
      return NextResponse.json({ error: ERROR_NOT_FOUND_MESSAGE }, { status: 404 });
    }

    return NextResponse.json(neuron);
  },
);
