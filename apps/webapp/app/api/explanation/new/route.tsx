import { badRequest } from '@/lib/api-error';
import { uploadExplanations, uploadExplanationsSchema } from '@/lib/db/explanation-upload';
import { RequestAuthedUser, withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const maxDuration = 120;

/**
{
  "/api/explanation/new": {
    "post": {
      "tags": [
        "Explanations"
      ],
      "summary": "Upload Explanations",
      "security": [{
          "apiKey": []
      }],
      "description": "Upload up to 500 explanations at once. Your account needs upload access, and you can only upload to explanation types that you own: contact us to get both. The batch is all or nothing. An explanation that is already there (same feature, type, text, pole, aspect and strength) is skipped, so you can safely retry.",
      "requestBody": {
        "required": true,
        "content": {
          "application/json": {
            "schema": {
              "type": "object",
              "properties": {
                "explanations": {
                  "type": "array",
                  "items": {
                    "type": "object",
                    "required": ["modelId", "layer", "index", "description", "typeName"],
                    "properties": {
                      "modelId": { "type": "string", "description": "The model ID, for example qwen3-8b." },
                      "layer": { "type": "string", "description": "The source ID. For MLP neurons, the layer number, for example 9." },
                      "index": { "type": "string", "description": "The feature index." },
                      "description": { "type": "string", "description": "The explanation. A short phrase is fine." },
                      "typeName": { "type": "string", "description": "Your explanation type." },
                      "explanationModelName": { "type": "string", "description": "Optional. The model that wrote the explanation." },
                      "pole": { "type": "string", "enum": ["POSITIVE", "NEGATIVE", "BOTH"], "description": "Optional. Which sign of the activation this describes. Default POSITIVE." },
                      "aspect": { "type": "string", "enum": ["ACTIVATION", "STEERING"], "description": "Optional. ACTIVATION is what makes the feature fire. STEERING is what the model does when the feature is pushed. Default ACTIVATION." },
                      "steerStrength": { "type": "number", "description": "Optional, for STEERING only. Signed, in the units of your explanation type." },
                      "sourceUrl": { "type": "string", "description": "Optional. The page for this feature on your site." },
                      "contributorName": { "type": "string", "description": "Optional. Who wrote this explanation on your site." },
                      "contributorUrl": { "type": "string", "description": "Optional. A link to that person." },
                      "metadata": { "type": "object", "description": "Optional. Other fields to show with the explanation, at most 16 KB as JSON." }
                    }
                  }
                }
              },
              "example": {
                "explanations": [
                  {
                    "modelId": "qwen3-8b",
                    "layer": "9",
                    "index": "10358",
                    "description": "wine",
                    "typeName": "my-site",
                    "pole": "NEGATIVE",
                    "sourceUrl": "https://example.com/qwen3-8b/9/10358"
                  }
                ]
              }
            }
          }
        }
      },
      "responses": {
        "200": {
          "description": "OK",
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "created": {
                    "type": "array",
                    "items": {
                      "type": "object",
                      "properties": {
                        "id": { "type": "string" },
                        "url": { "type": "string" }
                      }
                    }
                  },
                  "skipped": { "type": "number" }
                }
              }
            }
          }
        }
      }
    }
  }
}
*/

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const json = await request.json().catch(() => {
    throw badRequest('The body must be valid JSON.');
  });
  const body = await uploadExplanationsSchema.validate(json, { abortEarly: true, stripUnknown: true });
  return NextResponse.json(await uploadExplanations(body.explanations, request.user));
});
