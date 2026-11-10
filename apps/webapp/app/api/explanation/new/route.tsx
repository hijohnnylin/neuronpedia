import { badRequest } from '@/lib/api-error';
import { uploadExplanations, uploadExplanationsSchema } from '@/lib/db/explanation-upload';
import { RequestAuthedUser, withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const maxDuration = 120;

/**
@swagger
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
      "description": "Upload up to 500 explanations at once, for example labels from your own site or research.\n\n**You need approval first.** New accounts cannot upload. [Contact us](https://neuronpedia.org/contact) to get:\n- Upload access for your account.\n- An explanation type that you own. We make it with its display name, credit, link and license. You can only upload to explanation types that you own.\n- If you set `explanationModelName`, a name that Neuronpedia already knows. We can add one.\n\n**How uploads work:**\n- The feature must exist on Neuronpedia, and your account must be able to see its model and source.\n- Explanations are public as soon as they are uploaded. They show the credit of your explanation type, and your account is their author.\n- The batch is all or nothing. If one item fails a check, nothing is written.\n- An explanation that is already there (same feature, type, text, pole, aspect and strength) is skipped, so you can safely retry.\n- You cannot edit or delete explanations with the API. Contact us to fix or remove them.\n- For more than about 10,000 explanations, contact us first. We can load large sets in bulk.\n- Explanation search does not find uploaded explanations yet, because they have no embedding.\n\n**Limits:** 500 explanations per request, 2,000 characters per description, 16 KB of metadata as JSON, and http or https URLs only.\n\n**Pole and aspect:** by default an explanation describes what makes the feature fire on positive activations. Use `pole` NEGATIVE or BOTH for other signs, and `aspect` STEERING for what the model does when the feature is pushed, with an optional `steerStrength`. STEERING with pole BOTH means the feature was switched off (a knockout).",
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
