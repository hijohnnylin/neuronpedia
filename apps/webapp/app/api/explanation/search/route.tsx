import { MAX_EXPLANATION_SEARCH_RESULTS, searchExplanationsVec } from '@/lib/db/explanation';
import { getNeurons } from '@/lib/db/neuron';
import { getExplanationEmbeddingSql } from '@/lib/external/embedding';
import { NeuronIdentifier } from '@/lib/utils/neuron-identifier';
import { RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';
import { array, number, object, string, ValidationError } from 'yup';

export const maxDuration = 20;

const explanationSearchRequestSchema = object({
  modelId: string().required(),
  layers: array().of(string().required()).required(),
  query: string().required().min(3),
  offset: number().integer(),
});

const ACTS_TO_RETURN = 3;

/**
@swagger
{
  "/api/explanation/search": {
    "post": {
      "tags": [
        "Explanations"
      ],
      "summary": "Search by SAEs/Sources",
      "security": [
        {
          "apiKey": []
        }
      ],
      "description": "Search for explanations in the features of one or more SAEs/Sources in a model. Takes a query and returns up to 20 results at a time.",
      "requestBody": {
        "required": true,
        "content": {
          "application/json": {
            "schema": {
              "type": "object",
              "properties": {
                "modelId": {
                  "required": true,
                  "type": "string",
                  "description": "The ID of the model. For example, \"gemma-2b\".",
                  "default": "gemma-2-2b"
                },
                "layers": {
                  "required": true,
                  "type": "array",
                  "items": {
                    "type": "string"
                  },
                  "description": "The layers / SAE IDs to search. For example, [\"0-res-jb\", \"6-res-jb\"]",
                  "default": ["20-gemmascope-res-16k", "21-gemmascope-res-16k"]
                },
                "query": {
                  "required": true,
                  "type": "string",
                  "description": "The search query. For example, \"new york\".",
                  "default": "new york"
                },
                "offset": {
                  "type": "number",
                  "description": "Optional. The offset for pagination.",
                  "default": 0
                }
              },
              "example": {
                "modelId": "gemma-2-2b",
                "layers": [
                  "20-gemmascope-res-16k",
                  "21-gemmascope-res-16k"
                ],
                "query": "golden retrievers"
              }
            }
          }
        }
      },
      "responses": {
        "200": {
          "description": "Successful response",
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "request": {
                    "type": "object",
                    "properties": {
                      "modelId": {
                        "required": true,
                        "type": "string"
                      },
                      "layers": {
                        "items": {
                          "required": true,
                          "type": "string"
                        }
                      },
                      "query": {
                        "required": true,
                        "type": "string"
                      },
                      "offset": {
                        "type": "number"
                      }
                    }
                  },
                  "results": {
                    "type": "array",
                    "items": {
                      "$ref": null
                    }
                  },
                  "resultsCount": {
                    "type": "number"
                  },
                  "hasMore": {
                    "type": "boolean"
                  },
                  "nextOffset": {
                    "type": "number"
                  }
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

export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  const bodyJson = await request.json();
  if (!bodyJson) {
    return NextResponse.json({ message: 'Invalid JSON body' }, { status: 400 });
  }

  try {
    const body = await explanationSearchRequestSchema.validate(bodyJson);

    // // get the embeddings from OpenAI for the text
    const queryEmbeddingStr = await getExplanationEmbeddingSql(body.query);

    const result = await searchExplanationsVec(
      body.modelId,
      body.layers,
      queryEmbeddingStr,
      body.offset || 0,
      request.user,
    );

    // get the neurons
    const neurons = await getNeurons(
      result.map((r) => new NeuronIdentifier(r.modelId, r.layer, r.index)),
      request.user,
      ACTS_TO_RETURN,
    );
    // add the neurons to the result
    result.forEach((r) => {
      // eslint-disable-next-line no-param-reassign
      r.neuron = neurons.find((n) => n.modelId === r.modelId && n.layer === r.layer && n.index === r.index);
    });

    const hasMore = result.length === MAX_EXPLANATION_SEARCH_RESULTS;

    return NextResponse.json({
      request: body,
      results: result,
      resultsCount: result.length,
      hasMore,
      nextOffset: hasMore ? (body.offset || 0) + result.length : undefined,
    });
  } catch (error) {
    if (error instanceof ValidationError) {
      return NextResponse.json({ message: error.message }, { status: 400 });
    }
    return NextResponse.json({ message: 'Unknown Error' }, { status: 500 });
  }
});
