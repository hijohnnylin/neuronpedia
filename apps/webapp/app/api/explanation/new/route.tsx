/* eslint-disable */

// TODO: Bring back this endpoint but simplify (just explanation, no scoring)

import { RequestAuthedUser } from '@/lib/types/auth';
import { withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';
import { array, number, object, string } from 'yup';

// Hobby plans don't support > 60 seconds
// export const maxDuration = 120;

const uploadExplanationSchema = object({
  feature: object({
    modelId: string().required(),
    layer: string().required(),
    index: number().integer().required(),
    explanation: string().required(),

    // only required if it's autointerped
    explanationModel: string(),

    // only required for scoring
    activations: array().of(
      object({
        id: string().required(),
      }).required(),
    ),
    simulationScore: number(),
    simulationActivations: array().of(
      object({
        simulation: object({
          tokens: array().of(string().required()).required(),
          expected_activations: array().of(number().required()).required(),
        }).required(),
        ev_correlation_score: number().required(),
      }).required(),
    ),
    simulationModel: string(),
  }).required(),
});

/**
{
  "/api/explanation/new": {
    "post": {
      "tags": [
        "Explanations"
      ],
      "summary": "Create Explanation",
      "security": [{
          "apiKey": []
      }],
      "description": "Upload an explanation for a feature. This the simple version that doesn't upload scoring information. To upload with scoring data, use [`SAELens`](https://github.com/jbloomAus/SAELens), under `sae_lens.analysis.neuronpedia_integration.autointerp_neuronpedia_features`.",
      "requestBody": {
        "required": true,
        "content": {
          "application/json": {
            "schema": {
              "type": "object",
              "properties": {
                "feature": {
                  "type": "object",
                  "properties": {
                    "modelId": {
                      "required": true,
                      "type": "string",
                      "description": "The model ID"
                    },
                    "layer": {
                      "required": true,
                      "type": "string",
                      "description": "The layer or SAE ID. For example, 6-res-jb."
                    },
                    "index": {
                      "required": true,
                      "type": "string",
                      "description": "The index"
                    },
                    "explanation": {
                      "required": true,
                      "type": "string",
                      "description": "The explanation"
                    },
                    "explanationModel": {
                      "type": "string",
                      "description": "Optional. The model used to generate the explanation. For example, gpt3.5-turbo."
                    },
                  },
                  "example": {
                    "modelId": "example-model",
                    "layer": "0-test-np",
                    "index": "0",
                    "explanation": "This is an explanation text!",
                    "explanationModel": "gpt3.5-turbo",
                  }
                }
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
                  "message": {
                    "type": "string"
                  },
                  "url": {
                    "type": "string"
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

export const POST = withAuthedUser(
  async (request: RequestAuthedUser) =>
    NextResponse.json({
      message: 'This endpoint is not currently active.',
    }),
  // try {
  //   var bodyJson = await request.json();
  // } catch (error) {
  //   return NextResponse.json({ message: "Invalid JSON body" }, { status: 400 });
  // }

  // try {
  //   var body = await uploadExplanationSchema.validate(bodyJson);
  // } catch (error) {
  //   if (error instanceof ValidationError) {
  //     return NextResponse.json({ message: error.message }, { status: 400 });
  //   } else {
  //     return NextResponse.json({ message: "Unknown Error" }, { status: 500 });
  //   }
  // }

  // // ensure required values are there
  // const bodyFeat = body.feature;
  // const requirementsMetForScoring =
  //   bodyFeat.activations !== undefined &&
  //   bodyFeat.simulationScore !== undefined &&
  //   bodyFeat.simulationActivations !== undefined &&
  //   bodyFeat.simulationModel !== undefined;
  // const requirementsMetForNotScoring =
  //   bodyFeat.activations === undefined &&
  //   bodyFeat.simulationScore === undefined &&
  //   bodyFeat.simulationActivations === undefined &&
  //   bodyFeat.simulationModel === undefined;

  // if (!requirementsMetForScoring && !requirementsMetForNotScoring) {
  //   return NextResponse.json(
  //     {
  //       message:
  //         "Scoring requires that you supply all the following fields: activations, simulationScore, simulationActivations, and simulationModel.",
  //     },
  //     { status: 400 },
  //   );
  // }

  // // get explainer user ID
  // let explainerUser = await getExplai_UNUSED_nerUser();
  // let explainerUserId = explainerUser.id;

  // // validate feature exists and user has access
  // let feat = body.feature;
  // let feature = await neuronExistsAndUserHasAccess(
  //   feat.modelId,
  //   feat.layer,
  //   feat.index.toString(),
  //   request.user,
  // );
  // if (!feature) {
  //   return NextResponse.json(
  //     {
  //       message: `Feature not found - ${
  //         "modelId: " +
  //         feat.modelId +
  //         " layer: " +
  //         feat.layer +
  //         " index: " +
  //         feat.index
  //       }`,
  //     },
  //     { status: 400 },
  //   );
  // }

  // var explanationActivations: ExplanationActivationV1WithoutIdAndExplanation[] =
  //   [];
  // if (requirementsMetForScoring) {
  //   // add the explanation, score, and expactv1
  //   body.feature.simulationActivations?.forEach((simulation, i) => {
  //     if (
  //       body.feature.activations &&
  //       body.feature.simulationActivations &&
  //       body.feature.simulationModel
  //     ) {
  //       // activations are returned in the same order so just match the index
  //       explanationActivations.push({
  //         activationId: body.feature.activations[i].id,
  //         expectedValues:
  //           body.feature.simulationActivations[i].simulation
  //             .expected_activations,
  //         score: simulation.ev_correlation_score,
  //         scorerId: null, // means it's gpt - scorer is always same as explainer
  //         scorerAuto_UNUSED_InterpModel: body.feature.simulationModel,
  //         version: SCORER_VERSION,
  //       });
  //     }
  //   });
  // }

  // await createExplanation(
  //   feature.modelId,
  //   feature.layer,
  //   feature.index,
  //   body.feature.explanation,
  //   explainerUserId,
  //   body.feature.explanationModel,
  //   request.user.id,
  //   body.feature.simulationScore,
  //   explanationActivations,
  //   request.user,
  // );

  // return NextResponse.json({
  //   message: "Explanation created. Open url to see it in the example model.",
  //   url: `${NEXT_PUBLIC_URL}/${feature.modelId}/${feature.layer}/${feature.index}`,
  // });
);
