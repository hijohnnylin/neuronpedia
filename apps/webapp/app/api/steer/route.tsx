import { prisma } from '@/lib/db';
import { getModelById } from '@/lib/db/model';
import { neuronExistsAndUserHasAccess } from '@/lib/db/neuron';
import { ERROR_NOT_FOUND_MESSAGE } from '@/lib/db/userCanAccess';
import { DEMO_MODE, NEXT_PUBLIC_URL } from '@/lib/env';
import { steerCompletion } from '@/lib/utils/inference';
import {
  STEER_FREQUENCY_PENALTY_MAX,
  STEER_FREQUENCY_PENALTY_MIN,
  STEER_MAX_PROMPT_CHARS,
  STEER_METHOD,
  STEER_N_COMPLETION_TOKENS_MAX,
  STEER_N_COMPLETION_TOKENS_MAX_LARGE_LLM,
  STEER_STRENGTH_MULTIPLIER_MAX,
  STEER_TEMPERATURE_MAX,
  SteerFeature,
} from '@/lib/utils/steer';
import { AuthenticatedUser, RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { SteerOutputType } from '@prisma/client';
import { createHash } from 'crypto';
import { EventSourceMessage, EventSourceParserStream } from 'eventsource-parser/stream';
import { NPLogprob, NPSteerMethod, SteerCompletionPost200Response } from 'neuronpedia-inference-client';
import { NextResponse } from 'next/server';
import { array, bool, InferType, number, object, string, ValidationError } from 'yup';

const NNSIGHT_MODELS = ['llama3.3-70b-it', 'gpt-oss-20b'];
const STEERING_VERSION = 1;
const MAX_PROMPT_CHARS = STEER_MAX_PROMPT_CHARS;

function createStream(generator: AsyncGenerator<SteerResult>) {
  const encoder = new TextEncoder();
  return new ReadableStream({
    async start(controller) {
      // eslint-disable-next-line
      for await (const chunk of generator) {
        const dataString = `data: ${JSON.stringify(chunk)}\n\n`;
        controller.enqueue(encoder.encode(dataString));
      }
      controller.close();
    },
  });
}

async function* transformStream(
  stream: ReadableStreamDefaultReader<EventSourceMessage>,
): AsyncGenerator<SteerCompletionPost200Response> {
  while (true) {
    // eslint-disable-next-line
    const { done, value } = await stream.read();
    if (done) {
      break;
    }
    const parsed = JSON.parse(value.data);
    try {
      // we manually process because we aren't using the typescript client :(
      const toYield: SteerCompletionPost200Response = {
        outputs: parsed.outputs.map((output: any) => {
          const op = {
            type: output.type,
            output: output.output,
            logprobs: output.logprobs
              ? output.logprobs.map((logprob: any) => ({
                  token: logprob.token,
                  logprob: logprob.logprob,
                  topLogprobs: logprob.top_logprobs
                    ? logprob.top_logprobs.map((topLogprob: any) => ({
                        token: topLogprob.token,
                        logprob: topLogprob.logprob,
                      }))
                    : null,
                }))
              : null,
          };
          return op;
        }),
      };
      yield toYield;
    } catch (error) {
      console.error(error);
    }
  }
}

async function* generateResponse(
  body: SteerSchemaType,
  toReturnResult: SteerResult,
  steerTypesToRun: SteerOutputType[],
  features: SteerFeature[],
  user: AuthenticatedUser | null,
  hasVector: boolean,
): AsyncGenerator<SteerResult> {
  // NOW: always return the SteerResult modified to what we have so far
  // eslint-disable-next-line
  for (const steerType of steerTypesToRun) {
    // eslint-disable-next-line
    const steerCompletionResult = (await steerCompletion(
      body.modelId,
      steerTypesToRun,
      body.prompt,
      body.strength_multiplier,
      body.n_tokens,
      body.temperature,
      body.freq_penalty,
      body.seed,
      features,
      hasVector,
      user,
      body.steer_method,
      true,
    )) as ReadableStream<any>;

    const streamReader = steerCompletionResult
      .pipeThrough(new TextDecoderStream())
      .pipeThrough(new EventSourceParserStream())
      .getReader();
    // eslint-disable-next-line no-await-in-loop, no-restricted-syntax
    for await (const completionChunk of transformStream(streamReader)) {
      // find the output for the steerType
      const output = completionChunk.outputs.find((out) => out.type === steerType);
      if (!output) {
        throw new Error(`No output found for steerType: ${steerType}`);
      }
      // eslint-disable-next-line no-param-reassign
      toReturnResult[steerType] = output.output;
      if (steerType === SteerOutputType.STEERED) {
        // eslint-disable-next-line no-param-reassign
        toReturnResult.steeredLogProbs = output.logprobs ? output.logprobs : null;
      } else {
        // eslint-disable-next-line no-param-reassign
        toReturnResult.defaultLogProbs = output.logprobs ? output.logprobs : null;
      }
      yield toReturnResult;
    }
  }
  if (DEMO_MODE) {
    console.log('skipping saveSteerOutput in demo mode');
  } else {
    // eslint-disable-next-line
    toReturnResult = await saveSteerOutput(body, steerTypesToRun, toReturnResult, user?.id);
  }
  yield toReturnResult;
}

export type SteerResult = {
  [SteerOutputType.STEERED]: string | null;
  [SteerOutputType.DEFAULT]: string | null;
  steeredLogProbs: NPLogprob[] | null;
  defaultLogProbs: NPLogprob[] | null;
  id: string | null;
  shareUrl: string | null | undefined;
  limit: string | null;
};

const steerSchema = object({
  prompt: string().max(MAX_PROMPT_CHARS).required(),
  modelId: string().required(),
  features: array()
    .of(
      object({
        modelId: string().required(),
        layer: string().required(),
        index: number().integer().required(),
        strength: number()
          .required()
          // don't enforce min/max in the api
          // .min(STEER_STRENGTH_MIN)
          // .max(STEER_STRENGTH_MAX)
          .transform((value) => value),
      }).required(),
    )
    .required(),
  temperature: number().min(0).max(STEER_TEMPERATURE_MAX).required(),
  n_tokens: number().integer().min(1).max(STEER_N_COMPLETION_TOKENS_MAX).required(),
  freq_penalty: number().min(STEER_FREQUENCY_PENALTY_MIN).max(STEER_FREQUENCY_PENALTY_MAX).required(),
  seed: number().min(-100000000).max(100000000).required(),
  strength_multiplier: number().min(0).max(STEER_STRENGTH_MULTIPLIER_MAX).required(),
  stream: bool().default(false),
  steer_method: string().oneOf(Object.values(NPSteerMethod)).default(STEER_METHOD),
});

type SteerSchemaType = InferType<typeof steerSchema>;

async function saveSteerOutput(
  body: SteerSchemaType,
  steerTypesToRun: SteerOutputType[],
  toReturnResult: SteerResult,
  userId?: string | undefined,
) {
  // save the new outputs before returning
  const saveActions = steerTypesToRun.map((type) =>
    prisma.steerOutput.create({
      data: {
        // these two are different based on type
        outputText:
          type === SteerOutputType.STEERED
            ? toReturnResult[SteerOutputType.STEERED] || ''
            : toReturnResult[SteerOutputType.DEFAULT] || '',
        type,
        modelId: body.modelId,
        // rest is the same
        creatorId: userId,
        inputText: body.prompt,
        inputTextMd5: createHash('md5').update(body.prompt).digest('hex'),
        temperature: body.temperature,
        numTokens: body.n_tokens,
        freqPenalty: body.freq_penalty,
        seed: body.seed,
        strengthMultiplier: body.strength_multiplier,
        steerMethod: body.steer_method,
        version: STEERING_VERSION,
        logprobs:
          type === SteerOutputType.STEERED
            ? JSON.stringify(toReturnResult.steeredLogProbs)
            : JSON.stringify(toReturnResult.defaultLogProbs),
        toNeurons:
          type === SteerOutputType.DEFAULT
            ? {}
            : type === SteerOutputType.STEERED
              ? {
                  create: body.features.map((neuron) => ({
                    neuron: {
                      connect: {
                        modelId_layer_index: {
                          modelId: neuron.modelId,
                          layer: neuron.layer,
                          index: neuron.index.toString(),
                        },
                      },
                    },
                    strength: neuron.strength,
                  })),
                }
              : {},
      },
    }),
  );

  const saveResults = await Promise.all(saveActions);
  // eslint-disable-next-line no-restricted-syntax
  for (const saveResult of saveResults) {
    if (saveResult.type === SteerOutputType.STEERED) {
      // eslint-disable-next-line
      toReturnResult.id = saveResult.id;
      // eslint-disable-next-line
      toReturnResult.shareUrl = `${NEXT_PUBLIC_URL}/steer/${saveResult.id}`;
    }
  }
  console.log('SAVING AND RETURNING');
  return toReturnResult;
}

/**
@swagger
{
  "/api/steer": {
    "post": {
      "tags": [
        "Steering"
      ],
      "summary": "Steer With SAE Features (Non-Chat)",
      "security": [
        {
          "apiKey": []
        },
        {}
      ],
      "description": "Given a prompt and a set of SAE features, steer a model to generate both its default and steered text, as well as logprobs for each generated token. This is for completions, not chat.",
      "requestBody": {
        "required": true,
        "content": {
          "application/json": {
            "schema": {
              "type": "object",
              "example": {
                "prompt": "The most iconic structure on Earth is",
                "modelId": "gemma-2-2b",
                "features": [
                  {
                    "modelId": "gemma-2-2b",
                    "layer": "20-gemmascope-res-16k",
                    "index": 3124,
                    "strength": 38.5
                  }
                ],
                "temperature": 0.5,
                "n_tokens": 48,
                "freq_penalty": 2,
                "seed": 16,
                "strength_multiplier": 4,
                "steer_method": "SIMPLE_ADDITIVE"
              },
              "properties": {
                "prompt": {
                  "type": "string"
                },
                "modelId": {
                  "type": "string"
                },
                "features": {
                  "type": "array",
                  "items": {
                    "type": "object",
                    "required": [
                      "modelId",
                      "layer",
                      "index",
                      "strength"
                    ],
                    "properties": {
                      "modelId": {
                        "type": "string"
                      },
                      "layer": {
                        "type": "string"
                      },
                      "index": {
                        "type": "number"
                      },
                      "strength": {
                        "type": "number"
                      }
                    }
                  }
                },
                "temperature": {
                  "type": "number"
                },
                "n_tokens": {
                  "type": "number"
                },
                "freq_penalty": {
                  "type": "number"
                },
                "seed": {
                  "type": "number"
                },
                "strength_multiplier": {
                  "type": "number"
                },
                "steer_method": {
                  "type": "string",
                  "enum": ["SIMPLE_ADDITIVE", "ORTHOGONAL_DECOMP"]
                }
              }
            }
          }
        }
      },
      "responses": {
        "200": {
          "description": null
        }
      }
    }
  }
}
 */

export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  const bodyJson = await request.json();

  try {
    const body = await steerSchema.validate(bodyJson);
    const limit = request.headers.get('x-limit-remaining');
    let toReturnResult: SteerResult = {
      [SteerOutputType.STEERED]: null,
      [SteerOutputType.DEFAULT]: null,
      steeredLogProbs: null,
      defaultLogProbs: null,
      id: null,
      shareUrl: undefined,
      limit,
    };

    const { modelId } = body;
    if (NNSIGHT_MODELS.includes(modelId)) {
      if (body.n_tokens > STEER_N_COMPLETION_TOKENS_MAX_LARGE_LLM) {
        return NextResponse.json(
          { message: `For large LLM models the max n_tokens is ${STEER_N_COMPLETION_TOKENS_MAX_LARGE_LLM}` },
          { status: 400 },
        );
      }
    }

    // model access
    const modelAccess = await getModelById(body.modelId, request.user);
    if (!modelAccess) {
      return NextResponse.json({ message: ERROR_NOT_FOUND_MESSAGE }, { status: 404 });
    }
    // each feature access
    const featuresWithVectors: SteerFeature[] = [];
    // eslint-disable-next-line no-restricted-syntax
    for (const feature of body.features) {
      // eslint-disable-next-line no-await-in-loop
      const accessResult = await neuronExistsAndUserHasAccess(
        feature.modelId,
        feature.layer,
        feature.index.toString(),
        request.user,
      );
      if (!accessResult) {
        return NextResponse.json({ message: ERROR_NOT_FOUND_MESSAGE }, { status: 404 });
      }
      featuresWithVectors.push({ ...feature, neuron: accessResult });
    }

    // ensure that there is no mix of vector and non-vector features
    const hasVector = featuresWithVectors.some(
      (feature) => feature.neuron?.vector && feature.neuron?.vector.length > 0,
    );
    const hasNonVector = featuresWithVectors.some(
      (feature) => !feature.neuron?.vector || feature.neuron?.vector.length === 0,
    );
    if (hasVector && hasNonVector) {
      return NextResponse.json({ message: "Can't steer both vector and non-vector features" }, { status: 400 });
    }

    // check if it's saved already, if so, fill in the outputs we already have saved
    let steerTypesToRun: SteerOutputType[] = [SteerOutputType.STEERED, SteerOutputType.DEFAULT];
    // DEFAULT should not have ANY toNeurons
    const savedSteerOutputs = await prisma.steerOutput.findMany({
      where: {
        modelId: body.modelId,
        inputText: body.prompt,
        temperature: body.temperature,
        numTokens: body.n_tokens,
        freqPenalty: body.freq_penalty,
        seed: body.seed,
        strengthMultiplier: body.strength_multiplier,
        steerMethod: body.steer_method,
        version: STEERING_VERSION,
      },
      include: {
        toNeurons: true,
      },
    });

    let savedSteereds = savedSteerOutputs.filter((steerOutput) => steerOutput.type === SteerOutputType.STEERED);
    // savedSteered should also have the right ToNeurons
    savedSteereds = savedSteereds.filter((steerOutput) => {
      // first check same number of neurons
      if (steerOutput.toNeurons.length !== body.features.length) {
        return false;
      }
      // then check each to make sure they exist
      let hasMissingFeature = false;
      steerOutput.toNeurons.forEach((toNeuron) => {
        if (
          !body.features.some(
            (feature) =>
              toNeuron.modelId === feature.modelId &&
              toNeuron.layer === feature.layer &&
              toNeuron.index === feature.index.toString() &&
              toNeuron.strength === feature.strength,
          )
        ) {
          hasMissingFeature = true;
        }
      });
      if (hasMissingFeature) {
        return false;
      }
      return true;
    });
    // savedDefault shouldn't have any toNeurons (the features aren't applied)
    const savedDefault = savedSteerOutputs.filter((steerOutput) => steerOutput.type === SteerOutputType.DEFAULT);
    if (savedSteereds.length > 0) {
      toReturnResult[SteerOutputType.STEERED] = savedSteereds[0].outputText;
      toReturnResult.steeredLogProbs = savedSteereds[0].logprobs ? JSON.parse(savedSteereds[0].logprobs) : null;
      toReturnResult.id = savedSteereds[0].id;
      toReturnResult.shareUrl = `${NEXT_PUBLIC_URL}/steer/${savedSteereds[0].id}`;
      steerTypesToRun = steerTypesToRun.filter((type) => type !== SteerOutputType.STEERED);
    }
    if (savedDefault.length > 0) {
      toReturnResult[SteerOutputType.DEFAULT] = savedDefault[0].outputText;
      toReturnResult.defaultLogProbs = savedDefault[0].logprobs ? JSON.parse(savedDefault[0].logprobs) : null;
      steerTypesToRun = steerTypesToRun.filter((type) => type !== SteerOutputType.DEFAULT);
    }
    if (steerTypesToRun.length === 0) {
      return NextResponse.json(toReturnResult);
    }

    if (body.stream) {
      const generator = generateResponse(
        body,
        toReturnResult,
        steerTypesToRun,
        featuresWithVectors,
        request.user,
        hasVector,
      );
      const stream = createStream(generator);
      return new NextResponse(stream, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache, no-transform',
          Connection: 'keep-alive',
        },
      });
    }

    let steerCompletionResult = await steerCompletion(
      body.modelId,
      steerTypesToRun,
      body.prompt,
      body.strength_multiplier,
      body.n_tokens,
      body.temperature,
      body.freq_penalty,
      body.seed,
      featuresWithVectors,
      hasVector,
      request.user,
      body.steer_method,
      false, // non-streaming
    );
    steerCompletionResult = steerCompletionResult as SteerCompletionPost200Response;
    const steeredCompletionResult = steerCompletionResult.outputs.find(
      (output) => output.type === SteerOutputType.STEERED,
    );
    const defaultCompletionResult = steerCompletionResult.outputs.find(
      (output) => output.type === SteerOutputType.DEFAULT,
    );

    // fill in the results we got from inference server
    if (!toReturnResult[SteerOutputType.STEERED] && steeredCompletionResult) {
      console.log("didn't have steered, filling it");
      toReturnResult[SteerOutputType.STEERED] = steeredCompletionResult.output;
      // unfortunately we aren't using the typescript client, so we have to do this manually
      const logprobs =
        steeredCompletionResult.logprobs?.map((logprob: any) => ({
          token: logprob.token,
          logprob: logprob.logprob,
          topLogprobs: logprob.top_logprobs
            ? logprob.top_logprobs.map((topLogprob: any) => ({
                token: topLogprob.token,
                logprob: topLogprob.logprob,
              }))
            : null,
        })) || null;
      toReturnResult.steeredLogProbs = logprobs;
    }
    if (!toReturnResult[SteerOutputType.DEFAULT] && defaultCompletionResult) {
      console.log("didn't have default, filling it");
      toReturnResult[SteerOutputType.DEFAULT] = defaultCompletionResult.output;
      // unfortunately we aren't using the typescript client, so we have to do this manually
      const logprobs =
        defaultCompletionResult.logprobs?.map((logprob: any) => ({
          token: logprob.token,
          logprob: logprob.logprob,
          topLogprobs: logprob.top_logprobs
            ? logprob.top_logprobs.map((topLogprob: any) => ({
                token: topLogprob.token,
                logprob: topLogprob.logprob,
              }))
            : null,
        })) || null;
      toReturnResult.defaultLogProbs = logprobs;
    }

    toReturnResult = await saveSteerOutput(body, steerTypesToRun, toReturnResult, request.user?.id);

    return NextResponse.json(toReturnResult);
  } catch (error) {
    if (error instanceof ValidationError) {
      return NextResponse.json({ message: error.message }, { status: 400 });
    }
    console.log('unknown error', error);
    return NextResponse.json({ message: 'Unknown Error' }, { status: 500 });
  }
});
