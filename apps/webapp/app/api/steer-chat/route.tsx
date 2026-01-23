// TODO: clean this up

/* eslint-disable no-param-reassign */
/* eslint-disable no-restricted-syntax */

import { prisma } from '@/lib/db';
import { getModelById } from '@/lib/db/model';
import { neuronExistsAndUserHasAccess } from '@/lib/db/neuron';
import { ERROR_NOT_FOUND_MESSAGE } from '@/lib/db/userCanAccess';
import { DEMO_MODE, NEXT_PUBLIC_URL } from '@/lib/env';
import { steerCompletionChat } from '@/lib/utils/inference';
import {
  ChatMessage,
  ERROR_STEER_MAX_PROMPT_CHARS,
  STEER_MAX_PROMPT_CHARS,
  STEER_MAX_PROMPT_CHARS_ASSISTANT_AXIS,
  STEER_MAX_PROMPT_CHARS_THINKING,
  STEER_METHOD,
  STEER_N_COMPLETION_TOKENS_MAX,
  STEER_N_COMPLETION_TOKENS_MAX_ASSISTANT_AXIS,
  STEER_N_COMPLETION_TOKENS_MAX_LARGE_LLM,
  STEER_N_COMPLETION_TOKENS_MAX_THINKING,
  STEER_STRENGTH_MIN,
  STEER_STRENGTH_MULTIPLIER_MAX,
  STEER_TEMPERATURE_MAX,
  SteerFeature,
} from '@/lib/utils/steer';
import { AuthenticatedUser, RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { SteerOutputToNeuronWithPartialRelations } from '@/prisma/generated/zod';
import { SteerOutputType } from '@prisma/client';
import { createHash } from 'crypto';
import { EventSourceMessage } from 'eventsource-parser';
import { EventSourceParserStream } from 'eventsource-parser/stream';
import {
  NPLogprob,
  NPSteerChatMessage,
  NPSteerMethod,
  SteerCompletionChatPost200Response,
  SteerCompletionChatPost200ResponseAssistantAxisInner,
  SteerCompletionChatPost200ResponseAssistantAxisInnerFromJSON,
  SteerCompletionChatPost200ResponseAssistantAxisInnerToJSON,
  SteerCompletionChatPost200ResponseFromJSON,
} from 'neuronpedia-inference-client';
import { NextResponse } from 'next/server';
import { array, bool, InferType, number, object, string, ValidationError } from 'yup';

// Hobby plans don't support > 60 seconds
export const maxDuration = 180;

const NNSIGHT_MODELS = ['llama3.3-70b-it', 'gpt-oss-20b'];
const STEERING_VERSION = 1;

function sortChatMessages(chatMessages: ChatMessage[]) {
  const toReturn: ChatMessage[] = [];
  for (const message of chatMessages) {
    toReturn.push({
      content: message.content,
      role: message.role,
    });
  }
  return toReturn;
}

async function saveSteerChatOutput(
  body: SteerSchemaTypeChat,
  toReturnResult: SteerResultChat,
  existingDefaultOutputId: string | undefined,
  steerTypesRan: SteerOutputType[],
  input: { raw: string; chatTemplate: NPSteerChatMessage[] } | null,
  userId: string | undefined,
  assistantAxisArray?: SteerCompletionChatPost200ResponseAssistantAxisInner[],
) {
  let defaultOutputId = existingDefaultOutputId;

  // Helper to find capMonitorOutput for a given steer type from the assistant_axis array
  // We use ToJSON to save in snake_case format so FromJSON can correctly load it later
  const getCapMonitorOutput = (steerType: SteerOutputType): string | null => {
    if (!assistantAxisArray || !Array.isArray(assistantAxisArray)) return null;
    const axisItem = assistantAxisArray.find((item) => item.type === steerType);
    return axisItem ? JSON.stringify(SteerCompletionChatPost200ResponseAssistantAxisInnerToJSON(axisItem)) : null;
  };

  for (const steerTypeRan of steerTypesRan) {
    if (steerTypeRan === SteerOutputType.DEFAULT) {
      const output = toReturnResult[SteerOutputType.DEFAULT];
      if (!output) {
        throw new Error('No default output found');
      }
      console.log('saving default output');
      // eslint-disable-next-line no-await-in-loop
      const s1 = await prisma.steerOutput.create({
        data: {
          // these two are different based on type
          outputText: output.raw,
          outputTextChatTemplate: JSON.stringify(sortChatMessages(output.chatTemplate || [])),
          type: SteerOutputType.DEFAULT,
          modelId: body.modelId,
          // rest is the same
          creatorId: userId,
          inputText: input?.raw || '',
          inputTextMd5: createHash('md5')
            .update(input?.raw || '')
            .digest('hex'),
          inputTextChatTemplate: JSON.stringify(sortChatMessages(body.defaultChatMessages)),
          inputTextChatTemplateMd5: createHash('md5')
            .update(JSON.stringify(sortChatMessages(body.defaultChatMessages)))
            .digest('hex'),
          temperature: body.temperature,
          numTokens: body.n_tokens,
          freqPenalty: body.freq_penalty,
          seed: body.seed,
          strengthMultiplier: body.strength_multiplier,
          version: STEERING_VERSION,
          steerSpecialTokens: body.steer_special_tokens,
          steerMethod: body.steer_method,
          toNeurons: {},
          logprobs: output.logprobs ? JSON.stringify(output.logprobs) : null,
          capMonitorOutput: getCapMonitorOutput(SteerOutputType.DEFAULT),
        },
      });
      // update the default saved output id since we just saved it
      defaultOutputId = s1.id;
      console.log(`default saved: ${s1.id}`);
    } else if (steerTypeRan === SteerOutputType.STEERED) {
      console.log('saving steered output');
      const output = toReturnResult[SteerOutputType.STEERED];
      if (!output) {
        throw new Error('No steered output found');
      }
      // eslint-disable-next-line no-await-in-loop
      const dbResult = await prisma.steerOutput.create({
        data: {
          // these two are different based on type
          outputText: output.raw,
          outputTextChatTemplate: JSON.stringify(sortChatMessages(output.chatTemplate || [])),
          type: SteerOutputType.STEERED,
          modelId: body.modelId,
          // rest is the same
          creatorId: userId,
          inputText: input?.raw || '',
          inputTextMd5: createHash('md5')
            .update(input?.raw || '')
            .digest('hex'),
          inputTextChatTemplate: JSON.stringify(sortChatMessages(body.steeredChatMessages)),
          inputTextChatTemplateMd5: createHash('md5')
            .update(JSON.stringify(sortChatMessages(body.steeredChatMessages)))
            .digest('hex'),
          temperature: body.temperature,
          numTokens: body.n_tokens,
          freqPenalty: body.freq_penalty,
          seed: body.seed,
          strengthMultiplier: body.strength_multiplier,
          version: STEERING_VERSION,
          steerSpecialTokens: body.steer_special_tokens,
          steerMethod: body.steer_method,
          toNeurons: {
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
          },
          logprobs: output.logprobs ? JSON.stringify(output.logprobs) : null,
          capMonitorOutput: getCapMonitorOutput(SteerOutputType.STEERED),
        },
      });

      // eslint-disable-next-line no-param-reassign
      toReturnResult.id = dbResult.id;
      console.log(`steer saved: ${dbResult.id}`);
      // eslint-disable-next-line no-param-reassign
      toReturnResult.shareUrl = `${NEXT_PUBLIC_URL}/steer/${dbResult.id}`;
    }

    // update saved steered output with connected default output id
    if (toReturnResult.id) {
      // eslint-disable-next-line no-await-in-loop
      await prisma.steerOutput.update({
        where: {
          id: toReturnResult.id,
        },
        data: {
          connectedDefaultOutputId: defaultOutputId,
        },
      });
    }
  }
  return toReturnResult;
}

function createStream(generator: AsyncGenerator<SteerResultChat>) {
  const encoder = new TextEncoder();
  return new ReadableStream({
    async start(controller) {
      // eslint-disable-next-line
      for await (const chunk of generator) {
        const dataString = `data: ${JSON.stringify(chunk)}\n\n`;
        // console.log(JSON.stringify(chunk, null, 2));
        controller.enqueue(encoder.encode(dataString));
      }
      controller.close();
    },
  });
}

async function* transformStream(
  stream: ReadableStreamDefaultReader<EventSourceMessage>,
): AsyncGenerator<SteerCompletionChatPost200Response> {
  while (true) {
    // eslint-disable-next-line
    const { done, value } = await stream.read();
    if (done) {
      break;
    }

    try {
      const parsed = JSON.parse(value.data);
      // Use the TypeScript client's FromJSON function to transform snake_case to camelCase
      const toYield = SteerCompletionChatPost200ResponseFromJSON(parsed);
      yield toYield;
    } catch (error) {
      console.error(error);
    }
  }
}

async function* generateResponse(
  body: SteerSchemaTypeChat,
  toReturnResult: SteerResultChat,
  savedSteerDefaultOutputId: string | undefined,
  steerTypesToRun: SteerOutputType[],
  features: SteerFeature[],
  user: AuthenticatedUser | null,
  hasVector: boolean,
): AsyncGenerator<SteerResultChat> {
  console.log('steerTypesToRun', steerTypesToRun);
  const steerCompletionChatResults = (await steerCompletionChat(
    body.modelId,
    steerTypesToRun,
    body.defaultChatMessages,
    body.steeredChatMessages,
    body.strength_multiplier,
    body.n_tokens,
    body.temperature,
    body.freq_penalty,
    body.seed,
    body.steer_special_tokens,
    features,
    hasVector,
    user,
    true,
    body.steer_method,
    undefined,
    body.isAssistantAxis,
  )) as ReadableStream<any>[];

  const readableStreams = steerCompletionChatResults.map((stream) =>
    stream.pipeThrough(new TextDecoderStream()).pipeThrough(new EventSourceParserStream()),
  );
  const streamReaders = readableStreams.map((stream) => stream.getReader());

  // Check if this is a combined request (one stream with both types)
  const isCombinedRequest = steerCompletionChatResults.length === 1 && steerTypesToRun.length === 2;

  const streamProcessors = streamReaders.map((streamReader, index) => ({
    // For combined requests, the single stream handles both types
    steerTypes: isCombinedRequest ? steerTypesToRun : [steerTypesToRun[index]],
    done: false,
    generator: transformStream(streamReader),
    pendingPromise: null as Promise<{ processorIndex: number; value: any; done: boolean }> | null,
  }));

  let input: { raw: string; chatTemplate: NPSteerChatMessage[] } | null = null;

  // Helper to create a promise for reading from a processor
  const createReadPromise = (processor: (typeof streamProcessors)[0], processorIndex: number) =>
    processor.generator.next().then(({ value, done }) => ({
      processorIndex,
      value,
      done: done || false,
    }));

  // Initialize pending promises for all processors
  streamProcessors.forEach((processor, index) => {
    processor.pendingPromise = createReadPromise(processor, index);
  });

  // Continue until all streams are done - process in parallel using Promise.race
  while (streamProcessors.some((processor) => !processor.done)) {
    // Get all pending promises from non-done processors
    const activePromises = streamProcessors
      .map((processor, index) => ({ processor, index }))
      .filter(({ processor }) => !processor.done && processor.pendingPromise)
      .map(({ processor }) => processor.pendingPromise!);

    if (activePromises.length === 0) break;

    // Wait for whichever stream has data first
    const result = await Promise.race(activePromises);
    const processor = streamProcessors[result.processorIndex];

    if (result.done) {
      processor.done = true;
      processor.pendingPromise = null;
    } else {
      // Process the result from this processor
      const { value } = result;

      // Process all outputs for this processor's steer types
      for (const steerType of processor.steerTypes) {
        const output = value.outputs.find((out: any) => out.type === steerType);
        if (!output) {
          throw new Error(`No output found for steerType: ${steerType}`);
        }

        input = value.input;
        toReturnResult[steerType] = {
          raw: output.raw,
          chatTemplate: output.chatTemplate,
          logprobs: output.logprobs ? output.logprobs : null,
        };
      }
      // Pass through assistant_axis data from inference server
      // Merge assistant_axis arrays when we have separate streams for each type
      if (value.assistantAxis) {
        if (!toReturnResult.assistant_axis) {
          toReturnResult.assistant_axis = value.assistantAxis;
        } else if (Array.isArray(toReturnResult.assistant_axis) && Array.isArray(value.assistantAxis)) {
          // Merge arrays, replacing items with matching type
          for (const newItem of value.assistantAxis) {
            const existingIndex = toReturnResult.assistant_axis.findIndex((item: any) => item.type === newItem.type);
            if (existingIndex >= 0) {
              toReturnResult.assistant_axis[existingIndex] = newItem;
            } else {
              toReturnResult.assistant_axis.push(newItem);
            }
          }
        } else {
          toReturnResult.assistant_axis = value.assistantAxis;
        }
      }

      // Start reading the next chunk from this processor immediately
      processor.pendingPromise = createReadPromise(processor, result.processorIndex);

      // Yield the updated result
      yield toReturnResult;
    }
  }

  // Save final results after all streams are complete
  if (streamProcessors.every((processor) => processor.done)) {
    if (DEMO_MODE) {
      console.log('skipping saveSteerChatOutput in demo mode');
    } else {
      toReturnResult = await saveSteerChatOutput(
        body,
        toReturnResult,
        savedSteerDefaultOutputId,
        steerTypesToRun,
        input,
        user?.id,
        toReturnResult.assistant_axis,
      );
    }
    yield toReturnResult;
  }
}

export type SteerResultChat = {
  [SteerOutputType.STEERED]: {
    raw: string;
    chatTemplate: NPSteerChatMessage[] | undefined | null;
    logprobs: NPLogprob[] | null;
  } | null;
  [SteerOutputType.DEFAULT]: {
    raw: string;
    chatTemplate: NPSteerChatMessage[] | undefined | null;
    logprobs: NPLogprob[] | null;
  } | null;
  inputText?: string | null;
  id: string | null;
  shareUrl: string | null | undefined;
  limit: string | null;
  settings:
    | {
        temperature: number;
        n_tokens: number;
        freq_penalty: number;
        seed: number;
        strength_multiplier: number;
        steer_special_tokens: boolean;
        steer_method: NPSteerMethod;
      }
    | undefined;
  features?: SteerOutputToNeuronWithPartialRelations[];
  assistant_axis?: SteerCompletionChatPost200ResponseAssistantAxisInner[];
};

export type FeatureWithMaxActApprox = {
  modelId: string;
  layer: string;
  index: number;
  strength: number;
  maxActApprox: number;
};

const steerSchema = object({
  defaultChatMessages: array()
    .of(
      object({
        content: string().required(),
        role: string().oneOf(['user', 'assistant', 'system', 'model', 'developer']).required(),
      }),
    )
    .required(),
  steeredChatMessages: array()
    .of(
      object({
        content: string().required(),
        role: string().oneOf(['user', 'assistant', 'system', 'model', 'developer']).required(),
      }),
    )
    .required(),
  modelId: string().required(),
  features: array()
    .of(
      object({
        modelId: string().required(),
        layer: string().required(),
        index: number().integer().required(),
        strength: number()
          .required()
          .min(STEER_STRENGTH_MIN)
          .transform((value) => value),
      }).required(),
    )
    .required(),
  temperature: number().min(0).max(STEER_TEMPERATURE_MAX).required(),
  n_tokens: number().integer().min(1).required(),
  freq_penalty: number().min(-2).max(2).required(),
  seed: number().min(-100000000).max(100000000).required(),
  strength_multiplier: number().min(0).max(STEER_STRENGTH_MULTIPLIER_MAX).required(),
  steer_special_tokens: bool().required(),
  stream: bool().default(false),
  steer_method: string().oneOf(Object.values(NPSteerMethod)).default(STEER_METHOD),
  isAssistantAxis: bool().default(false),
});

export type SteerSchemaTypeChat = InferType<typeof steerSchema>;

/**
@swagger
{
  "/api/steer-chat": {
    "post": {
      "tags": [
        "Steering"
      ],
      "summary": "Steer With SAE Features (Chat)",
      "security": [
        {
          "apiKey": []
        },
        {}
      ],
      "description": "Given chat messages and a set of SAE features, steer a model to generate both its default and steered chat completions, as well as logprobs for each generated token. This is for chat, not completions.",
      "requestBody": {
        "required": true,
        "content": {
          "application/json": {
            "schema": {
              "type": "object",
              "example": {
                "defaultChatMessages": [
                  {
                    "role": "user",
                    "content": "hi"
                  }
                ],
                "steeredChatMessages": [
                  {
                    "role": "user", 
                    "content": "hi"
                  }
                ],
                "modelId": "gemma-2-9b-it",
                "features": [
                  {
                    "modelId": "gemma-2-9b-it",
                    "layer": "9-gemmascope-res-131k",
                    "index": 62610,
                    "strength": 48.0
                  }
                ],
                "temperature": 0.5,
                "n_tokens": 48,
                "freq_penalty": 2,
                "seed": 16,
                "strength_multiplier": 4,
                "steer_special_tokens": true,
                "steer_method": "SIMPLE_ADDITIVE"
              },
              "properties": {
                "defaultChatMessages": {
                  "type": "array",
                  "items": {
                    "type": "object",
                    "required": ["role", "content"],
                    "properties": {
                      "role": {
                        "type": "string",
                        "enum": ["user", "assistant", "system", "model"]
                      },
                      "content": {
                        "type": "string"
                      }
                    }
                  }
                },
                "steeredChatMessages": {
                  "type": "array", 
                  "items": {
                    "type": "object",
                    "required": ["role", "content"],
                    "properties": {
                      "role": {
                        "type": "string",
                        "enum": ["user", "assistant", "system", "model"]
                      },
                      "content": {
                        "type": "string"
                      }
                    }
                  }
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
                "steer_special_tokens": {
                  "type": "boolean"
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
          "description": "Successful steering response",
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "default": {
                    "type": "object",
                    "properties": {
                      "raw": {
                        "type": "string"
                      },
                      "chat_template": {
                        "type": "array"
                      }
                    }
                  },
                  "steered": {
                    "type": "object", 
                    "properties": {
                      "raw": {
                        "type": "string"
                      },
                      "chat_template": {
                        "type": "array"
                      }
                    }
                  },
                  "id": {
                    "type": "string"
                  },
                  "shareUrl": {
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

export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  const bodyJson = await request.json();

  try {
    const body = await steerSchema.validate(bodyJson);

    const { modelId } = body;
    const limit = request.headers.get('x-limit-remaining');

    // Calculate total length of all chat messages
    const totalDefaultChars = body.defaultChatMessages.reduce((sum, message) => sum + message.content.length, 0);
    const totalSteeredChars = body.steeredChatMessages.reduce((sum, message) => sum + message.content.length, 0);

    // Check if total length exceeds the maximum allowed
    let maxPromptChars = STEER_MAX_PROMPT_CHARS;
    if (body.isAssistantAxis) {
      maxPromptChars = STEER_MAX_PROMPT_CHARS_ASSISTANT_AXIS;
    } else if (NNSIGHT_MODELS.includes(modelId)) {
      maxPromptChars = STEER_MAX_PROMPT_CHARS_THINKING;
    }
    if (totalDefaultChars > maxPromptChars || totalSteeredChars > maxPromptChars) {
      console.log('total length exceeds the maximum allowed', totalDefaultChars, totalSteeredChars, maxPromptChars);
      return NextResponse.json({ message: ERROR_STEER_MAX_PROMPT_CHARS }, { status: 400 });
    }

    // check access
    // model access
    const modelAccess = await getModelById(modelId, request.user);
    if (!modelAccess) {
      return NextResponse.json({ message: ERROR_NOT_FOUND_MESSAGE }, { status: 404 });
    }
    // max completion tokens based on thinking or not
    if (modelAccess.thinking) {
      if (body.n_tokens > STEER_N_COMPLETION_TOKENS_MAX_THINKING) {
        return NextResponse.json(
          { message: `For thinking models the max n_tokens is ${STEER_N_COMPLETION_TOKENS_MAX_THINKING}` },
          { status: 400 },
        );
      }
    } else if (body.isAssistantAxis) {
      if (body.n_tokens > STEER_N_COMPLETION_TOKENS_MAX_ASSISTANT_AXIS) {
        return NextResponse.json(
          { message: `For assistant axis models the max n_tokens is ${STEER_N_COMPLETION_TOKENS_MAX_ASSISTANT_AXIS}` },
          { status: 400 },
        );
      }
    } else if (NNSIGHT_MODELS.includes(modelId)) {
      if (body.n_tokens > STEER_N_COMPLETION_TOKENS_MAX_LARGE_LLM) {
        return NextResponse.json(
          { message: `For large LLM models the max n_tokens is ${STEER_N_COMPLETION_TOKENS_MAX_LARGE_LLM}` },
          { status: 400 },
        );
      }
    } else if (body.n_tokens > STEER_N_COMPLETION_TOKENS_MAX) {
      return NextResponse.json(
        { message: `The max n_tokens for non-thinking models is ${STEER_N_COMPLETION_TOKENS_MAX}` },
        { status: 400 },
      );
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

    let toReturnResult: SteerResultChat = {
      [SteerOutputType.STEERED]: null,
      [SteerOutputType.DEFAULT]: null,
      id: null,
      shareUrl: undefined,
      limit,
      settings: {
        temperature: body.temperature,
        n_tokens: body.n_tokens,
        freq_penalty: body.freq_penalty,
        seed: body.seed,
        strength_multiplier: body.strength_multiplier,
        steer_special_tokens: body.steer_special_tokens,
        steer_method: body.steer_method,
      },
    };
    // check for saved outputs
    // if assistant axis, don't look it up

    // check for default saved output
    let steerTypesToRun: SteerOutputType[] = [SteerOutputType.STEERED, SteerOutputType.DEFAULT];
    // sort each chat message by content key, then role key so we can do an accurate lookup
    // this is because we store in the db using JSON.stringify and dictionaries are not ordered
    const defaultChatMessagesSorted = sortChatMessages(body.defaultChatMessages);
    const savedSteerDefaultOutput = await prisma.steerOutput.findFirst({
      where: {
        modelId,
        type: SteerOutputType.DEFAULT,
        inputTextChatTemplateMd5: createHash('md5').update(JSON.stringify(defaultChatMessagesSorted)).digest('hex'),
        temperature: body.temperature,
        numTokens: body.n_tokens,
        freqPenalty: body.freq_penalty,
        seed: body.seed,
        strengthMultiplier: body.strength_multiplier,
        version: STEERING_VERSION,
        steerSpecialTokens: body.steer_special_tokens,
        steerMethod: body.steer_method,
      },
    });
    // default already exists, set it to the output and don't run it
    if (savedSteerDefaultOutput) {
      console.log('has saved default output, setting it');
      toReturnResult[SteerOutputType.DEFAULT] = {
        raw: savedSteerDefaultOutput.outputText,
        chatTemplate: JSON.parse(savedSteerDefaultOutput.outputTextChatTemplate || '[]'),
        logprobs: savedSteerDefaultOutput.logprobs ? JSON.parse(savedSteerDefaultOutput.logprobs) : null,
      };
      // Set cached capMonitorOutput for assistant_axis
      if (savedSteerDefaultOutput.capMonitorOutput) {
        const cachedAxisItem = JSON.parse(savedSteerDefaultOutput.capMonitorOutput);
        // Use the TypeScript client's FromJSON function to handle both snake_case and camelCase
        const transformedItem = SteerCompletionChatPost200ResponseAssistantAxisInnerFromJSON(cachedAxisItem);
        if (!toReturnResult.assistant_axis) {
          toReturnResult.assistant_axis = [];
        }
        toReturnResult.assistant_axis.push(transformedItem);
      }
      steerTypesToRun = steerTypesToRun.filter((type) => type !== SteerOutputType.DEFAULT);
    }

    // check for steered saved output
    const steeredChatMessagesSorted = sortChatMessages(body.steeredChatMessages);
    let savedSteerSteeredOutputs = await prisma.steerOutput.findMany({
      where: {
        modelId,
        type: SteerOutputType.STEERED,
        inputTextChatTemplateMd5: createHash('md5').update(JSON.stringify(steeredChatMessagesSorted)).digest('hex'),
        temperature: body.temperature,
        numTokens: body.n_tokens,
        freqPenalty: body.freq_penalty,
        seed: body.seed,
        strengthMultiplier: body.strength_multiplier,
        version: STEERING_VERSION,
        steerSpecialTokens: body.steer_special_tokens,
        steerMethod: body.steer_method,
      },
      include: {
        toNeurons: true,
      },
    });

    // savedSteered should also have the right ToNeurons
    savedSteerSteeredOutputs = savedSteerSteeredOutputs.filter((steerOutput) => {
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

    if (savedSteerSteeredOutputs.length > 0) {
      console.log('has saved steered output, setting it');
      toReturnResult[SteerOutputType.STEERED] = {
        raw: savedSteerSteeredOutputs[0].outputText,
        chatTemplate: JSON.parse(savedSteerSteeredOutputs[0].outputTextChatTemplate || '[]'),
        logprobs: savedSteerSteeredOutputs[0].logprobs ? JSON.parse(savedSteerSteeredOutputs[0].logprobs) : null,
      };
      toReturnResult.id = savedSteerSteeredOutputs[0].id;
      toReturnResult.shareUrl = `${NEXT_PUBLIC_URL}/steer/${savedSteerSteeredOutputs[0].id}`;
      // Set cached capMonitorOutput for assistant_axis
      if (savedSteerSteeredOutputs[0].capMonitorOutput) {
        const cachedAxisItem = JSON.parse(savedSteerSteeredOutputs[0].capMonitorOutput);
        // Use the TypeScript client's FromJSON function to handle both snake_case and camelCase
        const transformedItem = SteerCompletionChatPost200ResponseAssistantAxisInnerFromJSON(cachedAxisItem);
        if (!toReturnResult.assistant_axis) {
          toReturnResult.assistant_axis = [];
        }
        toReturnResult.assistant_axis.push(transformedItem);
      }

      steerTypesToRun = steerTypesToRun.filter((type) => type !== SteerOutputType.STEERED);
    }

    if (steerTypesToRun.length === 0) {
      return NextResponse.json(toReturnResult);
    }

    if (body.stream) {
      const generator = generateResponse(
        body,
        toReturnResult,
        savedSteerDefaultOutput?.id,
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
    // if there are no featuresWithVectors, then steerTypesToRun should only be [SteerOutputType.DEFAULT]
    steerTypesToRun = featuresWithVectors.length === 0 ? [SteerOutputType.DEFAULT] : steerTypesToRun;
    let steerCompletionResults = await steerCompletionChat(
      modelId,
      steerTypesToRun,
      body.defaultChatMessages,
      body.steeredChatMessages,
      body.strength_multiplier,
      body.n_tokens,
      body.temperature,
      body.freq_penalty,
      body.seed,
      body.steer_special_tokens,
      featuresWithVectors,
      hasVector,
      request.user,
      body.stream,
      body.steer_method,
      undefined,
      body.isAssistantAxis,
    );
    steerCompletionResults = steerCompletionResults as SteerCompletionChatPost200Response[];
    for (let i = 0; i < steerCompletionResults.length; i += 1) {
      const result = steerCompletionResults[i];
      for (const output of result.outputs) {
        if (output.type === SteerOutputType.DEFAULT) {
          toReturnResult[SteerOutputType.DEFAULT] = {
            raw: output.raw,
            chatTemplate: output.chatTemplate,
            logprobs: output.logprobs ? output.logprobs : null,
          };
        } else if (output.type === SteerOutputType.STEERED) {
          toReturnResult[SteerOutputType.STEERED] = {
            raw: output.raw,
            chatTemplate: output.chatTemplate,
            logprobs: output.logprobs ? output.logprobs : null,
          };
        }
      }
      // Extract assistant_axis data from non-streaming response
      if (result.assistantAxis) {
        toReturnResult.assistant_axis = result.assistantAxis;
      }
    }
    let input: { raw: string; chatTemplate: NPSteerChatMessage[] } | null = null;
    steerCompletionResults.forEach((result) => {
      input = {
        raw: result.input.raw,
        chatTemplate: result.input.chatTemplate,
      };
    });

    // save the outputs
    toReturnResult = await saveSteerChatOutput(
      body,
      toReturnResult,
      savedSteerDefaultOutput?.id,
      steerTypesToRun,
      input,
      request.user?.id,
      toReturnResult.assistant_axis,
    );

    // return the result
    return NextResponse.json(toReturnResult);
  } catch (error) {
    if (error instanceof ValidationError) {
      console.log('validation error', error);
      return NextResponse.json({ message: error.message }, { status: 400 });
    }
    console.log('unknown error', error);
    return NextResponse.json({ message: 'Unknown Error' }, { status: 500 });
  }
});
