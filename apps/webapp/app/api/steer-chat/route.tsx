// TODO: clean this up

import {
  NPLogprob,
  NPSteerChatMessage,
  NPSteerMethod,
  NPVectorRead,
  SteerCompletionChatResponse,
  SteerVectorReadout,
} from '@/lib/api/inference-types';
import { prisma } from '@/lib/db';
import { getModelById } from '@/lib/db/model';
import { neuronExistsAndUserHasAccess } from '@/lib/db/neuron';
import { getPersonaAxisDefinitions, getPersonaAxisFits } from '@/lib/db/persona-axis';
import { ERROR_NOT_FOUND_MESSAGE } from '@/lib/db/userCanAccess';
import { DEMO_MODE, NEXT_PUBLIC_URL } from '@/lib/env';
import { InferenceServerError, steerCompletionChat } from '@/lib/utils/inference';
import { labelReadouts, personaAxisToVectorRead, type PersonaAxisDefinition } from '@/lib/utils/persona-axis';
import {
  ChatMessage,
  ERROR_STEER_MAX_PROMPT_CHARS,
  STEER_FREQUENCY_PENALTY,
  STEER_FREQUENCY_PENALTY_MAX,
  STEER_FREQUENCY_PENALTY_MIN,
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
import { axisReadoutsToLegacyAssistantAxis, LegacyAssistantAxis } from '@/lib/utils/steer-axis-legacy';
import { axisReadoutsToPublic, PublicAxisReadout } from '@/lib/utils/steer-axis-public';
import {
  AxisProvenance,
  axisReadoutsFromStored,
  mergeStoredAxes,
  storedAxisIds,
  storedAxisRowIds,
} from '@/lib/utils/steer-wire';
import { AuthenticatedUser, RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { SteerOutputToNeuronWithPartialRelations } from '@/prisma/generated/zod';
import { SteerOutputType } from '@prisma/client';
import { createHash } from 'crypto';
import { EventSourceMessage } from 'eventsource-parser';
import { EventSourceParserStream } from 'eventsource-parser/stream';
import { NextResponse } from 'next/server';
import { array, bool, InferType, number, object, string, ValidationError } from 'yup';

// Hobby plans don't support > 60 seconds
export const maxDuration = 180;

const NNSIGHT_MODELS = ['llama3.3-70b-it', 'gpt-oss-20b'];
// Part of the saved-output lookup key below, so bump this whenever the text or chatTemplate we
// store changes shape -- otherwise rows written under the old semantics keep being served as hits.
const STEERING_VERSION = 2;

// What the deprecated `isAssistantAxis: true` asks for. Readouts are named now, so the flag is one
// particular name; it stays accepted because `/api/*` has callers outside this repo.
const ASSISTANT_AXIS_ID = 'lu_assistant-axis';

/** The axis ids a request asks for, honouring the deprecated boolean. */
function requestedAxisIds(body: { axes?: string[] | undefined; isAssistantAxis?: boolean | undefined }): string[] {
  if (body.axes && body.axes.length > 0) return body.axes;
  return body.isAssistantAxis ? [ASSISTANT_AXIS_ID] : [];
}

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
  axisReadouts?: SteerVectorReadout[],
  supersededRows?: Partial<Record<SteerOutputType, { outputText: string; capMonitorOutput: string | null }>>,
  measuredWith?: AxisProvenance,
) {
  let defaultOutputId = existingDefaultOutputId;

  // The readouts measured for one steer type, in the stored snake_case shape keyed by axis id, so
  // rows written before the wire changed stay readable.
  //
  // `supersededRows` is the cached row this generation replaced because it was missing a requested
  // axis. Carrying its readouts across means asking for a second axis adds to what is stored
  // rather than trading one for the other -- otherwise a caller alternating between two axes
  // regenerates forever, each request replacing the row the previous one wrote. Only merged when
  // the regenerated text came out identical, since a readout describes the turns it was measured
  // on; if generation was not reproducible, the old numbers belong to text that no longer exists.
  const getCapMonitorOutput = (steerType: SteerOutputType, outputText: string): string | null => {
    const forType = (axisReadouts ?? []).filter((readout) => readout.type === steerType);
    const superseded = supersededRows?.[steerType];
    const carryOver = superseded && superseded.outputText === outputText ? superseded.capMonitorOutput : null;
    if (forType.length === 0 && !carryOver) return null;
    return JSON.stringify(mergeStoredAxes(carryOver ? JSON.parse(carryOver) : null, forType, steerType, measuredWith));
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
          capMonitorOutput: getCapMonitorOutput(SteerOutputType.DEFAULT, output.raw),
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
          capMonitorOutput: getCapMonitorOutput(SteerOutputType.STEERED, output.raw),
        },
      });

      toReturnResult.id = dbResult.id;
      console.log(`steer saved: ${dbResult.id}`);

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
): AsyncGenerator<SteerCompletionChatResponse> {
  while (true) {
    // eslint-disable-next-line
    const { done, value } = await stream.read();
    if (done) {
      break;
    }

    try {
      const parsed = JSON.parse(value.data);
      const toYield = parsed as SteerCompletionChatResponse;
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
  supersededRows?: Partial<Record<SteerOutputType, { outputText: string; capMonitorOutput: string | null }>>,
  /** Readout axes, sent with the request: inference resolves none by name. */
  reads: NPVectorRead[] = [],
  measuredWith?: AxisProvenance,
  /**
   * Readouts already on `toReturnResult` from a cache hit on the other steer type.
   *
   * Passed in rather than read back off `toReturnResult.axes`, which is now a mapped public shape.
   * Without them a request that found DEFAULT cached and generated STEERED would store only the
   * half it generated.
   */
  cachedReadouts: SteerVectorReadout[] = [],
  /** The rows behind `reads`, for the labels inference does not send back. */
  definitions: PersonaAxisDefinition[] = [],
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
    reads,
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

  // The readouts in the inference shape, which is what gets persisted and what the two public views
  // are mapped from. Seeded with whatever a cache hit on the other steer type already produced.
  const axisReadouts: SteerVectorReadout[] = [...cachedReadouts];

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
      // Axis readouts arrive on the last frame of each stream, and default and steered are
      // separate streams, so a readout is identified by (axis id, steer type): merging on the
      // type alone would let the second stream's DEFAULT frame drop the first's STEERED one.
      if (Array.isArray(value.readouts) && value.readouts.length > 0) {
        for (const readout of labelReadouts(value.readouts as SteerVectorReadout[], definitions)) {
          const at = axisReadouts.findIndex((item) => item.id === readout.id && item.type === readout.type);
          if (at >= 0) {
            axisReadouts[at] = readout;
          } else {
            axisReadouts.push(readout);
          }
        }
        setAxesOnResult(toReturnResult, axisReadouts);
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
        axisReadouts,
        supersededRows,
        measuredWith,
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
  // Set by /api/steer-load for completions saved before STEER_COMPLETION_VERSION, whose raw text
  // already starts with inputText. Those must be rendered as-is rather than after the prompt.
  outputTextIncludesPrompt?: boolean;
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
  /** One entry per requested axis per steer type. */
  axes?: PublicAxisReadout[];
  /**
   * The pre-`axes` view of the same readouts, one entry per steer type with values keyed by display
   * title. Deprecated and derived, never a separate measurement, but this endpoint has callers
   * outside this repo whose field names are a contract of their own.
   */
  assistant_axis?: LegacyAssistantAxis[];
};

/**
 * Set both the current and the deprecated view of a set of readouts on a result.
 *
 * Takes the inference readouts and owns the mapping into both public shapes, so the accumulation a
 * caller does stays in the inference type. Reading the accumulated set back off `result.axes` is no
 * longer possible, deliberately: it is a mapped shape now, and mapping back would be a second
 * translation to keep in step.
 */
function setAxesOnResult(result: SteerResultChat, readouts: SteerVectorReadout[]) {
  result.axes = axisReadoutsToPublic(readouts);
  result.assistant_axis = readouts.length > 0 ? axisReadoutsToLegacyAssistantAxis(readouts) : undefined;
}

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
  // See the note in /api/steer: no backend applies this any more, so it is undocumented and
  // optional, but it keeps a default because it is part of the saved-output lookup key.
  freq_penalty: number()
    .min(STEER_FREQUENCY_PENALTY_MIN)
    .max(STEER_FREQUENCY_PENALTY_MAX)
    .default(STEER_FREQUENCY_PENALTY),
  seed: number().min(-100000000).max(100000000).required(),
  strength_multiplier: number().min(0).max(STEER_STRENGTH_MULTIPLIER_MAX).required(),
  steer_special_tokens: bool().required(),
  stream: bool().default(false),
  steer_method: string().oneOf(Object.values(NPSteerMethod)).default(STEER_METHOD),
  // Names of readout axes to measure on the generated turns, resolved against the `Vector` rows
  // for this model. A name with no row is a 400; inference resolves none of its own.
  axes: array().of(string().required()).optional(),
  // Deprecated in favour of `axes: ['lu_assistant-axis']`, and still accepted because this endpoint
  // has callers outside this repo. Ignored when `axes` is non-empty.
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

    // Readout requests get their own prompt and completion limits: a readout is measured per
    // assistant turn, so the conversations are longer than a one-shot steer. Keyed off whether
    // any axis was asked for rather than off the deprecated boolean, so the same work gets the
    // same limits whichever field the caller used to ask for it.
    const axisIds = requestedAxisIds(body);
    const hasAxes = axisIds.length > 0;

    // Calculate total length of all chat messages
    const totalDefaultChars = body.defaultChatMessages.reduce((sum, message) => sum + message.content.length, 0);
    const totalSteeredChars = body.steeredChatMessages.reduce((sum, message) => sum + message.content.length, 0);

    // Check if total length exceeds the maximum allowed
    let maxPromptChars = STEER_MAX_PROMPT_CHARS;
    if (hasAxes) {
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
    } else if (hasAxes) {
      if (body.n_tokens > STEER_N_COMPLETION_TOKENS_MAX_ASSISTANT_AXIS) {
        return NextResponse.json(
          { message: `For readout requests the max n_tokens is ${STEER_N_COMPLETION_TOKENS_MAX_ASSISTANT_AXIS}` },
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
    // The readouts in the inference shape, accumulated across the two steer types as cache hits and
    // fresh generations land. `toReturnResult.axes` is a mapped public shape and cannot be read back
    // as this, so what is stored and what is displayed both derive from here.
    const axisReadouts: SteerVectorReadout[] = [];
    // A requested axis is a `Vector` row, and is sent to inference with the request. Adding an
    // axis is then a row rather than a deploy, and which fit measured a turn is something this
    // database knows rather than something a serving pod's disk decides.
    //
    // Only the labels are read here. Deciding whether a cached row still answers the request needs
    // the live version and nothing else, and most turns of a conversation are answered from cache
    // -- reading the vectors now would pull a few hundred kilobytes of floats out of Postgres for
    // every one of them. The fits are read once there is a generation to run.
    const axisDefinitions = await getPersonaAxisDefinitions(modelId, axisIds, request.user);
    const measuredWith: AxisProvenance = Object.fromEntries(axisDefinitions.map((axis) => [axis.name, axis.id]));
    const rowAxisIds = axisIds.filter((id) => id in measuredWith);

    // A name with no row is refused rather than dropped. Inference resolves no axis by name, so
    // an unmatched name would otherwise reach it as nothing at all and come back as a chart
    // silently missing the axis the caller asked about -- indistinguishable from one that read
    // zero. This used to be the pod's 400, and it has to stay somebody's.
    const unknownAxisIds = axisIds.filter((id) => !(id in measuredWith));
    if (unknownAxisIds.length > 0) {
      return NextResponse.json(
        { message: `Unknown readout axis ${unknownAxisIds.join(', ')} for model ${modelId}` },
        { status: 400 },
      );
    }

    // check for saved outputs

    /**
     * Whether a cached row already holds every axis this request asks for.
     *
     * The lookup key below is the generation's settings -- prompt, seed, temperature, features --
     * and says nothing about which readouts were measured. So a row saved when only one axis was
     * asked for is a perfect hit for the text and silently short of the readouts, and returning it
     * leaves the caller with a chart missing the axis it asked about and no way to tell that from
     * an axis that read zero. A row that does not cover the request is regenerated, and its stored
     * readouts are merged with the new ones rather than replaced.
     *
     * A reading that did not record which row measured it does not cover the request either: it
     * was taken by whatever asset a serving pod had on disk at the time, which is not this fit and
     * often reported no percentile at all -- so a first turn answered from such a row came back
     * with raw measurements while every later turn of the same conversation was regenerated and
     * came back with percentiles. Regenerating those costs one generation per stored conversation
     * and does not repeat, since the row written in its place records the id.
     */
    const coversRequestedAxes = (capMonitorOutput: string | null): boolean => {
      if (axisIds.length === 0) return true;
      if (!capMonitorOutput) return false;
      const parsed = JSON.parse(capMonitorOutput);
      const stored = new Set(storedAxisIds(parsed));
      const storedRows = storedAxisRowIds(parsed);
      return axisIds.every((id) => stored.has(id) && storedRows[id] === measuredWith[id]);
    };

    // Rows that matched the generation key but lacked a requested readout, so this request is
    // regenerating over them. Their readouts are carried into the new rows; see
    // `saveSteerChatOutput`.
    const supersededRows: Partial<Record<SteerOutputType, { outputText: string; capMonitorOutput: string | null }>> =
      {};

    // check for default saved output
    let steerTypesToRun: SteerOutputType[] = [SteerOutputType.STEERED, SteerOutputType.DEFAULT];
    // sort each chat message by content key, then role key so we can do an accurate lookup
    // this is because we store in the db using JSON.stringify and dictionaries are not ordered
    const defaultChatMessagesSorted = sortChatMessages(body.defaultChatMessages);
    // findMany rather than findFirst: saving always creates a row, so several rows can share this
    // generation key while differing in which readouts they stored. Picking an arbitrary one can
    // return a stale row short of a requested axis while a complete row sits next to it, and since
    // that miss regenerates and writes yet another row, the request never converges. Prefer a row
    // that covers the request, exactly as the steered lookup below does.
    const savedSteerDefaultOutputs = await prisma.steerOutput.findMany({
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
    // default already exists, and covers the requested readouts, so don't run it
    const savedSteerDefaultOutput =
      savedSteerDefaultOutputs.find((output) => coversRequestedAxes(output.capMonitorOutput)) ??
      savedSteerDefaultOutputs[0] ??
      null;
    const defaultCovers = savedSteerDefaultOutput
      ? coversRequestedAxes(savedSteerDefaultOutput.capMonitorOutput)
      : false;
    if (savedSteerDefaultOutput && defaultCovers) {
      console.log('has saved default output, setting it');
      toReturnResult[SteerOutputType.DEFAULT] = {
        raw: savedSteerDefaultOutput.outputText,
        chatTemplate: JSON.parse(savedSteerDefaultOutput.outputTextChatTemplate || '[]'),
        logprobs: savedSteerDefaultOutput.logprobs ? JSON.parse(savedSteerDefaultOutput.logprobs) : null,
      };
      if (savedSteerDefaultOutput.capMonitorOutput) {
        const cached = axisReadoutsFromStored(
          JSON.parse(savedSteerDefaultOutput.capMonitorOutput),
          SteerOutputType.DEFAULT,
        );
        // Only what was asked for: a row may cover more axes than this request wants, and
        // returning the extras would put lines on the chart nobody selected.
        const wanted = cached.filter((readout) => axisIds.includes(readout.id));
        axisReadouts.push(...wanted);
        setAxesOnResult(toReturnResult, axisReadouts);
      }
      steerTypesToRun = steerTypesToRun.filter((type) => type !== SteerOutputType.DEFAULT);
    } else if (savedSteerDefaultOutput) {
      console.log('has saved default output but it lacks a requested axis; regenerating');
      supersededRows[SteerOutputType.DEFAULT] = {
        outputText: savedSteerDefaultOutput.outputText,
        capMonitorOutput: savedSteerDefaultOutput.capMonitorOutput,
      };
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

    // Prefer a row that already holds every requested readout. Several rows can match the
    // generation key, and they need not have been measured for the same axes, so this picks a
    // usable one instead of taking the first and finding it short.
    const savedSteered =
      savedSteerSteeredOutputs.find((output) => coversRequestedAxes(output.capMonitorOutput)) ?? null;
    if (savedSteered) {
      console.log('has saved steered output, setting it');
      toReturnResult[SteerOutputType.STEERED] = {
        raw: savedSteered.outputText,
        chatTemplate: JSON.parse(savedSteered.outputTextChatTemplate || '[]'),
        logprobs: savedSteered.logprobs ? JSON.parse(savedSteered.logprobs) : null,
      };
      toReturnResult.id = savedSteered.id;
      toReturnResult.shareUrl = `${NEXT_PUBLIC_URL}/steer/${savedSteered.id}`;
      if (savedSteered.capMonitorOutput) {
        const cached = axisReadoutsFromStored(JSON.parse(savedSteered.capMonitorOutput), SteerOutputType.STEERED);
        const wanted = cached.filter((readout) => axisIds.includes(readout.id));
        axisReadouts.push(...wanted);
        setAxesOnResult(toReturnResult, axisReadouts);
      }

      steerTypesToRun = steerTypesToRun.filter((type) => type !== SteerOutputType.STEERED);
    } else if (savedSteerSteeredOutputs.length > 0) {
      console.log('has saved steered output but none covers a requested axis; regenerating');
      supersededRows[SteerOutputType.STEERED] = {
        outputText: savedSteerSteeredOutputs[0].outputText,
        capMonitorOutput: savedSteerSteeredOutputs[0].capMonitorOutput,
      };
    }

    // Nothing to steer with means nothing to generate a steered column from, so drop it. This has
    // to happen before the branch below rather than after: a streamed request sent one inference
    // call per type, so a STEERED type left in here asked the inference server to steer with no
    // features and came back an error, which is what a readout-only chat looks like.
    if (featuresWithVectors.length === 0) {
      steerTypesToRun = steerTypesToRun.filter((type) => type !== SteerOutputType.STEERED);
    }

    if (steerTypesToRun.length === 0) {
      return NextResponse.json(toReturnResult);
    }

    // Something is being generated, so the axes have to be measured: read the vectors now.
    const reads = (await getPersonaAxisFits(modelId, rowAxisIds, request.user)).map(personaAxisToVectorRead);

    if (body.stream) {
      const generator = generateResponse(
        body,
        toReturnResult,
        savedSteerDefaultOutput?.id,
        steerTypesToRun,
        featuresWithVectors,
        request.user,
        hasVector,
        supersededRows,
        reads,
        measuredWith,
        axisReadouts,
        axisDefinitions,
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
      reads,
    );
    steerCompletionResults = steerCompletionResults as SteerCompletionChatResponse[];
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
      if (result.readouts && result.readouts.length > 0) {
        axisReadouts.push(...labelReadouts(result.readouts, axisDefinitions));
        setAxesOnResult(toReturnResult, axisReadouts);
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
      axisReadouts,
      supersededRows,
      measuredWith,
    );

    // return the result
    return NextResponse.json(toReturnResult);
  } catch (error) {
    if (error instanceof ValidationError) {
      console.log('validation error', error);
      return NextResponse.json({ message: error.message }, { status: 400 });
    }
    // The inference server's own message is the actionable one — e.g. that this model
    // has no chat template and the caller wants /api/steer instead.
    if (error instanceof InferenceServerError) {
      console.log('inference error', error.status, error.message);
      return NextResponse.json({ message: error.message }, { status: error.status });
    }
    console.log('unknown error', error);
    return NextResponse.json({ message: 'Unknown Error' }, { status: 500 });
  }
});
