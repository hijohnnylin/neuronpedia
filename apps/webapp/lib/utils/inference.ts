/* eslint-disable no-var */

import type { paths } from '@/lib/api/inference';
import type {
  ActivationAttentionResponse,
  ActivationTopkByTokenResponse,
  NPSteerMethod,
  SteerCompletionResponse,
  UtilSaeVectorResponse,
} from '@/lib/api/inference-types';
import { NPSteerType } from '@/lib/api/inference-types';
import { getTransformerLensModelIdIfExists } from '@/lib/db/model';
import { getNeuronOnly } from '@/lib/db/neuron';
import { getSourceSetNameFromSource } from '@/lib/utils/source';
import {
  ChatMessage,
  replaceSteerModelIdIfNeeded,
  STEER_METHOD,
  STEER_N_LOGPROBS,
  SteerFeature,
} from '@/lib/utils/steer';
import { AuthenticatedUser } from '@/lib/with-user';
import { NeuronPartial, NeuronPartialWithRelations } from '@/prisma/generated/zod';
import { InferenceEngine, SteerOutputType } from '@prisma/client';
import * as Sentry from '@sentry/nextjs';
import createClient from 'openapi-fetch';
import {
  getAllInstanceHostsForModel,
  getAllServerHostsForModel,
  getFirstInstanceHostForModel,
  getOneRandomServerHostForModel,
  getOneRandomServerHostForSource,
  getOneRandomServerHostForSourceSet,
  getTwoRandomServerHostsForModel,
  getTwoRandomServerHostsForSourceSet,
  LOCALHOST_INFERENCE_HOST,
} from '../db/inference-host-source';
import { INFERENCE_SERVER_SECRET, USE_LOCALHOST_INFERENCE } from '../env';
import { LensPromptRequest } from './lens';
import { NeuronIdentifier } from './neuron-identifier';

/**
 * An error the inference server returned, carrying its status and message.
 *
 * Without this the routes collapse every non-2xx into `500 Unknown Error`, which
 * throws away the half of the response that tells the caller what to do — e.g.
 * "this model's tokenizer has no chat template, use /api/steer with a raw prompt".
 */
export class InferenceServerError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'InferenceServerError';
    this.status = status;
  }
}

/** Inference errors are `{ error: string }` on every endpoint; fall back to the status text. */
const messageFromInferenceErrorBody = (body: unknown, response: Response): string => {
  if (body && typeof body === 'object') {
    const { error, detail } = body as { error?: unknown; detail?: unknown };
    if (typeof error === 'string') {
      return error;
    }
    if (typeof detail === 'string') {
      return detail;
    }
  }
  return response.statusText || `Inference request failed (${response.status})`;
};

const readInferenceError = async (response: Response): Promise<string> => {
  try {
    return messageFromInferenceErrorBody(await response.json(), response);
  } catch {
    // Non-JSON body (a proxy error page, an empty 502).
    return response.statusText || `Inference request failed (${response.status})`;
  }
};

/** Raise a forwardable error for a non-2xx response from a raw `fetch` to inference. */
export const throwIfInferenceError = async (response: Response): Promise<void> => {
  if (response.ok) {
    return;
  }
  throw new InferenceServerError(response.status, await readInferenceError(response));
};

/**
 * Rethrow a transport-level failure as an {@link InferenceServerError} where it carries a
 * response, and untouched otherwise so a DNS or connect failure still reads as one.
 */
export const rethrowAsInferenceError = async (error: unknown): Promise<never> => {
  const response = (error as { response?: Response } | null)?.response;
  if (response instanceof Response) {
    throw new InferenceServerError(response.status, await readInferenceError(response));
  }
  throw error;
};

// The /v1 prefix is part of the paths in the spec, so it is not in the base URL.
export const makeInferenceServerApiWithServerHost = (serverHost: string) =>
  createClient<paths>({
    baseUrl: USE_LOCALHOST_INFERENCE ? LOCALHOST_INFERENCE_HOST : serverHost,
    headers: {
      'X-SECRET-KEY': INFERENCE_SERVER_SECRET,
      'Accept-Encoding': 'gzip',
    },
  });

type InferenceResult<T> = { data?: T; error?: unknown; response: Response };

// openapi-fetch reports a failed request in the result rather than by rejecting, where the
// generated client threw a ResponseError. Every caller above these signals failure by letting
// an exception reach a catch block, so without this a 500 from inference would arrive as
// `data === undefined` and surface much later as a TypeError on a missing field.
//
// The message comes from inference's own `{"error": ...}`, which is written for a caller to
// read -- "this model's tokenizer has no chat template, use /api/steer with a raw prompt" is
// the kind of thing that would be lost by collapsing everything into a generic 502.
export async function unwrapInferenceResponse<T>(result: Promise<InferenceResult<T>>): Promise<T> {
  const { data, error, response } = await result;
  if (error !== undefined || data === undefined) {
    throw new InferenceServerError(response.status, messageFromInferenceErrorBody(error, response));
  }
  return data;
}

/**
 * The JSON request body the spec declares for a POST path, with every field optional.
 *
 * `Partial` is deliberate. `openapi-typescript` marks a field required whenever the schema
 * gives it a default, because it is describing the shape the server has *after* applying
 * defaults — but a client is free to omit exactly those. Requiring them here would mean
 * restating the server's defaults in the webapp, which is the duplication this whole setup
 * exists to remove. Unknown and misspelled keys are still rejected, which is the failure this
 * needs to catch; a genuinely missing required field surfaces immediately as a 422.
 */
type InferenceRequestBody<P extends keyof paths> = paths[P] extends {
  post: { requestBody: { content: { 'application/json': infer B } } };
}
  ? Partial<B>
  : never;

/**
 * POST to inference with a spec-checked body, returning the raw `Response`.
 *
 * The streaming endpoints cannot go through `openapi-fetch`, which parses a whole body before
 * handing it back — these need the `ReadableStream` intact. That used to mean a hand-rolled
 * `fetch` with a hand-written object literal and no checking at all, which is how several of
 * them drifted into sending snake_case field names that only still worked because the server
 * accepts either. Typing the body against `paths` closes that hole without touching how the
 * response is consumed.
 */
function postInferenceStreaming<P extends keyof paths>(
  host: string,
  path: P,
  body: InferenceRequestBody<P>,
  init?: { signal?: AbortSignal },
): Promise<Response> {
  // `paths` keys already carry the /v1 prefix.
  return fetch(`${host}${String(path)}`, {
    method: 'POST',
    cache: 'no-cache',
    headers: {
      'Content-Type': 'application/json',
      'X-SECRET-KEY': INFERENCE_SERVER_SECRET,
    },
    body: JSON.stringify(body),
    signal: init?.signal,
  });
}

export type InferenceActivationResultMultiple = {
  tokens: string[];
  activations: {
    layer: string;
    index: number;
    values: number[];
    maxValue: number;
    maxValueIndex: number;
    sumValues?: number | undefined;
    dfaValues?: number[] | undefined;
    dfaTargetIndex?: number | undefined;
    dfaMaxValue?: number | undefined;
  }[];
  error: string | undefined;
};

export type SearchTopKResult = {
  source: string;
  results: {
    position: number;
    token: string;
    // From the inference server's tokenizer: BOS, EOS, padding, turn markers and
    // the like, as opposed to content.
    isSpecial: boolean;
    topFeatures: {
      activationValue: number;
      featureIndex: number;
      feature: NeuronPartialWithRelations | undefined;
    }[];
  }[];
};

function convertSteerFeatureVectorsToInferenceVectors(steerFeatures: SteerFeature[]) {
  // Features with no vector are dropped rather than sent with the field missing, which the
  // server rejects as a 422. Callers gate on `hasVector`, so this should not fire in practice.
  return steerFeatures.flatMap((feature) =>
    feature.neuron?.vector
      ? [
          {
            hook: feature.neuron.hookName || '',
            steeringVector: feature.neuron.vector,
            strength: feature.strength,
          },
        ]
      : [],
  );
}

export const getCosSimForFeature = async (
  feature: NeuronIdentifier,
  targetModelId: string,
  targetSourceId: string,
  user: AuthenticatedUser | null,
) => {
  // get if it's a feature/vector first
  const result = await getNeuronOnly(feature.modelId, feature.layer, feature.index);

  if (result?.hasVector) {
    // if it's a vector, then we can use any server that has the same modelId, since we don't need the SAE to be loaded

    var [serverHost, _] = await getTwoRandomServerHostsForModel(targetModelId);
  } else {
    // if it's not a vector, then we need to use the source set's host
    var serverHost = await getOneRandomServerHostForSource(targetModelId, targetSourceId, user);
  }

  const transformerLensModelId = await getTransformerLensModelIdIfExists(targetModelId);

  return makeInferenceServerApiWithServerHost(serverHost).POST('/v1/util/sae-topk-by-decoder-cossim', {
    body: {
      ...(result?.hasVector
        ? {
            vector: result.vector,
          }
        : {
            feature: {
              model: feature.modelId,
              source: feature.layer,
              index: parseInt(feature.index, 10),
            },
          }),
      model: transformerLensModelId,
      source: targetSourceId,
      numResults: 10,
    },
  });
};

type ActivationForFeatureResult = {
  tokens: string[];
  values: number[];
  maxValue: number;
  minValue: number;
  maxValueTokenIndex: number;
  dfaValues?: number[];
  dfaTargetIndex?: number;
  dfaMaxValue?: number;
};

// Drop a leading special token from an activation result. The inference server
// returns it because the model actually sees it, but for demo/quiz surfaces it's
// noise, and dropping it here means clients never have to know what one looks
// like. `isSpecial` is the server's own answer, derived from the tokenizer's
// special-token ids, so this needs no list of literals. Recomputes the max/DFA
// indices so they still point at the right token after the shift.
function dropBosFromActivation<T extends ActivationForFeatureResult>(
  activation: T,
  isSpecial: boolean[] | undefined,
): T {
  if (activation.tokens.length === 0 || !isSpecial?.[0]) {
    return activation;
  }
  const values = activation.values.slice(1);
  const dfaValues = activation.dfaValues ? activation.dfaValues.slice(1) : undefined;
  const maxValue = values.length > 0 ? Math.max(...values) : 0;
  return {
    ...activation,
    tokens: activation.tokens.slice(1),
    values,
    maxValue,
    minValue: values.length > 0 ? Math.min(...values) : 0,
    maxValueTokenIndex: values.indexOf(maxValue),
    ...(dfaValues
      ? (() => {
          const dfaMaxValue = dfaValues.length > 0 ? Math.max(...dfaValues) : 0;
          return { dfaValues, dfaMaxValue, dfaTargetIndex: dfaValues.indexOf(dfaMaxValue) };
        })()
      : {}),
  };
}

export const getActivationForFeature = async (
  feature: NeuronPartial,
  defaultTestText: string | string[],
  user: AuthenticatedUser | null,
  // When true, strip the leading BOS token from the result (see
  // `dropBosFromActivation`). Used by the Gemma Scope demo surfaces.
  ignoreBos = false,
) => {
  if (!feature.modelId || !feature.layer || !feature.index) {
    throw new Error('Invalid feature');
  }

  // Inference failures here name a model internal and nothing else — "hook
  // 'blocks.19.ln2.hook_normalized' has no canonical point" says which hook was rejected but not
  // which feature asked for it, and Sentry receives no request body for App Router routes, so
  // without this the report is unactionable. These three fields are enough to find the neuron, and
  // deliberately exclude the caller's text.
  Sentry.setContext('feature', {
    modelId: feature.modelId,
    source: feature.layer,
    index: feature.index,
  });

  // get if it's a feature/vector first
  const result = await getNeuronOnly(feature.modelId, feature.layer, feature.index);

  if (result?.hasVector) {
    // if it's a vector, then we can use any server that has the same modelId, since we don't need the SAE to be loaded

    var [serverHost, _] = await getTwoRandomServerHostsForModel(feature.modelId);
  } else {
    // if it's not a vector, then we need to use the source set's host
    var serverHost = await getOneRandomServerHostForSource(feature.modelId, feature.layer, user);
  }

  const modelIdForSearcher = replaceSteerModelIdIfNeeded(feature.modelId);
  const transformerLensModelId = await getTransformerLensModelIdIfExists(modelIdForSearcher);

  if (Array.isArray(defaultTestText)) {
    return unwrapInferenceResponse(
      makeInferenceServerApiWithServerHost(serverHost).POST('/v1/activation/single-batch', {
        body: result?.hasVector
          ? {
              prompts: defaultTestText,
              model: transformerLensModelId,
              vector: result.vector,
              hook: result.hookName || '',
            }
          : {
              prompts: defaultTestText,
              model: transformerLensModelId,
              source: feature.layer,
              index: feature.index,
            },
      }),
    )
      .then((result) =>
        result.results.map((result) => {
          const { tokens } = result;
          const activations = result.activation.values;
          const activation = {
            tokens,
            values: activations,
            maxValue: Math.max(...activations),
            minValue: Math.min(...activations),
            modelId: feature.modelId || '',
            layer: feature.layer || '',
            index: feature.index || '',
            creatorId: user?.id || '',
            dataIndex: null,
            dataSource: 'Neuronpedia',
            maxValueTokenIndex: activations.indexOf(Math.max(...activations)),
            createdAt: new Date(),
            dfaValues: result.activation.dfaValues ?? undefined,
            dfaTargetIndex: result.activation.dfaTargetIndex ?? undefined,
            dfaMaxValue: result.activation.dfaMaxValue ?? undefined,
          };
          return ignoreBos ? dropBosFromActivation(activation, result.tokensIsSpecial) : activation;
        }),
      )
      .catch((error) => {
        console.error(error);
        throw error;
      });
  }
  return unwrapInferenceResponse(
    makeInferenceServerApiWithServerHost(serverHost).POST('/v1/activation/single', {
      body: result?.hasVector
        ? {
            prompt: defaultTestText,
            model: transformerLensModelId,
            vector: result.vector,
            hook: result.hookName || '',
          }
        : {
            prompt: defaultTestText,
            model: transformerLensModelId,
            source: feature.layer,
            index: feature.index,
          },
    }),
  )
    .then((result) => {
      const { tokens } = result;
      const activations = result.activation.values;
      const activation = {
        tokens,
        values: activations,
        maxValue: Math.max(...activations),
        minValue: Math.min(...activations),
        modelId: feature.modelId || '',
        layer: feature.layer || '',
        index: feature.index || '',
        creatorId: user?.id || '',
        dataIndex: null,
        dataSource: 'Neuronpedia',
        maxValueTokenIndex: activations.indexOf(Math.max(...activations)),
        createdAt: new Date(),
        dfaValues: result.activation.dfaValues ?? undefined,
        dfaTargetIndex: result.activation.dfaTargetIndex ?? undefined,
        dfaMaxValue: result.activation.dfaMaxValue ?? undefined,
      };
      return ignoreBos ? dropBosFromActivation(activation, result.tokensIsSpecial) : activation;
    })
    .catch((error) => {
      console.error(error);
      throw error;
    });
};

export const runInferenceActivationSource = async (
  modelId: string,
  source: string,
  prompts: string[],
  user: AuthenticatedUser | null,
) => {
  const serverHost = await getOneRandomServerHostForSource(modelId, source, user);
  if (!serverHost) {
    throw new Error('No server host found');
  }

  const transformerLensModelId = await getTransformerLensModelIdIfExists(modelId);

  return makeInferenceServerApiWithServerHost(serverHost).POST('/v1/activation/source', {
    body: {
      prompts,
      model: transformerLensModelId,
      source,
    },
  });
};

export const runInferenceActivationAll = async (
  modelId: string,
  sourceSetName: string,
  text: string | string[],
  numResults: number,
  selectedLayers: string[],
  sortIndexes: number[],
  ignoreBos: boolean,
  user: AuthenticatedUser | null,
) => {
  // TODO: we don't currently support search-all on different instances
  const serverHost = await getOneRandomServerHostForSourceSet(modelId, sourceSetName, user);
  if (!serverHost) {
    throw new Error('No server host found');
  }

  const transformerLensModelId = await getTransformerLensModelIdIfExists(modelId);

  if (Array.isArray(text)) {
    return unwrapInferenceResponse(
      makeInferenceServerApiWithServerHost(serverHost).POST('/v1/activation/all-batch', {
        body: {
          prompts: text,
          model: transformerLensModelId,
          selectedSources: selectedLayers,
          sortByTokenIndexes: sortIndexes,
          sourceSet: sourceSetName,
          ignoreBos,
          numResults,
        },
      }),
    );
  }
  return unwrapInferenceResponse(
    makeInferenceServerApiWithServerHost(serverHost).POST('/v1/activation/all', {
      body: {
        prompt: text,
        model: transformerLensModelId,
        selectedSources: selectedLayers,
        sortByTokenIndexes: sortIndexes,
        sourceSet: sourceSetName,
        ignoreBos,
        numResults,
      },
    }),
  );
};

// TODO: steerCompletion should also support parallel inference with two servers
export const steerCompletion = async (
  modelId: string,
  steerTypesToRun: SteerOutputType[],
  prompt: string,
  strengthMultiplier: number,
  n_tokens: number,
  temperature: number,
  freq_penalty: number,
  seed: number,
  steerFeatures: SteerFeature[],
  hasVector: boolean,
  user: AuthenticatedUser | null,
  steerMethod: NPSteerMethod = STEER_METHOD,
  stream: boolean = true,
  n_logprobs: number = STEER_N_LOGPROBS,
) => {
  // get the sae set's host
  const firstFeatureLayer = steerFeatures[0].layer;

  let serverHost: string | null = null;
  if (hasVector) {
    // if we have the vectors, then we can use any server that has the same modelId, since we don't need the SAE to be loaded
    serverHost = await getOneRandomServerHostForModel(modelId);
  } else {
    serverHost = await getOneRandomServerHostForSourceSet(modelId, getSourceSetNameFromSource(firstFeatureLayer), user);
  }
  if (!serverHost) {
    throw new Error('No server host found');
  }

  const transformerLensModelId = await getTransformerLensModelIdIfExists(modelId);

  const response = await postInferenceStreaming(serverHost, '/v1/steer/completion', {
    types: steerTypesToRun.map((type) =>
      type === SteerOutputType.DEFAULT ? NPSteerType.DEFAULT : NPSteerType.STEERED,
    ),
    prompt,
    model: transformerLensModelId,
    features: hasVector
      ? undefined
      : steerFeatures.map((feature) => ({
          model: feature.modelId,
          source: feature.layer,
          index: feature.index,
          strength: feature.strength,
        })),
    vectors: hasVector ? convertSteerFeatureVectorsToInferenceVectors(steerFeatures) : undefined,
    strengthMultiplier,
    nCompletionTokens: n_tokens,
    temperature,
    freqPenalty: freq_penalty,
    seed,
    steerMethod,
    normalizeSteering: false,
    stream,
    nLogprobs: n_logprobs,
  });
  await throwIfInferenceError(response);
  if (!response.body) {
    throw new Error('No response body');
  }

  if (stream) {
    return response.body;
  }
  const result = await response.json();
  return result as SteerCompletionResponse;
};

export const steerCompletionChat = async (
  modelId: string,
  steerTypesToRun: SteerOutputType[],
  defaultChatMessages: ChatMessage[],
  steeredChatMessages: ChatMessage[],
  strengthMultiplier: number,
  nTokens: number,
  temperature: number,
  freqPenalty: number,
  seed: number,
  steerSpecialTokens: boolean,
  steerFeatures: SteerFeature[],
  hasVector: boolean,
  user: AuthenticatedUser | null,
  stream: boolean,
  steerMethod: NPSteerMethod = STEER_METHOD,
  n_logprobs: number = STEER_N_LOGPROBS,
  isAssistantAxis: boolean = false,
) => {
  // record start time
  const startTime = new Date().getTime();

  if (isAssistantAxis) {
    // The axis is a vector, so any instance of the model can serve it: take the first one
    // registered for the model rather than requiring a particular engine.
    const assistantAxisHost = await getFirstInstanceHostForModel(modelId);
    if (!assistantAxisHost) {
      throw new Error('No hosts found.');
    }
    var [serverHostDefault, serverHostSteered] = [assistantAxisHost, assistantAxisHost];
  } else if (hasVector || steerFeatures.length === 0) {
    // if we have the vectors, then we can use any server that has the same modelId, since we don't need the SAE to be loaded
    [serverHostDefault, serverHostSteered] = await getTwoRandomServerHostsForModel(modelId);
  } else {
    // get the sae set's host
    const firstFeatureLayer = steerFeatures[0].layer;
    // if we have just one server, then just use that server
    [serverHostDefault, serverHostSteered] = await getTwoRandomServerHostsForSourceSet(
      modelId,
      getSourceSetNameFromSource(firstFeatureLayer),
      user,
    );
  }

  // make the promises to run
  // check if we need to replace "gemma-2-2b-it" with "gemma-2-2b", since we don't have SAEs for "-it"
  const modelIdForSearcher = replaceSteerModelIdIfNeeded(modelId);
  const transformerLensModelId = await getTransformerLensModelIdIfExists(modelIdForSearcher);

  if (stream) {
    const hasTwoServers = serverHostDefault !== serverHostSteered;

    // Always send one request per steer type so default and steered generate simultaneously.
    // A combined request makes the inference server loop over the types one after the other,
    // so the second column only starts filling once the first has finished.
    console.log(
      `completion chat - sending separate requests (hasTwoServers: ${hasTwoServers}, isAssistantAxis: ${isAssistantAxis})`,
    );
    const toRunPromises = steerTypesToRun.map((type) => {
      const host = type === SteerOutputType.DEFAULT ? serverHostDefault : serverHostSteered;
      console.log(`completion chat - sending ${type} to ${host}`);
      return postInferenceStreaming(host, '/v1/steer/completion-chat', {
        types: [type === SteerOutputType.DEFAULT ? NPSteerType.DEFAULT : NPSteerType.STEERED],
        prompt: type === SteerOutputType.DEFAULT ? defaultChatMessages : steeredChatMessages,
        model: transformerLensModelId,
        features: hasVector
          ? undefined
          : steerFeatures.map((feature) => ({
              model: feature.modelId,
              source: feature.layer,
              index: feature.index,
              strength: feature.strength,
            })),
        vectors: hasVector ? convertSteerFeatureVectorsToInferenceVectors(steerFeatures) : undefined,
        strengthMultiplier,
        nCompletionTokens: nTokens,
        temperature,
        freqPenalty,
        seed,
        steerSpecialTokens,
        steerMethod,
        normalizeSteering: false,
        stream: true,
        nLogprobs: n_logprobs,
        isAssistantAxis,
      });
    });
    const responses = await Promise.all(toRunPromises);
    // Checked before any stream is handed back: once the route starts piping bodies to
    // the browser it can no longer set a status code.
    await Promise.all(responses.map(throwIfInferenceError));
    return responses.map((response) => {
      if (!response.body) {
        throw new Error('No response body');
      }
      return response.body;
    });
  }
  const toRunPromises = steerTypesToRun.map((type) => {
    if (type === SteerOutputType.DEFAULT) {
      console.log('does not have saved default output, running it');
      return unwrapInferenceResponse(
        makeInferenceServerApiWithServerHost(serverHostDefault).POST('/v1/steer/completion-chat', {
          body: {
            types: [NPSteerType.DEFAULT],
            prompt: defaultChatMessages,
            model: transformerLensModelId,
            features: hasVector
              ? undefined
              : steerFeatures.map((feature) => ({
                  model: feature.modelId,
                  source: feature.layer,
                  index: feature.index,
                  strength: feature.strength,
                })),
            vectors: hasVector ? convertSteerFeatureVectorsToInferenceVectors(steerFeatures) : undefined,
            strengthMultiplier,
            nCompletionTokens: nTokens,
            temperature,
            freqPenalty,
            seed,
            steerSpecialTokens,
            steerMethod,
            normalizeSteering: false,
            nLogprobs: n_logprobs,
            isAssistantAxis,
            // This path collects whole responses; the SSE variant is lensPromptStream's job.
            stream: false,
          },
        }),
      );
    }
    if (type === SteerOutputType.STEERED) {
      console.log('does not have saved steered output, running it');
      return unwrapInferenceResponse(
        makeInferenceServerApiWithServerHost(serverHostSteered).POST('/v1/steer/completion-chat', {
          body: {
            types: [NPSteerType.STEERED],
            prompt: steeredChatMessages,
            model: transformerLensModelId,
            features: hasVector
              ? undefined
              : steerFeatures.map((feature) => ({
                  model: feature.modelId,
                  source: feature.layer,
                  index: feature.index,
                  strength: feature.strength,
                })),
            vectors: hasVector ? convertSteerFeatureVectorsToInferenceVectors(steerFeatures) : undefined,
            strengthMultiplier,
            nCompletionTokens: nTokens,
            temperature,
            freqPenalty,
            seed,
            steerSpecialTokens,
            steerMethod,
            normalizeSteering: false,
            nLogprobs: n_logprobs,
            isAssistantAxis,
            // This path collects whole responses; the SSE variant is lensPromptStream's job.
            stream: false,
          },
        }),
      );
    }
    throw new Error('Invalid steer type');
  });

  // run the promises
  const inferenceCompletionChatResponses = await Promise.all(toRunPromises).catch(rethrowAsInferenceError);

  // record end time
  const endTime = new Date().getTime();
  console.log(`Time taken: ${endTime - startTime}ms`);

  // No emptiness check: unwrapInferenceResponse throws on a non-2xx or missing body, so a
  // failed pod has already surfaced by here rather than arriving as an undefined entry.
  return inferenceCompletionChatResponses;
};

export const getActivationsTopKByToken = async (
  modelId: string,
  layer: string,
  text: string | string[],
  topK: number,
  ignoreBos: boolean,
  user: AuthenticatedUser | null,
) => {
  const sourceSet = getSourceSetNameFromSource(layer);
  const serverHost = await getOneRandomServerHostForSourceSet(modelId, sourceSet, user);
  if (!serverHost) {
    throw new Error('No server host found');
  }

  const transformerLensModelId = await getTransformerLensModelIdIfExists(modelId);

  if (Array.isArray(text)) {
    return unwrapInferenceResponse(
      makeInferenceServerApiWithServerHost(serverHost).POST('/v1/activation/topk-by-token-batch', {
        body: {
          prompts: text,
          model: transformerLensModelId,
          source: layer,
          topK,
          ignoreBos,
        },
      }),
    );
  }
  const result: ActivationTopkByTokenResponse = await unwrapInferenceResponse(
    makeInferenceServerApiWithServerHost(serverHost).POST('/v1/activation/topk-by-token', {
      body: {
        prompt: text,
        model: transformerLensModelId,
        source: layer,
        topK,
        ignoreBos,
      },
    }),
  );
  return result;
};

export type InferenceAttentionResult = ActivationAttentionResponse;

// Runs custom-text attention for a single (layer, head) on the model's inference
// server. Attention heads aren't tied to a Source, so we use any model-level host
// on a supported engine (TransformerLens or NNsight; not nnsight-vllm/chatspace).
// The /activation/attention endpoint isn't in the typed client, so we call it with a raw fetch
// (like the lens endpoint); the response is still typed from the spec.
export const getAttentionForHead = async (
  modelId: string,
  layer: number,
  headIndex: number,
  prompt: string,
): Promise<InferenceAttentionResult> => {
  let host: string | null = null;
  if (USE_LOCALHOST_INFERENCE) {
    host = LOCALHOST_INFERENCE_HOST;
  } else {
    for (const engine of [InferenceEngine.TRANSFORMER_LENS, InferenceEngine.NNSIGHT]) {
      // eslint-disable-next-line no-await-in-loop
      let hosts = await getAllInstanceHostsForModel(modelId, engine);
      if (hosts.length === 0) {
        // eslint-disable-next-line no-await-in-loop
        hosts = [...new Set(await getAllServerHostsForModel(modelId, engine))];
      }
      if (hosts.length > 0) {
        host = hosts[Math.floor(Math.random() * hosts.length)];
        break;
      }
    }
  }
  if (!host) {
    throw new Error('No inference server host found for this model.');
  }

  const transformerLensModelId = await getTransformerLensModelIdIfExists(modelId);

  const response = await postInferenceStreaming(host, '/v1/activation/attention', {
    model: transformerLensModelId,
    prompt,
    layer,
    head: headIndex,
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => null);
    throw new Error(errorBody?.error || `Inference server error (${response.status})`);
  }

  return (await response.json()) as InferenceAttentionResult;
};

export const tokenizeText = async (modelId: string, text: string, prependBos: boolean) => {
  const serverHost = await getOneRandomServerHostForModel(modelId);
  const transformerLensModelId = await getTransformerLensModelIdIfExists(modelId);

  const result = await makeInferenceServerApiWithServerHost(serverHost).POST('/v1/tokenize', {
    body: {
      model: transformerLensModelId,
      text,
      prependBos,
    },
  });

  return result;
};

export const getVectorFromInstance = async (
  modelId: string,
  source: string,
  index: string,
): Promise<UtilSaeVectorResponse> => {
  const serverHost = await getOneRandomServerHostForSource(modelId, source, null);
  if (!serverHost) {
    throw new Error('No server host found');
  }
  const transformerLensModelId = await getTransformerLensModelIdIfExists(modelId);

  return unwrapInferenceResponse(
    makeInferenceServerApiWithServerHost(serverHost).POST('/v1/util/sae-vector', {
      body: {
        model: transformerLensModelId,
        source,
        index: parseInt(index, 10),
      },
    }),
  );
};

// Streaming logit/Jacobian lens for a prompt. The lens endpoint is not in the
// generated inference client yet, so we call it with a raw fetch (like
// steerCompletion). The endpoint streams NDJSON (one message per line); this
// returns the raw `fetch` Response so the API route can pipe the body straight
// through to the browser without buffering the (potentially large) stream.
//
// A single inference server processes one request at a time (a global model
// lock shared across all endpoints, e.g. /steer and /lens), so a server can be
// busy even when it isn't serving a lens request. To avoid failing when the
// first-chosen server is busy, we try each known host for the model in random
// order, asking each to fail fast (`fail_if_busy` -> HTTP 429) if it's already
// occupied. The first host that accepts the request wins. If every host is
// busy, we fall back to queueing on one host (waiting for the lock, as before)
// so the request is still served rather than rejected. We only surface an error
// when every host hard-fails (connection error / 5xx). Deterministic client
// errors (4xx other than 429) are returned immediately, since retrying another
// host wouldn't change the outcome.
//
// The caller is responsible for handling a non-ok response (`response.ok`).
export const lensPromptStream = async (
  modelId: string,
  request: Omit<LensPromptRequest, 'model'>,
  // Tie the upstream request to the caller's abort signal so a client abort
  // (e.g. the user pressing "Stop") closes the connection to the inference
  // server, letting it stop generating and release its model lock.
  signal?: AbortSignal,
): Promise<Response> => {
  const transformerLensModelId = await getTransformerLensModelIdIfExists(modelId);

  // Build the ordered list of candidate hosts to try.
  let hosts: string[];
  if (USE_LOCALHOST_INFERENCE) {
    hosts = [LOCALHOST_INFERENCE_HOST];
  } else {
    // Use every instance registered against the model (not just those linked to
    // a Source via InferenceHostSourceOnSource) so all interchangeable jlens
    // instances are candidates. Fall back to the source-linked hosts if none.
    hosts = await getAllInstanceHostsForModel(modelId);
    if (hosts.length === 0) {
      hosts = [...new Set(await getAllServerHostsForModel(modelId))];
    }
    // Shuffle (Fisher-Yates) so load is spread across hosts rather than always
    // hammering the first one.
    for (let i = hosts.length - 1; i > 0; i -= 1) {
      const j = Math.floor(Math.random() * (i + 1));
      [hosts[i], hosts[j]] = [hosts[j], hosts[i]];
    }
  }
  if (hosts.length === 0) {
    throw new Error('No server host found');
  }

  const sendRequest = (host: string, failIfBusy: boolean) =>
    postInferenceStreaming(
      host,
      '/v1/lens/prompt',
      { ...request, model: transformerLensModelId, stream: true, failIfBusy },
      { signal },
    );

  let lastErrorResponse: Response | null = null;
  let lastError: unknown = null;
  let firstBusyHost: string | null = null;

  // Pass 1: try each host, skipping any that report busy (429) or hard-fail
  // (connection error / 5xx / 404). Return on the first success or deterministic
  // client error (other 4xx).
  for (let i = 0; i < hosts.length; i += 1) {
    try {
      // eslint-disable-next-line no-await-in-loop
      const response = await sendRequest(hosts[i], true);
      if (response.status === 429) {
        if (firstBusyHost === null) {
          firstBusyHost = hosts[i];
        }
        // Free the connection since we're moving on to the next host.
        void response.body?.cancel();

        continue;
      }
      // A 404 means this host is unavailable (e.g. the instance went down and
      // its proxy/gateway returns "Not Found"), not a deterministic client
      // error — so fall through and try the next host, like a 5xx.
      if (response.status === 404) {
        void lastErrorResponse?.body?.cancel();
        lastErrorResponse = response;

        continue;
      }
      // Success, or a deterministic client error (other 4xx) that won't differ
      // across hosts — return either way. 5xx (and 404 above) fall through to
      // try another host.
      if (response.ok || (response.status >= 400 && response.status < 500)) {
        return response;
      }
      void lastErrorResponse?.body?.cancel();
      lastErrorResponse = response;
    } catch (error) {
      // Network/connection error to this host; try the next one.
      lastError = error;
    }
  }

  // Pass 2: every host was busy and/or hard-failed. If at least one was merely
  // busy, fall back to queueing on that busy host (fail_if_busy=false) so the
  // request is still served (matching the previous single-server behavior of
  // waiting for the lock) rather than rejected. We queue on a host we know was
  // reachable-but-busy rather than one that hard-failed (e.g. 404 because it's
  // down), which would just fail again.
  if (firstBusyHost !== null) {
    void lastErrorResponse?.body?.cancel();
    return sendRequest(firstBusyHost, false);
  }

  // Every host hard-failed (no busy responses): surface the last failure.
  if (lastErrorResponse) {
    return lastErrorResponse;
  }
  throw lastError instanceof Error ? lastError : new Error('All inference servers failed for the lens request');
};
