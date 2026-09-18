import type { paths } from '@/lib/api/inference';
import type {
  ActivationAttentionResponse,
  ActivationTopkByTokenResponse,
  NPSteerMethod,
  NPVectorRead,
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
  STEER_FREQUENCY_PENALTY,
  STEER_METHOD,
  STEER_N_LOGPROBS,
  SteerFeature,
} from '@/lib/utils/steer';
import { AuthenticatedUser } from '@/lib/with-user';
import { NeuronPartial, NeuronPartialWithRelations } from '@/prisma/generated/zod';
import { ComputeService, SteerOutputType } from '@prisma/client';
import * as Sentry from '@sentry/nextjs';
import createClient from 'openapi-fetch';
import { NoComputeHostError, resolveHost, resolveHosts } from '../db/compute-host';
import { INFERENCE_SERVER_SECRET } from '../env';
import { LensPromptRequest } from './lens';
import { NeuronIdentifier } from './neuron-identifier';

// Every lookup in this file is for the inference service; the rest of the
// target varies.
const inferenceTarget = (
  modelId: string,
  narrow?: { sourceId?: string; sourceSetName?: string; user?: AuthenticatedUser | null },
) => ({
  service: ComputeService.INFERENCE,
  modelId,
  ...narrow,
});

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

/**
 * How long a pod has to answer with response HEADERS before we give up on it.
 *
 * Not a limit on the work: a lens run is minutes of streaming, and this stops counting the
 * moment the headers land. It bounds only the wait to find out whether a pod is going to serve
 * us at all, which is what makes failover reachable — undici's default is 300s, longer than the
 * route's own `maxDuration`, so before this a wedged pod took the whole request down with it
 * and the remaining pods were never tried.
 *
 * 8s is short because every fast-fail path on the inference server is now genuinely fast:
 * `failIfBusy` answers 429 without waiting for either a request slot or VRAM. So a pod that has
 * sent nothing in 8s is not busy, it is wedged.
 */
const HEADERS_TIMEOUT_MS = 8_000;

/**
 * Whole-request ceiling for the calls that do not stream.
 *
 * These buffer the entire response, so there is no headers/body split to exploit and no
 * partial result worth waiting for. Exported for the two hand-written fetches that have not
 * moved to the typed client yet.
 */
export const INFERENCE_REQUEST_TIMEOUT_MS = 60_000;

/** True when `error` is this module's own deadline rather than an abort from the caller. */
const isTimeoutAbort = (error: unknown) => (error as { name?: string } | null)?.name === 'TimeoutError';

const withDeadline = (timeoutMs: number, callerSignal?: AbortSignal | null): AbortSignal => {
  const deadline = AbortSignal.timeout(timeoutMs);
  return callerSignal ? AbortSignal.any([callerSignal, deadline]) : deadline;
};

// The /v1 prefix is part of the paths in the spec, so it is not in the base URL.
export const makeInferenceServerApiWithServerHost = (serverHost: string) =>
  createClient<paths>({
    baseUrl: serverHost,
    headers: {
      'X-SECRET-KEY': INFERENCE_SERVER_SECRET,
      'Accept-Encoding': 'gzip',
    },
    // openapi-fetch has no timeout of its own, so without this every non-streaming call
    // inherits undici's 300s and hangs past whatever `maxDuration` the route declared.
    fetch: (request: Request) => fetch(request, { signal: withDeadline(INFERENCE_REQUEST_TIMEOUT_MS, request.signal) }),
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
async function postInferenceStreaming<P extends keyof paths>(
  host: string,
  path: P,
  body: InferenceRequestBody<P>,
  // `headersTimeoutMs` defaults to 0, meaning wait as long as it takes. Opt in only when the
  // reply really does stream: a caller whose whole body lands at once gets its headers when the
  // work is finished, so a deadline there measures how long the answer took and aborts every
  // request slower than it. Not every caller of this helper streams — the attention endpoint
  // uses it only because it is missing from the typed client.
  init?: { signal?: AbortSignal; headersTimeoutMs?: number },
): Promise<Response> {
  // The deadline covers only the wait for headers: `fetch` resolves as soon as they arrive, and
  // the timer is cleared there, so the stream that follows runs for as long as it needs. Built
  // from a controller rather than `AbortSignal.timeout` for exactly that reason — a timeout
  // signal stays armed and would cut the body mid-generation.
  const headersTimeoutMs = init?.headersTimeoutMs ?? 0;
  const deadline = headersTimeoutMs > 0 ? new AbortController() : null;
  const timer = deadline
    ? setTimeout(
        () => deadline.abort(new DOMException(`No response headers after ${headersTimeoutMs}ms`, 'TimeoutError')),
        headersTimeoutMs,
      )
    : undefined;
  const signals = [init?.signal, deadline?.signal].filter((signal) => signal !== undefined);

  try {
    // `paths` keys already carry the /v1 prefix.
    return await fetch(`${host}${String(path)}`, {
      method: 'POST',
      cache: 'no-cache',
      headers: {
        'Content-Type': 'application/json',
        'X-SECRET-KEY': INFERENCE_SERVER_SECRET,
      },
      body: JSON.stringify(body),
      signal: signals.length > 0 ? AbortSignal.any(signals) : undefined,
    });
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Headers deadline for the streaming steer endpoints in pass 1. Off until the pods are rolled.
 *
 * A pod that predates `fail_if_busy` on steer ignores the field, so a saturated one queues
 * silently instead of refusing. The deadline cannot tell that apart from a wedged pod, so it
 * fires, pass 1 walks every host, and each abandoned attempt leaves a generation running on a
 * server that cannot yet notice the client left — N+1 generations for one request, exactly when
 * the fleet is already busy.
 *
 * Little is given up by waiting. Against an upgraded pod the 429 does the real work: busy is
 * answered in milliseconds and pass 1 moves on. The deadline only adds the wedged-pod case, where
 * a host accepts the connection and then never speaks, and the route's `maxDuration` still bounds
 * that. So set this to HEADERS_TIMEOUT_MS once every pod serving steer honours the flag.
 */
const STEER_HEADERS_TIMEOUT_MS = 0;

/**
 * Whether a failed response says anything about the host, or would fail the same everywhere.
 *
 * 404 counts, unlike in `computeFetch`: inference itself never returns one, so a 404 here is a
 * dead pod's gateway answering in its place.
 */
const shouldTryAnotherHost = (status: number) => status >= 500 || status === 408 || status === 429 || status === 404;

/**
 * Candidate hosts starting at `offset`, wrapping around.
 *
 * Steering runs a default and a steered completion at once, and wants them on different pods so
 * they generate side by side rather than queueing behind each other. Giving each a rotation of
 * the same list keeps that while leaving both a full set of fallbacks.
 */
const rotate = (hosts: string[], offset: number) => hosts.map((_, i) => hosts[(i + offset) % hosts.length]);

/**
 * Send to each candidate in turn until one answers, and return that response.
 *
 * Two passes. Pass 1 asks every host to refuse rather than queue (`failIfBusy`), so an occupied
 * or memory-starved pod costs milliseconds and the next one gets a turn. A 2xx or a deterministic
 * 4xx comes back straight away; a 429, a 5xx, a 404 from a dead pod's gateway, or anything thrown
 * moves on.
 *
 * Pass 2 exists because pass 1 refuses far more readily than it fails. Once every host has
 * declined, the fleet is busy rather than broken, so we go back and queue on the most promising
 * one — which is what the request would have done all along. Without it a short deadline turns a
 * request that used to wait and succeed into an outright failure.
 *
 * `send` therefore takes the flag rather than closing over it, since the same host is asked twice
 * with different answers wanted. Anything the caller wants bounded in pass 1 must be unbounded in
 * pass 2: that attempt is meant to wait.
 */
const streamWithFailover = async (
  hosts: string[],
  send: (host: string, failIfBusy: boolean) => Promise<Response>,
  callerSignal?: AbortSignal,
  label = 'request',
): Promise<Response> => {
  let lastResponse: Response | null = null;
  let lastError: unknown = null;
  let firstBusyHost: string | null = null;
  // A host that went quiet is still a better bet than one that refused the connection:
  // something is listening there.
  let firstSilentHost: string | null = null;

  for (let i = 0; i < hosts.length; i += 1) {
    const remaining = hosts.length - i - 1;
    try {
      // eslint-disable-next-line no-await-in-loop
      const response = await send(hosts[i], true);
      if (response.status === 429) {
        firstBusyHost ??= hosts[i];
        // Free the connection, since we are moving on.
        void response.body?.cancel();
        continue;
      }
      if (!shouldTryAnotherHost(response.status)) {
        void lastResponse?.body?.cancel();
        return response;
      }
      void lastResponse?.body?.cancel();
      lastResponse = response;
      console.warn(`[inference] ${hosts[i]} -> ${response.status}; ${remaining} host(s) remaining`);
    } catch (error) {
      // The client has gone, so no other host would have anyone to answer.
      if (callerSignal?.aborted) {
        throw error;
      }
      if (isTimeoutAbort(error)) {
        firstSilentHost ??= hosts[i];
      }
      lastError = error;
      const why = isTimeoutAbort(error) ? 'sent no headers in time' : `threw: ${error}`;
      console.warn(`[inference] ${hosts[i]} ${why}; ${remaining} host(s) remaining`);
    }
  }

  // Prefer a host that answered 429: it is demonstrably healthy and merely occupied. Neither it
  // nor a quiet host is worth less than one that hard-failed, which would only fail again.
  const queueHost = firstBusyHost ?? firstSilentHost;
  if (queueHost !== null) {
    void lastResponse?.body?.cancel();
    return send(queueHost, false);
  }

  if (lastResponse) {
    return lastResponse;
  }
  throw lastError instanceof Error ? lastError : new Error(`All inference servers failed for the ${label}`);
};

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

  // A vector needs no SAE loaded, so any host serving the model will do.
  const serverHost = await resolveHost(
    result?.hasVector
      ? inferenceTarget(targetModelId)
      : inferenceTarget(targetModelId, { sourceId: targetSourceId, user }),
  );

  const transformerLensModelId = await getTransformerLensModelIdIfExists(targetModelId);

  // Callers read the payload's fields directly, so the openapi-fetch envelope comes off here.
  return unwrapInferenceResponse(
    makeInferenceServerApiWithServerHost(serverHost).POST('/v1/util/sae-topk-by-decoder-cossim', {
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
    }),
  );
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

  // A vector needs no SAE loaded, so any host serving the model will do.
  const serverHost = await resolveHost(
    result?.hasVector
      ? inferenceTarget(feature.modelId)
      : inferenceTarget(feature.modelId, { sourceId: feature.layer, user }),
  );

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
  const serverHost = await resolveHost(inferenceTarget(modelId, { sourceId: source, user }));

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
  const serverHost = await resolveHost(inferenceTarget(modelId, { sourceSetName, user }));

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

// TODO: steerCompletion should also run its two completions on two servers, the way
// steerCompletionChat does. It already fails over across every host it knows.
export const steerCompletion = async (
  modelId: string,
  steerTypesToRun: SteerOutputType[],
  prompt: string,
  strengthMultiplier: number,
  n_tokens: number,
  temperature: number,
  presence_penalty: number,
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

  // Vectors need no SAE loaded, so any host serving the model will do.
  const target = hasVector
    ? inferenceTarget(modelId)
    : inferenceTarget(modelId, { sourceSetName: getSourceSetNameFromSource(firstFeatureLayer), user });
  const hosts = await resolveHosts(target);
  if (hosts.length === 0) {
    throw new NoComputeHostError(target);
  }

  const transformerLensModelId = await getTransformerLensModelIdIfExists(modelId);

  // A non-streamed completion sends no headers until the whole generation is done, so "no
  // headers yet" says nothing about the host and a headers deadline would fire on every
  // completion longer than the deadline. Bound the call as a whole instead, like the other
  // non-streaming inference calls. `failIfBusy` still gives pass 1 its fast refusal, which is
  // what the deadline was standing in for.
  const wholeCallDeadline = stream ? undefined : AbortSignal.timeout(INFERENCE_REQUEST_TIMEOUT_MS);

  const response = await streamWithFailover(
    hosts,
    (host, failIfBusy) =>
      postInferenceStreaming(
        host,
        '/v1/steer/completion',
        {
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
          presencePenalty: presence_penalty,
          // Inference servers before 1.11.1 require this field; no version applies it.
          freqPenalty: STEER_FREQUENCY_PENALTY,
          seed,
          steerMethod,
          normalizeSteering: false,
          stream,
          nLogprobs: n_logprobs,
          failIfBusy,
        },
        // Pass 2 asks to queue, so it must be allowed to wait for its turn.
        {
          signal: wholeCallDeadline,
          headersTimeoutMs: stream && failIfBusy ? STEER_HEADERS_TIMEOUT_MS : 0,
        },
      ),
    wholeCallDeadline,
    'steer completion',
  );
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
  presencePenalty: number,
  seed: number,
  steerSpecialTokens: boolean,
  steerFeatures: SteerFeature[],
  hasVector: boolean,
  user: AuthenticatedUser | null,
  stream: boolean,
  steerMethod: NPSteerMethod = STEER_METHOD,
  n_logprobs: number = STEER_N_LOGPROBS,
  /**
   * Vectors to read off the generated conversation, sent with the request. Inference ships none of
   * its own, so a `Vector` row is how any of them reaches it.
   */
  reads: NPVectorRead[] = [],
) => {
  // record start time
  const startTime = new Date().getTime();

  // A read projects a vector, and vectors need no SAE loaded, so in every case but the last any host
  // serving the model will do.
  const target =
    reads.length > 0 || hasVector || steerFeatures.length === 0
      ? inferenceTarget(modelId)
      : inferenceTarget(modelId, { sourceSetName: getSourceSetNameFromSource(steerFeatures[0].layer), user });
  const hosts = await resolveHosts(target);
  if (hosts.length === 0) {
    throw new NoComputeHostError(target);
  }
  const [serverHostDefault, serverHostSteered] = [hosts[0], hosts[hosts.length > 1 ? 1 : 0]];

  // make the promises to run
  // check if we need to replace "gemma-2-2b-it" with "gemma-2-2b", since we don't have SAEs for "-it"
  const modelIdForSearcher = replaceSteerModelIdIfNeeded(modelId);
  const transformerLensModelId = await getTransformerLensModelIdIfExists(modelIdForSearcher);

  if (stream) {
    const hasTwoServers = serverHostDefault !== serverHostSteered;

    // Always send one request per steer type so default and steered generate simultaneously.
    // A combined request makes the inference server loop over the types one after the other,
    // so the second column only starts filling once the first has finished.
    const readLog = reads.map((read) => read.id).join(',') || 'none';
    console.log(`completion chat - sending separate requests (hasTwoServers: ${hasTwoServers}, reads: ${readLog})`);
    const toRunPromises = steerTypesToRun.map((type) => {
      // Each type starts on its own host and falls back through the rest, so one dead pod
      // costs a retry rather than the whole completion.
      const candidates = rotate(hosts, type === SteerOutputType.DEFAULT ? 0 : 1);
      console.log(`completion chat - sending ${type} to ${candidates[0]}`);
      return streamWithFailover(
        candidates,
        (host, failIfBusy) =>
          postInferenceStreaming(
            host,
            '/v1/steer/completion-chat',
            {
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
              presencePenalty,
              // Inference servers before 1.11.1 require this field; no version applies it.
              freqPenalty: STEER_FREQUENCY_PENALTY,
              seed,
              steerSpecialTokens,
              steerMethod,
              normalizeSteering: false,
              stream: true,
              nLogprobs: n_logprobs,
              reads,
              failIfBusy,
            },
            // Pass 2 asks to queue, so it must be allowed to wait for its turn.
            { headersTimeoutMs: failIfBusy ? STEER_HEADERS_TIMEOUT_MS : 0 },
          ),
        undefined,
        'steer chat completion',
      );
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
            presencePenalty,
            // Inference servers before 1.11.1 require this field; no version applies it.
            freqPenalty: STEER_FREQUENCY_PENALTY,
            seed,
            steerSpecialTokens,
            steerMethod,
            normalizeSteering: false,
            nLogprobs: n_logprobs,
            reads,
            // This path collects whole responses; the SSE variant is lensPromptStream's job.
            stream: false,
            // No failover on this path, so queueing is the only way to be served.
            failIfBusy: false,
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
            presencePenalty,
            // Inference servers before 1.11.1 require this field; no version applies it.
            freqPenalty: STEER_FREQUENCY_PENALTY,
            seed,
            steerSpecialTokens,
            steerMethod,
            normalizeSteering: false,
            nLogprobs: n_logprobs,
            reads,
            // This path collects whole responses; the SSE variant is lensPromptStream's job.
            stream: false,
            // No failover on this path, so queueing is the only way to be served.
            failIfBusy: false,
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
  const serverHost = await resolveHost(inferenceTarget(modelId, { sourceSetName: sourceSet, user }));

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
// server. Attention heads aren't tied to a Source, so any host serving the model
// will do. The /activation/attention endpoint isn't in the typed client, so we
// call it with a raw fetch (like the lens endpoint); the response is still typed
// from the spec.
export const getAttentionForHead = async (
  modelId: string,
  layer: number,
  headIndex: number,
  prompt: string,
): Promise<InferenceAttentionResult> => {
  const host = await resolveHost(inferenceTarget(modelId));

  const transformerLensModelId = await getTransformerLensModelIdIfExists(modelId);

  const response = await postInferenceStreaming(
    host,
    '/v1/activation/attention',
    {
      model: transformerLensModelId,
      prompt,
      layer,
      head: headIndex,
    },
    // Not a stream despite the helper: the whole reply lands at once, so headers arrive only
    // when the work is finished. Bound the call, not the wait for its first byte.
    { signal: AbortSignal.timeout(INFERENCE_REQUEST_TIMEOUT_MS), headersTimeoutMs: 0 },
  );

  if (!response.ok) {
    const errorBody = await response.json().catch(() => null);
    throw new Error(errorBody?.error || `Inference server error (${response.status})`);
  }

  return (await response.json()) as InferenceAttentionResult;
};

export const tokenizeText = async (modelId: string, text: string, prependBos: boolean) => {
  const serverHost = await resolveHost(inferenceTarget(modelId));
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
  const serverHost = await resolveHost(inferenceTarget(modelId, { sourceId: source }));
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
// Pass 1 carries HEADERS_TIMEOUT_MS, which is what makes any of this reachable on a pod that
// accepts the connection and then goes quiet. Pass 2 does not: queueing is the whole point
// there, so waiting is the answer rather than the failure.
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

  // Any host serving the model is a candidate: jlens serves any request from
  // any instance. Already ordered and shuffled by the resolver.
  const hosts = await resolveHosts(inferenceTarget(modelId));
  if (hosts.length === 0) {
    throw new Error('No server host found');
  }

  return streamWithFailover(
    hosts,
    (host, failIfBusy) =>
      postInferenceStreaming(
        host,
        '/v1/lens/prompt',
        { ...request, model: transformerLensModelId, stream: true, failIfBusy },
        // Pass 2 asks to queue, so it must be allowed to wait for its turn.
        { signal, headersTimeoutMs: failIfBusy ? HEADERS_TIMEOUT_MS : 0 },
      ),
    signal,
    'lens request',
  );
};
