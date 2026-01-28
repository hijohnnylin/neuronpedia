/* eslint-disable no-var */
/* eslint-disable vars-on-top */
/* eslint-disable no-param-reassign */
/* eslint-disable block-scoped-var */
/* eslint-disable @typescript-eslint/no-shadow */
/* eslint-disable @typescript-eslint/no-redeclare */

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
import {
  ActivationSingleBatchPost200Response,
  ActivationSinglePost200Response,
  ActivationTopkByTokenPost200Response,
  BASE_PATH,
  Configuration,
  DefaultApi,
  NPSteerMethod,
  NPSteerType,
  NPSteerVector,
  SteerCompletionChatPost200Response,
  SteerCompletionPost200Response,
  UtilSaeVectorPost200Response,
} from 'neuronpedia-inference-client';
import runpodSdk from 'runpod-sdk';
import {
  getOneRandomServerHostForModel,
  getOneRandomServerHostForSource,
  getOneRandomServerHostForSourceSet,
  getRunpodServerlessUrlForModel,
  getTwoRandomServerHostsForModel,
  getTwoRandomServerHostsForSourceSet,
  LOCALHOST_INFERENCE_HOST,
} from '../db/inference-host-source';
import { INFERENCE_RUNPOD_API_KEY, INFERENCE_SERVER_SECRET, USE_LOCALHOST_INFERENCE } from '../env';
import { NeuronIdentifier } from './neuron-identifier';

// ============================================================================
// RUNPOD MODE CONFIGURATION
// Set to true to use RunPod Load Balancing mode (direct HTTP, better concurrency)
// Set to false to use RunPod Queue-based mode (SDK with job polling)
// ============================================================================
const USE_RUNPOD_LOAD_BALANCING = true;

// Maximum retries for load balancing mode cold starts
const RUNPOD_LB_MAX_RETRIES = 3;
const RUNPOD_LB_RETRY_DELAY_MS = 2000;

export const makeInferenceServerApiWithServerHost = (serverHost: string) =>
  new DefaultApi(
    new Configuration({
      basePath: (USE_LOCALHOST_INFERENCE ? LOCALHOST_INFERENCE_HOST : serverHost) + BASE_PATH,
      headers: {
        'X-SECRET-KEY': INFERENCE_SERVER_SECRET,
        'Accept-Encoding': 'gzip',
      },
    }),
  );

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
  counts?: number[][];
  error: string | undefined;
};

export type SearchTopKResult = {
  source: string;
  results: {
    position: number;
    token: string;
    topFeatures: {
      activationValue: number;
      featureIndex: number;
      feature: NeuronPartialWithRelations | undefined;
    }[];
  }[];
};

function convertSteerFeatureVectorsToInferenceVectors(steerFeatures: SteerFeature[]) {
  return steerFeatures.map((feature) => ({
    hook: feature.neuron?.hookName || '',
    steering_vector: feature.neuron?.vector,
    steeringVector: feature.neuron?.vector,
    strength: feature.strength,
  }));
}

/**
 * Extract the endpoint ID from a RunPod serverless URL.
 * URL format: https://api.runpod.ai/v2/{endpointId}
 */
function extractRunpodEndpointId(runpodServerlessUrl: string): string {
  const parts = runpodServerlessUrl.split('/');
  return parts[parts.length - 1];
}

/**
 * Creates a ReadableStream from a RunPod job that streams SSE data.
 * This converts the RunPod SDK stream format to match our existing SSE format.
 */
function createRunpodStreamingResponse(
  runpodServerlessUrl: string,
  payload: Record<string, unknown>,
): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const endpointId = extractRunpodEndpointId(runpodServerlessUrl);
  const sdk = runpodSdk(INFERENCE_RUNPOD_API_KEY);
  const endpoint = sdk.endpoint(endpointId);

  if (!endpoint) {
    throw new Error(`Failed to create RunPod endpoint for ${endpointId}`);
  }

  return new ReadableStream({
    async start(controller) {
      try {
        const RUNPOD_TIMEOUT_MS = 120000;
        const job = await endpoint.run({ input: payload }, RUNPOD_TIMEOUT_MS);

        for await (const output of endpoint.stream(job.id, RUNPOD_TIMEOUT_MS)) {
          // RunPod wraps stream output in an "output" property
          // The output.output contains the SSE data string like "data: {...}"
          if (output && typeof output === 'object' && 'output' in output) {
            const innerOutput = output.output;

            if (typeof innerOutput === 'string') {
              if (innerOutput.startsWith('data: ')) {
                // Extract the JSON part after "data: "
                const jsonStr = innerOutput.substring(6);
                try {
                  const parsed = JSON.parse(jsonStr);
                  controller.enqueue(encoder.encode(`data: ${JSON.stringify(parsed)}\n\n`));
                } catch {
                  controller.enqueue(encoder.encode(`${innerOutput}\n\n`));
                }
              } else {
                controller.enqueue(encoder.encode(`data: ${innerOutput}\n\n`));
              }
            } else if (innerOutput && typeof innerOutput === 'object') {
              controller.enqueue(encoder.encode(`data: ${JSON.stringify(innerOutput)}\n\n`));
            }
          } else if (typeof output === 'string') {
            if (output.startsWith('data: ')) {
              controller.enqueue(encoder.encode(`${output}\n\n`));
            } else {
              try {
                const data = JSON.parse(output);
                controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
              } catch {
                controller.enqueue(encoder.encode(`data: ${output}\n\n`));
              }
            }
          } else if (output && typeof output === 'object') {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(output)}\n\n`));
          }
        }

        controller.close();
      } catch (error) {
        console.error('RunPod streaming error:', error);
        controller.error(error);
      }
    },
  });
}

/**
 * Creates a ReadableStream from a RunPod Load Balancing endpoint.
 * This uses direct HTTP requests instead of the SDK for better concurrency.
 * URL format: https://{endpointId}.api.runpod.ai/generate
 */
function createRunpodLoadBalancingStreamingResponse(
  runpodServerlessUrl: string,
  payload: Record<string, unknown>,
): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  // Extract endpoint ID from the stored URL format (https://api.runpod.ai/v2/{endpointId})
  const endpointId = extractRunpodEndpointId(runpodServerlessUrl);
  // Construct load balancing URL format
  const loadBalancingUrl = `https://${endpointId}.api.runpod.ai/generate`;

  return new ReadableStream({
    async start(controller) {
      let lastError: Error | null = null;

      // eslint-disable-next-line no-plusplus
      for (let attempt = 0; attempt < RUNPOD_LB_MAX_RETRIES; attempt += 1) {
        try {
          // eslint-disable-next-line no-await-in-loop
          const response = await fetch(loadBalancingUrl, {
            method: 'POST',
            headers: {
              Authorization: `Bearer ${INFERENCE_RUNPOD_API_KEY}`,
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
          });

          // Handle cold start / no workers available (400 error)
          if (response.status === 400) {
            // eslint-disable-next-line no-await-in-loop
            const errorText = await response.text();
            // eslint-disable-next-line no-console
            console.warn(
              `RunPod LB attempt ${attempt + 1}/${RUNPOD_LB_MAX_RETRIES}: No workers available, retrying...`,
              errorText,
            );
            lastError = new Error(`No workers available: ${errorText}`);
            if (attempt < RUNPOD_LB_MAX_RETRIES - 1) {
              // eslint-disable-next-line no-await-in-loop, no-promise-executor-return
              await new Promise<void>((resolve) => {
                setTimeout(resolve, RUNPOD_LB_RETRY_DELAY_MS * (attempt + 1));
              });
            } else {
              throw lastError;
            }
          } else if (!response.ok) {
            throw new Error(`RunPod LB error: ${response.status} ${response.statusText}`);
          } else if (!response.body) {
            throw new Error('No response body from RunPod LB');
          } else {
            // Stream the SSE response directly
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = ''; // Buffer for incomplete messages

            // eslint-disable-next-line no-constant-condition
            while (true) {
              // eslint-disable-next-line no-await-in-loop
              const { done, value } = await reader.read();
              if (done) {
                // Process any remaining data in buffer
                if (buffer.trim()) {
                  if (buffer.startsWith('data: ')) {
                    controller.enqueue(encoder.encode(`${buffer}\n\n`));
                  } else {
                    controller.enqueue(encoder.encode(`data: ${buffer}\n\n`));
                  }
                }
                break;
              }

              // Append new data to buffer
              buffer += decoder.decode(value, { stream: true });

              // Split by double newlines to find complete SSE messages
              const parts = buffer.split('\n\n');

              // Process all complete messages (all but the last part)
              for (let i = 0; i < parts.length - 1; i += 1) {
                const message = parts[i].trim();
                if (message) {
                  if (message.startsWith('data: ')) {
                    controller.enqueue(encoder.encode(`${message}\n\n`));
                  } else {
                    controller.enqueue(encoder.encode(`data: ${message}\n\n`));
                  }
                }
              }

              // Keep the last part (potentially incomplete) in buffer
              buffer = parts[parts.length - 1];
            }

            controller.close();
            return; // Success, exit retry loop
          }
        } catch (error) {
          // eslint-disable-next-line no-console
          console.error(`RunPod LB attempt ${attempt + 1}/${RUNPOD_LB_MAX_RETRIES} failed:`, error);
          lastError = error instanceof Error ? error : new Error(String(error));

          if (attempt < RUNPOD_LB_MAX_RETRIES - 1) {
            // eslint-disable-next-line no-await-in-loop, no-promise-executor-return
            await new Promise<void>((resolve) => {
              setTimeout(resolve, RUNPOD_LB_RETRY_DELAY_MS * (attempt + 1));
            });
          }
        }
      }

      // All retries exhausted
      // eslint-disable-next-line no-console
      console.error('RunPod LB streaming failed after all retries:', lastError);
      controller.error(lastError);
    },
  });
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
    // eslint-disable-next-line
    var [serverHost, _] = await getTwoRandomServerHostsForModel(targetModelId);
  } else {
    // if it's not a vector, then we need to use the source set's host
    var serverHost = await getOneRandomServerHostForSource(targetModelId, targetSourceId, user);
  }

  const transformerLensModelId = await getTransformerLensModelIdIfExists(targetModelId);

  return makeInferenceServerApiWithServerHost(serverHost).utilSaeTopkByDecoderCossimPost({
    utilSaeTopkByDecoderCossimPostRequest: {
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

export const getActivationForFeature = async (
  feature: NeuronPartial,
  defaultTestText: string | string[],
  user: AuthenticatedUser | null,
) => {
  if (!feature.modelId || !feature.layer || !feature.index) {
    throw new Error('Invalid feature');
  }

  // get if it's a feature/vector first
  const result = await getNeuronOnly(feature.modelId, feature.layer, feature.index);

  if (result?.hasVector) {
    // if it's a vector, then we can use any server that has the same modelId, since we don't need the SAE to be loaded
    // eslint-disable-next-line
    var [serverHost, _] = await getTwoRandomServerHostsForModel(feature.modelId);
  } else {
    // if it's not a vector, then we need to use the source set's host
    var serverHost = await getOneRandomServerHostForSource(feature.modelId, feature.layer, user);
  }

  const modelIdForSearcher = replaceSteerModelIdIfNeeded(feature.modelId);
  const transformerLensModelId = await getTransformerLensModelIdIfExists(modelIdForSearcher);

  if (Array.isArray(defaultTestText)) {
    return makeInferenceServerApiWithServerHost(serverHost)
      .activationSingleBatchPost({
        activationSingleBatchPostRequest: result?.hasVector
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
      })
      .then((result: ActivationSingleBatchPost200Response) =>
        result.results.map((result) => {
          const { tokens } = result;
          const activations = result.activation.values;
          return {
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
            dfaValues: result.activation.dfaValues,
            dfaTargetIndex: result.activation.dfaTargetIndex,
            dfaMaxValue: result.activation.dfaMaxValue,
          };
        }),
      )
      .catch((error) => {
        console.error(error);
        throw error;
      });
  }
  return makeInferenceServerApiWithServerHost(serverHost)
    .activationSinglePost({
      activationSinglePostRequest: result?.hasVector
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
    })
    .then((result: ActivationSinglePost200Response) => {
      const { tokens } = result;
      const activations = result.activation.values;
      return {
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
        dfaValues: result.activation.dfaValues,
        dfaTargetIndex: result.activation.dfaTargetIndex,
        dfaMaxValue: result.activation.dfaMaxValue,
      };
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

  return makeInferenceServerApiWithServerHost(serverHost).activationSourcePost({
    activationSourcePostRequest: {
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
    return makeInferenceServerApiWithServerHost(serverHost).activationAllBatchPost({
      activationAllBatchPostRequest: {
        prompts: text,
        model: transformerLensModelId,
        selectedSources: selectedLayers,
        sortByTokenIndexes: sortIndexes,
        sourceSet: sourceSetName,
        ignoreBos,
        numResults,
      },
    });
  }
  return makeInferenceServerApiWithServerHost(serverHost).activationAllPost({
    activationAllPostRequest: {
      prompt: text,
      model: transformerLensModelId,
      selectedSources: selectedLayers,
      sortByTokenIndexes: sortIndexes,
      sourceSet: sourceSetName,
      ignoreBos,
      numResults,
    },
  });
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

  // TODO: use typescript client instead of hardcoding for streaming

  const response = await fetch(`${serverHost}/v1/steer/completion`, {
    method: 'POST',
    cache: 'no-cache',
    headers: {
      'Content-Type': 'application/json',
      'X-SECRET-KEY': INFERENCE_SERVER_SECRET,
    },
    body: JSON.stringify({
      types: steerTypesToRun,
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
      strength_multiplier: strengthMultiplier,
      n_completion_tokens: n_tokens,
      temperature,
      freq_penalty,
      seed,
      steer_method: steerMethod,
      normalize_steering: false,
      stream,
      n_logprobs,
    }),
  });
  if (!response.body) {
    throw new Error('No response body');
  }

  if (stream) {
    return response.body;
  }
  const result = await response.json();
  return result as SteerCompletionPost200Response;
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

  if (hasVector || steerFeatures.length === 0) {
    // if we have the vectors, then we can use any server that has the same modelId, since we don't need the SAE to be loaded
    var [serverHostDefault, serverHostSteered] = await getTwoRandomServerHostsForModel(
      modelId,
      isAssistantAxis ? InferenceEngine.CSPACE : InferenceEngine.TRANSFORMER_LENS,
    );
  } else {
    // get the sae set's host
    const firstFeatureLayer = steerFeatures[0].layer;
    // if we have just one server, then just use that server
    [serverHostDefault, serverHostSteered] = await getTwoRandomServerHostsForSourceSet(
      modelId,
      getSourceSetNameFromSource(firstFeatureLayer),
      user,
      isAssistantAxis ? InferenceEngine.CSPACE : InferenceEngine.TRANSFORMER_LENS,
    );
  }

  // make the promises to run
  // check if we need to replace "gemma-2-2b-it" with "gemma-2-2b", since we don't have SAEs for "-it"
  const modelIdForSearcher = replaceSteerModelIdIfNeeded(modelId);
  const transformerLensModelId = await getTransformerLensModelIdIfExists(modelIdForSearcher);

  if (stream) {
    // Check if we can combine default and steered into one request
    const messagesAreEqual = JSON.stringify(defaultChatMessages) === JSON.stringify(steeredChatMessages);
    const hasBothTypes =
      steerTypesToRun.includes(SteerOutputType.DEFAULT) && steerTypesToRun.includes(SteerOutputType.STEERED);

    // Check if we should use RunPod for assistant axis requests
    if (isAssistantAxis && INFERENCE_RUNPOD_API_KEY) {
      const runpodServerlessUrl = await getRunpodServerlessUrlForModel(modelId, InferenceEngine.CSPACE);

      if (runpodServerlessUrl) {
        const mode = USE_RUNPOD_LOAD_BALANCING ? 'load-balancing' : 'queue-based';
        console.log(`Using RunPod serverless (${mode}) for assistant axis streaming request`);

        // Always send separate requests for each type to enable parallel streaming
        // Even when messages are equal, we want two simultaneous requests for better parallelism
        const runpodStreams = steerTypesToRun.map((type) => {
          console.log(`completion chat (runpod ${mode}) - sending ${type} request`);
          const payload = {
            prompt: type === SteerOutputType.DEFAULT ? defaultChatMessages : steeredChatMessages,
            types: [type === SteerOutputType.DEFAULT ? 'DEFAULT' : 'STEERED'],
            vectors: hasVector ? convertSteerFeatureVectorsToInferenceVectors(steerFeatures) : [],
            n_completion_tokens: nTokens,
            temperature,
            steer_method: steerMethod,
            normalize_steering: false,
            stream: true,
            n_logprobs,
            steer_special_tokens: steerSpecialTokens,
          };

          // Use load balancing or queue-based mode based on flag
          if (USE_RUNPOD_LOAD_BALANCING) {
            return createRunpodLoadBalancingStreamingResponse(runpodServerlessUrl, payload);
          }
          return createRunpodStreamingResponse(runpodServerlessUrl, payload);
        });
        return runpodStreams;
      }
    }

    let toRunPromises: Promise<Response>[];

    // Check if we have two different servers available
    const hasTwoServers = serverHostDefault !== serverHostSteered;

    // Only combine into single request if messages are equal, both types needed, AND we only have one server
    // If we have two servers, always send separate requests to use both servers in parallel
    // For assistant-axis, always send separate requests even with one server for parallel streaming
    if (messagesAreEqual && hasBothTypes && !hasTwoServers && !isAssistantAxis) {
      // Send a single request with both types (only when using a single server and not assistant-axis)
      console.log('completion chat - messages are equal and single server, sending combined request');
      toRunPromises = [
        fetch(`${serverHostDefault}/v1/steer/completion-chat`, {
          method: 'POST',
          cache: 'no-cache',
          headers: {
            'Content-Type': 'application/json',
            'X-SECRET-KEY': INFERENCE_SERVER_SECRET,
          },
          body: JSON.stringify({
            types: [NPSteerType.Steered, NPSteerType.Default],
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
            strength_multiplier: strengthMultiplier,
            n_completion_tokens: nTokens,
            temperature,
            freq_penalty: freqPenalty,
            seed,
            steer_special_tokens: steerSpecialTokens,
            steer_method: steerMethod,
            normalize_steering: false,
            stream: true,
            n_logprobs,
            is_assistant_axis: isAssistantAxis,
          }),
        }),
      ];
    } else {
      // Send separate requests for each type (use two servers when available for parallelism)
      // For assistant-axis, always send separate requests even when messages are equal
      console.log(
        `completion chat - sending separate requests (hasTwoServers: ${hasTwoServers}, isAssistantAxis: ${isAssistantAxis})`,
      );
      toRunPromises = steerTypesToRun.map((type) => {
        const host = type === SteerOutputType.DEFAULT ? serverHostDefault : serverHostSteered;
        console.log(`completion chat - sending ${type} to ${host}`);
        return fetch(`${host}/v1/steer/completion-chat`, {
          method: 'POST',
          cache: 'no-cache',
          headers: {
            'Content-Type': 'application/json',
            'X-SECRET-KEY': INFERENCE_SERVER_SECRET,
          },
          body: JSON.stringify({
            types: [type === SteerOutputType.DEFAULT ? NPSteerType.Default : NPSteerType.Steered],
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
            strength_multiplier: strengthMultiplier,
            n_completion_tokens: nTokens,
            temperature,
            freq_penalty: freqPenalty,
            seed,
            steer_special_tokens: steerSpecialTokens,
            steer_method: steerMethod,
            normalize_steering: false,
            stream: true,
            n_logprobs,
            is_assistant_axis: isAssistantAxis,
          }),
        });
      });
    }
    const responses = await Promise.all(toRunPromises);
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
      return makeInferenceServerApiWithServerHost(serverHostDefault).steerCompletionChatPost({
        steerCompletionChatPostRequest: {
          types: [NPSteerType.Default],
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
          vectors: hasVector
            ? (convertSteerFeatureVectorsToInferenceVectors(steerFeatures) as NPSteerVector[])
            : undefined,
          strengthMultiplier,
          nCompletionTokens: nTokens,
          temperature,
          freqPenalty,
          seed,
          steerSpecialTokens,
          steerMethod,
          normalizeSteering: false,
          nLogprobs: n_logprobs,
          // @ts-ignore we'll fix this later with typescript client
          isAssistantAxis,
        },
      });
    }
    if (type === SteerOutputType.STEERED) {
      console.log('does not have saved steered output, running it');
      return makeInferenceServerApiWithServerHost(serverHostSteered).steerCompletionChatPost({
        steerCompletionChatPostRequest: {
          types: [NPSteerType.Steered],
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
          vectors: hasVector
            ? (convertSteerFeatureVectorsToInferenceVectors(steerFeatures) as NPSteerVector[])
            : undefined,
          strengthMultiplier,
          nCompletionTokens: nTokens,
          temperature,
          freqPenalty,
          seed,
          steerSpecialTokens,
          steerMethod,
          normalizeSteering: false,
          nLogprobs: n_logprobs,
          // @ts-ignore we'll fix this later with typescript client
          isAssistantAxis,
        },
      });
    }
    throw new Error('Invalid steer type');
  });

  // run the promises
  const inferenceCompletionChatResponses = await Promise.all(toRunPromises);

  // record end time
  const endTime = new Date().getTime();
  console.log(`Time taken: ${endTime - startTime}ms`);

  if (inferenceCompletionChatResponses.some((result) => !result)) {
    throw new Error('Error running inference server on a result.');
  }

  return inferenceCompletionChatResponses as SteerCompletionChatPost200Response[];
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
    return makeInferenceServerApiWithServerHost(serverHost).activationTopkByTokenBatchPost({
      activationTopkByTokenBatchPostRequest: {
        prompts: text,
        model: transformerLensModelId,
        source: layer,
        topK,
        ignoreBos,
      },
    });
  }
  const result: ActivationTopkByTokenPost200Response = await makeInferenceServerApiWithServerHost(
    serverHost,
  ).activationTopkByTokenPost({
    activationTopkByTokenPostRequest: {
      prompt: text,
      model: transformerLensModelId,
      source: layer,
      topK,
      ignoreBos,
    },
  });
  return result;
};

export const tokenizeText = async (modelId: string, text: string, prependBos: boolean) => {
  const serverHost = await getOneRandomServerHostForModel(modelId);
  const transformerLensModelId = await getTransformerLensModelIdIfExists(modelId);

  const result = await makeInferenceServerApiWithServerHost(serverHost).tokenizePost({
    tokenizePostRequest: {
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
): Promise<UtilSaeVectorPost200Response> => {
  const serverHost = await getOneRandomServerHostForSource(modelId, source, null);
  if (!serverHost) {
    throw new Error('No server host found');
  }
  const transformerLensModelId = await getTransformerLensModelIdIfExists(modelId);

  return makeInferenceServerApiWithServerHost(serverHost).utilSaeVectorPost({
    utilSaeVectorPostRequest: {
      model: transformerLensModelId,
      source,
      index: parseInt(index, 10),
    },
  });
};
