/* eslint-disable no-var */
/* eslint-disable vars-on-top */
/* eslint-disable no-param-reassign */
/* eslint-disable block-scoped-var */
/* eslint-disable @typescript-eslint/no-shadow */
/* eslint-disable @typescript-eslint/no-redeclare */

import { getTransformerLensModelIdIfExists } from '@/lib/db/model';
import { getNeuronOnly } from '@/lib/db/neuron';
import { getSourceSetNameFromSource } from '@/lib/utils/source';
import { ChatMessage, replaceSteerModelIdIfNeeded, STEER_METHOD, SteerFeature } from '@/lib/utils/steer';
import { AuthenticatedUser } from '@/lib/with-user';
import { NeuronPartial, NeuronPartialWithRelations } from '@/prisma/generated/zod';
import { SteerOutputType } from '@prisma/client';
import {
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
} from 'neuronpedia-inference-client';
import {
  getOneRandomServerHostForModel,
  getOneRandomServerHostForSource,
  getOneRandomServerHostForSourceSet,
  getTwoRandomServerHostsForModel,
  getTwoRandomServerHostsForSourceSet,
  LOCALHOST_INFERENCE_HOST,
} from '../db/inference-host-source';
import { INFERENCE_SERVER_SECRET, USE_LOCALHOST_INFERENCE } from '../env';
import { NeuronIdentifier } from './neuron-identifier';

export const makeInferenceServerApiWithServerHost = (serverHost: string) =>
  new DefaultApi(
    new Configuration({
      basePath: (USE_LOCALHOST_INFERENCE ? LOCALHOST_INFERENCE_HOST : serverHost) + BASE_PATH,
      headers: {
        'X-SECRET-KEY': INFERENCE_SERVER_SECRET,
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
  defaultTestText: string,
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
            index: Number(feature.index),
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

export const runInferenceActivationAll = async (
  modelId: string,
  sourceSetName: string,
  text: string,
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
) => {
  // record start time
  const startTime = new Date().getTime();

  // get the sae set's host
  const firstFeatureLayer = steerFeatures[0].layer;

  if (hasVector) {
    // if we have the vectors, then we can use any server that has the same modelId, since we don't need the SAE to be loaded
    var [serverHostDefault, serverHostSteered] = await getTwoRandomServerHostsForModel(modelId);
  } else {
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
    const toRunPromises = steerTypesToRun.map((type) => {
      console.log(`completion chat - does not have saved ${type} output, running it`);
      return fetch(
        `${type === SteerOutputType.DEFAULT ? serverHostDefault : serverHostSteered}/v1/steer/completion-chat`,
        {
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
          }),
        },
      );
    });
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
  text: string,
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
