import { UserSecretType } from '@prisma/client';
import { BASE_PATH, Configuration, DefaultApi } from 'neuronpedia-autointerp-client';
import {
  AUTOINTERP_SERVER,
  AUTOINTERP_SERVER_SECRET,
  IS_DOCKER_COMPOSE,
  IS_LOCALHOST,
  OLD_SCORER_SERVER,
  USE_LOCALHOST_AUTOINTERP,
} from '../env';

export const AUTOINTERP_SERVER_API = new DefaultApi(
  new Configuration({
    basePath:
      (USE_LOCALHOST_AUTOINTERP
        ? IS_DOCKER_COMPOSE
          ? 'http://autointerp:5003'
          : 'http://127.0.0.1:5003'
        : AUTOINTERP_SERVER) + BASE_PATH,
    headers: {
      'X-SECRET-KEY': AUTOINTERP_SERVER_SECRET,
    },
  }),
);

export const EXPLANATIONTYPE_HUMAN = 'human';
export const SCORER_VERSION = 2;

export const ACTIVATIONS_TO_USE_OLD_SCORER = 20;
export const OLD_SCORER_URL = IS_LOCALHOST ? 'http://127.0.0.1:5001/score' : `${OLD_SCORER_SERVER}/score`;
export enum AutoInterpModelType {
  OPENAI = 'openai',
  ANTHROPIC = 'anthropic',
  GOOGLE = 'google',
  UNKNOWN = 'unknown',
}

// TODO: put this in database
export const isReasoningModel = (modelId: string) =>
  modelId.startsWith('o1-') ||
  modelId.startsWith('o3-') ||
  modelId.startsWith('deepseek-r1') ||
  modelId.indexOf('-thinking') !== -1;

// TODO: this should be in the database
export const getAutoInterpModelTypeFromModelId = (modelId: string) => {
  if (modelId.startsWith('gpt') || modelId.startsWith('o1') || modelId.startsWith('o3') || modelId.startsWith('o4')) {
    return AutoInterpModelType.OPENAI;
  }
  if (modelId.startsWith('claude')) {
    return AutoInterpModelType.ANTHROPIC;
  }
  if (modelId.startsWith('gemini')) {
    return AutoInterpModelType.GOOGLE;
  }
  return AutoInterpModelType.UNKNOWN;
};
export const ERROR_NO_AUTOINTERP_KEY = 'No auto-interp key found for user.';
export const ERROR_REQUIRES_OPENROUTER = 'This autointerp type requires an OpenRouter key.';
export const ERROR_RECALL_ALT_FAILED =
  'All scoring requests failed. Check that you have enough credits in your API key (Either OpenRouter or others), and that your key has not been revoked.';
export function getKeyTypeForAutoInterpModelType(modelType: AutoInterpModelType) {
  if (modelType === AutoInterpModelType.OPENAI) {
    return UserSecretType.OPENAI;
  }
  if (modelType === AutoInterpModelType.ANTHROPIC) {
    return UserSecretType.ANTHROPIC;
  }
  if (modelType === AutoInterpModelType.GOOGLE) {
    return UserSecretType.GOOGLE;
  }
  return UserSecretType.OPENROUTER;
}
export const OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1';
export function requiresOpenRouterForExplanationType(explanationType: string) {
  return explanationType === 'eleuther_acts_top20';
}

export function requiresOpenRouterForExplanationScoreType(explanationScoreType: string) {
  return (
    explanationScoreType === 'recall_alt' ||
    explanationScoreType === 'eleuther_fuzz' ||
    explanationScoreType === 'eleuther_recall'
  );
}
