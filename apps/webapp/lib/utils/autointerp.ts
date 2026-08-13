import type { paths } from '@/lib/api/autointerp';
import { UserSecretType } from '@prisma/client';
import createClient from 'openapi-fetch';
import { upstreamError } from '../api-error';
import { AUTOINTERP_SERVER, AUTOINTERP_SERVER_SECRET, USE_LOCALHOST_AUTOINTERP } from '../env';

// The version prefix is part of the paths in the spec, so it is not in the base URL.
export const AUTOINTERP_SERVER_API = createClient<paths>({
  baseUrl: USE_LOCALHOST_AUTOINTERP ? 'http://127.0.0.1:5003' : AUTOINTERP_SERVER,
  headers: {
    'X-SECRET-KEY': AUTOINTERP_SERVER_SECRET,
  },
});

type AutointerpResult<T> = { data?: T; error?: unknown; response: Response };

// openapi-fetch reports a failed request in the result rather than by rejecting, but the API
// routes that reach these calls report failure by letting an exception reach their catch block.
// This is the adapter between the two.
//
// The upstream body goes in `cause` rather than the message: it is FastAPI's {"detail": str(e)},
// a raw python exception string that can carry absolute server paths. `upstreamError` keeps that
// for Sentry and gives the caller a 502 with a message written for them.
export async function unwrapAutointerpResponse<T>(result: Promise<AutointerpResult<T>>): Promise<T> {
  const { data, error, response } = await result;
  if (error !== undefined || data === undefined) {
    throw upstreamError('autointerp', { status: response.status, body: error ?? response.statusText });
  }
  return data;
}

export const EXPLANATIONTYPE_HUMAN = 'human';

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
// DB has model names with version-first ordering (e.g. claude-4-5-haiku) but
// the Anthropic API expects family-first ordering (e.g. claude-haiku-4-5-20251001).
const ANTHROPIC_MODEL_ID_FIXES: Record<string, string> = {
  'claude-4-5-haiku': 'claude-haiku-4-5-20251001',
  'claude-4-5-sonnet': 'claude-sonnet-4-5-20250929',
};

export function getAnthropicModelId(modelName: string): string {
  return ANTHROPIC_MODEL_ID_FIXES[modelName] || modelName;
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
