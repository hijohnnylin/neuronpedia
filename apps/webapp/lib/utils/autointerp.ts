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
// np_max-act-logits on the neuron with its sign flipped: the most negative examples and the negative logits
export const EXPLANATIONTYPE_NP_MIN_ACT_LOGITS = 'np_min-act-logits';

type SignedActivation = { values: number[]; maxValue: number; minValue: number };

// One example with the sign of the neuron flipped, so that its lowest token reads as the peak
export function flipActivationSign<T extends SignedActivation>(activation: T): T {
  const values = activation.values.map((v) => -v);
  return {
    ...activation,
    values,
    maxValue: values.length > 0 ? Math.max(...values) : -activation.minValue,
    minValue: values.length > 0 ? Math.min(...values) : -activation.maxValue,
  };
}

// The examples with the most negative values first, with the sign flipped
export function flipToMostNegative<T extends SignedActivation>(activations: T[], count: number): T[] {
  return activations
    .map((activation) => ({ activation, min: Math.min(...activation.values) }))
    .filter(({ min }) => min < 0)
    .sort((a, b) => a.min - b.min)
    .slice(0, count)
    .map(({ activation }) => flipActivationSign(activation));
}

const SCORING_ACTIVATIONS_TO_LOAD = 20;
const SCORING_ZERO_ACTIVATIONS_TO_LOAD_MAX = 5;

// The examples that a scorer reads: the top 20 deduplicated, normalized to a max of 10, and up to
// 5 all-zero ones. A min explanation describes the flipped neuron, so flip the examples first.
export function selectScoringActivations<T extends SignedActivation & { tokens: string[] }>(
  activations: T[],
  isMin: boolean,
): { activations: T[]; zeroActivations: T[] } {
  const flipped = isMin ? activations.map(flipActivationSign) : activations;
  const deduped = [...new Map(flipped.map((item) => [item.tokens.join(''), item])).values()];
  deduped.sort((a, b) => b.maxValue - a.maxValue);
  return {
    activations: deduped
      .slice(0, SCORING_ACTIVATIONS_TO_LOAD)
      .filter((activation) => activation.maxValue > 0)
      .map((activation) => ({
        ...activation,
        values: activation.values.map((value) => (value * 10) / activation.maxValue),
      })),
    zeroActivations: deduped
      .filter((activation) => activation.maxValue === 0)
      .slice(0, SCORING_ZERO_ACTIVATIONS_TO_LOAD_MAX),
  };
}

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
  'Scoring requests failed. Check that you have enough credits in your API key (Either OpenRouter or others), and that your key has not been revoked.';
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

// Claude and Gemini explainers run without reasoning, also when the call goes through OpenRouter
export function getExplainerReasoningEffort(modelType: AutoInterpModelType): 'low' | undefined {
  return modelType === AutoInterpModelType.ANTHROPIC || modelType === AutoInterpModelType.GOOGLE ? undefined : 'low';
}
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
