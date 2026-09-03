/**
 * Named aliases for the inference wire types the webapp actually uses.
 *
 * `inference.d.ts` is generated from `apps/inference/openapi.json` and reaches everything
 * through `components['schemas'][...]`, which is unreadable at a call site. These aliases are
 * the import surface the old generated npm package used to provide, so a consumer changes its
 * import path rather than the shape of its code.
 *
 * Field names here are the wire names, and the wire is camelCase -- `maxValue`, `sourceSet`. The
 * python models keep snake_case attributes and alias on the way out, so what you write here is
 * literally what goes over the wire; nothing rewrites keys in between.
 *
 * These are hand-listed on purpose. Renaming a schema in the Python models breaks this file at
 * compile time, which is where you want to find out.
 */
import type { components } from '@/lib/api/inference';

type Schemas = components['schemas'];

export type ActivationAllResponse = Schemas['ActivationAllResponse'];
export type ActivationAllBatchResponse = Schemas['ActivationAllBatchResponse'];
export type ActivationSingleResponse = Schemas['ActivationSingleResponse'];
export type ActivationSingleBatchResponse = Schemas['ActivationSingleBatchResponse'];
export type ActivationSourceResponse = Schemas['ActivationSourceResponse'];
export type ActivationTopkByTokenResponse = Schemas['ActivationTopkByTokenResponse'];
export type ActivationTopkByTokenBatchResponse = Schemas['ActivationTopkByTokenBatchResponse'];
export type ActivationAttentionResponse = Schemas['ActivationAttentionResponse'];

export type NPLogprob = Schemas['NPLogprob'];
export type NPNormalize = Schemas['NPNormalize'];
export type NPSteerChatMessage = Schemas['NPSteerChatMessage'];
export type NPSteerFeature = Schemas['NPSteerFeature'];
export type NPSteerMethod = Schemas['NPSteerMethod'];
export type NPSteerType = Schemas['NPSteerType'];
export type NPSteerVector = Schemas['NPSteerVector'];
export type NPVectorRead = Schemas['NPVectorRead'];
export type SteerReadoutTurn = Schemas['SteerReadoutTurn'];
export type SteerVectorReadout = Schemas['SteerVectorReadout'];
export type SteerCompletionResponse = Schemas['SteerCompletionResponse'];
export type SteerCompletionChatResponse = Schemas['SteerCompletionChatResponse'];

export type TokenizeResponse = Schemas['TokenizeResponse'];
export type UtilSaeVectorResponse = Schemas['UtilSaeVectorResponse'];
export type UtilSaeTopkByDecoderCossimResponse = Schemas['UtilSaeTopkByDecoderCossimResponse'];

/**
 * `openapi-typescript` emits enums as string unions, which are types only -- but call sites
 * iterate these to build dropdowns and yup `oneOf` lists, and reference members by name. These
 * objects restore that, declared alongside a type of the same name so `NPSteerType.STEERED` and
 * `x: NPSteerType` both read the way they did under the generated client.
 *
 * The `{ [K in Union]: K }` annotation is what makes them safe to hand-write: every member of the
 * spec union is a required key, and each value must equal its own key. Add a steer method in the
 * Python models and this file stops compiling, instead of the option silently vanishing from the
 * UI.
 */
/**
 * Path prefix every inference route sits under. The typed client carries it in the path
 * literal; this is for the streaming endpoints that build a URL by hand.
 */
export const INFERENCE_BASE_PATH = '/v1';

export const NPSteerMethod: { [K in NPSteerMethod]: K } = {
  SIMPLE_ADDITIVE: 'SIMPLE_ADDITIVE',
  ORTHOGONAL_DECOMP: 'ORTHOGONAL_DECOMP',
  PROJECTION_CAP: 'PROJECTION_CAP',
};

export const NPSteerType: { [K in NPSteerType]: K } = {
  STEERED: 'STEERED',
  DEFAULT: 'DEFAULT',
};

/** Display labels for the steer-method dropdown, kept exhaustive by the same trick. */
export const STEER_METHOD_LABELS: Record<NPSteerMethod, string> = {
  SIMPLE_ADDITIVE: 'Simple Additive',
  ORTHOGONAL_DECOMP: 'Orthogonal Decomp',
  PROJECTION_CAP: 'Projection Cap',
};
