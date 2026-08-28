import type {
  ForwardPassResponse,
  GraphChatMessage,
  SteerResponse as GraphSteerResponse,
  LogitsByToken,
  ParseChatPromptResponse,
  SalientLogit,
  SteerFeature,
} from '@/lib/api/graph-types';
import { ComputeService } from '@prisma/client';
import * as yup from 'yup';
import { computeFetch } from '../db/compute-host';
import {
  STEER_FREEZE_ATTENTION,
  STEER_FREQUENCY_PENALTY,
  STEER_FREQUENCY_PENALTY_MAX,
  STEER_FREQUENCY_PENALTY_MIN,
  STEER_N_COMPLETION_TOKENS,
  STEER_N_COMPLETION_TOKENS_MAX,
  STEER_SEED,
  STEER_TEMPERATURE,
  STEER_TEMPERATURE_MAX,
  STEER_TOPK_LOGITS,
  STEER_TOPK_LOGITS_MAX,
} from './steer';

export const GRAPH_MAX_PROMPT_LENGTH_CHARS = 10000;
export const GRAPH_BATCH_SIZE = 48;
// this time estimate comes from testing different prompt lengths with batch size 48, and is only valid for gemma-2-2b, for a40
// with nnsight we are using lazy encoder, so we need to add time for this
// lorsa models always run with QK tracing enabled, which adds ~40s on top of the baseline lorsa overhead
export const getEstimatedTimeFromNumTokens = (numTokens: number, slower: boolean = false, isLorsa: boolean = false) =>
  11.2 * Math.log2(Math.max(numTokens, 4)) - 7 + (slower ? 20 : 0) + (isLorsa ? 20 : 0);
export const GRAPH_MAX_TOKENS = 64;
export const GRAPH_GENERATION_ENABLED_MODELS = ['gemma-2-2b', 'gemma-3-4b-it', 'qwen3-4b', 'qwen3-1.7b'];

export const LORSA_MODELS = ['qwen3-1.7b'];
export const LORSA_BATCH_SIZE = 16;
export const LORSA_MAX_TOKENS = 10;
export const GRAPH_MODEL_MAP = {
  'gemma-2-2b': 'google/gemma-2-2b',
  'gemma-3-4b-it': 'google/gemma-3-4b-it',
  'qwen3-4b': 'Qwen/Qwen3-4B',
  'qwen3-1.7b': 'Qwen/Qwen3-1.7B',
};

export const GRAPH_S3_USER_GRAPHS_DIR = 'user-graphs';

export const GRAPH_MAXNLOGITS_MIN = 5;
export const GRAPH_MAXNLOGITS_MAX = 15;
export const GRAPH_MAXNLOGITS_DEFAULT = 10;
export const GRAPH_DESIREDLOGITPROB_MIN = 0.6;
export const GRAPH_DESIREDLOGITPROB_MAX = 0.99;
export const GRAPH_DESIREDLOGITPROB_DEFAULT = 0.95;
export const GRAPH_NODETHRESHOLD_MIN = 0.5;
export const GRAPH_NODETHRESHOLD_MIN_LORSA = 0.3;
export const GRAPH_NODETHRESHOLD_MAX = 0.95;
export const GRAPH_NODETHRESHOLD_MAX_LORSA = 0.7;
export const GRAPH_NODETHRESHOLD_DEFAULT = 0.8;
export const GRAPH_NODETHRESHOLD_DEFAULT_LORSA = 0.7;
export const GRAPH_EDGETHRESHOLD_MIN = 0.65;
export const GRAPH_EDGETHRESHOLD_MIN_LORSA = 0.5;
export const GRAPH_EDGETHRESHOLD_MAX = 0.98;
export const GRAPH_EDGETHRESHOLD_MAX_LORSA = 0.8;
export const GRAPH_EDGETHRESHOLD_DEFAULT = 0.85;
export const GRAPH_EDGETHRESHOLD_DEFAULT_LORSA = 0.8;
export const GRAPH_MAXFEATURENODES_MIN = 1500;
export const GRAPH_MAXFEATURENODES_MIN_LORSA = 1500;
export const GRAPH_MAXFEATURENODES_MAX = 10000;
export const GRAPH_MAXFEATURENODES_MAX_LORSA = 3000;
export const GRAPH_MAXFEATURENODES_DEFAULT = 5000;
export const GRAPH_MAXFEATURENODES_DEFAULT_LORSA = 3000;
export const GRAPH_SLUG_MIN = 2;

// QK tracing is lorsa-only (passed as `enable_qk_tracing=true` for LORSA_MODELS)
export const GRAPH_QKTOPFRACTION_MIN = 0.05;
export const GRAPH_QKTOPFRACTION_MAX = 0.5;
export const GRAPH_QKTOPFRACTION_DEFAULT = 0.5;
export const GRAPH_QKTOPK_MIN = 1;
export const GRAPH_QKTOPK_MAX = 5;
export const GRAPH_QKTOPK_DEFAULT = 5;

export const GRAPH_DYNAMIC_PRUNING_THRESHOLD_DEFAULT = 0.6;

export const GRAPH_ANONYMOUS_USER_ID = 'anonymous';

export const MAX_PUT_REQUESTS_PER_DAY = 200;

export const graphGenerateSchemaClient = yup.object({
  prompt: yup
    .string()
    .max(GRAPH_MAX_PROMPT_LENGTH_CHARS, `Prompt cannot exceed ${GRAPH_MAX_PROMPT_LENGTH_CHARS} characters.`)
    .min(1, 'Prompt is required.')
    .required(),
  // Structured chat turns (instruct models). When present, the graph server
  // renders the prompt string via the model's real chat template — the frontend
  // does no chat-template building. `prompt` still carries a plain-text version
  // of the turns for length/required validation and is ignored server-side.
  messages: yup
    .array()
    .of(
      yup.object({
        role: yup.string().required(),
        // `defined`, not `required`: an empty final assistant turn is how the
        // caller asks for that turn to be left open for the model to continue,
        // and yup's `required` rejects the empty string.
        content: yup.string().defined(),
      }),
    )
    .nullable(),
  modelId: yup.string().min(1, 'Model is required.').oneOf(GRAPH_GENERATION_ENABLED_MODELS).required(),
  sourceSetName: yup.string().nullable(),
  maxNLogits: yup
    .number()
    .integer('Must be an integer.')
    .min(GRAPH_MAXNLOGITS_MIN, `Must be at least ${GRAPH_MAXNLOGITS_MIN}.`)
    .max(GRAPH_MAXNLOGITS_MAX, `Must be at most ${GRAPH_MAXNLOGITS_MAX}.`)
    .default(GRAPH_MAXNLOGITS_DEFAULT)
    .required('This field is required.'),
  desiredLogitProb: yup
    .number()
    .min(GRAPH_DESIREDLOGITPROB_MIN, `Must be at least ${GRAPH_DESIREDLOGITPROB_MIN}.`)
    .max(GRAPH_DESIREDLOGITPROB_MAX, `Must be at most ${GRAPH_DESIREDLOGITPROB_MAX}.`)
    .default(GRAPH_DESIREDLOGITPROB_DEFAULT)
    .required('This field is required.'),
  nodeThreshold: yup
    .number()
    .min(GRAPH_NODETHRESHOLD_MIN, `Must be at least ${GRAPH_NODETHRESHOLD_MIN}.`)
    .max(GRAPH_NODETHRESHOLD_MAX, `Must be at most ${GRAPH_NODETHRESHOLD_MAX}.`)
    .default(GRAPH_NODETHRESHOLD_DEFAULT)
    .required('This field is required.'),
  edgeThreshold: yup
    .number()
    .min(GRAPH_EDGETHRESHOLD_MIN, `Must be at least ${GRAPH_EDGETHRESHOLD_MIN}.`)
    .max(GRAPH_EDGETHRESHOLD_MAX, `Must be at most ${GRAPH_EDGETHRESHOLD_MAX}.`)
    .default(GRAPH_EDGETHRESHOLD_DEFAULT)
    .required('This field is required.'),
  maxFeatureNodes: yup
    .number()
    .integer('Must be an integer.')
    .min(GRAPH_MAXFEATURENODES_MIN, `Must be at least ${GRAPH_MAXFEATURENODES_MIN}.`)
    .max(GRAPH_MAXFEATURENODES_MAX, `Must be at most ${GRAPH_MAXFEATURENODES_MAX}.`)
    .default(GRAPH_MAXFEATURENODES_DEFAULT)
    .required('This field is required.'),
  qkTopFraction: yup
    .number()
    .min(GRAPH_QKTOPFRACTION_MIN, `Must be at least ${GRAPH_QKTOPFRACTION_MIN}.`)
    .max(GRAPH_QKTOPFRACTION_MAX, `Must be at most ${GRAPH_QKTOPFRACTION_MAX}.`)
    .default(GRAPH_QKTOPFRACTION_DEFAULT),
  qkTopk: yup
    .number()
    .integer('Must be an integer.')
    .min(GRAPH_QKTOPK_MIN, `Must be at least ${GRAPH_QKTOPK_MIN}.`)
    .max(GRAPH_QKTOPK_MAX, `Must be at most ${GRAPH_QKTOPK_MAX}.`)
    .default(GRAPH_QKTOPK_DEFAULT),
  slug: yup.string(),
});

export type { SalientLogit };

// POST to the graph server for a source set, with failover across its hosts.
const graphFetch = (modelId: string, sourceSetName: string, action: string, body: unknown) =>
  computeFetch({ service: ComputeService.GRAPH, modelId, sourceSetName }, `/${action}`, {
    method: 'POST',
    body: JSON.stringify(body),
  });

// `/forward-pass` on the graph server. The name is the webapp's, kept because every caller and
// route uses it; the shape is the generated one.
export type GraphTokenizeResponse = ForwardPassResponse;

// Structured chat turn sent to the graph server, which applies the model's real
// chat template server-side (the frontend does no chat-template special-casing).
export type { GraphChatMessage };

export const getGraphTokenize = async (
  prompt: string,
  maxNLogits: number,
  desiredLogitProb: number,
  modelId: string,
  sourceSetName: string,
  // Structured chat turns. When provided, the graph server renders the prompt
  // string via the model's real chat template (no frontend chat templating);
  // the rendered string is returned as `prompt`.
  messages?: GraphChatMessage[] | null,
): Promise<GraphTokenizeResponse> => {
  const action = 'forward-pass';
  const response = await graphFetch(modelId, sourceSetName, action, {
    ...(messages && messages.length > 0 ? { messages } : { prompt }),
    max_n_logits: maxNLogits,
    desired_logit_prob: desiredLogitProb,
    request_type: action,
  });

  const responseText = await response.text();
  let json;
  try {
    json = JSON.parse(responseText);
  } catch {
    throw new Error(
      `Graph server returned non-JSON response (HTTP ${response.status}${responseText ? `: ${responseText.slice(0, 200)}` : ''})`,
    );
  }
  if (json.error) {
    throw new Error(json.error);
  }
  if (!response.ok) {
    throw new Error(`External API returned ${response.status}: ${response.statusText}`);
  }

  const salientLogits: SalientLogit[] = json.salient_logits.map((logit: SalientLogit) => ({
    token: logit.token,
    token_id: logit.token_id,
    probability: logit.probability,
  }));

  const toReturn: GraphTokenizeResponse = {
    // Prefer the server-rendered prompt (when `messages` was sent); else echo.
    prompt: typeof json.prompt === 'string' ? json.prompt : prompt,
    input_tokens: json.input_tokens,
    salient_logits: salientLogits,
    total_salient_tokens: json.total_salient_tokens,
    cumulative_probability: json.cumulative_probability,
  };

  return toReturn;
};

// Field-level documentation now lives on the python model these are generated from.
export type GraphParseChatPromptResponse = ParseChatPromptResponse;

// The inverse of the `messages` rendering done by `getGraphTokenize` /
// `generateGraphAndUploadToS3`: recover the structured turns behind a stored
// (already-rendered) graph prompt. The graph server does this against the
// model's own chat template, which is what keeps per-family special tokens out
// of the frontend.
export const getGraphParseChatPrompt = async (
  prompt: string,
  modelId: string,
  sourceSetName: string,
): Promise<GraphParseChatPromptResponse> => {
  const action = 'parse-chat-prompt';
  const response = await graphFetch(modelId, sourceSetName, action, { prompt, request_type: action });

  const responseText = await response.text();
  let json;
  try {
    json = JSON.parse(responseText);
  } catch {
    throw new Error(
      `Graph server returned non-JSON response (HTTP ${response.status}${responseText ? `: ${responseText.slice(0, 200)}` : ''})`,
    );
  }
  if (json.error) {
    throw new Error(json.error);
  }
  if (!response.ok) {
    throw new Error(`External API returned ${response.status}: ${response.statusText}`);
  }

  const messages = Array.isArray(json.messages) ? (json.messages as GraphChatMessage[]) : null;
  // Both fallbacks cover a graph server that predates these fields, and both
  // degrade the same way: toward doing nothing. An echoed prompt costs a visible
  // BOS, and an empty position list costs a slider that the server ignores.
  return {
    messages,
    is_chat: messages !== null,
    prompt: typeof json.prompt === 'string' ? json.prompt : prompt,
    unsteerable_positions: Array.isArray(json.unsteerable_positions) ? (json.unsteerable_positions as number[]) : [],
  };
};

export const generateGraphAndUploadToS3 = async (
  prompt: string,
  modelId: string,
  sourceSetName: string,
  maxNLogits: number,
  desiredLogitProb: number,
  nodeThreshold: number,
  edgeThreshold: number,
  slugIdentifier: string,
  maxFeatureNodes: number,
  signedUrl: string,
  userId: string | undefined,
  qkTopFraction: number = GRAPH_QKTOPFRACTION_DEFAULT,
  qkTopk: number = GRAPH_QKTOPK_DEFAULT,
  // Structured chat turns; when set the graph server renders the prompt string
  // via the model's real chat template (the generated graph's metadata.prompt
  // holds the rendered string, which is what gets stored).
  messages?: GraphChatMessage[] | null,
) => {
  const action = 'generate-graph';
  const isLorsa = LORSA_MODELS.includes(modelId);
  const batchSize = isLorsa ? LORSA_BATCH_SIZE : GRAPH_BATCH_SIZE;

  const body = {
    ...(messages && messages.length > 0 ? { messages } : { prompt }),
    model_id: GRAPH_MODEL_MAP[modelId as keyof typeof GRAPH_MODEL_MAP],
    batch_size: batchSize,
    max_n_logits: maxNLogits,
    desired_logit_prob: desiredLogitProb,
    node_threshold: nodeThreshold,
    edge_threshold: edgeThreshold,
    slug_identifier: slugIdentifier,
    max_feature_nodes: maxFeatureNodes,
    signed_url: signedUrl,
    user_id: userId,
    // QK tracing is always enabled for lorsa models
    ...(isLorsa
      ? {
          enable_qk_tracing: true,
          qk_top_fraction: qkTopFraction,
          qk_topk: qkTopk,
        }
      : {}),
  };
  const response = await graphFetch(modelId, sourceSetName, action, body);

  const json = await response.json();
  if (json.error) {
    throw new Error(json.error);
  }

  if (!response.ok) {
    throw new Error(`External API returned ${response.status}: ${response.statusText}`);
  }

  // no need to return anything since it was uploaded to s3
};

// Annotated with the generated type rather than inferred from the schema. These fields are
// forwarded to the graph server untouched AND documented as public request fields on
// /api/steer-logits, so the yup schema and the python model have to describe the same thing --
// the annotation is what makes a rename in `schemas.py` a compile error here instead of a 422
// at runtime. The runtime validation itself is still yup's job; this only pins the shape.
export const SteerLogitFeatureSchema: yup.ObjectSchema<SteerFeature> = yup.object({
  layer: yup.number().required('Layer is required'),
  index: yup.number().required('Index is required'),
  token_active_position: yup.number().required('Token active position is required'),
  steer_position: yup.number().nullable(),
  steer_generated_tokens: yup.boolean().required('Steer generated tokens is required'),
  delta: yup.number().nullable(),
  ablate: yup.boolean().required('Ablate is required'),
});

export type SteerLogitFeature = SteerFeature;

export const SteerLogitsRequestSchema = yup.object({
  modelId: yup.string().required('Model ID is required'),
  sourceSetName: yup.string().nullable(),
  prompt: yup.string().required('Prompt is required'),
  features: yup.array().of(SteerLogitFeatureSchema).required('Features are required'),
  nTokens: yup.number().default(STEER_N_COMPLETION_TOKENS).min(1).max(STEER_N_COMPLETION_TOKENS_MAX),
  topK: yup.number().default(STEER_TOPK_LOGITS).min(0).max(STEER_TOPK_LOGITS_MAX),
  freezeAttention: yup.boolean().default(STEER_FREEZE_ATTENTION),
  temperature: yup.number().default(STEER_TEMPERATURE).min(0.0).max(STEER_TEMPERATURE_MAX),
  freqPenalty: yup
    .number()
    .default(STEER_FREQUENCY_PENALTY)
    .min(STEER_FREQUENCY_PENALTY_MIN)
    .max(STEER_FREQUENCY_PENALTY_MAX),
  seed: yup.number().default(STEER_SEED).nullable(),
});

export type SteerLogitsRequest = yup.InferType<typeof SteerLogitsRequestSchema>;

// Left inferred: the pin that matters is on SteerResponseSchema below, which checks this
// schema's output against the generated LogitsByToken as part of the whole object.
const SteerResponseLogitsByTokenSchema = yup
  .array()
  .of(
    yup
      .object({
        token: yup.string().required(),
        top_logits: yup
          .array()
          .of(
            yup.object({
              prob: yup.number().required(),
              token: yup.string().required(),
            }),
          )
          .required(),
      })
      .required(),
  )
  .required();

export type SteerResponseLogitsByToken = LogitsByToken[];

// Same reasoning as SteerLogitFeatureSchema: /api/steer-logits returns this nearly verbatim, so
// the annotation pins it to the python model while yup keeps validating at runtime.
export const SteerResponseSchema: yup.ObjectSchema<GraphSteerResponse> = yup.object({
  DEFAULT_GENERATION: yup.string().required('Default generation is required'),
  STEERED_GENERATION: yup.string().required('Steered generation is required'),
  DEFAULT_LOGITS_BY_TOKEN: SteerResponseLogitsByTokenSchema,
  STEERED_LOGITS_BY_TOKEN: SteerResponseLogitsByTokenSchema,
});

export type SteerResponse = GraphSteerResponse;

export type SteeredPositionIdentifier = {
  modelId: string;
  layer: number;
  index: number;
  tokenActivePosition: number;
};

export const steerLogits = async (
  modelId: string,
  sourceSetName: string,
  prompt: string,
  features: SteerLogitFeature[],
  nTokens: number,
  topK: number,
  freezeAttention: boolean,
  temperature: number,
  freqPenalty: number,
  seed: number | null,
) => {
  const action = 'steer';
  // TODO: clean up model id usage
  const mappedModelId = GRAPH_GENERATION_ENABLED_MODELS.includes(modelId)
    ? GRAPH_MODEL_MAP[modelId as keyof typeof GRAPH_MODEL_MAP]
    : modelId;
  const body = {
    model_id: mappedModelId,
    prompt,
    features,
    n_tokens: nTokens,
    top_k: topK,
    freeze_attention: freezeAttention,
    temperature,
    freq_penalty: freqPenalty,
    seed,
    request_type: action,
  };

  const response = await graphFetch(modelId, sourceSetName, action, body);

  const json = await response.json();
  if (json.error) {
    throw new Error(json.error);
  }
  if (!response.ok) {
    throw new Error(`External API returned ${response.status}: ${response.statusText}`);
  }

  const validatedResponse = SteerResponseSchema.validateSync(json);

  return validatedResponse;
};
