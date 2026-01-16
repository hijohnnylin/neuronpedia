import { GRAPH_RUNPOD_SECRET } from '@/lib/env';
import * as yup from 'yup';
import {
  getAuthHeaderForGraphServerRequest,
  getGraphServerRequestUrlForSourceSet,
  getIsRunpodServerlessHostForSourceSet,
  wrapRequestBodyForRunpodIfNeeded,
} from '../db/graph-host-source';
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

export const MAX_RUNPOD_JOBS_IN_QUEUE = 1000;
export const RUNPOD_BUSY_ERROR = 'RUNPOD_BUSY';

export const GRAPH_MAX_PROMPT_LENGTH_CHARS = 10000;
export const GRAPH_BATCH_SIZE = 48;
// this time estimate comes from testing different prompt lengths with batch size 48, and is only valid for gemma-2-2b, for a40
// with nnsight we are using lazy encoder, so we need to add time for this
export const getEstimatedTimeFromNumTokens = (numTokens: number, nnsight: boolean = false) =>
  11.2 * Math.log2(Math.max(numTokens, 4)) - 7 + (nnsight ? 20 : 0); // add a few seconds buffer
export const GRAPH_MAX_TOKENS = 64;
export const GRAPH_GENERATION_ENABLED_MODELS = ['gemma-2-2b', 'gemma-3-4b-it', 'qwen3-4b'];
export const GRAPH_MODEL_MAP = {
  'gemma-2-2b': 'google/gemma-2-2b',
  'gemma-3-4b-it': 'google/gemma-3-4b-it',
  'qwen3-4b': 'Qwen/Qwen3-4B',
};

export const GRAPH_S3_USER_GRAPHS_DIR = 'user-graphs';

export const GRAPH_MAXNLOGITS_MIN = 5;
export const GRAPH_MAXNLOGITS_MAX = 15;
export const GRAPH_MAXNLOGITS_DEFAULT = 10;
export const GRAPH_DESIREDLOGITPROB_MIN = 0.6;
export const GRAPH_DESIREDLOGITPROB_MAX = 0.99;
export const GRAPH_DESIREDLOGITPROB_DEFAULT = 0.95;
export const GRAPH_NODETHRESHOLD_MIN = 0.5;
export const GRAPH_NODETHRESHOLD_MAX = 1.0;
export const GRAPH_NODETHRESHOLD_DEFAULT = 0.8;
export const GRAPH_EDGETHRESHOLD_MIN = 0.8;
export const GRAPH_EDGETHRESHOLD_MAX = 1.0;
export const GRAPH_EDGETHRESHOLD_DEFAULT = 0.85;
export const GRAPH_MAXFEATURENODES_MIN = 3000;
export const GRAPH_MAXFEATURENODES_MAX = 10000;
export const GRAPH_MAXFEATURENODES_DEFAULT = 5000;
export const GRAPH_SLUG_MIN = 2;

export const GRAPH_DYNAMIC_PRUNING_THRESHOLD_DEFAULT = 0.6;

export const GRAPH_ANONYMOUS_USER_ID = 'anonymous';

export const MAX_PUT_REQUESTS_PER_DAY = 200;

export const graphGenerateSchemaClient = yup.object({
  prompt: yup
    .string()
    .max(GRAPH_MAX_PROMPT_LENGTH_CHARS, `Prompt cannot exceed ${GRAPH_MAX_PROMPT_LENGTH_CHARS} characters.`)
    .min(1, 'Prompt is required.')
    .required(),
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
  slug: yup.string(),
});

export const checkRunpodQueueJobs = async (host: string) => {
  const response = await fetch(`${host}/health`, {
    headers: {
      Authorization: `Bearer ${GRAPH_RUNPOD_SECRET}`,
    },
  });

  if (!response.ok) {
    throw new Error(`RunPod health check failed: ${response.status}`);
  }

  const data = await response.json();

  if (data.jobs !== undefined && data.jobs.inQueue !== undefined) {
    return data.jobs.inQueue;
  }
  throw new Error('RunPod health check failed: jobs not found');
};

export type SalientLogit = { token: string; token_id: number; probability: number };

export type GraphTokenizeResponse = {
  prompt: string;
  input_tokens: string[];
  salient_logits: SalientLogit[];
  total_salient_tokens: number;
  cumulative_probability: number;
};

export interface GraphGenerateRunpodResponse {
  error?: string;
  output: GraphTokenizeResponse;
}

export const getGraphTokenize = async (
  prompt: string,
  maxNLogits: number,
  desiredLogitProb: number,
  modelId: string,
  sourceSetName: string,
): Promise<GraphTokenizeResponse> => {
  const isRunpodServerlessHost = await getIsRunpodServerlessHostForSourceSet(modelId, sourceSetName);
  const action = 'forward-pass';
  const body = {
    prompt,
    max_n_logits: maxNLogits,
    desired_logit_prob: desiredLogitProb,
    request_type: action,
  };

  const response = await fetch(
    `${await getGraphServerRequestUrlForSourceSet(modelId, sourceSetName, action, isRunpodServerlessHost)}`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaderForGraphServerRequest(isRunpodServerlessHost),
      },
      body: JSON.stringify(wrapRequestBodyForRunpodIfNeeded(body, isRunpodServerlessHost)),
    },
  );

  let json = await response.json();
  if (json.error) {
    throw new Error(json.error);
  }
  if (!response.ok) {
    throw new Error(`External API returned ${response.status}: ${response.statusText}`);
  }

  if (isRunpodServerlessHost) {
    json = json.output;
  }

  const salientLogits: SalientLogit[] = json.salient_logits.map((logit: SalientLogit) => ({
    token: logit.token,
    token_id: logit.token_id,
    probability: logit.probability,
  }));

  const toReturn: GraphTokenizeResponse = {
    prompt,
    input_tokens: json.input_tokens,
    salient_logits: salientLogits,
    total_salient_tokens: json.total_salient_tokens,
    cumulative_probability: json.cumulative_probability,
  };

  return toReturn;
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
) => {
  const isRunpodServerlessHost = await getIsRunpodServerlessHostForSourceSet(modelId, sourceSetName);
  const action = 'generate-graph';
  const body = {
    prompt,
    model_id: GRAPH_MODEL_MAP[modelId as keyof typeof GRAPH_MODEL_MAP],
    batch_size: GRAPH_BATCH_SIZE,
    max_n_logits: maxNLogits,
    desired_logit_prob: desiredLogitProb,
    node_threshold: nodeThreshold,
    edge_threshold: edgeThreshold,
    slug_identifier: slugIdentifier,
    max_feature_nodes: maxFeatureNodes,
    signed_url: signedUrl,
    user_id: userId,
  };
  const response = await fetch(
    `${await getGraphServerRequestUrlForSourceSet(modelId, sourceSetName, action, isRunpodServerlessHost)}`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaderForGraphServerRequest(isRunpodServerlessHost),
      },
      body: JSON.stringify(wrapRequestBodyForRunpodIfNeeded(body, isRunpodServerlessHost)),
    },
  );

  const json = await response.json();
  if (json.error) {
    throw new Error(json.error);
  }

  if (!response.ok) {
    throw new Error(`External API returned ${response.status}: ${response.statusText}`);
  }

  // no need to return anything since it was uploaded to s3
};

export const SteerLogitFeatureSchema = yup.object({
  layer: yup.number().required('Layer is required'),
  index: yup.number().required('Index is required'),
  token_active_position: yup.number().required('Token active position is required'),
  steer_position: yup.number().nullable(),
  steer_generated_tokens: yup.boolean().required('Steer generated tokens is required'),
  delta: yup.number().nullable(),
  ablate: yup.boolean().required('Ablate is required'),
});

export type SteerLogitFeature = yup.InferType<typeof SteerLogitFeatureSchema>;

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
  steeredOutputOnly: yup.boolean().default(false),
});

export type SteerLogitsRequest = yup.InferType<typeof SteerLogitsRequestSchema>;

export const SteerResponseLogitsByTokenSchema = yup
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

export type SteerResponseLogitsByToken = yup.InferType<typeof SteerResponseLogitsByTokenSchema>;

export const SteerResponseSchema = yup.object({
  DEFAULT_GENERATION: yup.string().required('Default generation is required'),
  STEERED_GENERATION: yup.string().required('Steered generation is required'),
  DEFAULT_LOGITS_BY_TOKEN: SteerResponseLogitsByTokenSchema,
  STEERED_LOGITS_BY_TOKEN: SteerResponseLogitsByTokenSchema,
});

export type SteerResponse = yup.InferType<typeof SteerResponseSchema>;

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
  steeredOutputOnly: boolean,
) => {
  const isRunpodServerlessHost = await getIsRunpodServerlessHostForSourceSet(modelId, sourceSetName);

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
    steered_output_only: steeredOutputOnly,
    request_type: action,
  };

  const response = await fetch(
    `${await getGraphServerRequestUrlForSourceSet(modelId, sourceSetName, action, isRunpodServerlessHost)}`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaderForGraphServerRequest(isRunpodServerlessHost),
      },
      body: JSON.stringify(wrapRequestBodyForRunpodIfNeeded(body, isRunpodServerlessHost)),
    },
  );

  let json = await response.json();
  if (json.error) {
    throw new Error(json.error);
  }
  if (!response.ok) {
    throw new Error(`External API returned ${response.status}: ${response.statusText}`);
  }

  if (isRunpodServerlessHost) {
    json = json.output;
  }

  const validatedResponse = SteerResponseSchema.validateSync(json);

  return validatedResponse;
};
