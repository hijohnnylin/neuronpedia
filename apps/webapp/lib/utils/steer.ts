import { NeuronPartial } from '@/prisma/generated/zod';
import { Model } from '@prisma/client';
import { NPSteerMethod } from 'neuronpedia-inference-client';
import { STEER_FORCE_ALLOW_INSTRUCT_MODELS } from '../env';

export const STEER_N_COMPLETION_TOKENS = 256;
export const STEER_N_COMPLETION_TOKENS_LARGE_LLM = 1024;
export const STEER_N_COMPLETION_TOKENS_GRAPH = 10;
export const STEER_N_COMPLETION_TOKENS_THINKING = 512;
export const STEER_N_COMPLETION_TOKENS_GRAPH_MAX = 20;
export const STEER_N_COMPLETION_TOKENS_MAX = 256;
export const STEER_N_COMPLETION_TOKENS_MAX_ASSISTANT_AXIS = 512;
export const STEER_N_COMPLETION_TOKENS_MAX_LARGE_LLM = 1024;
export const STEER_N_COMPLETION_TOKENS_MAX_THINKING = 768;
export const STEER_TEMPERATURE = 0.5;
export const STEER_TEMPERATURE_GRAPH = 0;
export const STEER_TEMPERATURE_MAX = 2;
export const STEER_STRENGTH_MULTIPLIER = 1;
export const STEER_STRENGTH_MULTIPLIER_MAX = 10;
export const STEER_STRENGTH_MIN = -300;
export const STEER_STRENGTH_MAX = 300;
export const STEER_STRENGTH_GRAPH = 100;
export const STEER_STRENGTH_ADDED_MULTIPLIER_MIN = -5;
export const STEER_STRENGTH_ADDED_MULTIPLIER_MAX = 5;
export const STEER_STRENGTH_ADDED_MULTIPLIER_GRAPH = -1;
export const STEER_STRENGTH_ADDED_MULTIPLIER_CUSTOM_GRAPH = 1;
export const STEER_MULTIPLIER_STEP = 0.1;
export const STEER_SPECIAL_TOKENS = true;
export const STEER_FREQUENCY_PENALTY_GRAPH = 0;
export const STEER_FREQUENCY_PENALTY = 1.0;
export const STEER_FREQUENCY_PENALTY_MIN = -2;
export const STEER_FREQUENCY_PENALTY_MAX = 2;
export const STEER_MAX_PROMPT_CHARS = 2048;
export const STEER_MAX_PROMPT_CHARS_ASSISTANT_AXIS = 16384; // average 4 tokens = 6144 tokens max per conversation
export const STEER_SEED = 16;
export const STEER_METHOD = NPSteerMethod.SimpleAdditive;
export const STEER_METHOD_ASSISTANT_CAP = NPSteerMethod.ProjectionCap;
export const STEER_TOPK_LOGITS = 5;
export const STEER_TOPK_LOGITS_MAX = 10;
export const STEER_FREEZE_ATTENTION = true;
export const STEER_N_LOGPROBS = 5;

export const ERROR_STEER_MAX_PROMPT_CHARS = 'Total conversation length exceeds the maximum number of characters allowed. Please click Reset to start a new conversation.';

export function replaceSteerModelIdIfNeeded(modelId: string) {
  if (STEER_FORCE_ALLOW_INSTRUCT_MODELS.includes(modelId)) {
    // Only remove -it if it's at the end of the string
    return modelId.endsWith('-it') ? modelId.slice(0, -3) : modelId;
  }
  return modelId;
}

export type ChatMessage = {
  // role: "user" | "assistant" | "model" | "system";
  content: string;
  role: string;
};

export type SteerFeature = {
  modelId: string;
  layer: string;
  index: number;
  explanation?: string;
  strength: number;
  hasVector?: boolean;
  neuron?: NeuronPartial;
};

export type PromptPreset = { name: string; prompt: string };
export type FeaturePreset = {
  name: string;
  features: SteerFeature[];
  isUserVector?: boolean;
  exampleSteerOutputId?: string;
  exampleDefaultOutputId?: string;
  steerMethod?: NPSteerMethod;
  alias?: string; // can directly link to this preset from URL
};

export type SteerPreset = {
  model: Model;
  defaultPrompt: string;
  promptPresets: PromptPreset[];
  featurePresets: FeaturePreset[];
  defaultSelectedFeatures: SteerFeature[];
};

// we only have to do this for old steer outputs. new ones are saved as chat templates.
export function convertOldSteerOutputToChatMessages(raw: string): ChatMessage[] {
  let newRaw = raw.replaceAll('<bos>', '');
  newRaw = newRaw.replaceAll('<eos>', '');
  // split by message
  const split = newRaw.split('<end_of_turn>');
  const toReturn: ChatMessage[] = split.map((s) => {
    // eslint-disable-next-line no-param-reassign
    s = s.trimStart();
    // remove the <start_of_turn>
    // eslint-disable-next-line no-param-reassign
    s = s.replaceAll('<start_of_turn>', '');
    const splitByLineBreak = s.split('\n');
    if (splitByLineBreak.length === 0) {
      return { content: '', role: 'system' as ChatMessage['role'] };
    }
    let role = splitByLineBreak[0].trim();
    if (role !== 'user' && role !== 'model' && role !== 'assistant') {
      role = 'system' as ChatMessage['role'];
    }
    const content = splitByLineBreak.slice(1).join('\n');
    return { content, role: role as ChatMessage['role'] };
  });
  return toReturn;
}
