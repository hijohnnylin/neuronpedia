import { ACTIVATIONS_SERVER, ACTIVATIONS_SERVER_SECRET } from '@/lib/env';

export const ACTIVATION_RAW_MAX_PROMPT_CHAR_LENGTH = 8000;
export const ACTIVATION_RAW_MAX_PROMPTS_PER_BATCH = 16;

type ActivationRawHookPoint = 'residual_stream';
type ActivationRawType = 'final_output_token';

export type ActivationRawRequest = {
  model: string;
  prompts: string[];
  hook_point?: ActivationRawHookPoint;
  type?: ActivationRawType;
};

export type ActivationRawLayer = {
  layer: number;
  token_indices: number[];
  values: number[][];
};

export type ActivationRawPromptResult = {
  token_strings: string[];
  token_ids: number[];
  activations: ActivationRawLayer[];
};

export type ActivationRawResponse = {
  hook_point: string;
  type: string;
  dtype: string;
  device: string;
  results: ActivationRawPromptResult[];
};

export async function getRawActivations(request: ActivationRawRequest): Promise<ActivationRawResponse> {
  const response = await fetch(`${ACTIVATIONS_SERVER}/raw`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(ACTIVATIONS_SERVER_SECRET ? { 'X-SECRET-KEY': ACTIVATIONS_SERVER_SECRET } : {}),
    },
    body: JSON.stringify({
      model: request.model,
      prompts: request.prompts,
      hook_point: request.hook_point ?? 'residual_stream',
      type: request.type ?? 'final_output_token',
    }),
    cache: 'no-store',
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Activations server error (${response.status}): ${errorText}`);
  }

  return (await response.json()) as ActivationRawResponse;
}
