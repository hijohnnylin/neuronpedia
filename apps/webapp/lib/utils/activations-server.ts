import { getOneRandomServerHostForModel } from '@/lib/db/inference-host-source';
import { getTransformerLensModelIdIfExists } from '@/lib/db/model';
import { INFERENCE_SERVER_SECRET } from '@/lib/env';
import { throwIfInferenceError } from '@/lib/utils/inference';

export const ACTIVATION_RAW_MAX_PROMPT_CHAR_LENGTH = 8000;
export const ACTIVATION_RAW_MAX_PROMPTS_PER_BATCH = 16;

type ActivationRawHookPoint = 'residual_stream';
type ActivationRawType = 'final_output_token';

export type ActivationRawRequest = {
  model: string;
  prompts: string[];
  /** Omit for every layer. */
  layers?: number[];
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

/**
 * Residual stream vectors at each prompt's final token, straight from the model.
 *
 * Still a hand-written fetch, though `/v1/activation/raw` is now in the spec and could go
 * through the typed client like the rest.
 */
export async function getRawActivations(request: ActivationRawRequest): Promise<ActivationRawResponse> {
  const transformerLensModelId = await getTransformerLensModelIdIfExists(request.model);
  const host = await getOneRandomServerHostForModel(request.model);

  const response = await fetch(`${host}/v1/activation/raw`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-SECRET-KEY': INFERENCE_SERVER_SECRET,
      'Accept-Encoding': 'gzip',
    },
    body: JSON.stringify({
      model: transformerLensModelId,
      prompts: request.prompts,
      layers: request.layers,
      hook_point: request.hook_point ?? 'residual_stream',
      type: request.type ?? 'final_output_token',
    }),
    cache: 'no-store',
  });
  await throwIfInferenceError(response);

  return (await response.json()) as ActivationRawResponse;
}
