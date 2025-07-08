import {
  ChatMessage,
  SteerFeature,
  STEER_TEMPERATURE,
  STEER_N_COMPLETION_TOKENS,
  STEER_FREQUENCY_PENALTY,
  STEER_SEED,
  STEER_STRENGTH_MULTIPLIER,
  STEER_SPECIAL_TOKENS,
} from './steer'; // Assuming steer.ts is in the same directory

export interface SteerChatApiRequestPayload {
  modelId: string; // The model ID for generation
  features: SteerFeature[];
  defaultChatMessages: ChatMessage[];
  steeredChatMessages: ChatMessage[];
  temperature?: number;
  nTokens?: number;
  freqPenalty?: number;
  seed?: number;
  strengthMultiplier?: number;
  steerSpecialTokens: boolean; // Made non-optional as it's required by the API
}

export interface SteerChatApiResponse {
  STEERED?: {
    chatTemplate: ChatMessage[];
    // other fields might exist but are not needed for this step
  } | null;
  DEFAULT?: {
    chatTemplate: ChatMessage[];
    // other fields might exist
  } | null;
  // Potentially other fields like error messages if the API returns them in a structured way
}

export async function callSteerChatApi(
  params: SteerChatApiRequestPayload,
): Promise<SteerChatApiResponse> {
  const {
    modelId,
    features,
    defaultChatMessages,
    steeredChatMessages,
    temperature = STEER_TEMPERATURE,
    nTokens = STEER_N_COMPLETION_TOKENS,
    freqPenalty = STEER_FREQUENCY_PENALTY,
    seed = STEER_SEED,
    strengthMultiplier = STEER_STRENGTH_MULTIPLIER,
    steerSpecialTokens = STEER_SPECIAL_TOKENS,
  } = params;

  const neuronpediaApiKey = process.env.NEXT_PUBLIC_NEURONPEDIA_APIKEY;
  let steerChatEndpoint = '/api/steer-chat';
  const requestHeaders: HeadersInit = { 'Content-Type': 'application/json' };

  if (neuronpediaApiKey && neuronpediaApiKey.trim() !== '') {
    steerChatEndpoint = '/api/internal-proxy/steer-chat';
    requestHeaders['X-Forwarded-Client-API-Key'] = neuronpediaApiKey.trim();
  }

  const requestBodyJson = {
    modelId,
    features,
    defaultChatMessages,
    steeredChatMessages,
    temperature, // Assuming API accepts 'temperature' as is
    n_tokens: nTokens,
    freq_penalty: freqPenalty,
    seed, // Assuming API accepts 'seed' as is
    strength_multiplier: strengthMultiplier,
    steer_special_tokens: steerSpecialTokens, // Ensure API receives snake_case
  };

  const response = await fetch(steerChatEndpoint, {
    method: 'POST',
    headers: requestHeaders,
    body: JSON.stringify(requestBodyJson),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `API Error: ${response.status} ${response.statusText}. Details: ${errorText}`,
    );
  }

  return response.json() as Promise<SteerChatApiResponse>;
}