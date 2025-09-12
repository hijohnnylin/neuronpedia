import OpenAI, { AzureOpenAI } from 'openai';
import type {
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionCreateParams,
  CreateEmbeddingResponse,
  EmbeddingCreateParams,
} from 'openai/resources';
import type { Stream } from 'openai/streaming';
import {
  AZURE_API_VERSION,
  AZURE_OPENAI_API_KEY,
  AZURE_OPENAI_ENDPOINT,
  NEXT_PUBLIC_URL,
  OPENAI_API_KEY,
  OPENAI_DEPLOYMENT_NAME,
  OPENROUTER_API_KEY,
  SITE_NAME_VERCEL_DEPLOY,
  USE_AZURE_OPENAI,
  USE_OPENAI,
  USE_OPENROUTER,
} from './env';

const OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1';

export type Provider = 'openai' | 'azure' | 'openrouter';

export interface OpenAIClientConfig {
  provider?: Provider;
  apiKey?: string;
  deploymentName?: string;
  azureConfig?: {
    endpoint?: string;
    apiVersion?: string;
  };
}

/**
 * Unified OpenAI client supporting multiple providers (OpenAI, Azure, OpenRouter).
 * 
 * Provider selection priority:
 * 1. Explicit provider in constructor config
 * 2. USE_* environment flags
 * 3. API key availability: Azure > OpenRouter > OpenAI
 */
export class OpenAIClient {
  private client: OpenAI | AzureOpenAI;

  private provider: Provider;

  private defaultDeploymentName?: string;

  constructor(config?: OpenAIClientConfig) {
    this.provider = config?.provider || OpenAIClient.determineProvider();
    this.defaultDeploymentName = config?.deploymentName || OPENAI_DEPLOYMENT_NAME;

    switch (this.provider) {
      case 'azure':
        this.client = OpenAIClient.createAzureClient(config);
        break;
      case 'openrouter':
        this.client = OpenAIClient.createOpenRouterClient(config);
        break;
      case 'openai':
      default:
        this.client = OpenAIClient.createOpenAIClient(config);
        break;
    }
  }

  /**
   * Determines provider based on environment configuration.
   * Checks USE_* flags and API key availability in priority order.
   */
  static determineProvider(): Provider {
    if (USE_AZURE_OPENAI && AZURE_OPENAI_API_KEY && AZURE_OPENAI_ENDPOINT) return 'azure';
    if (USE_OPENROUTER && OPENROUTER_API_KEY) return 'openrouter';
    if (USE_OPENAI && OPENAI_API_KEY) return 'openai';
    return 'openai'; // Default fallback
  }

  private static createOpenAIClient(config?: OpenAIClientConfig): OpenAI {
    const apiKey = config?.apiKey || OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error('OpenAI API key not found. Set OPENAI_API_KEY in env.ts.');
    }
    return new OpenAI({ apiKey });
  }

  private static createAzureClient(config?: OpenAIClientConfig): AzureOpenAI {
    const apiKey = config?.apiKey || AZURE_OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error('Azure OpenAI API key not found. Set AZURE_OPENAI_API_KEY in env.ts.');
    }

    const endpoint = config?.azureConfig?.endpoint || AZURE_OPENAI_ENDPOINT;
    const apiVersion = config?.azureConfig?.apiVersion || AZURE_API_VERSION;

    return new AzureOpenAI({
      apiKey,
      apiVersion,
      endpoint,
    });
  }

  private static createOpenRouterClient(config?: OpenAIClientConfig): OpenAI {
    const apiKey = config?.apiKey || OPENROUTER_API_KEY;
    if (!apiKey) {
      throw new Error('OpenRouter API key not found. Set OPENROUTER_API_KEY in env.ts.');
    }

    return new OpenAI({
      baseURL: OPENROUTER_BASE_URL,
      apiKey,
      defaultHeaders: {
        'HTTP-Referer': NEXT_PUBLIC_URL || '',
        'X-Title': SITE_NAME_VERCEL_DEPLOY || 'Neuronpedia',
      },
    });
  }

  /**
   * Create chat completions. Uses default deployment name if model not specified.
   */
  async createChatCompletion<T extends ChatCompletionCreateParams>(
    params: T,
  ): Promise<T['stream'] extends true ? Stream<ChatCompletionChunk> : ChatCompletion> {
    const finalParams =
      this.defaultDeploymentName && !params.model ? { ...params, model: this.defaultDeploymentName } : params;

    return this.client.chat.completions.create(finalParams) as any;
  }

  /**
   * Create embeddings. `params.model` is required.
   */
  async createEmbedding(params: EmbeddingCreateParams): Promise<CreateEmbeddingResponse> {
    return this.client.embeddings.create(params);
  }

  getProvider(): Provider {
    return this.provider;
  }

  getClient(): OpenAI | AzureOpenAI {
    return this.client;
  }
}

// Singleton instance
// eslint-disable-next-line @typescript-eslint/naming-convention, no-underscore-dangle
let _instance: OpenAIClient | null = null;

/**
 * Get or create singleton OpenAI client instance.
 */
export const getOpenAIClient = (): OpenAIClient => {
  if (!_instance) {
    _instance = new OpenAIClient({ provider: OpenAIClient.determineProvider() });
  }
  return _instance;
};

// Example usage:
// 
// // Using singleton
// const client = getOpenAIClient();
// const completion = await client.createChatCompletion({
//   messages: [{ role: 'user', content: 'Hello!' }],
//   model: 'gpt-4o-mini',
// });
//
// // Custom instance
// const azureClient = new OpenAIClient({
//   provider: 'azure',
//   apiKey: 'custom-key',
//   azureConfig: {
//     endpoint: 'https://custom.openai.azure.com/',
//   }
// });