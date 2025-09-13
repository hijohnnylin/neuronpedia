import OpenAI, { AzureOpenAI } from 'openai';
import {
  AZURE_OPENAI_API_KEY,
  AZURE_OPENAI_ENDPOINT,
  EMBEDDING_PROVIDER,
  NEXT_PUBLIC_URL,
  OPENAI_API_KEY,
  OPENROUTER_API_KEY,
  SITE_NAME_VERCEL_DEPLOY,
} from './env';

const OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1';

export type Provider = 'openai' | 'azure' | 'openrouter';

export interface OpenAIClientConfig {
  provider?: Provider;
  apiKey?: string;
  azureConfig?: {
    endpoint?: string;
    apiVersion?: string;
  };
}

/**
 * Configuration and factory for OpenAI clients supporting multiple providers.
 * Returns the appropriate OpenAI/AzureOpenAI client instance based on provider.
 *
 * Provider selection priority:
 * 1. Explicit provider in config
 * 2. USE_* environment flags
 * 3. API key availability: Azure > OpenRouter > OpenAI
 */
export class OpenAIClientFactory {
  static createClient(config?: OpenAIClientConfig): OpenAI | AzureOpenAI {
    const provider = config?.provider || OpenAIClientFactory.determineProvider();

    switch (provider) {
      case 'azure':
        return OpenAIClientFactory.createAzureClient(config);
      case 'openrouter':
        return OpenAIClientFactory.createOpenRouterClient(config);
      case 'openai':
      default:
        return OpenAIClientFactory.createOpenAIClient(config);
    }
  }

  /**
   * Determines provider based on environment configuration.
   * Uses EMBEDDING_PROVIDER environment variable.
   */
  static determineProvider(): Provider {
    return EMBEDDING_PROVIDER;
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

    return new AzureOpenAI({
      apiKey,
      endpoint,
      apiVersion: '2024-10-21' // <-- this is required, inference wont work without it
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
   * Get the provider type for a given client instance.
   */
  static getProvider(config?: OpenAIClientConfig): Provider {
    return config?.provider || OpenAIClientFactory.determineProvider();
  }
}

// Singleton instance
// eslint-disable-next-line @typescript-eslint/naming-convention, no-underscore-dangle
let _instance: OpenAI | AzureOpenAI | null = null;

/**
 * Get or create singleton OpenAI client instance.
 * Returns the underlying OpenAI/AzureOpenAI client directly.
 */
export const getOpenAIClient = (): OpenAI | AzureOpenAI => {
  if (!_instance) {
    _instance = OpenAIClientFactory.createClient({ provider: OpenAIClientFactory.determineProvider() });
  }
  return _instance;
};

// Example usage:
//
// // Using singleton
// const client = getOpenAIClient();
// const response = await client.embeddings.create({
//     input: text,
//     model: embeddingModel,
//     dimensions,
//   });
//
// // Custom instance
// const azureClient = OpenAIClientFactory.createClient({
//   provider: 'azure',
//   apiKey: 'custom-key',
//   azureConfig: {
//     endpoint: 'https://custom.openai.azure.com/',
//   }
// });