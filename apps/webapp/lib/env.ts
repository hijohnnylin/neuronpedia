import { config } from 'dotenv';
import { z } from "zod";
import { createEnv } from "@t3-oss/env-nextjs";

// If it's not undefined, then it's a one click deploy. It doesn't matter what the value itself is.
// Also, if it's one-click-deploy on Vercel, we always use the demo environment variables.
export const SITE_NAME_VERCEL_DEPLOY = process.env.NEXT_PUBLIC_SITE_NAME_VERCEL_DEPLOY;
export const IS_ONE_CLICK_VERCEL_DEPLOY = SITE_NAME_VERCEL_DEPLOY !== undefined;
if (SITE_NAME_VERCEL_DEPLOY) {
  // @ts-ignore
  if (typeof EdgeRuntime !== 'string') {
    config({ path: '.env.demo', override: true });
  }
}

// Custom transformer for string to boolean
const onlyBool = z
  .string()
  // only allow "true" or "false"
  .refine((s) => s === 'true' || s === 'false')
  // transform to boolean
  .transform((v) => v === 'true');

export const env = createEnv({
  server: {
    // All non-NEXT_PUBLIC_ variables go here
    // Domain and Auth
    NEXTAUTH_URL: z.string().default(''),

    // Feature Flags
    ENABLE_RATE_LIMITER: onlyBool.default('false'),
    ENABLE_VERCEL_ANALYTICS: onlyBool.default('false'),
    
    // Default User IDs
    DEFAULT_CREATOR_USER_ID: z.string().default('clkht01d40000jv08hvalcvly'),
    INFERENCE_ACTIVATION_USER_ID: z.string().default('cljgamm90000076zdchicy6zj'),
    PUBLIC_ACTIVATIONS_USER_IDS: z
      .string()
      .default('')
      .transform((v) =>
        v ? v.split(',').map((id) => id.trim()) : ['cljj57d3c000076ei38vwnv35', 'clkht01d40000jv08hvalcvly'],
      ),

    // Email Sending Providers
    USE_AWS_SES: onlyBool.default('false'),
    USE_RESEND: onlyBool.default('false'),
    USE_SMTP: onlyBool.default('false'),

    // AWS SES
    AWS_ACCESS_KEY_ID: z.string().default(''),
    AWS_SECRET_ACCESS_KEY: z.string().default(''),
    // Resend.com
    RESEND_EMAIL_API_KEY: z.string().default(''),
    // SMTP
    SMTP_SERVER_HOST: z.string().default(''),
    SMTP_SERVER_PORT: z.coerce.number().min(1).max(65535).default(25),
    SMTP_SERVER_FROM: z.string().email().default('noreply@neuronpedia.org'),

    // AI API Keys
    OPENAI_API_KEY: z.string().default(''),
    GEMINI_API_KEY: z.string().default(''),
    ANTHROPIC_API_KEY: z.string().default(''),
    OPENROUTER_API_KEY: z.string().default(''),
    AZURE_OPENAI_API_KEY: z.string().default(''),

    // Azure-specific provider config
    AZURE_OPENAI_ENDPOINT: z.string().url().or(z.literal('')).default(''),
    AZURE_API_VERSION: z.string().default('2024-02-01'),
    OPENAI_DEPLOYMENT_NAME: z.string().default('gpt-4o-mini'),

    // Provider selection flags
    USE_AZURE_OPENAI: onlyBool.default('false'),
    USE_OPENROUTER: onlyBool.default('false'),
    USE_OPENAI: onlyBool.default('true'),

    // Support Servers - Inference
    USE_LOCALHOST_INFERENCE: onlyBool.default('false'),
    INFERENCE_SERVER_SECRET: z.string().default(''),

    // Support Servers - Autointerp
    USE_LOCALHOST_AUTOINTERP: onlyBool.default('false'),
    AUTOINTERP_SERVER: z.string().default(''),
    AUTOINTERP_SERVER_SECRET: z.string().default(''),

    // Support Servers - Graph
    USE_LOCALHOST_GRAPH: onlyBool.default('false'),
    GRAPH_SERVER: z.string().default(''),
    GRAPH_SERVER_SECRET: z.string().default(''),
    USE_RUNPOD_GRAPH: onlyBool.default('false'),
    GRAPH_RUNPOD_SECRET: z.string().default(''),
    GRAPH_RUNPOD_SERVER: z.string().default(''),

    // Authentication - Apple
    APPLE_CLIENT_ID: z.string().default(''),
    APPLE_CLIENT_SECRET: z.string().default(''),
    // Authentication - GitHub
    GITHUB_ID: z.string().default(''),
    GITHUB_SECRET: z.string().default(''),
    // Authentication - Google
    GOOGLE_CLIENT_ID: z.string().default(''),
    GOOGLE_CLIENT_SECRET: z.string().default(''),

    // Misc
    NODE_ENV: z.string().default(''),
    IS_DOCKER_COMPOSE: onlyBool.default('false'),
    GRAPH_ADMIN_BROWSE_KEY: z.string().default(''),
    HIGHER_LIMIT_API_TOKENS: z
      .string()
      .default('')
      .transform((v) => (v ? v.split(',').map((t) => t.trim()) : [])),
  },
  client: {
    // All NEXT_PUBLIC_ prefixed variables go here
    NEXT_PUBLIC_URL: z.string().default(''),
    NEXT_PUBLIC_ENABLE_SIGNIN: onlyBool.default('false'),
    
    // Default Values
    NEXT_PUBLIC_CONTACT_EMAIL_ADDRESS: z.string().email().default('johnny@neuronpedia.org'),
    NEXT_PUBLIC_DEFAULT_RELEASE_NAME: z.string().default(''),
    NEXT_PUBLIC_DEFAULT_MODELID: z.string().default(''),
    NEXT_PUBLIC_DEFAULT_SOURCESET: z.string().default(''),
    NEXT_PUBLIC_DEFAULT_SOURCE: z.string().default(''),
    NEXT_PUBLIC_DEFAULT_STEER_MODEL: z.string().default(''),
    NEXT_PUBLIC_STEER_FORCE_ALLOW_INSTRUCT_MODELS: z
      .string()
      .default('')
      .transform((v) => (v ? v.split(',').map((m) => m.trim()) : [])),
    
    NEXT_PUBLIC_SEARCH_TOPK_MAX_CHAR_LENGTH: z
      .string()
      .transform((v) => (v ? parseInt(v, 10) : 1024))
      .default('1024'),
    
    NEXT_PUBLIC_DEMO_MODE: onlyBool.default('false'),
  },
  // For Next.js >= 13.4.4, we only need to specify client variables
  experimental__runtimeEnv: {
    NEXT_PUBLIC_URL: process.env.NEXT_PUBLIC_URL,
    NEXT_PUBLIC_ENABLE_SIGNIN: process.env.NEXT_PUBLIC_ENABLE_SIGNIN,
    NEXT_PUBLIC_CONTACT_EMAIL_ADDRESS: process.env.NEXT_PUBLIC_CONTACT_EMAIL_ADDRESS,
    NEXT_PUBLIC_DEFAULT_RELEASE_NAME: process.env.NEXT_PUBLIC_DEFAULT_RELEASE_NAME,
    NEXT_PUBLIC_DEFAULT_MODELID: process.env.NEXT_PUBLIC_DEFAULT_MODELID,
    NEXT_PUBLIC_DEFAULT_SOURCESET: process.env.NEXT_PUBLIC_DEFAULT_SOURCESET,
    NEXT_PUBLIC_DEFAULT_SOURCE: process.env.NEXT_PUBLIC_DEFAULT_SOURCE,
    NEXT_PUBLIC_DEFAULT_STEER_MODEL: process.env.NEXT_PUBLIC_DEFAULT_STEER_MODEL,
    NEXT_PUBLIC_STEER_FORCE_ALLOW_INSTRUCT_MODELS: process.env.NEXT_PUBLIC_STEER_FORCE_ALLOW_INSTRUCT_MODELS,
    NEXT_PUBLIC_SEARCH_TOPK_MAX_CHAR_LENGTH: process.env.NEXT_PUBLIC_SEARCH_TOPK_MAX_CHAR_LENGTH,
    NEXT_PUBLIC_DEMO_MODE: process.env.NEXT_PUBLIC_DEMO_MODE,
  },
  skipValidation: !!process.env.SKIP_ENV_VALIDATION,
  onValidationError: (error: any) => {
    console.error('Environment validation failed:', error);
    throw error;
  },
  onInvalidAccess: (_variable: any) => {
    // We don't need onInvalidAccess because Next.js already handles the security by setting 
    // server-only variables to undefined on the client. Since the code is open source, 
    // clients knowing the variable names is not a security issue - they just can't access 
    // the values. The Zod default values ensure the app doesn't crash when these are undefined.
    // console.warn(
    //   `❌ Attempted to access server-side environment variable '${_variable}' on the client.`
    // );
  },
});

// Validation logic after parsing
const IS_SERVER = typeof window === 'undefined';
if (IS_SERVER) {
  // Provider validation
  const trueCount = [env.USE_AZURE_OPENAI, env.USE_OPENROUTER, env.USE_OPENAI].filter(Boolean).length;
  if (trueCount !== 1) {
    throw new Error('Exactly one provider must be enabled');
  }

  // OpenAI validation
  if (env.USE_OPENAI && !env.OPENAI_API_KEY) {
    throw new Error('OPENAI_API_KEY is required when USE_OPENAI is enabled');
  }

  // OpenRouter validation
  if (env.USE_OPENROUTER && !env.OPENROUTER_API_KEY) {
    throw new Error('OPENROUTER_API_KEY is required when USE_OPENROUTER is enabled');
  }

  // Azure OpenAI validation
  if (env.USE_AZURE_OPENAI) {
    if (!env.AZURE_OPENAI_API_KEY) {
      throw new Error('AZURE_OPENAI_API_KEY is required when USE_AZURE_OPENAI is enabled');
    }
    if (!env.AZURE_OPENAI_ENDPOINT) {
      throw new Error('AZURE_OPENAI_ENDPOINT is required when USE_AZURE_OPENAI is enabled');
    }
    if (!env.OPENAI_DEPLOYMENT_NAME) {
      throw new Error('OPENAI_DEPLOYMENT_NAME is required when USE_AZURE_OPENAI is enabled');
    }
  }

  // Email provider validation
  const emailProviderCount = [env.USE_AWS_SES, env.USE_RESEND, env.USE_SMTP].filter(Boolean).length;
  if (emailProviderCount > 1) {
    throw new Error('Only one email provider can be enabled at a time');
  }

  // AWS SES validation
  if (env.USE_AWS_SES) {
    if (!env.AWS_ACCESS_KEY_ID) {
      throw new Error('AWS_ACCESS_KEY_ID is required when USE_AWS_SES is enabled');
    }
    if (!env.AWS_SECRET_ACCESS_KEY) {
      throw new Error('AWS_SECRET_ACCESS_KEY is required when USE_AWS_SES is enabled');
    }
  }

  // Resend validation
  if (env.USE_RESEND && !env.RESEND_EMAIL_API_KEY) {
    throw new Error('RESEND_EMAIL_API_KEY is required when USE_RESEND is enabled');
  }

  // SMTP validation
  if (env.USE_SMTP) {
    if (!env.SMTP_SERVER_HOST) {
      throw new Error('SMTP_SERVER_HOST is required when USE_SMTP is enabled');
    }
    if (!env.SMTP_SERVER_PORT) {
      throw new Error('SMTP_SERVER_PORT is required when USE_SMTP is enabled');
    }
    if (!env.SMTP_SERVER_FROM) {
      throw new Error('SMTP_SERVER_FROM is required when USE_SMTP is enabled');
    }
  }

  // Graph server validation
  if (env.USE_RUNPOD_GRAPH && env.USE_LOCALHOST_GRAPH) {
    throw new Error('USE_LOCALHOST_GRAPH and USE_RUNPOD_GRAPH cannot both be true');
  }
}

// Export all variables with their original names for backward compatibility
// TODO stop exporting each of these. Instead import ``env`` in target modules and index into the desired key, e.g. ``env.OPENAI_API_KEY``
export const {
  // Domain and Auth
  NEXT_PUBLIC_URL,
  NEXTAUTH_URL,

  // Feature Flags
  ENABLE_RATE_LIMITER,
  ENABLE_VERCEL_ANALYTICS,
  NEXT_PUBLIC_ENABLE_SIGNIN: NEXT_PUBLIC_ENABLE_SIGNIN_RAW,

  // Default Values
  NEXT_PUBLIC_CONTACT_EMAIL_ADDRESS: CONTACT_EMAIL_ADDRESS,
  NEXT_PUBLIC_DEFAULT_RELEASE_NAME: DEFAULT_RELEASE_NAME,
  NEXT_PUBLIC_DEFAULT_MODELID: DEFAULT_MODELID,
  NEXT_PUBLIC_DEFAULT_SOURCESET: DEFAULT_SOURCESET,
  NEXT_PUBLIC_DEFAULT_SOURCE: DEFAULT_SOURCE,
  NEXT_PUBLIC_DEFAULT_STEER_MODEL: DEFAULT_STEER_MODEL,
  NEXT_PUBLIC_STEER_FORCE_ALLOW_INSTRUCT_MODELS: STEER_FORCE_ALLOW_INSTRUCT_MODELS,

  // Default User IDs
  DEFAULT_CREATOR_USER_ID,
  INFERENCE_ACTIVATION_USER_ID: INFERENCE_ACTIVATION_USER_ID_DO_NOT_INCLUDE_IN_PUBLIC_ACTIVATIONS,
  PUBLIC_ACTIVATIONS_USER_IDS,

  // Email Provider selection flags
  USE_AWS_SES,
  USE_RESEND,
  USE_SMTP,

  // AWS
  AWS_ACCESS_KEY_ID,
  AWS_SECRET_ACCESS_KEY,

  // Resend
  RESEND_EMAIL_API_KEY,

  // SMTP
  SMTP_SERVER_HOST,
  SMTP_SERVER_PORT,
  SMTP_SERVER_FROM,

  // AI API Keys
  OPENAI_API_KEY,
  GEMINI_API_KEY,
  ANTHROPIC_API_KEY,
  OPENROUTER_API_KEY,

  // Azure OpenAI
  AZURE_OPENAI_API_KEY,
  AZURE_OPENAI_ENDPOINT,
  AZURE_API_VERSION,
  OPENAI_DEPLOYMENT_NAME,

  // Provider selection flags
  USE_AZURE_OPENAI,
  USE_OPENROUTER,
  USE_OPENAI,

  // Support Servers - Inference
  USE_LOCALHOST_INFERENCE,
  INFERENCE_SERVER_SECRET,
  NEXT_PUBLIC_SEARCH_TOPK_MAX_CHAR_LENGTH,

  // Support Servers - Autointerp
  USE_LOCALHOST_AUTOINTERP,
  AUTOINTERP_SERVER,
  AUTOINTERP_SERVER_SECRET,

  // Support Servers - Graph
  USE_LOCALHOST_GRAPH,
  GRAPH_SERVER,
  GRAPH_SERVER_SECRET,
  USE_RUNPOD_GRAPH,
  GRAPH_RUNPOD_SECRET,
  GRAPH_RUNPOD_SERVER,

  // Authentication - Apple
  APPLE_CLIENT_ID,
  APPLE_CLIENT_SECRET,

  // Authentication - GitHub
  GITHUB_ID,
  GITHUB_SECRET,

  // Authentication - Google
  GOOGLE_CLIENT_ID,
  GOOGLE_CLIENT_SECRET,

  // Misc
  NODE_ENV,
  IS_DOCKER_COMPOSE,
  NEXT_PUBLIC_DEMO_MODE,
  GRAPH_ADMIN_BROWSE_KEY,
  HIGHER_LIMIT_API_TOKENS,
} = env;

// Computed values that depend on other variables
export const NEXT_PUBLIC_ENABLE_SIGNIN = NEXT_PUBLIC_ENABLE_SIGNIN_RAW && !IS_ONE_CLICK_VERCEL_DEPLOY;

// Derived values
export const IS_LOCALHOST = NEXT_PUBLIC_URL === 'http://localhost:3000';
export const IS_ACTUALLY_NEURONPEDIA_ORG =
  NEXT_PUBLIC_URL === 'https://neuronpedia.org' || NEXT_PUBLIC_URL === 'https://www.neuronpedia.org';
export const DEMO_MODE = NEXT_PUBLIC_DEMO_MODE || IS_ONE_CLICK_VERCEL_DEPLOY;

// Constants
export const ASSET_BASE_URL = 'https://neuronpedia.s3.us-east-1.amazonaws.com/site-assets';
export const API_KEY_HEADER_NAME = 'x-api-key';
