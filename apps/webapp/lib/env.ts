import { createEnv } from '@t3-oss/env-nextjs';
import { config } from 'dotenv';
import { z } from 'zod';

// If it's a one-click deploy on Vercel, we always use the demo environment variables.
if (process.env.NEXT_PUBLIC_SITE_NAME_VERCEL_DEPLOY) {
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

    // AWS SES
    AWS_ACCESS_KEY_ID: z.string().default(''),
    AWS_SECRET_ACCESS_KEY: z.string().default(''),
    // Resend.com
    RESEND_EMAIL_API_KEY: z.string().default(''),

    // AI API Keys
    OPENAI_API_KEY: z.string().default(''),
    GEMINI_API_KEY: z.string().default(''),
    ANTHROPIC_API_KEY: z.string().default(''),
    OPENROUTER_API_KEY: z.string().default(''),

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
    GRAPH_SERVER_QWEN3_4B: z.string().default(''),
    GRAPH_SERVER_SECRET: z.string().default(''),
    USE_RUNPOD_GRAPH: onlyBool.default('false'),
    GRAPH_RUNPOD_SECRET: z.string().default(''),
    GRAPH_RUNPOD_SERVER: z.string().default(''),
    GRAPH_RUNPOD_SERVER_QWEN3_4B: z.string().default(''),

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

    // Computed/derived values
    IS_ONE_CLICK_VERCEL_DEPLOY: z
      .string()
      .optional()
      .transform(() => process.env.NEXT_PUBLIC_SITE_NAME_VERCEL_DEPLOY !== undefined),
    IS_LOCALHOST: z
      .string()
      .default('')
      .transform(() => process.env.NEXT_PUBLIC_URL === 'http://localhost:3000'),
    IS_ACTUALLY_NEURONPEDIA_ORG: z
      .string()
      .default('')
      .transform(
        () =>
          process.env.NEXT_PUBLIC_URL === 'https://neuronpedia.org' ||
          process.env.NEXT_PUBLIC_URL === 'https://www.neuronpedia.org',
      ),
    DEMO_MODE: z
      .string()
      .default('')
      .transform(
        () =>
          process.env.NEXT_PUBLIC_DEMO_MODE === 'true' || process.env.NEXT_PUBLIC_SITE_NAME_VERCEL_DEPLOY !== undefined,
      ),
  },
  client: {
    // All NEXT_PUBLIC_ prefixed variables go here
    NEXT_PUBLIC_URL: z.string().default(''),
    NEXT_PUBLIC_ENABLE_SIGNIN: onlyBool
      .default('false')
      .transform((v) => v && !process.env.NEXT_PUBLIC_SITE_NAME_VERCEL_DEPLOY),

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
    NEXT_PUBLIC_SITE_NAME_VERCEL_DEPLOY: z.string().optional(),
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
    NEXT_PUBLIC_SITE_NAME_VERCEL_DEPLOY: process.env.NEXT_PUBLIC_SITE_NAME_VERCEL_DEPLOY,
  },
  skipValidation: !!process.env.SKIP_ENV_VALIDATION,
  onValidationError: (error: any) => {
    console.error('Environment validation failed:', error);
    throw error;
  },
  onInvalidAccess: ((variable: string) => {
    // We don't need onInvalidAccess because Next.js already handles the security by setting
    // server-only variables to undefined on the client. Since the code is open source,
    // clients knowing the variable names is not a security issue - they just can't access
    // the values. The Zod default values ensure the app doesn't crash when these are undefined.
    console.warn(`⚠️ Attempted to access server-side environment variable '${variable}' on the client.`);
  }) as (variable: string) => never,
});

/** ********* POST PARSING VALIDATION LOGIC ********* */
const IS_SERVER = typeof window === 'undefined';
if (IS_SERVER) {
  // Provider validation
  const trueCount = [env.OPENAI_API_KEY, env.OPENROUTER_API_KEY].filter(Boolean).length;
  if (trueCount === 0) {
    throw new Error('At least one OpenAI provider must be enabled');
  }

  // Graph server validation
  if (env.USE_RUNPOD_GRAPH && env.USE_LOCALHOST_GRAPH) {
    throw new Error('USE_LOCALHOST_GRAPH and USE_RUNPOD_GRAPH cannot both be true');
  }
}
