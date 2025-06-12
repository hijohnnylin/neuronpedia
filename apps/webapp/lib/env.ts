import { config } from 'dotenv';

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

// Domain of your main site
export const NEXT_PUBLIC_URL = process.env.NEXT_PUBLIC_URL || '';

// Auth will redirect to this domain
export const NEXTAUTH_URL = process.env.NEXTAUTH_URL || '';

// Secret for hashing auth tokens (Used by NextAuth, not our code directly)
// export const NEXTAUTH_SECRET = process.env.NEXTAUTH_SECRET || '';

// Database (Used by Prisma, not our code directly)
// export const POSTGRES_PRISMA_URL = process.env.POSTGRES_PRISMA_URL || '';
// export const POSTGRES_URL_NON_POOLING = process.env.POSTGRES_URL_NON_POOLING || '';

// Feature Flags
export const ENABLE_RATE_LIMITER = process.env.ENABLE_RATE_LIMITER === 'true';
export const ENABLE_VERCEL_ANALYTICS = process.env.ENABLE_VERCEL_ANALYTICS === 'true';
export const NEXT_PUBLIC_ENABLE_SIGNIN =
  process.env.NEXT_PUBLIC_ENABLE_SIGNIN === 'true' && !IS_ONE_CLICK_VERCEL_DEPLOY;

// Default Values
export const NEURONPEDIA_EMAIL_ADDRESS = 'johnny@neuronpedia.org';
export const CONTACT_EMAIL_ADDRESS = process.env.NEXT_PUBLIC_CONTACT_EMAIL_ADDRESS || NEURONPEDIA_EMAIL_ADDRESS;
export const DEFAULT_RELEASE_NAME = process.env.NEXT_PUBLIC_DEFAULT_RELEASE_NAME || '';
export const DEFAULT_MODELID = process.env.NEXT_PUBLIC_DEFAULT_MODELID || '';
export const DEFAULT_SOURCESET = process.env.NEXT_PUBLIC_DEFAULT_SOURCESET || '';
export const DEFAULT_SOURCE = process.env.NEXT_PUBLIC_DEFAULT_SOURCE || '';
export const DEFAULT_STEER_MODEL = process.env.NEXT_PUBLIC_DEFAULT_STEER_MODEL || '';
export const STEER_FORCE_ALLOW_INSTRUCT_MODELS =
  process.env.NEXT_PUBLIC_STEER_FORCE_ALLOW_INSTRUCT_MODELS?.split(',').map((m) => m.trim()) || [];

// Default User IDs
// The fallback values are users in the seeded database.
export const DEFAULT_CREATOR_USER_ID = process.env.DEFAULT_CREATOR_USER_ID || 'clkht01d40000jv08hvalcvly';
export const INFERENCE_ACTIVATION_USER_ID_DO_NOT_INCLUDE_IN_PUBLIC_ACTIVATIONS =
  process.env.INFERENCE_ACTIVATION_USER_ID || 'cljgamm90000076zdchicy6zj';
export const PUBLIC_ACTIVATIONS_USER_IDS = process.env.PUBLIC_ACTIVATIONS_USER_IDS
  ? process.env.PUBLIC_ACTIVATIONS_USER_IDS.split(',').map((id) => id.trim())
  : ['cljj57d3c000076ei38vwnv35', 'clkht01d40000jv08hvalcvly'];

// Email
// For email sending providers, choose EITHER AWS SES or Resend.com.
// Resend is easier to set up, but AWS is more reliable.
// If both are defined, AWS will be used.
// AWS
export const AWS_ACCESS_KEY_ID = process.env.AWS_ACCESS_KEY_ID || '';
export const AWS_SECRET_ACCESS_KEY = process.env.AWS_SECRET_ACCESS_KEY || '';
// Resend
export const RESEND_EMAIL_API_KEY = process.env.RESEND_EMAIL_API_KEY || '';

// External Services
// AI API Keys (Mostly for auto-interp for whitelisted accounts)
export const OPENAI_API_KEY = process.env.OPENAI_API_KEY || '';
// if (!process.env.OPENAI_API_KEY) {
//   console.warn(
//     'OPENAI_API_KEY is not set. Search Explanations will not work. Set the key in the file neuronpedia/apps/webapp/.env',
//   );
// }
export const GEMINI_API_KEY = process.env.GEMINI_API_KEY || '';
export const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY || '';
export const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY || '';

// Sentry (Crash Reporting - Used by Sentry, not by us directly)
// export const SENTRY_DSN = process.env.SENTRY_DSN || '';
// export const SENTRY_AUTH_TOKEN = process.env.SENTRY_AUTH_TOKEN || '';

// Rate Limiting - Redis (Used by Upstash, not by us directly)
// export const KV_URL = process.env.KV_URL || '';
// export const KV_REST_API_URL = process.env.KV_REST_API_URL || '';
// export const KV_REST_API_TOKEN = process.env.KV_REST_API_TOKEN || '';
// export const KV_REST_API_READ_ONLY_TOKEN = process.env.KV_REST_API_READ_ONLY_TOKEN || '';

// Support Servers
// Inference Server
export const USE_LOCALHOST_INFERENCE = process.env.USE_LOCALHOST_INFERENCE === 'true';
export const INFERENCE_SERVER_SECRET = process.env.INFERENCE_SERVER_SECRET || '';

export const NEXT_PUBLIC_SEARCH_TOPK_MAX_CHAR_LENGTH = process.env.NEXT_PUBLIC_SEARCH_TOPK_MAX_CHAR_LENGTH
  ? parseInt(process.env.NEXT_PUBLIC_SEARCH_TOPK_MAX_CHAR_LENGTH, 10)
  : 1024;

// Autointerp Server
export const USE_LOCALHOST_AUTOINTERP = process.env.USE_LOCALHOST_AUTOINTERP === 'true';
export const AUTOINTERP_SERVER = process.env.AUTOINTERP_SERVER || '';
export const AUTOINTERP_SERVER_SECRET = process.env.AUTOINTERP_SERVER_SECRET || '';

// Graph Server
export const USE_LOCALHOST_GRAPH = process.env.USE_LOCALHOST_GRAPH === 'true';
export const GRAPH_SERVER = process.env.GRAPH_SERVER || '';
export const GRAPH_SERVER_SECRET = process.env.GRAPH_SERVER_SECRET || '';
export const GRAPH_RUNPOD_SECRET = process.env.GRAPH_RUNPOD_SECRET || '';
export const GRAPH_RUNPOD_SERVER = process.env.GRAPH_RUNPOD_SERVER || '';

// Deprecated - Scorer Server
export const OLD_SCORER_SERVER = process.env.OLD_SCORER_SERVER || '';

// Authentication Methods
// Apple
export const APPLE_CLIENT_ID = process.env.APPLE_CLIENT_ID || '';
export const APPLE_CLIENT_SECRET = process.env.APPLE_CLIENT_SECRET || '';
// GitHub
export const GITHUB_ID = process.env.GITHUB_ID || '';
export const GITHUB_SECRET = process.env.GITHUB_SECRET || '';
// Google
export const GOOGLE_CLIENT_ID = process.env.GOOGLE_CLIENT_ID || '';
export const GOOGLE_CLIENT_SECRET = process.env.GOOGLE_CLIENT_SECRET || '';

// Authentication Refresh - for updating the Apple Client Secret every 6 months (using ./node scripts/apple-gen-secret.js and .secret.apple.p8)
// export const APPLE_KEY_ID = process.env.APPLE_KEY_ID || '';
// export const APPLE_TEAM_ID = process.env.APPLE_TEAM_ID || '';

export const IS_LOCALHOST = process.env.NEXT_PUBLIC_URL === 'http://localhost:3000';
export const IS_ACTUALLY_NEURONPEDIA_ORG =
  process.env.NEXT_PUBLIC_URL === 'https://neuronpedia.org' ||
  process.env.NEXT_PUBLIC_URL === 'https://www.neuronpedia.org';

// Misc
export const NODE_ENV = process.env.NODE_ENV || '';
export const IS_DOCKER_COMPOSE = process.env.IS_DOCKER_COMPOSE === 'true';
export const DEMO_MODE = process.env.NEXT_PUBLIC_DEMO_MODE === 'true' || IS_ONE_CLICK_VERCEL_DEPLOY;
export const ASSET_BASE_URL = 'https://neuronpedia.s3.us-east-1.amazonaws.com/site-assets';
export const GRAPH_ADMIN_BROWSE_KEY = process.env.GRAPH_ADMIN_BROWSE_KEY || '';

export const API_KEY_HEADER_NAME = 'x-api-key';
export const HIGHER_LIMIT_API_TOKENS = process.env.HIGHER_LIMIT_API_TOKENS
  ? process.env.HIGHER_LIMIT_API_TOKENS.split(',').map((t) => t.trim())
  : [];
