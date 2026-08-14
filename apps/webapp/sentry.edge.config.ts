// This file configures the initialization of Sentry for edge features (middleware, edge routes, and so on).
// The config you add here will be used whenever one of the edge features is loaded.
// Note that this config is unrelated to the Vercel Edge Runtime and is also required when running locally.
// https://docs.sentry.io/platforms/javascript/guides/nextjs/

import { tracesSampler } from '@/lib/sentry-sampling';
import * as Sentry from '@sentry/nextjs';

Sentry.init({
  dsn: process.env.SENTRY_DSN,
  // `!== undefined` treats an env var that is present but empty as configured, which starts the
  // SDK with no DSN instead of leaving it off.
  enabled: !!process.env.SENTRY_DSN,

  tracesSampler,

  // Setting this option to true will print useful information to the console while you're setting up Sentry.
  debug: false,
});
