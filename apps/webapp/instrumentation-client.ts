// This file configures the initialization of Sentry on the client.
// The config you add here will be used whenever a users loads a page in their browser.
// https://docs.sentry.io/platforms/javascript/guides/nextjs/

import * as Sentry from '@sentry/nextjs';

export const onRouterTransitionStart = Sentry.captureRouterTransitionStart;

// Must be NEXT_PUBLIC_: this file is bundled for the browser, and Next only inlines env vars
// carrying that prefix. Reading the bare `SENTRY_DSN` here leaves `dsn` undefined in every
// browser, which is why nothing client-side has ever reached Sentry. A DSN is a public
// identifier, not a secret, so exposing it in the bundle is how it is meant to be shipped.
Sentry.init({
  dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
  enabled: !!process.env.NEXT_PUBLIC_SENTRY_DSN,

  // Add optional integrations for additional features
  // integrations: [Sentry.replayIntegration()],

  // Matches the 25% the server and edge runtimes sample at (see lib/sentry-sampling.ts). A flat
  // rate rather than that sampler because its route exclusions are server-side routes, and
  // because the env var it reads is not exposed to the browser.
  tracesSampleRate: 0.25,

  // // Define how likely Replay events are sampled.
  // // This sets the sample rate to be 10%. You may want this to be 100% while
  // // in development and sample at a lower rate in production
  // replaysSessionSampleRate: 0.1,

  // // Define how likely Replay events are sampled when an error occurs.
  // replaysOnErrorSampleRate: 1.0,

  // Setting this option to true will print useful information to the console while you're setting up Sentry.
  debug: false,
});
