// This file configures the initialization of Sentry on the server.
// The config you add here will be used whenever the server handles a request.
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
