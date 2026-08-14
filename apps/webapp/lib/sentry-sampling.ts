/**
 * How much of the traffic gets a performance trace, shared by the server and edge runtimes.
 *
 * Tracing used to sample everything, which on this app is mostly waste: the routes below carry
 * the bulk of the volume and none of the insight. `opengraph-image` is fetched by crawlers and
 * link unfurlers rather than users, and it produced ~30k "failed to pipe response" errors in a
 * quarter from clients hanging up mid-stream. Health checks are polled on a fixed interval, so
 * they say nothing about how the app is performing under real traffic.
 *
 * Everything else is sampled at SENTRY_TRACES_SAMPLE_RATE, defaulting to 25%.
 */

// Structural rather than imported: `@sentry/nextjs` does not re-export the sampling context type,
// and `@sentry/core`, which declares it, is a transitive dependency we should not import from
// directly. Declaring only the fields used here keeps the real context assignable to it.
type SamplingContext = {
  /** The name of the span being sampled, e.g. `GET /[modelId]/[layer]`. */
  name: string;
  /**
   * Sentry's own helper: returns the rate the incoming trace was sampled at when this span
   * continues one, and the fallback otherwise. Using it rather than branching on `parentSampled`
   * is what keeps a distributed trace from arriving with holes that look like missing
   * instrumentation.
   */
  inheritOrSampleWith: (fallbackSampleRate: number) => number;
};

const NEVER_SAMPLED = [/\/opengraph-image$/, /^GET \/api\/health/, /^GET \/health/];

const DEFAULT_RATE = 0.25;

function configuredRate(): number {
  // Checked for emptiness first: `Number('')` is 0, so an env var that is set but blank would
  // otherwise read as a valid rate of zero and turn tracing off altogether.
  const raw = process.env.SENTRY_TRACES_SAMPLE_RATE?.trim();
  if (!raw) {
    return DEFAULT_RATE;
  }
  const parsed = Number(raw);
  return Number.isFinite(parsed) && parsed >= 0 && parsed <= 1 ? parsed : DEFAULT_RATE;
}

export function tracesSampler(samplingContext: SamplingContext): number {
  // Ahead of the inherit call on purpose: these are entry points hit directly by crawlers and
  // monitors, never a continuation of a trace we started, so there is nothing to keep whole.
  if (NEVER_SAMPLED.some((pattern) => pattern.test(samplingContext.name))) {
    return 0;
  }

  return samplingContext.inheritOrSampleWith(configuredRate());
}
