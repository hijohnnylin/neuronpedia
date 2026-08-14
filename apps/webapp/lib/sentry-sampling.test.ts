import { afterEach, describe, expect, it, vi } from 'vitest';

import { tracesSampler } from './sentry-sampling';

// Stands in for Sentry's context. `inheritOrSampleWith` returns the fallback by default, which is
// what it does for a request that is not continuing an incoming trace.
const context = (name: string, inherit?: (fallback: number) => number) => ({
  name,
  inheritOrSampleWith: inherit ?? ((fallback: number) => fallback),
});

const originalRate = process.env.SENTRY_TRACES_SAMPLE_RATE;

afterEach(() => {
  if (originalRate === undefined) {
    delete process.env.SENTRY_TRACES_SAMPLE_RATE;
  } else {
    process.env.SENTRY_TRACES_SAMPLE_RATE = originalRate;
  }
});

describe('tracesSampler', () => {
  it('defaults to 25%', () => {
    delete process.env.SENTRY_TRACES_SAMPLE_RATE;
    expect(tracesSampler(context('GET /'))).toBe(0.25);
  });

  it('drops the crawler-driven opengraph-image route', () => {
    expect(tracesSampler(context('GET /[modelId]/[layer]/[index]/opengraph-image'))).toBe(0);
  });

  it('drops health checks', () => {
    expect(tracesSampler(context('GET /api/health'))).toBe(0);
    expect(tracesSampler(context('GET /health'))).toBe(0);
  });

  it('samples ordinary routes', () => {
    expect(tracesSampler(context('GET /[modelId]/[layer]'))).toBe(0.25);
    expect(tracesSampler(context('POST /api/search-all'))).toBe(0.25);
  });

  it('defers to the incoming trace rather than re-deciding', () => {
    const inherit = vi.fn(() => 1);
    expect(tracesSampler(context('GET /[modelId]/[layer]', inherit))).toBe(1);
    expect(inherit).toHaveBeenCalledWith(0.25);
  });

  it('drops an excluded route without consulting the incoming trace', () => {
    const inherit = vi.fn(() => 1);
    expect(tracesSampler(context('GET /api/health', inherit))).toBe(0);
    expect(inherit).not.toHaveBeenCalled();
  });

  it('honours SENTRY_TRACES_SAMPLE_RATE', () => {
    process.env.SENTRY_TRACES_SAMPLE_RATE = '0.05';
    expect(tracesSampler(context('GET /'))).toBe(0.05);
  });

  it('falls back to the default when the env var is not a usable rate', () => {
    for (const value of ['not-a-number', '-1', '2', '', '   ']) {
      process.env.SENTRY_TRACES_SAMPLE_RATE = value;
      expect(tracesSampler(context('GET /'))).toBe(0.25);
    }
  });
});
