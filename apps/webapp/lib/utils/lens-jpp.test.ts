import { afterEach, describe, expect, it, vi } from 'vitest';

// `JPP_LENS_MODEL_IDS` is read when `@/lib/env` loads, so each case loads the modules fresh.
async function load(models: string | undefined) {
  vi.resetModules();
  vi.stubEnv('NEXT_PUBLIC_JPP_LENS_MODEL_IDS', models);
  return import('./lens');
}

afterEach(() => {
  vi.unstubAllEnvs();
});

describe('the J++ Lens', () => {
  it('is requested only for the models whose servers have it', async () => {
    const { requestLensTypes } = await load(' qwen3.6-27b , qwen3.5-2b');
    expect(requestLensTypes('qwen3.6-27b')).toEqual(['JACOBIAN_LENS', 'JPP_LENS', 'LOGIT_LENS']);
    expect(requestLensTypes('gemma-3-4b')).toEqual(['JACOBIAN_LENS', 'LOGIT_LENS']);
  });

  it('is requested for no model when the list is unset', async () => {
    const { requestLensTypes } = await load(undefined);
    expect(requestLensTypes('qwen3.6-27b')).toEqual(['JACOBIAN_LENS', 'LOGIT_LENS']);
  });
});
