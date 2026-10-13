import { describe, expect, it } from 'vitest';
import { getLegacyModelIdRedirect } from './legacy-model-id';

function redirectFor(url: string) {
  const parsed = new URL(url, 'https://www.neuronpedia.org');
  return getLegacyModelIdRedirect(parsed.pathname, parsed.searchParams);
}

describe('getLegacyModelIdRedirect', () => {
  it('ignores model IDs that are not legacy', () => {
    expect(redirectFor('/llama3.1-8b-it/steer')).toBeNull();
    expect(redirectFor('/gemma-2-2b/20-gemmascope-res-16k/123')).toBeNull();
    expect(redirectFor('/')).toBeNull();
    expect(redirectFor('/api/steer')).toBeNull();
  });

  it('ignores object prototype keys', () => {
    expect(redirectFor('/constructor/steer')).toBeNull();
    expect(redirectFor('/__proto__')).toBeNull();
  });

  it('replaces the model ID and keeps the path and query', () => {
    expect(redirectFor('/llama31-8b-it')).toBe('/llama3.1-8b-it');
    expect(redirectFor('/qwen25-7b-it/steer?saved=abc&strength=5')).toBe('/qwen2.5-7b-it/steer?saved=abc&strength=5');
    expect(redirectFor('/llama31-8b-it/15-llamascope-res-32k?embed=true')).toBe(
      '/llama3.1-8b-it/15-llamascope-res-32k?embed=true',
    );
    expect(redirectFor('/llama31-8b-it/15-llamascope-res-32k/123?embed=true&embedtest=false')).toBe(
      '/llama3.1-8b-it/15-llamascope-res-32k/123?embed=true&embedtest=false',
    );
  });

  it('sends a feature with redirectToSteer to the steer embed', () => {
    expect(
      redirectFor('/llama31-8b-it/15-llamascope-res-32k/123?embed=true&redirectToSteer=true&strength=4&saved=abc'),
    ).toBe(
      '/llama3.1-8b-it/steer?embed=true&source=15-llamascope-res-32k&index=123&hideInitialSettingsOnMobile=true&strength=4&saved=abc',
    );
  });

  it('uses the default strength and no saved ID when they are absent', () => {
    expect(redirectFor('/qwen25-7b-it/20-res/7?redirectToSteer=true')).toBe(
      '/qwen2.5-7b-it/steer?embed=true&source=20-res&index=7&hideInitialSettingsOnMobile=true&strength=10',
    );
  });

  it('ignores redirectToSteer on pages that are not features', () => {
    expect(redirectFor('/llama31-8b-it/steer?redirectToSteer=true')).toBe('/llama3.1-8b-it/steer?redirectToSteer=true');
  });
});
