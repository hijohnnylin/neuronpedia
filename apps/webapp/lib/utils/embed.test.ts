import { describe, expect, it } from 'vitest';
import { getFullPageHref, isInteractiveEmbedPath } from './embed';

describe('isInteractiveEmbedPath', () => {
  it.each([
    '/qwen3.6-27b/jlens',
    '/gemma-2-9b-it/steer',
    '/gemma-2-2b/graph',
    '/graph',
    '/qwen3.6-27b/nla',
    '/llama3.3-70b-it/assistant-axis',
  ])('accepts %s', (pathname) => {
    expect(isInteractiveEmbedPath(pathname)).toBe(true);
  });

  it.each([
    '/',
    '/gemma-2-2b',
    '/gemma-2-2b/20-gemmascope-res-16k',
    '/gemma-2-2b/20-gemmascope-res-16k/123',
    '/a/b/steer',
  ])('rejects %s', (pathname) => {
    expect(isInteractiveEmbedPath(pathname)).toBe(false);
  });
});

describe('getFullPageHref', () => {
  it('removes only the embed parameter', () => {
    expect(
      getFullPageHref(
        '/qwen3.6-27b/jlens',
        new URLSearchParams('shareId=abc&embed=true&hideInitialSettingsOnMobile=true'),
      ),
    ).toBe('/qwen3.6-27b/jlens?shareId=abc&hideInitialSettingsOnMobile=true');
  });

  it('returns the bare path when no other parameters remain', () => {
    expect(getFullPageHref('/gemma-2-9b-it/steer', new URLSearchParams('embed=true'))).toBe('/gemma-2-9b-it/steer');
  });
});
