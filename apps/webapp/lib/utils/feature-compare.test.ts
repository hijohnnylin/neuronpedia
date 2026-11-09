import { describe, expect, it } from 'vitest';
import { getCompareLinks, getComparePath, getCompareValue, parseCompareValue } from './feature-compare';

const LEFT = { modelId: 'gemma-2-2b', layer: '20-gemmascope-res-16k', index: '1234' };
const RIGHT = { modelId: 'gpt2-small', layer: '8-res-jb', index: '7' };

describe('parseCompareValue', () => {
  it('reads model, layer and index', () => {
    expect(parseCompareValue('gpt2-small/8-res-jb/7')).toEqual(RIGHT);
  });

  it('round-trips getCompareValue', () => {
    expect(parseCompareValue(getCompareValue(LEFT))).toEqual(LEFT);
  });

  it.each([undefined, '', 'gpt2-small', 'gpt2-small/8-res-jb', 'a/b/c/d', 'gpt2-small//7', ' /8-res-jb/7'])(
    'rejects %j',
    (value) => {
      expect(parseCompareValue(value)).toBeNull();
    },
  );

  it('rejects a repeated param', () => {
    expect(parseCompareValue(['gpt2-small/8-res-jb/7', 'gpt2-small/8-res-jb/8'])).toBeNull();
  });
});

describe('getComparePath', () => {
  it('keeps the left feature in the path', () => {
    expect(getComparePath(LEFT, RIGHT)).toBe('/gemma-2-2b/20-gemmascope-res-16k/1234?compare=gpt2-small/8-res-jb/7');
  });

  it('encodes each segment of the compare value', () => {
    expect(getCompareValue({ modelId: 'm', layer: 'a?b', index: '1&x=2' })).toBe('m/a%3Fb/1%26x%3D2');
  });
});

describe('getCompareLinks', () => {
  const links = getCompareLinks(LEFT, RIGHT);

  it('changes only the left feature from the left column', () => {
    expect(links.left.hrefFor('gemma-2-2b', '20-gemmascope-res-16k', 5)).toBe(
      '/gemma-2-2b/20-gemmascope-res-16k/5?compare=gpt2-small/8-res-jb/7',
    );
  });

  it('changes only the right feature from the right column', () => {
    expect(links.right.hrefFor('gpt2-small', '8-res-jb', 8)).toBe(
      '/gemma-2-2b/20-gemmascope-res-16k/1234?compare=gpt2-small/8-res-jb/8',
    );
  });

  it('closing a column leaves the other feature on its own', () => {
    expect(links.left.closeHref).toBe('/gpt2-small/8-res-jb/7');
    expect(links.right.closeHref).toBe('/gemma-2-2b/20-gemmascope-res-16k/1234');
  });
});
