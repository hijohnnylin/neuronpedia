import { describe, expect, it } from 'vitest';
import { parseFixture } from './jlens-export';

const BASE = { version: 1, kind: 'chat', modelId: 'm', exportedAt: '', meta: null, tokens: [], messages: [] };

describe('parseFixture oracle field', () => {
  it('reads a file with no oracle field (every share before the oracle)', () => {
    expect(parseFixture(BASE).oracle).toBeUndefined();
  });

  it('keeps a good oracle field', () => {
    const oracle = { maxBullets: 2, reads: [] };
    expect(parseFixture({ ...BASE, oracle }).oracle).toEqual(oracle);
  });

  it('drops a bad oracle field and keeps the run', () => {
    const parsed = parseFixture({ ...BASE, oracle: { reads: 'x' } });
    expect(parsed.oracle).toBeUndefined();
    expect(parsed.kind).toBe('chat');
  });
});

describe('parseFixture layer range field', () => {
  it('reads a file with no layer range', () => {
    expect(parseFixture(BASE).layerRange).toBeUndefined();
  });

  it('keeps a good layer range', () => {
    expect(parseFixture({ ...BASE, layerRange: [0, 21] }).layerRange).toEqual([0, 21]);
  });

  it('drops a bad layer range and keeps the run', () => {
    for (const layerRange of [[21, 0], [0], [-1, 5], [0.5, 3], 'x']) {
      const parsed = parseFixture({ ...BASE, layerRange });
      expect(parsed.layerRange).toBeUndefined();
      expect(parsed.kind).toBe('chat');
    }
  });
});
