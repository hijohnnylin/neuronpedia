import { describe, expect, test } from 'vitest';
import type { LensTokenMessage } from '@/lib/utils/lens';
import {
  buildFieldSteps,
  cleanLabel,
  conceptsForToken,
  DEFAULT_FIELD_CONFIG,
  FieldConfig,
  FieldSim,
  hasAnyConcepts,
  isJunkToken,
  normalizeKey,
  rampColor,
} from '../jlens-field-sim';

// Build a single-lens token whose per-layer read-outs are the same rows on every
// layer, so tests can reason about the layer window and weighting directly.
function makeToken(
  token: string,
  rows: { tokens: string[]; probs: number[] }[],
  opts: { position?: number; isGenerated?: boolean; type?: 'JACOBIAN_LENS' | 'LOGIT_LENS' } = {},
): LensTokenMessage {
  return {
    kind: 'token',
    position: opts.position ?? 0,
    token,
    id: opts.position ?? 0,
    is_generated: opts.isGenerated ?? true,
    results: [
      {
        type: opts.type ?? 'JACOBIAN_LENS',
        top_tokens: rows.map((r) => r.tokens),
        top_probs: rows.map((r) => r.probs),
      },
    ],
  };
}

const cfg = (over: Partial<FieldConfig> = {}): FieldConfig => ({ ...DEFAULT_FIELD_CONFIG, ...over });

describe('isJunkToken', () => {
  test('filters whitespace, punctuation, specials, single ascii, replacement char', () => {
    expect(isJunkToken('   ')).toBe(true);
    expect(isJunkToken('...')).toBe(true);
    expect(isJunkToken('<|endoftext|>')).toBe(true);
    expect(isJunkToken('a')).toBe(true);
    expect(isJunkToken('�')).toBe(true);
  });

  test('keeps real words and CJK tokens', () => {
    expect(isJunkToken(' Rayleigh')).toBe(false);
    expect(isJunkToken('scattering')).toBe(false);
    expect(isJunkToken('色散')).toBe(false);
  });
});

describe('normalizeKey / cleanLabel', () => {
  test('strips tokenizer boundary markers and lowercases the key', () => {
    expect(normalizeKey('ĠRayleigh')).toBe('rayleigh');
    expect(normalizeKey('▁blue')).toBe('blue');
    expect(normalizeKey('  Blue ')).toBe('blue');
  });

  test('cleanLabel strips markers but keeps case', () => {
    expect(cleanLabel('ĠRayleigh')).toBe('Rayleigh');
    expect(cleanLabel('▁Blue')).toBe('Blue');
  });
});

describe('rampColor', () => {
  test('returns the ramp endpoints and clamps out-of-range input', () => {
    expect(rampColor(0)).toEqual([148, 163, 184]);
    expect(rampColor(1)).toEqual([245, 158, 11]);
    expect(rampColor(-1)).toEqual(rampColor(0));
    expect(rampColor(2)).toEqual(rampColor(1));
  });
});

describe('conceptsForToken', () => {
  test('reads only layers above the start fraction and weights later layers more', () => {
    // 4 layers; layerStartFraction 0.5 -> cut at layer 2, so layers 0/1 ignored.
    const token = makeToken(' output', [
      { tokens: [' noiseA'], probs: [0.9] }, // layer 0, ignored
      { tokens: [' noiseB'], probs: [0.9] }, // layer 1, ignored
      { tokens: [' Rayleigh'], probs: [0.4] }, // layer 2, ramp 0
      { tokens: [' Rayleigh'], probs: [0.5] }, // layer 3, ramp 1
    ]);
    const concepts = conceptsForToken(token, cfg({ layerStartFraction: 0.5 }));
    const keys = concepts.map((c) => c.key);
    expect(keys).toContain('rayleigh');
    expect(keys).not.toContain('noisea');
    expect(keys).not.toContain('noiseb');
    const rayleigh = concepts.find((c) => c.key === 'rayleigh')!;
    // Peak probability/layer is the strongest layer for the concept.
    expect(rayleigh.peakProb).toBeCloseTo(0.5, 6);
    expect(rayleigh.peakLayer).toBe(3);
  });

  test('drops junk, sub-threshold probs, and (optionally) the emitted token', () => {
    const token = makeToken(' blue', [
      { tokens: [' blue', ' ...', ' tiny'], probs: [0.5, 0.5, 0.001] },
      { tokens: [' blue', ' scattering', ' tiny'], probs: [0.5, 0.4, 0.001] },
    ]);
    const withEmitted = conceptsForToken(token, cfg({ layerStartFraction: 0, hideEmitted: false }));
    expect(withEmitted.map((c) => c.key)).toContain('blue');
    expect(withEmitted.map((c) => c.key)).not.toContain('...'); // punctuation filtered
    expect(withEmitted.map((c) => c.key)).not.toContain('tiny'); // below prob floor

    const hidden = conceptsForToken(token, cfg({ layerStartFraction: 0, hideEmitted: true }));
    expect(hidden.map((c) => c.key)).not.toContain('blue'); // emitted token hidden
    expect(hidden.map((c) => c.key)).toContain('scattering');
  });

  test('returns concepts sorted by descending weight', () => {
    const token = makeToken(' x', [
      { tokens: [' weak', ' strong'], probs: [0.05, 0.6] },
      { tokens: [' weak', ' strong'], probs: [0.05, 0.6] },
    ]);
    const concepts = conceptsForToken(token, cfg({ layerStartFraction: 0 }));
    expect(concepts[0].key).toBe('strong');
    for (let i = 1; i < concepts.length; i++) {
      expect(concepts[i - 1].weight).toBeGreaterThanOrEqual(concepts[i].weight);
    }
  });
});

describe('buildFieldSteps / hasAnyConcepts', () => {
  const rows = [
    { tokens: [' Rayleigh'], probs: [0.4] },
    { tokens: [' Rayleigh'], probs: [0.5] },
  ];

  test('produces one step per token and reports concept presence', () => {
    const tokens = [makeToken(' a', rows, { position: 0 }), makeToken(' b', rows, { position: 1 })];
    const steps = buildFieldSteps(tokens, cfg({ layerStartFraction: 0 }));
    expect(steps).toHaveLength(2);
    expect(hasAnyConcepts(steps)).toBe(true);
  });

  test('reports no concepts when everything is junk', () => {
    const junk = [makeToken(' a', [{ tokens: [' ...'], probs: [0.9] }], { position: 0 })];
    const steps = buildFieldSteps(junk, cfg({ layerStartFraction: 0 }));
    expect(hasAnyConcepts(steps)).toBe(false);
  });
});

describe('FieldSim', () => {
  const bounds = { x0: 0, y0: 0, x1: 400, y1: 300 };
  const rows = [
    { tokens: [' Rayleigh', ' scattering'], probs: [0.4, 0.3] },
    { tokens: [' Rayleigh', ' scattering'], probs: [0.5, 0.35] },
  ];
  const steps = buildFieldSteps(
    Array.from({ length: 6 }, (_, i) => makeToken(' t', rows, { position: i })),
    cfg({ layerStartFraction: 0 }),
  );

  test('is deterministic: two sims seeked to the same time match exactly', () => {
    const a = new FieldSim(steps, cfg(), bounds);
    const b = new FieldSim(steps, cfg(), bounds);
    a.seek(3);
    b.seek(3);
    const keysA = [...a.bubbles.keys()].sort();
    const keysB = [...b.bubbles.keys()].sort();
    expect(keysA).toEqual(keysB);
    for (const k of keysA) {
      expect(a.bubbles.get(k)!.x).toBeCloseTo(b.bubbles.get(k)!.x, 9);
      expect(a.bubbles.get(k)!.y).toBeCloseTo(b.bubbles.get(k)!.y, 9);
      expect(a.bubbles.get(k)!.energy).toBeCloseTo(b.bubbles.get(k)!.energy, 9);
    }
  });

  test('seeking backward replays from the start (same state as a fresh seek)', () => {
    const a = new FieldSim(steps, cfg(), bounds);
    a.seek(3);
    a.seek(1);
    const b = new FieldSim(steps, cfg(), bounds);
    b.seek(1);
    expect([...a.bubbles.keys()].sort()).toEqual([...b.bubbles.keys()].sort());
    for (const k of a.bubbles.keys()) {
      expect(a.bubbles.get(k)!.x).toBeCloseTo(b.bubbles.get(k)!.x, 9);
    }
  });

  test('surfaces concepts and keeps bubbles within bounds', () => {
    const sim = new FieldSim(steps, cfg(), bounds);
    sim.seek(2);
    expect(sim.bubbles.size).toBeGreaterThan(0);
    for (const b of sim.bubbles.values()) {
      expect(b.x).toBeGreaterThanOrEqual(bounds.x0 - 1);
      expect(b.x).toBeLessThanOrEqual(bounds.x1 + 1);
      expect(b.y).toBeGreaterThanOrEqual(bounds.y0 - 1);
      expect(b.y).toBeLessThanOrEqual(bounds.y1 + 1);
    }
  });

  test('has a positive duration and a clamped token counter', () => {
    const sim = new FieldSim(steps, cfg(), bounds);
    expect(sim.duration).toBeGreaterThan(0);
    expect(sim.stepCount).toBe(6);
    expect(sim.tokenIndexAt(-5)).toBe(0);
    expect(sim.tokenIndexAt(sim.duration + 100)).toBe(6);
  });
});
