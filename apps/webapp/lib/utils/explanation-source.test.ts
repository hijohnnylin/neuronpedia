import { describe, expect, it } from 'vitest';
import {
  externalFirst,
  formatMetadataKey,
  getExplanationBadge,
  hasMetadata,
  orderMetadataEntries,
  safeHttpUrl,
} from './explanation-source';

describe('getExplanationBadge', () => {
  it('shows nothing for a positive activation explanation', () => {
    expect(getExplanationBadge({ typeName: 'np_max-act-logits' })).toBeNull();
    expect(getExplanationBadge({ pole: 'POSITIVE', aspect: 'ACTIVATION' })).toBeNull();
  });

  it('marks negative explanations, by pole or by the old negative type', () => {
    expect(getExplanationBadge({ pole: 'NEGATIVE' })?.label).toBe('Neg Acts');
    expect(getExplanationBadge({ typeName: 'np_min-act-logits' })?.label).toBe('Neg Acts');
  });

  it('marks labels that cover both signs', () => {
    expect(getExplanationBadge({ pole: 'BOTH', aspect: 'ACTIVATION' })?.label).toBe('Both Signs');
  });

  it('shows the signed strength for steering', () => {
    expect(getExplanationBadge({ pole: 'NEGATIVE', aspect: 'STEERING', steerStrength: -0.8 })?.label).toBe(
      'Steer \u22120.8',
    );
    expect(getExplanationBadge({ pole: 'POSITIVE', aspect: 'STEERING', steerStrength: 1.25 })?.label).toBe(
      'Steer +1.25',
    );
  });

  it('shows an arrow for steering without a strength', () => {
    expect(getExplanationBadge({ pole: 'NEGATIVE', aspect: 'STEERING' })?.label).toBe('Steer \u2193');
    expect(getExplanationBadge({ aspect: 'STEERING' })?.label).toBe('Steer \u2191');
  });
});

describe('formatMetadataKey', () => {
  it('turns camelCase and snake_case keys into words', () => {
    expect(formatMetadataKey('knockoutConcepts')).toBe('Knockout concepts');
    expect(formatMetadataKey('base_output')).toBe('Base output');
    expect(formatMetadataKey('title')).toBe('Title');
  });
});

describe('orderMetadataEntries', () => {
  it('puts the description first and long text and lists of objects last', () => {
    const keys = orderMetadataEntries({
      tags: ['a'],
      reply: 'x'.repeat(200),
      examples: [{ prompt: 'p' }],
      category: 'persona',
      description: 'short',
    }).map(([k]) => k);
    expect(keys).toEqual(['description', 'tags', 'category', 'reply', 'examples']);
  });
});

describe('externalFirst', () => {
  it('puts external explanations first and keeps the order within each group', () => {
    const ids = externalFirst([
      { id: 'np1' },
      { id: 'ext1', sourceUrl: 'https://a' },
      { id: 'np2', sourceUrl: null },
      { id: 'ext2', sourceUrl: 'https://b' },
    ]).map((e) => e.id);
    expect(ids).toEqual(['ext1', 'ext2', 'np1', 'np2']);
  });

  it('sorts external explanations by type name and keeps Neuronpedia ones in order at the end', () => {
    const ids = externalFirst([
      { id: 'np1', typeName: 'np_max-act-logits' },
      { id: 'persona1', typeName: 'neuron-personas', sourceUrl: 'https://a' },
      { id: 'lens1', typeName: 'neuron-lens', sourceUrl: 'https://b' },
      { id: 'np2', typeName: 'claude-autointerp' },
      { id: 'mapper1', typeName: 'modelmapper', sourceUrl: 'https://c' },
      { id: 'lens2', typeName: 'neuron-lens', sourceUrl: 'https://d' },
    ]).map((e) => e.id);
    expect(ids).toEqual(['mapper1', 'lens1', 'lens2', 'persona1', 'np1', 'np2']);
  });
});

describe('safeHttpUrl', () => {
  it('keeps http and https links and drops other schemes', () => {
    expect(safeHttpUrl('https://neuronary.org/r/abc')).toBe('https://neuronary.org/r/abc');
    expect(safeHttpUrl('javascript:alert(1)')).toBeUndefined();
    expect(safeHttpUrl(null)).toBeUndefined();
  });
});

describe('hasMetadata', () => {
  it('is false for empty or non-object values', () => {
    expect(hasMetadata(null)).toBe(false);
    expect(hasMetadata([])).toBe(false);
    expect(hasMetadata({})).toBe(false);
    expect(hasMetadata({ note: '' })).toBe(false);
  });

  it('is true when one value is set', () => {
    expect(hasMetadata({ title: 'Wine Bottle' })).toBe(true);
  });
});
