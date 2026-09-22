import { describe, expect, it } from 'vitest';
import {
  ACTIVATING_PER_GROUP,
  buildGroup,
  buildGroups,
  buildQuestions,
  cropAround,
  firingIndices,
  GROUP_SIZE,
  INTRUDER_GROUPS,
  INTRUDER_MAX_TOKENS,
  markTokens,
  MAX_MARKS,
} from './interpretability-jev';

const tokens = (n: number, prefix = 't') => Array.from({ length: n }, (_, i) => ` ${prefix}${i}`);
const withPeak = (n: number, peak: number, prefix = 't') => ({
  tokens: tokens(n, prefix),
  values: Array.from({ length: n }, (_, i) => (i === peak ? 8 : i === peak + 1 ? 3 : 0)),
});
const zeros = (n: number, prefix = 'z') => ({ tokens: tokens(n, prefix), values: Array(n).fill(0) });

describe('cropAround', () => {
  it('centers the anchor when there is room on both sides', () => {
    const c = cropAround(tokens(20), Array(20).fill(0), 10, 6);
    expect(c.tokens).toEqual(tokens(20).slice(7, 13));
    expect(c.anchor).toBe(3);
  });

  it('clamps to the start and end', () => {
    expect(cropAround(tokens(10), Array(10).fill(0), 1, 6).tokens).toEqual(tokens(10).slice(0, 6));
    expect(cropAround(tokens(10), Array(10).fill(0), 9, 6).tokens).toEqual(tokens(10).slice(4, 10));
    expect(cropAround(tokens(10), Array(10).fill(0), 9, 6).anchor).toBe(5);
  });

  it('never asks for more tokens than exist', () => {
    expect(cropAround(tokens(3), [0, 0, 0], 1, 10).tokens).toHaveLength(3);
  });
});

describe('firingIndices and markTokens', () => {
  it('marks tokens at or above 3/10 of the max', () => {
    expect([...firingIndices([0, 10, 3, 2.9])].sort()).toEqual([1, 2]);
    expect(firingIndices([0, 0]).size).toBe(0);
  });

  it('keeps only the MAX_MARKS strongest tokens on dense texts', () => {
    const dense = Array.from({ length: 30 }, (_, i) => 5 + (i % 5));
    const marks = firingIndices(dense);
    expect(marks.size).toBe(MAX_MARKS);
    // Six 9s and two 8s: the strongest values, none of the 5s, 6s or 7s.
    [...marks].forEach((i) => expect(dense[i]).toBeGreaterThanOrEqual(8));
  });

  it('wraps marked tokens and flattens newlines', () => {
    expect(markTokens(['Gold', 'Ċ', ' fell'], new Set([2]))).toBe('Gold <<fell>>'.replace('<<fell>>', '<< fell>>'));
  });
});

describe('buildGroup', () => {
  const rng = () => 0.5;

  it('crops every member to the shortest length, capped at INTRUDER_MAX_TOKENS', () => {
    const activating = [withPeak(50, 25), withPeak(40, 5), withPeak(60, 59), withPeak(45, 20)];
    const group = buildGroup(activating, zeros(12), 'zero', rng);
    expect(group.examples).toHaveLength(GROUP_SIZE);
    group.examples.forEach((e) => expect(e.tokens).toHaveLength(12));

    const long = buildGroup(activating, zeros(100), 'zero', rng);
    long.examples.forEach((e) => expect(e.tokens).toHaveLength(INTRUDER_MAX_TOKENS));
  });

  it('keeps the peak token inside the window and marks it', () => {
    const activating = [withPeak(50, 25), withPeak(40, 5), withPeak(60, 59), withPeak(45, 20)];
    const group = buildGroup(activating, zeros(100), 'zero', rng);
    group.examples
      .filter((e) => !e.is_intruder)
      .forEach((e) => {
        expect(Math.max(...e.values)).toBe(8);
        expect(e.marked_text).toContain('<<');
      });
  });

  it('gives the intruder about as many marks as the activating texts, at least one', () => {
    const activating = [withPeak(30, 10), withPeak(30, 10), withPeak(30, 10), withPeak(30, 10)];
    const group = buildGroup(activating, zeros(30), 'zero', rng);
    const intruder = group.examples[group.intruder_index];
    expect(intruder.is_intruder).toBe(true);
    expect(intruder.kind).toBe('zero');
    // Each activating text marks the peak and the 3/10 neighbour, so the intruder gets two.
    expect(intruder.marked_text.match(/<</g)).toHaveLength(2);
    expect(Math.max(...intruder.values)).toBe(0);
  });
});

describe('buildGroups', () => {
  const top = Array.from({ length: 7 }, (_, i) => withPeak(30, 10 + i, `a${i}`));

  it('builds INTRUDER_GROUPS groups and reuses activating texts when there are too few', () => {
    const groups = buildGroups(top, [zeros(30)], 'seed');
    expect(groups).toHaveLength(INTRUDER_GROUPS);
    groups.forEach((g) => {
      expect(g.examples).toHaveLength(GROUP_SIZE);
      expect(g.examples.filter((e) => !e.is_intruder)).toHaveLength(ACTIVATING_PER_GROUP);
      expect(g.examples[g.intruder_index].is_intruder).toBe(true);
    });
    // The first group has four distinct activating texts even though the deck must recycle later.
    const firstTexts = new Set(groups[0].examples.filter((e) => !e.is_intruder).map((e) => e.tokens.join('')));
    expect(firstTexts.size).toBe(ACTIVATING_PER_GROUP);
  });

  it('is deterministic for a seed and varies across seeds', () => {
    const a = buildGroups(top, [zeros(30)], 'gpt2/0/1');
    const b = buildGroups(top, [zeros(30)], 'gpt2/0/1');
    const c = buildGroups(top, [zeros(30)], 'gpt2/0/2');
    expect(a).toEqual(b);
    expect(a.map((g) => g.intruder_index)).not.toEqual(c.map((g) => g.intruder_index));
  });

  it('falls back to decoys when the feature has no zero-activation texts', () => {
    const groups = buildGroups(top, [], 'seed');
    groups.forEach((g) => expect(g.examples[g.intruder_index].kind).toBe('decoy'));
  });

  it('rejects a feature with no activating texts', () => {
    expect(() => buildGroups([], [zeros(30)], 'seed')).toThrow();
  });
});

describe('buildQuestions', () => {
  it('asks one choice question per group with one option per index', () => {
    const groups = buildGroups([withPeak(30, 10)], [], 'seed');
    const questions = buildQuestions(groups);
    expect(Object.keys(questions)).toEqual(groups.map((_, g) => `group_${g}`));
    expect(questions.group_0.type).toBe('choice');
    expect(Object.keys(questions.group_0.criteria)).toEqual(['0', '1', '2', '3', '4']);
    expect(questions.group_2.instructions).toContain('`groups[2]`');
  });
});
