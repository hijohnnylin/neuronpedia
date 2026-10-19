import { describe, expect, it } from 'vitest';
import { flipActivationSign, flipToMostNegative, selectScoringActivations } from './autointerp';

const row = (id: string, values: number[]) => ({
  id,
  tokens: [id],
  values,
  maxValue: values.length > 0 ? Math.max(...values) : 0,
  minValue: values.length > 0 ? Math.min(...values) : 0,
});

describe('flipActivationSign', () => {
  it('negates the values and swaps the max and the min', () => {
    const out = flipActivationSign(row('a', [0, -2, 1]));
    expect(out.values).toEqual([-0, 2, -1]);
    expect(out.maxValue).toBe(2);
    expect(out.minValue).toBe(-1);
  });

  it('keeps an all-zero example at a max of zero', () => {
    expect(flipActivationSign(row('a', [0, 0])).maxValue === 0).toBe(true);
  });
});

describe('flipToMostNegative', () => {
  it('orders by the most negative value and negates the values', () => {
    const out = flipToMostNegative([row('a', [0, -2, 1]), row('b', [-9, 0]), row('c', [3, 4])], 10);
    expect(out.map((r) => r.id)).toEqual(['b', 'a']);
    expect(out[0].values).toEqual([9, -0]);
    expect(out[0].maxValue).toBe(9);
    expect(out[1].values.indexOf(Math.max(...out[1].values))).toBe(1);
  });

  it('keeps only examples below zero, up to count', () => {
    expect(flipToMostNegative([row('a', [0, 1]), row('b', [])], 10)).toEqual([]);
    expect(flipToMostNegative([row('a', [-1]), row('b', [-3]), row('c', [-2])], 2).map((r) => r.id)).toEqual([
      'b',
      'c',
    ]);
  });

  it('does not change the input rows', () => {
    const input = [row('a', [-1, 2])];
    flipToMostNegative(input, 10);
    expect(input[0].values).toEqual([-1, 2]);
  });
});

describe('selectScoringActivations', () => {
  const rows = [row('pos', [1, 4]), row('neg', [-2, -1]), row('zero', [0, 0])];

  it('scores a max explanation on the positive examples', () => {
    const { activations, zeroActivations } = selectScoringActivations(rows, false);
    expect(activations.map((r) => r.id)).toEqual(['pos']);
    expect(activations[0].values).toEqual([2.5, 10]);
    expect(zeroActivations.map((r) => r.id)).toEqual(['zero']);
  });

  it('scores a min explanation on the flipped negative examples', () => {
    const { activations, zeroActivations } = selectScoringActivations(rows, true);
    expect(activations.map((r) => r.id)).toEqual(['neg']);
    expect(activations[0].values).toEqual([10, 5]);
    expect(zeroActivations.map((r) => r.id)).toEqual(['zero']);
  });
});
