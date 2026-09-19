import { describe, expect, it } from 'vitest';
import { decodeMixedToken } from '../utils/byte-level-tokens';
import { balancedAccuracy, markedText, plainText } from './autointerp-scorer-jev';

describe('decodeMixedToken', () => {
  it('decodes byte-level newline and multi-byte punctuation', () => {
    expect(decodeMixedToken('Ċ')).toBe('\n');
    expect(decodeMixedToken('âĢĻs')).toBe('\u2019s');
    expect(decodeMixedToken('Ġthe')).toBe(' the');
  });

  it('passes already-decoded text through', () => {
    expect(decodeMixedToken(' weakness')).toBe(' weakness');
    expect(decodeMixedToken('café')).toBe('café');
  });

  it('turns the SentencePiece word marker into a space', () => {
    expect(decodeMixedToken('▁dog')).toBe(' dog');
    expect(decodeMixedToken('▁')).toBe(' ');
    expect(plainText(['▁your', '▁dog', 'Ċ'])).toBe(' your dog ');
  });
});

describe('plainText', () => {
  it('joins decoded tokens and flattens newlines', () => {
    expect(plainText(['Gold', 'Ċ', ' fell', 'âĢĻ'])).toBe('Gold  fell\u2019');
  });
});

describe('markedText', () => {
  it('marks only tokens at or above 3/10 of the max', () => {
    expect(markedText([' a', ' weak', ' dollar'], [0, 10, 2])).toBe(' a<< weak>> dollar');
    expect(markedText([' a', ' weak', ' dollar'], [0, 10, 3])).toBe(' a<< weak>><< dollar>>');
  });

  it('marks exactly one word when nothing activates', () => {
    const out = markedText(['no', ' signal', ' here'], [0, 0, 0]);
    expect(out.match(/<</g)).toHaveLength(1);
    expect(out.replace(/<<|>>/g, '')).toBe('no signal here');
  });

  it('is stable for the same input', () => {
    const tokens = ['alpha', ' beta', ' gamma', ' delta'];
    expect(markedText(tokens, [0, 0, 0, 0])).toBe(markedText(tokens, [0, 0, 0, 0]));
  });
});

describe('balancedAccuracy', () => {
  it('is 1 for perfect predictions and 0.5 for a constant answer', () => {
    expect(balancedAccuracy([true, true, false], [true, true, false])).toBe(1);
    expect(balancedAccuracy([true, true, true], [true, true, false])).toBe(0.5);
  });

  it('weights the two classes equally', () => {
    // 4 positives all right, 1 negative wrong: TPR 1, TNR 0.
    expect(balancedAccuracy([true, true, true, true, true], [true, true, true, true, false])).toBe(0.5);
  });

  it('is 0 when a class is missing', () => {
    expect(balancedAccuracy([true, true], [true, true])).toBe(0);
  });
});
