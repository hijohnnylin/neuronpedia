import { describe, expect, it } from 'vitest';
import { isNeuronInModelRange } from './source';

const QWEN3_32B = { layers: 64, neuronsPerLayer: 25600 };

describe('isNeuronInModelRange', () => {
  it('is true inside the model', () => {
    expect(isNeuronInModelRange(QWEN3_32B, '0', '0')).toBe(true);
    expect(isNeuronInModelRange(QWEN3_32B, '63', '25599')).toBe(true);
  });

  it('is false past the last layer or neuron', () => {
    expect(isNeuronInModelRange(QWEN3_32B, '64', '0')).toBe(false);
    expect(isNeuronInModelRange(QWEN3_32B, '388', '3223')).toBe(false);
    expect(isNeuronInModelRange(QWEN3_32B, '0', '25600')).toBe(false);
  });

  it('is false for a negative layer or index', () => {
    expect(isNeuronInModelRange(QWEN3_32B, '-232', '3224')).toBe(false);
    expect(isNeuronInModelRange(QWEN3_32B, '-0', '1')).toBe(false);
    expect(isNeuronInModelRange(QWEN3_32B, '3', '-1')).toBe(false);
  });

  it('is false for a number that is not in plain form', () => {
    expect(isNeuronInModelRange(QWEN3_32B, '07', '1')).toBe(false);
    expect(isNeuronInModelRange(QWEN3_32B, '7', '01')).toBe(false);
    expect(isNeuronInModelRange(QWEN3_32B, '7-res-jb', '1')).toBe(false);
  });

  it('does not bound a count that is not known', () => {
    expect(isNeuronInModelRange({ layers: 0, neuronsPerLayer: 0 }, '500', '999999')).toBe(true);
    expect(isNeuronInModelRange({}, '5', '5')).toBe(true);
  });
});
