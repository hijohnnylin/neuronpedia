import { describe, expect, it } from 'vitest';
import { formatLayerSelection, selectLayerRange } from './layer-selection';

const layers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'];

describe('selectLayerRange', () => {
  it('adds a range in either drag direction, in layer order', () => {
    expect(selectLayerRange(layers, ['9'], 4, 2, true)).toEqual(['2', '3', '4', '9']);
    expect(selectLayerRange(layers, ['9'], 2, 4, true)).toEqual(['2', '3', '4', '9']);
  });

  it('removes a range', () => {
    expect(selectLayerRange(layers, ['1', '2', '3', '4'], 2, 3, false)).toEqual(['1', '4']);
  });

  it('sorts by layer order, not as strings', () => {
    expect(selectLayerRange(layers, ['10'], 9, 9, true)).toEqual(['9', '10']);
  });

  it('gives [] (all layers) when nothing or everything is selected', () => {
    expect(selectLayerRange(layers, ['5'], 5, 5, false)).toEqual([]);
    expect(selectLayerRange(layers, [], 0, 11, true)).toEqual([]);
  });
});

describe('formatLayerSelection', () => {
  it('shows runs as ranges', () => {
    expect(formatLayerSelection(layers, [])).toBe('All Layers');
    expect(formatLayerSelection(layers, ['5'])).toBe('Layer 5');
    expect(formatLayerSelection(layers, ['1', '2', '3', '7', '9', '10'])).toBe('Layers 1–3, 7, 9–10');
  });

  it('uses the layer number and extra info of a source id', () => {
    const sources = ['5-gemmascope-res-16k', '6-gemmascope-res-16k', '7-gemmascope-res-16k'];
    expect(formatLayerSelection(sources, ['5-gemmascope-res-16k', '6-gemmascope-res-16k'])).toBe('Layers 5–6');
  });
});
