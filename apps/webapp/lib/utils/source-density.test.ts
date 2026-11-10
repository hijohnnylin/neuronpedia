import { describe, expect, it } from 'vitest';
import { getSourceSetNameFromSource, sourceSetIgnoresDensity } from './source';

describe('sourceSetIgnoresDensity', () => {
  it('is true for neuron layers', () => {
    expect(sourceSetIgnoresDensity(getSourceSetNameFromSource('17'))).toBe(true);
  });

  it('is true for ICA Lens layers', () => {
    expect(sourceSetIgnoresDensity(getSourceSetNameFromSource('32-icalens-res'))).toBe(true);
  });

  it('is false for SAE layers', () => {
    expect(sourceSetIgnoresDensity(getSourceSetNameFromSource('20-gemmascope-res-16k'))).toBe(false);
    expect(sourceSetIgnoresDensity(getSourceSetNameFromSource('7-res-jb'))).toBe(false);
  });
});
