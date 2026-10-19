import { ModelWithPartialRelations } from '@/prisma/generated/zod';
import { describe, expect, it } from 'vitest';
import { getDefaultSourceForModel, getDefaultSourceSetForModel } from './source';

function sourceSet(name: string, sourceIds: string[], visibility: 'PUBLIC' | 'UNLISTED' = 'PUBLIC') {
  return {
    name,
    visibility,
    hasDashboards: true,
    sources: sourceIds.map((id) => ({ id, hasDashboards: true, visibility })),
  };
}

function model(sourceSets: ReturnType<typeof sourceSet>[], defaults: Partial<ModelWithPartialRelations> = {}) {
  return { id: 'm', sourceSets, ...defaults } as unknown as ModelWithPartialRelations;
}

const NEURONS = sourceSet('neurons', ['10', '2', '0'], 'UNLISTED');
const SAE_B = sourceSet('res-b', ['10-res-b', '2-res-b']);
const SAE_A = sourceSet('res-a', ['5-res-a']);

describe('getDefaultSourceSetForModel', () => {
  it('starts on neurons when the model has them, also for a public-only picker', () => {
    expect(getDefaultSourceSetForModel(model([SAE_B, NEURONS, SAE_A]))?.name).toBe('neurons');
    expect(getDefaultSourceSetForModel(model([SAE_B, NEURONS, SAE_A]), true)?.name).toBe('neurons');
  });

  it('uses the model defaultSourceSetName over neurons', () => {
    expect(getDefaultSourceSetForModel(model([SAE_A, NEURONS, SAE_B], { defaultSourceSetName: 'res-b' }))?.name).toBe(
      'res-b',
    );
  });

  it('skips a default that the picker does not show', () => {
    const hidden = sourceSet('res-hidden', ['1-res-hidden'], 'UNLISTED');
    expect(
      getDefaultSourceSetForModel(model([hidden, NEURONS], { defaultSourceSetName: 'res-hidden' }), true)?.name,
    ).toBe('neurons');
  });

  it('falls back to the first source set by name without neurons', () => {
    expect(getDefaultSourceSetForModel(model([SAE_B, SAE_A]))?.name).toBe('res-a');
    expect(getDefaultSourceSetForModel(model([]))).toBeUndefined();
  });
});

describe('getDefaultSourceForModel', () => {
  it('picks the first neuron layer in number order', () => {
    expect(getDefaultSourceForModel(model([SAE_A, NEURONS]))?.id).toBe('0');
  });

  it('uses the model defaultSourceId when it is in the default source set', () => {
    const m = model([SAE_A, NEURONS, SAE_B], { defaultSourceSetName: 'res-b', defaultSourceId: '10-res-b' });
    expect(getDefaultSourceForModel(m)?.id).toBe('10-res-b');
    expect(getDefaultSourceForModel(model([SAE_B], { defaultSourceId: '5-res-a' }))?.id).toBe('2-res-b');
  });
});
