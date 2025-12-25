import { describe, expect, test } from 'vitest';
import { formatToGlobalModels, UNNAMED_AUTHOR_NAME } from '@/lib/utils/general';

describe('formatToGlobalModels', () => {
  test('converts array to object keyed by id', () => {
    const models = [
      { id: 'gemma-2-2b', displayName: 'Gemma 2 2B' },
      { id: 'gpt2-small', displayName: 'GPT-2 Small' },
    ] as any[];

    const result = formatToGlobalModels(models);

    expect(result['gemma-2-2b'].displayName).toBe('Gemma 2 2B');
    expect(result['gpt2-small'].displayName).toBe('GPT-2 Small');
  });

  test('returns empty object for empty array', () => {
    const result = formatToGlobalModels([]);
    expect(result).toEqual({});
  });

  test('overwrites duplicate ids with last value', () => {
    const models = [
      { id: 'model-1', displayName: 'First' },
      { id: 'model-1', displayName: 'Second' },
    ] as any[];

    const result = formatToGlobalModels(models);

    expect(result['model-1'].displayName).toBe('Second');
  });
});

describe('UNNAMED_AUTHOR_NAME', () => {
  test('has expected value', () => {
    expect(UNNAMED_AUTHOR_NAME).toBe('Unnamed');
  });
});

