import { groupTokenLists, tokenListKeyForExplanation } from '@/components/token-list-columns';
import { NeuronTokenListPartialWithRelations } from 'prisma/generated/zod';
import { describe, expect, it } from 'vitest';

function list(typeName: string, name: string, pole: string, aspect: string) {
  return { typeName, name, pole, aspect, tokens: ['a'], values: [] } as unknown as NeuronTokenListPartialWithRelations;
}

const groups = groupTokenLists([
  list('neuron-lens', 'drivers', 'NEGATIVE', 'ACTIVATION'),
  list('neuron-lens', 'drivers', 'POSITIVE', 'ACTIVATION'),
  list('neuron-lens', 'Output Read', 'NEGATIVE', 'STEERING'),
  list('neuron-lens', 'Output Read', 'POSITIVE', 'STEERING'),
  list('modelmapper', 'Generated Words', 'BOTH', 'ACTIVATION'),
]);
const key = (typeName: string, name: string) => JSON.stringify([typeName, name]);

describe('tokenListKeyForExplanation', () => {
  it('picks the list of the same type with the same aspect', () => {
    expect(tokenListKeyForExplanation(groups, { typeName: 'neuron-lens', pole: 'POSITIVE', aspect: 'STEERING' })).toBe(
      key('neuron-lens', 'Output Read'),
    );
    expect(
      tokenListKeyForExplanation(groups, { typeName: 'neuron-lens', pole: 'NEGATIVE', aspect: 'ACTIVATION' }),
    ).toBe(key('neuron-lens', 'drivers'));
  });

  it('falls back to a list of the same type when no aspect matches', () => {
    expect(tokenListKeyForExplanation(groups, { typeName: 'modelmapper', pole: 'POSITIVE', aspect: 'STEERING' })).toBe(
      key('modelmapper', 'Generated Words'),
    );
  });

  it('gives nothing for a type with no lists', () => {
    expect(
      tokenListKeyForExplanation(groups, { typeName: 'np_max-act-logits', pole: null, aspect: null }),
    ).toBeUndefined();
  });
});
