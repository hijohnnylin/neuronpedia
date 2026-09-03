import { describe, expect, it } from 'vitest';
import { axisReadoutsToPublic } from './steer-axis-public';

describe('axisReadoutsToPublic', () => {
  it('reproduces the readout the endpoint used to forward verbatim', () => {
    // This field was the inference type assigned straight onto the response, so the bytes below are
    // the existing public contract rather than a new one.
    expect(
      axisReadoutsToPublic([
        {
          id: 'lu_assistant-axis',
          author: 'lu',
          title: '- Role-playing \u2194\ufe0f + Assistant-like',
          type: 'STEERED',
          layer: 40,
          caveat: 'Fitted on one model',
          polePositive: 'Assistant-like',
          poleNegative: 'Role-playing',
          polePositiveDescription: 'answers as itself',
          poleNegativeDescription: 'answers in character',
          sourceRevision: 'abc123',
          turns: [{ value: 0.5, valuePostCap: 0.2, percentile: 0.9, percentilePostCap: 0.4, snippet: 'hello' }],
        },
      ]),
    ).toEqual([
      {
        id: 'lu_assistant-axis',
        author: 'lu',
        title: '- Role-playing \u2194\ufe0f + Assistant-like',
        type: 'STEERED',
        layer: 40,
        caveat: 'Fitted on one model',
        polePositive: 'Assistant-like',
        poleNegative: 'Role-playing',
        polePositiveDescription: 'answers as itself',
        poleNegativeDescription: 'answers in character',
        sourceRevision: 'abc123',
        turns: [{ value: 0.5, valuePostCap: 0.2, percentile: 0.9, percentilePostCap: 0.4, snippet: 'hello' }],
      },
    ]);
  });

  it('keeps null distinct from absent, since the two serialize differently', () => {
    // `JSON.stringify` omits an undefined key and writes a null one, and inference sends both. So a
    // mapper that defaulted either way would change the response bytes for existing callers.
    const [mapped] = axisReadoutsToPublic([
      { id: 'mit_toxic', author: 'mit', title: 'mit_toxic', layer: null, caveat: undefined, turns: [] },
    ]);

    expect(JSON.parse(JSON.stringify(mapped))).toEqual({
      id: 'mit_toxic',
      author: 'mit',
      title: 'mit_toxic',
      layer: null,
      turns: [],
    });
  });

  it('carries a null turn list through rather than inventing an empty one', () => {
    // An axis whose layer failed to capture is dropped upstream, but a readout with no turns is a
    // different statement from one with none measured, and callers test for the field.
    expect(axisReadoutsToPublic([{ id: 'a', author: 'a', title: 'a', turns: null }])[0].turns).toBeNull();
    expect(axisReadoutsToPublic([{ id: 'a', author: 'a', title: 'a' }])[0].turns).toBeUndefined();
  });

  it('drops a field inference adds until someone maps it deliberately', () => {
    // The point of the type: a new field on the pydantic model reaches the public response only by
    // being named here, so growing the inference shape is not the same as growing this one.
    const withExtra = { id: 'a', author: 'a', title: 'a', somethingNew: 42 } as never;
    expect(axisReadoutsToPublic([withExtra])[0]).not.toHaveProperty('somethingNew');
  });
});
