import { describe, expect, it } from 'vitest';
import { axisReadoutsToLegacyAssistantAxis } from './steer-axis-legacy';

describe('axisReadoutsToLegacyAssistantAxis', () => {
  it('reproduces the shape a single-axis response has always had', () => {
    // The exact payload `/api/steer-chat` returned before readouts were keyed by id, down to the
    // key: users of the public API split that string on its arrow, so it has to come out unchanged.
    const title = '- Role-playing \u2194\ufe0f + Assistant-like';
    expect(
      axisReadoutsToLegacyAssistantAxis([
        {
          id: 'lu_assistant-axis',
          author: 'lu',
          title: 'Assistant axis',
          type: 'STEERED',
          layer: 40,
          turns: [
            { value: 0.5, valuePostCap: 0.2, snippet: 'hello' },
            { value: -1, valuePostCap: -0.5, snippet: 'bye' },
          ],
        },
      ]),
    ).toEqual([
      {
        pcTitles: [title],
        turns: [
          {
            pcValues: { [title]: 0.5 },
            pcValuesPostCap: { [title]: 0.2 },
            snippet: 'hello',
          },
          {
            pcValues: { [title]: -1 },
            pcValuesPostCap: { [title]: -0.5 },
            snippet: 'bye',
          },
        ],
        type: 'STEERED',
      },
    ]);
  });

  it('folds several axes into one entry per steer type', () => {
    // Six traits at different layers become one entry whose turns hold all six values, which is
    // what the old shape could express and is why this view survives the rename.
    const legacy = axisReadoutsToLegacyAssistantAxis([
      {
        id: 'mit_empathy',
        author: 'mit',
        title: 'empathetic',
        type: 'DEFAULT',
        layer: 19,
        turns: [{ value: 0.25, snippet: 'a' }],
      },
      {
        id: 'mit_toxic',
        author: 'mit',
        title: 'toxicity',
        type: 'DEFAULT',
        layer: 13,
        turns: [{ value: -0.75, snippet: 'a' }],
      },
    ]);
    expect(legacy).toHaveLength(1);
    expect(legacy[0].pcTitles).toEqual(['mit_empathy', 'mit_toxic']);
    expect(legacy[0].turns?.[0].pcValues).toEqual({ mit_empathy: 0.25, mit_toxic: -0.75 });
  });

  it('keeps the steer types apart', () => {
    const legacy = axisReadoutsToLegacyAssistantAxis([
      { id: 'mit_empathy', author: 'mit', title: 'empathetic', type: 'DEFAULT', turns: [{ value: 1 }] },
      { id: 'mit_empathy', author: 'mit', title: 'empathetic', type: 'STEERED', turns: [{ value: -1 }] },
    ]);
    expect(legacy.map((entry) => entry.type)).toEqual(['DEFAULT', 'STEERED']);
    expect(legacy[0].turns?.[0].pcValues).toEqual({ mit_empathy: 1 });
    expect(legacy[1].turns?.[0].pcValues).toEqual({ mit_empathy: -1 });
  });

  it('omits the post-cap map when nothing was steered', () => {
    // The old field was absent rather than empty in this case, and callers test for it rather
    // than for its size.
    const legacy = axisReadoutsToLegacyAssistantAxis([
      { id: 't_a', author: 't', title: 'a', type: 'DEFAULT', turns: [{ value: 0.5, snippet: 'x' }] },
    ]);
    expect(legacy[0].turns?.[0].pcValuesPostCap).toBeUndefined();
  });

  it('leaves a short axis out of the turns it has no value for', () => {
    // Rather than reporting zero, which would draw as a real measurement.
    const legacy = axisReadoutsToLegacyAssistantAxis([
      { id: 't_long', author: 't', title: 'long', type: 'DEFAULT', turns: [{ value: 1 }, { value: 2 }] },
      { id: 't_short', author: 't', title: 'short', type: 'DEFAULT', turns: [{ value: 9 }] },
    ]);
    expect(legacy[0].turns?.[0].pcValues).toEqual({ t_long: 1, t_short: 9 });
    expect(legacy[0].turns?.[1].pcValues).toEqual({ t_long: 2 });
  });

  it('is empty for no readouts', () => {
    expect(axisReadoutsToLegacyAssistantAxis([])).toEqual([]);
  });

  it('keys the one axis with outside callers by the string they already parse', () => {
    // The exact bytes the 70B axis has always been keyed by, from the table rather than from this
    // readout: no poles, no matching title, and the key comes out right anyway. Rewording a pole or
    // a display name is then a thing this view cannot notice, which is the point of the table.
    const legacy = axisReadoutsToLegacyAssistantAxis([
      {
        id: 'lu_assistant-axis',
        author: 'lu',
        title: 'Some later wording of the assistant axis',
        type: 'STEERED',
        turns: [{ value: 0.5 }],
      },
    ]);
    const title = '- Role-playing \u2194\ufe0f + Assistant-like';
    expect(legacy[0].pcTitles).toEqual([title]);
    expect(legacy[0].turns?.[0].pcValues).toEqual({ [title]: 0.5 });
  });

  it('keys every other axis by its id', () => {
    // Nothing was parsing these keys before ids existed, so the id is what this view can say about
    // them honestly -- and unlike a title, two axes cannot collide on it.
    const legacy = axisReadoutsToLegacyAssistantAxis([
      { id: 'me_curiosity', author: 'me', title: 'how curious', type: 'DEFAULT', turns: [{ value: 1 }] },
    ]);
    expect(legacy[0].pcTitles).toEqual(['me_curiosity']);
  });
});
