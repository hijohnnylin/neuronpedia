import { describe, expect, it } from 'vitest';
import { axisReadoutsToLegacyAssistantAxis } from './steer-axis-legacy';

describe('axisReadoutsToLegacyAssistantAxis', () => {
  it('reproduces the shape a single-axis response has always had', () => {
    // The exact payload `/api/steer-chat` returned before readouts were keyed by id. Users of the
    // public API and the existing chart both read this, so it has to come out unchanged.
    expect(
      axisReadoutsToLegacyAssistantAxis([
        {
          id: 'lu_assistant-axis',
          author: 'lu',
          title: 'Role-playing to Assistant-like',
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
        pcTitles: ['Role-playing to Assistant-like'],
        turns: [
          {
            pcValues: { 'Role-playing to Assistant-like': 0.5 },
            pcValuesPostCap: { 'Role-playing to Assistant-like': 0.2 },
            snippet: 'hello',
          },
          {
            pcValues: { 'Role-playing to Assistant-like': -1 },
            pcValuesPostCap: { 'Role-playing to Assistant-like': -0.5 },
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
    expect(legacy[0].pcTitles).toEqual(['empathetic', 'toxicity']);
    expect(legacy[0].turns?.[0].pcValues).toEqual({ empathetic: 0.25, toxicity: -0.75 });
  });

  it('keeps the steer types apart', () => {
    const legacy = axisReadoutsToLegacyAssistantAxis([
      { id: 'mit_empathy', author: 'mit', title: 'empathetic', type: 'DEFAULT', turns: [{ value: 1 }] },
      { id: 'mit_empathy', author: 'mit', title: 'empathetic', type: 'STEERED', turns: [{ value: -1 }] },
    ]);
    expect(legacy.map((entry) => entry.type)).toEqual(['DEFAULT', 'STEERED']);
    expect(legacy[0].turns?.[0].pcValues).toEqual({ empathetic: 1 });
    expect(legacy[1].turns?.[0].pcValues).toEqual({ empathetic: -1 });
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
    expect(legacy[0].turns?.[0].pcValues).toEqual({ long: 1, short: 9 });
    expect(legacy[0].turns?.[1].pcValues).toEqual({ long: 2 });
  });

  it('is empty for no readouts', () => {
    expect(axisReadoutsToLegacyAssistantAxis([])).toEqual([]);
  });

  it('builds the title from the poles, which is what an outside caller already parses', () => {
    // The exact bytes the 70B axis has always been keyed by. Inference reports an axis sent with
    // the request under its id, so a payload keyed by `title` would rename every key of this view
    // the day the demo started sending rows -- for callers whose code splits on that arrow.
    const legacy = axisReadoutsToLegacyAssistantAxis([
      {
        id: 'lu_assistant-axis',
        author: 'lu',
        title: 'lu_assistant-axis',
        polePositive: 'Assistant-like',
        poleNegative: 'Role-playing',
        type: 'STEERED',
        turns: [{ value: 0.5 }],
      },
    ]);
    const title = '- Role-playing \u2194\ufe0f + Assistant-like';
    expect(legacy[0].pcTitles).toEqual([title]);
    expect(legacy[0].turns?.[0].pcValues).toEqual({ [title]: 0.5 });
  });

  it('falls back to the title for an axis that names no poles', () => {
    // All this view ever had, and all a stranger's axis need supply.
    const legacy = axisReadoutsToLegacyAssistantAxis([
      { id: 'me_curiosity', author: 'me', title: 'how curious', type: 'DEFAULT', turns: [{ value: 1 }] },
    ]);
    expect(legacy[0].pcTitles).toEqual(['how curious']);
  });
});
