import { describe, expect, it } from 'vitest';
import { STEER_COMPLETION_VERSION } from './steer';
import {
  authorFromAxisId,
  axisReadoutsFromStored,
  axisReadoutsToStored,
  mergeStoredAxes,
  storedAxisIds,
  storedAxisRowIds,
  storedOutputTextIncludesPrompt,
} from './steer-wire';

// The title the 70B asset shipped, spelled with escapes so the file has no combining marks.
const ASSISTANT_AXIS_TITLE = '- Role-playing \u2194\ufe0f + Assistant-like';

// A row as it was written before the wire became camelCase, and before readouts were keyed by axis
// id. Rows like this are still in the database and are never migrated, so reading one has to keep
// working indefinitely.
const LEGACY_ROW = {
  pc_titles: [ASSISTANT_AXIS_TITLE],
  turns: [
    {
      pc_values: { [ASSISTANT_AXIS_TITLE]: 0.5 },
      pc_values_post_cap: { [ASSISTANT_AXIS_TITLE]: 0.2 },
      snippet: 'hello',
    },
    { pc_values: { [ASSISTANT_AXIS_TITLE]: -1 }, pc_values_post_cap: { [ASSISTANT_AXIS_TITLE]: -1 }, snippet: 'bye' },
  ],
  type: 'STEERED',
};

describe('axisReadoutsFromStored', () => {
  it('reads a legacy snake_case row', () => {
    expect(axisReadoutsFromStored(LEGACY_ROW)).toEqual([
      {
        id: 'lu_assistant-axis',
        author: 'lu',
        title: ASSISTANT_AXIS_TITLE,
        type: 'STEERED',
        turns: [
          { value: 0.5, valuePostCap: 0.2, snippet: 'hello' },
          { value: -1, valuePostCap: -1, snippet: 'bye' },
        ],
      },
    ]);
  });

  it('maps the one legacy title to the axis id now serving it', () => {
    // Without this, every conversation already in the database would look like a cache miss for
    // `lu_assistant-axis` and be regenerated.
    expect(storedAxisIds(LEGACY_ROW)).toEqual(['lu_assistant-axis']);
  });

  it('keeps an unrecognized legacy title as its own id, rather than dropping it', () => {
    const readouts = axisReadoutsFromStored({ pc_titles: ['some other fit'], turns: [] });
    expect(readouts[0].id).toBe('some other fit');
    expect(readouts[0].title).toBe('some other fit');
  });

  it('reads a legacy camelCase row, from the window where the writer had stopped converting', () => {
    const readouts = axisReadoutsFromStored({
      pcTitles: ['warmth'],
      turns: [{ pcValues: { warmth: 1 }, pcValuesPostCap: { warmth: 0.5 }, snippet: 'hi' }],
      type: 'DEFAULT',
    });
    expect(readouts).toEqual([
      {
        id: 'warmth',
        // An id from before ids carried an author, so there is nobody to credit.
        author: '',
        title: 'warmth',
        type: 'DEFAULT',
        turns: [{ value: 1, valuePostCap: 0.5, snippet: 'hi' }],
      },
    ]);
  });

  it('splits a multi-component legacy row into one readout per component', () => {
    // The values were stored as one map per turn keyed by title, so this is where a reworded
    // title used to orphan a row.
    const readouts = axisReadoutsFromStored({
      pc_titles: ['warmth', 'formality'],
      turns: [{ pc_values: { warmth: 0.5, formality: -0.5 } }],
    });
    expect(readouts.map((readout) => readout.id)).toEqual(['warmth', 'formality']);
    expect(readouts[0].turns?.[0].value).toBe(0.5);
    expect(readouts[1].turns?.[0].value).toBe(-0.5);
  });

  it('reads the current axes shape', () => {
    const readouts = axisReadoutsFromStored({
      axes: {
        mit_empathy: { title: 'empathetic', layer: 19, turns: [{ value: 0.25, value_post_cap: 0.5, snippet: 'hi' }] },
        mit_sycophantic: { title: 'sycophancy', layer: 15, caveat: 'Turn bias 0.40', turns: [{ value: -0.1 }] },
      },
      type: 'DEFAULT',
    });
    expect(readouts.map((readout) => readout.id)).toEqual(['mit_empathy', 'mit_sycophantic']);
    expect(readouts[0]).toEqual({
      id: 'mit_empathy',
      author: 'mit',
      title: 'empathetic',
      type: 'DEFAULT',
      layer: 19,
      caveat: undefined,
      turns: [{ value: 0.25, valuePostCap: 0.5, snippet: 'hi' }],
    });
    expect(readouts[1].caveat).toBe('Turn bias 0.40');
  });

  it('reads both readings of a turn', () => {
    // `value` is the measurement and `percentile` is where it falls in the calibration corpus.
    // The chart shows the second and the row keeps the first, so a read that dropped either
    // would either put "102%" back on screen or lose how far past the corpus the turn sat.
    const readouts = axisReadoutsFromStored({
      axes: {
        mit_toxic: {
          title: 'toxicity',
          turns: [{ value: 1.19, value_post_cap: 0.4, percentile: 1, percentile_post_cap: 0.62 }],
        },
      },
      type: 'STEERED',
    });
    expect(readouts[0].turns?.[0]).toEqual({
      value: 1.19,
      valuePostCap: 0.4,
      percentile: 1,
      percentilePostCap: 0.62,
      snippet: undefined,
    });
  });

  it('reads a row written before percentiles existed', () => {
    // Every row already in the database. Absent rather than 0, which would draw as dead centre.
    const readouts = axisReadoutsFromStored({ axes: { mit_toxic: { turns: [{ value: 0.4 }] } } });
    expect(readouts[0].turns?.[0].percentile).toBeUndefined();
    expect(readouts[0].turns?.[0].percentilePostCap).toBeUndefined();
  });

  it('falls back to the row\u2019s own steer type when the payload carries none', () => {
    // Which column a readout belongs to is not recoverable from the payload, and early rows
    // stored no type at all.
    const readouts = axisReadoutsFromStored({ axes: { a: { turns: [] } } }, 'STEERED');
    expect(readouts[0].type).toBe('STEERED');
  });

  it('survives the shapes a nullable json column actually yields', () => {
    for (const empty of [null, undefined]) {
      expect(axisReadoutsFromStored(empty)).toEqual([]);
    }
    expect(axisReadoutsFromStored({})).toEqual([]);
    expect(axisReadoutsFromStored({ axes: {} })).toEqual([]);
    expect(axisReadoutsFromStored({ axes: { a: {} } })[0].turns).toEqual([]);
  });
});

describe('axisReadoutsToStored', () => {
  it('writes snake_case keyed by axis id, not by title', () => {
    const stored = axisReadoutsToStored([
      {
        id: 'mit_empathy',
        author: 'mit',
        title: 'empathetic',
        polePositive: 'empathetic',
        poleNegative: 'unempathetic',
        layer: 19,
        turns: [{ value: 0.5, valuePostCap: 0.2, snippet: 'hello' }],
      },
    ]);
    expect(stored).toEqual({
      axes: {
        mit_empathy: {
          axis_id: undefined,
          // The poles rather than the title: what the two ends were called is structure a later
          // reader can label a reading with, and a display string is not.
          pole_positive: 'empathetic',
          pole_negative: 'unempathetic',
          layer: 19,
          caveat: undefined,
          turns: [{ value: 0.5, value_post_cap: 0.2, snippet: 'hello' }],
        },
      },
      type: undefined,
    });
    expect(JSON.stringify(stored)).not.toContain('valuePostCap');
    expect(JSON.stringify(stored)).not.toContain('title');
  });

  it('records the row that measured each reading', () => {
    // The point of storing it: a percentile is a position in the corpus one particular fit was
    // calibrated on, so a reading is only interpretable against the row that produced it.
    const stored = axisReadoutsToStored(
      [{ id: 'mit_toxic', author: 'mit', title: 'mit_toxic', turns: [{ value: 0.3 }] }],
      'DEFAULT',
      { mit_toxic: 'row_abc' },
    );
    expect(stored.axes.mit_toxic.axis_id).toBe('row_abc');
  });

  it('leaves the row unrecorded for an axis the server resolved itself', () => {
    // An axis named by id and served from a pod's own assets has no row behind it. Recording one
    // anyway would claim a provenance the reading does not have.
    const stored = axisReadoutsToStored([
      { id: 'lu_assistant-axis', author: 'lu', title: 'lu_assistant-axis', turns: [{ value: 0.3 }] },
    ]);
    expect(stored.axes['lu_assistant-axis'].axis_id).toBeUndefined();
  });

  it('writes the percentile snake_case beside the value', () => {
    const stored = axisReadoutsToStored([
      {
        id: 'mit_toxic',
        author: 'mit',
        title: 'toxicity',
        turns: [{ value: 1.19, percentile: 1, percentilePostCap: 0.62, valuePostCap: 0.4 }],
      },
    ]);
    expect(stored.axes.mit_toxic.turns).toEqual([
      { value: 1.19, value_post_cap: 0.4, percentile: 1, percentile_post_cap: 0.62, snippet: undefined },
    ]);
    expect(JSON.stringify(stored)).not.toContain('percentilePostCap');
  });

  it('keeps the unclipped measurement rather than storing only the bounded one', () => {
    // The stored value is what a later read has to work from, and a percentile saturates: every
    // turn past the corpus reads 1. Storing that alone would flatten a 1.02 and a 1.35 into the
    // same row forever, and the distance past the corpus is what says an axis is off distribution.
    const round = axisReadoutsToStored(
      axisReadoutsFromStored({
        axes: { mit_toxic: { turns: [{ value: 1.35, percentile: 1 }] } },
      }),
    );
    expect(round.axes.mit_toxic.turns?.[0].value).toBe(1.35);
    expect(round.axes.mit_toxic.turns?.[0].percentile).toBe(1);
  });

  it('takes the stored type from the readouts when not given one', () => {
    const stored = axisReadoutsToStored([{ id: 't_a', author: 't', title: 'a', type: 'STEERED', turns: [] }]);
    expect(stored.type).toBe('STEERED');
  });

  it('round-trips the current shape without changing it', () => {
    const stored = {
      axes: { mit_empathy: { title: 'empathetic', layer: 19, caveat: undefined, turns: [{ value: 1, snippet: 'x' }] } },
      type: 'DEFAULT',
    };
    const round = axisReadoutsToStored(axisReadoutsFromStored(stored), 'DEFAULT');
    expect(round.axes.mit_empathy.turns).toEqual([{ value: 1, value_post_cap: undefined, snippet: 'x' }]);
    expect(round.type).toBe('DEFAULT');
  });

  it('writes an empty axes map rather than dropping the key', () => {
    expect(axisReadoutsToStored([])).toEqual({ axes: {}, type: undefined });
  });
});

describe('mergeStoredAxes', () => {
  it('keeps axes the row already had alongside the newly measured ones', () => {
    // The point of merging: a row measured for one axis and later asked about another must end up
    // holding both, or the second request's work is discarded.
    const merged = mergeStoredAxes(
      { axes: { mit_empathy: { title: 'empathetic', turns: [{ value: 1 }] } }, type: 'DEFAULT' },
      [{ id: 'mit_toxic', author: 'mit', title: 'toxicity', turns: [{ value: -1 }] }],
      'DEFAULT',
    );
    expect(Object.keys(merged.axes).sort()).toEqual(['mit_empathy', 'mit_toxic']);
    expect(merged.axes.mit_empathy.turns).toEqual([{ value: 1, value_post_cap: undefined, snippet: undefined }]);
  });

  it('lets a fresh measurement replace a stored one for the same axis', () => {
    const merged = mergeStoredAxes({ axes: { mit_empathy: { turns: [{ value: 1 }] } } }, [
      { id: 'mit_empathy', author: 'mit', title: 'empathetic', turns: [{ value: 0.25 }] },
    ]);
    expect(merged.axes.mit_empathy.turns?.[0].value).toBe(0.25);
  });

  it('upgrades a legacy row in place', () => {
    const merged = mergeStoredAxes(
      LEGACY_ROW,
      [{ id: 'mit_toxic', author: 'mit', title: 'toxicity', turns: [] }],
      'STEERED',
    );
    expect(Object.keys(merged.axes).sort()).toEqual(['lu_assistant-axis', 'mit_toxic']);
    expect(merged.axes['lu_assistant-axis'].turns?.[0].value).toBe(0.5);
  });

  it('merges into an empty column', () => {
    const merged = mergeStoredAxes(null, [{ id: 't_a', author: 't', title: 'a', turns: [] }], 'DEFAULT');
    expect(Object.keys(merged.axes)).toEqual(['t_a']);
  });

  it('keeps the provenance of an axis this request did not measure', () => {
    // A readout has no field for the row that measured it, so carrying a stored axis across by
    // reshaping it through one would silently drop the only record of which fit produced it.
    const merged = mergeStoredAxes(
      { axes: { mit_empathy: { axis_id: 'row_empathy', turns: [{ value: 1 }] } } },
      [{ id: 'mit_toxic', author: 'mit', title: 'mit_toxic', turns: [{ value: -1 }] }],
      'DEFAULT',
      { mit_toxic: 'row_toxic' },
    );
    expect(merged.axes.mit_empathy.axis_id).toBe('row_empathy');
    expect(merged.axes.mit_toxic.axis_id).toBe('row_toxic');
  });
});

describe('storedAxisRowIds', () => {
  it('reports the row each reading was measured with', () => {
    const rows = storedAxisRowIds({
      axes: { mit_toxic: { axis_id: 'row_toxic' }, mit_empathy: { axis_id: 'row_empathy' } },
    });
    expect(rows).toEqual({ mit_toxic: 'row_toxic', mit_empathy: 'row_empathy' });
  });

  it('omits an axis that recorded no row, rather than inventing one', () => {
    // Those readings predate axes being rows, and a cache has to be able to tell them apart from a
    // reading this scheme took -- they were measured by whatever a pod had on disk.
    expect(storedAxisRowIds({ axes: { mit_toxic: { turns: [] } } })).toEqual({});
    expect(storedAxisRowIds(LEGACY_ROW)).toEqual({});
    expect(storedAxisRowIds(null)).toEqual({});
  });
});

describe('authorFromAxisId', () => {
  it('reads the author off the id', () => {
    expect(authorFromAxisId('mit_empathy')).toBe('mit');
    expect(authorFromAxisId('lu_assistant-axis')).toBe('lu');
  });

  it('splits on the first underscore, so a name may contain one', () => {
    expect(authorFromAxisId('mit_turn_bias')).toBe('mit');
  });

  it('is empty for an id predating the convention, rather than guessing', () => {
    // These come from legacy rows whose title did not map. Someone fitted them; nothing
    // recorded who, and inventing an author would be worse than showing none.
    expect(authorFromAxisId('some other fit')).toBe('');
    expect(authorFromAxisId('_leading')).toBe('');
  });
});

describe('storedAxisIds', () => {
  it('reads ids without reshaping the payload', () => {
    expect(storedAxisIds({ axes: { a: {}, b: {} } }).sort()).toEqual(['a', 'b']);
  });

  it('is empty for an absent column', () => {
    expect(storedAxisIds(null)).toEqual([]);
    expect(storedAxisIds({})).toEqual([]);
  });
});

describe('storedOutputTextIncludesPrompt', () => {
  it('flags a completion saved before inference stopped returning the prompt', () => {
    expect(
      storedOutputTextIncludesPrompt({ version: STEER_COMPLETION_VERSION - 1, outputTextChatTemplate: null }),
    ).toBe(true);
  });

  it('does not flag a completion saved at the current version', () => {
    expect(storedOutputTextIncludesPrompt({ version: STEER_COMPLETION_VERSION, outputTextChatTemplate: null })).toBe(
      false,
    );
  });

  it('never flags a chat row, whose outputText is not rendered on its own', () => {
    // Chat rows share this table and are still at older versions, but they render from
    // outputTextChatTemplate, so treating one as prompt-prefixed would strip a real message.
    expect(
      storedOutputTextIncludesPrompt({ version: STEER_COMPLETION_VERSION - 1, outputTextChatTemplate: '[]' }),
    ).toBe(false);
  });
});
