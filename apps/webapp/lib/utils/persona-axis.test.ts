/**
 * A `Vector` row as the payload inference measures with.
 *
 * This mapping is where a stored axis stops being a database row, and every field it gets wrong is
 * wrong quietly: a swapped pole reports each reading against the opposite trait, a dropped scale
 * reports raw projection units as though they were calibrated, and a stored parameter that reaches the
 * payload under the wrong name is ignored by the server rather than refused.
 */

import { describe, expect, it } from 'vitest';
import { personaAxisToNPAxis, type PersonaAxisFit } from './persona-axis';

function fit(overrides: Partial<PersonaAxisFit> = {}): PersonaAxisFit {
  return {
    id: 'row_toxic',
    name: 'mit_toxic',
    author: 'mit',
    layer: 13,
    polePositive: 'toxic',
    poleNegative: 'respectful',
    polePositiveDescription: 'harmful or offensive',
    poleNegativeDescription: 'considerate',
    displayName: null,
    caveat: null,
    values: [0.5, -0.25],
    projectionParams: {},
    ...overrides,
  };
}

describe('personaAxisToNPAxis', () => {
  it('sends the axis under its name, not the row id', () => {
    // `id` is what the readout comes back under and what a chart matches on. A cuid there would
    // make every stored reading unreadable to the next request, which asks by name.
    const payload = personaAxisToNPAxis(fit());
    expect(payload.id).toBe('mit_toxic');
    expect(JSON.stringify(payload)).not.toContain('row_toxic');
  });

  it('carries the stored parameters through to the payload', () => {
    const payload = personaAxisToNPAxis(
      fit({ layer: 29, author: 'lu', projectionParams: { center: 0.4, scalePos: 2, scaleNeg: 3, normalize: 'l2' } }),
    );
    expect(payload).toMatchObject({
      author: 'lu',
      layer: 29,
      direction: [0.5, -0.25],
      center: 0.4,
      scalePos: 2,
      scaleNeg: 3,
      normalize: 'l2',
    });
  });

  it('defaults the four fields the payload requires and a row may omit', () => {
    // What omitting them means: the reading is the bare dot product of the direction with the
    // activation, in the axis's own units. Defaulted here rather than in every writer of a row.
    expect(personaAxisToNPAxis(fit())).toMatchObject({
      normalize: 'none',
      center: 0,
      scalePos: 1,
      scaleNeg: 1,
    });
  });

  it('cannot be told what to call itself by its own stored parameters', () => {
    // The stored parameters are spread before the row's own fields, so a stale or hostile key cannot rename the
    // axis or replace its numbers. Ordering in one object literal is all that enforces this, which
    // is exactly the kind of thing that survives a refactor only if a test is watching.
    const payload = personaAxisToNPAxis(fit({ projectionParams: { id: 'other', direction: [9, 9], layer: 99 } }));
    expect(payload).toMatchObject({ id: 'mit_toxic', direction: [0.5, -0.25], layer: 13 });
  });

  it('carries both poles and what they mean', () => {
    const payload = personaAxisToNPAxis(fit());
    expect(payload.polePositive).toBe('toxic');
    expect(payload.poleNegative).toBe('respectful');
    expect(payload.polePositiveDescription).toBe('harmful or offensive');
    expect(payload.poleNegativeDescription).toBe('considerate');
  });

  it('leaves out what the row leaves out', () => {
    // Absence is the whole reason these live in a blob rather than in columns: a `Float[]` column
    // can only be empty, and an empty quantile table is refused outright by a server that would
    // have measured the axis fine without one.
    const payload = personaAxisToNPAxis(fit());
    expect(payload.preNormMean).toBeUndefined();
    expect(payload.postNormMean).toBeUndefined();
    expect(payload.quantilesPos).toBeUndefined();
    expect(payload.quantilesNeg).toBeUndefined();
    expect(payload.quantileLevels).toBeUndefined();
  });

  it('sends the means an axis was fitted with', () => {
    const payload = personaAxisToNPAxis(fit({ projectionParams: { preNormMean: [1, 2], postNormMean: [3, 4] } }));
    expect(payload.preNormMean).toEqual([1, 2]);
    expect(payload.postNormMean).toEqual([3, 4]);
  });

  it('sends both tables and their levels together', () => {
    const payload = personaAxisToNPAxis(
      fit({ projectionParams: { quantileLevels: [0, 0.5, 1], quantilesPos: [0, 1, 2], quantilesNeg: [0, 1, 3] } }),
    );
    expect(payload.quantileLevels).toEqual([0, 0.5, 1]);
    expect(payload.quantilesPos).toEqual([0, 1, 2]);
    expect(payload.quantilesNeg).toEqual([0, 1, 3]);
  });

  it('leaves the levels out for tables sampled on an even grid', () => {
    // Which is what inference defaults to, so an axis that stored no grid sends no grid.
    const payload = personaAxisToNPAxis(fit({ projectionParams: { quantilesPos: [0, 1], quantilesNeg: [0, 2] } }));
    expect(payload.quantileLevels).toBeUndefined();
    expect(payload.quantilesPos).toEqual([0, 1]);
  });

  it('sends the render conditions the fit was made under', () => {
    const payload = personaAxisToNPAxis(
      fit({
        projectionParams: { render: { blankSystemPrompt: true, templateKwargs: { date_string: '26 Jul 2024' } } },
      }),
    );
    expect(payload.render).toEqual({ blankSystemPrompt: true, templateKwargs: { date_string: '26 Jul 2024' } });
  });

  it('sends an empty template map for an axis with no conditions', () => {
    expect(personaAxisToNPAxis(fit()).render).toEqual({ blankSystemPrompt: false, templateKwargs: {} });
  });

  it('drops a non-string template value rather than coercing it', () => {
    // A chat template takes strings. `date_string: 26` would render as a plausible date and move
    // the fit off the distribution it was measured on, which is what these conditions prevent.
    const payload = personaAxisToNPAxis(
      fit({ projectionParams: { render: { templateKwargs: { date_string: 26, tools: 'none' } } } }),
    );
    expect(payload.render?.templateKwargs).toEqual({ tools: 'none' });
  });

  it('survives the shapes a json column actually yields', () => {
    // A row written before the column had a default, or by something that put the wrong thing
    // there. Every one of these has to come back as the uncalibrated axis rather than as a throw.
    for (const projectionParams of [null, undefined, 'a string', 42, ['a', 'list']]) {
      const payload = personaAxisToNPAxis(fit({ projectionParams }));
      expect(payload).toMatchObject({ normalize: 'none', center: 0, scalePos: 1, scaleNeg: 1 });
      expect(payload.render).toEqual({ blankSystemPrompt: false, templateKwargs: {} });
    }
  });
});
