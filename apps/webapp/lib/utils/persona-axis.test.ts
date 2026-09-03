/**
 * A `Vector` row as the payload inference measures with.
 *
 * This mapping is where a stored axis stops being a database row, and every field it gets wrong is
 * wrong quietly: a swapped pole reports each reading against the opposite trait, a dropped scale
 * reports raw projection units as though they were calibrated, and a stored parameter that reaches the
 * payload under the wrong name is ignored by the server rather than refused.
 */

import type { SteerVectorReadout } from '@/lib/api/inference-types';
import { describe, expect, it } from 'vitest';
import { labelReadouts, personaAxisToVectorRead, type PersonaAxisFit } from './persona-axis';

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

describe('personaAxisToVectorRead', () => {
  it('sends the axis under its name, not the row id', () => {
    // `id` is what the readout comes back under and what a chart matches on. A cuid there would
    // make every stored reading unreadable to the next request, which asks by name.
    const payload = personaAxisToVectorRead(fit());
    expect(payload.id).toBe('mit_toxic');
    expect(JSON.stringify(payload)).not.toContain('row_toxic');
  });

  it('carries the stored parameters through to the payload', () => {
    const payload = personaAxisToVectorRead(
      fit({ layer: 29, projectionParams: { center: 0.4, scalePos: 2, scaleNeg: 3, normalize: 'l2' } }),
    );
    expect(payload).toMatchObject({
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
    expect(personaAxisToVectorRead(fit())).toMatchObject({
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
    const payload = personaAxisToVectorRead(fit({ projectionParams: { id: 'other', direction: [9, 9], layer: 99 } }));
    expect(payload).toMatchObject({ id: 'mit_toxic', direction: [0.5, -0.25], layer: 13 });
  });

  it('sends no labels at all', () => {
    // The payload is a direction and the arithmetic to apply it. Display text stays in this table:
    // it is already in hand wherever a reading is shown or stored, and a compute server cannot
    // check it. `labelReadouts` is what puts it on the response.
    // Named `np_tone` so that no label is a substring of the one field that does get sent: with
    // `mit_toxic` the id itself contains both the author and the positive pole.
    const payload = personaAxisToVectorRead(
      fit({ name: 'np_tone', displayName: 'Tone', caveat: 'fitted on 200 turns' }),
    );
    const sent = JSON.stringify(payload);
    for (const label of ['toxic', 'respectful', 'harmful or offensive', 'considerate', 'Tone', '200 turns', 'mit']) {
      expect(sent).not.toContain(label);
    }
  });

  it('leaves out what the row leaves out', () => {
    // Absence is the whole reason these live in a blob rather than in columns: a `Float[]` column
    // can only be empty, and an empty quantile table is refused outright by a server that would
    // have measured the axis fine without one.
    const payload = personaAxisToVectorRead(fit());
    expect(payload.preNormMean).toBeUndefined();
    expect(payload.postNormMean).toBeUndefined();
    expect(payload.quantilesPos).toBeUndefined();
    expect(payload.quantilesNeg).toBeUndefined();
    expect(payload.quantileLevels).toBeUndefined();
  });

  it('sends the means an axis was fitted with', () => {
    const payload = personaAxisToVectorRead(fit({ projectionParams: { preNormMean: [1, 2], postNormMean: [3, 4] } }));
    expect(payload.preNormMean).toEqual([1, 2]);
    expect(payload.postNormMean).toEqual([3, 4]);
  });

  it('sends both tables and their levels together', () => {
    const payload = personaAxisToVectorRead(
      fit({ projectionParams: { quantileLevels: [0, 0.5, 1], quantilesPos: [0, 1, 2], quantilesNeg: [0, 1, 3] } }),
    );
    expect(payload.quantileLevels).toEqual([0, 0.5, 1]);
    expect(payload.quantilesPos).toEqual([0, 1, 2]);
    expect(payload.quantilesNeg).toEqual([0, 1, 3]);
  });

  it('leaves the levels out for tables sampled on an even grid', () => {
    // Which is what inference defaults to, so an axis that stored no grid sends no grid.
    const payload = personaAxisToVectorRead(fit({ projectionParams: { quantilesPos: [0, 1], quantilesNeg: [0, 2] } }));
    expect(payload.quantileLevels).toBeUndefined();
    expect(payload.quantilesPos).toEqual([0, 1]);
  });

  it('sends the render conditions the fit was made under', () => {
    const payload = personaAxisToVectorRead(
      fit({
        projectionParams: { render: { blankSystemPrompt: true, templateKwargs: { date_string: '26 Jul 2024' } } },
      }),
    );
    expect(payload.render).toEqual({ blankSystemPrompt: true, templateKwargs: { date_string: '26 Jul 2024' } });
  });

  it('sends an empty template map for an axis with no conditions', () => {
    expect(personaAxisToVectorRead(fit()).render).toEqual({ blankSystemPrompt: false, templateKwargs: {} });
  });

  it('drops a non-string template value rather than coercing it', () => {
    // A chat template takes strings. `date_string: 26` would render as a plausible date and move
    // the fit off the distribution it was measured on, which is what these conditions prevent.
    const payload = personaAxisToVectorRead(
      fit({ projectionParams: { render: { templateKwargs: { date_string: 26, tools: 'none' } } } }),
    );
    expect(payload.render?.templateKwargs).toEqual({ tools: 'none' });
  });

  it('survives the shapes a json column actually yields', () => {
    // A row written before the column had a default, or by something that put the wrong thing
    // there. Every one of these has to come back as the uncalibrated axis rather than as a throw.
    for (const projectionParams of [null, undefined, 'a string', 42, ['a', 'list']]) {
      const payload = personaAxisToVectorRead(fit({ projectionParams }));
      expect(payload).toMatchObject({ normalize: 'none', center: 0, scalePos: 1, scaleNeg: 1 });
      expect(payload.render).toEqual({ blankSystemPrompt: false, templateKwargs: {} });
    }
  });
});

function readout(overrides: Partial<SteerVectorReadout> = {}): SteerVectorReadout {
  // What inference answers for a read sent inline: the id it was given, a placeholder author, the
  // id again as a title, and no labels.
  return { id: 'mit_toxic', author: 'custom', title: 'mit_toxic', turns: [{ value: 0.4 }], ...overrides };
}

describe('labelReadouts', () => {
  it('puts the row’s labels on the reading', () => {
    const [labelled] = labelReadouts([readout()], [fit()]);
    expect(labelled).toMatchObject({
      author: 'mit',
      polePositive: 'toxic',
      poleNegative: 'respectful',
      polePositiveDescription: 'harmful or offensive',
      poleNegativeDescription: 'considerate',
    });
  });

  it('keeps the numbers exactly as they came back', () => {
    const [labelled] = labelReadouts([readout({ turns: [{ value: 1.35, percentile: 1 }] })], [fit()]);
    expect(labelled.turns).toEqual([{ value: 1.35, percentile: 1 }]);
  });

  it('titles a row by its display name, or by the id when it has none', () => {
    expect(labelReadouts([readout()], [fit({ displayName: 'Tone' })])[0].title).toBe('Tone');
    expect(labelReadouts([readout()], [fit()])[0].title).toBe('mit_toxic');
  });

  it('matches on the name a read was sent under, not the row id', () => {
    // `id` on a readout is what the request asked for, which is `Vector.name`. Matching on
    // `Vector.id` would label nothing and quietly return a wheel with no pole names on it.
    expect(labelReadouts([readout({ id: 'row_toxic' })], [fit()])[0].polePositive).toBeUndefined();
  });

  it('labels a row that names no poles, without inventing any', () => {
    // A probe or a plain steering vector. It reads to the same number as an axis does; what it
    // lacks is ends to name, and `undefined` is what every shape downstream falls back from.
    const probe = fit({ name: 'mit_deception', polePositive: null, poleNegative: null, displayName: 'Deception' });
    const [labelled] = labelReadouts([readout({ id: 'mit_deception' })], [probe]);
    expect(labelled).toMatchObject({ author: 'mit', title: 'Deception' });
    expect(labelled.polePositive).toBeUndefined();
    expect(labelled.poleNegative).toBeUndefined();
  });

  it('leaves a readout with no row of its own untouched', () => {
    // A vector fetched from a published artifact: there inference did read a manifest, so what it
    // sent is the artifact’s own labelling and this has nothing better to say.
    const fetched = readout({ id: 'mit_empathy', author: 'mit', title: 'empathetic', polePositive: 'empathetic' });
    expect(labelReadouts([fetched], [fit()])[0]).toEqual(fetched);
  });
});
