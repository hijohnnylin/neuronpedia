/**
 * What the wheel is made of: the axis rows, not the readouts.
 *
 * The layout is the one piece of this chart that is data rather than geometry, and it is where the
 * two halves of the page meet -- definitions read from the database decide what is drawn, readouts
 * from inference decide how far each wedge reaches. Both failures that matters here are silent on
 * screen: a wheel that draws the wrong pole's name still looks like a wheel, and one whose wedges
 * come from whatever inference answered still looks fine until a later turn rearranges it.
 */

import type { PersonaAxisDefinition } from '@/lib/utils/persona-axis';
import type { PublicAxisReadout } from '@/lib/utils/steer-axis-public';
import { describe, expect, it } from 'vitest';
import { TRAIT_EXAMPLES } from './neural-transparency-snippets';
import { findBehaviorSwing, layoutPoles, namedPoles } from './persona-sunburst';

function definition(name: string, overrides: Partial<PersonaAxisDefinition> = {}): PersonaAxisDefinition {
  return {
    id: `row_${name}`,
    name,
    author: 'mit',
    layer: 13,
    polePositive: 'toxic',
    poleNegative: 'respectful',
    polePositiveDescription: 'harmful or offensive',
    poleNegativeDescription: 'considerate',
    displayName: null,
    caveat: null,
    ...overrides,
  };
}

/** One measured turn on one axis. `percentile` is what the wheel draws where an axis reports one. */
function readout(name: string, value: number, percentile?: number): PublicAxisReadout {
  return {
    id: name,
    author: 'mit',
    title: name,
    layer: 13,
    turns: [{ value, percentile }],
  };
}

const poleBy = (poles: ReturnType<typeof layoutPoles>['poles'], key: string) => {
  const found = poles.find((pole) => pole.key === key);
  if (!found) throw new Error(`no pole ${key} in ${poles.map((pole) => pole.key).join(', ')}`);
  return found;
};

describe('the frame comes from the definitions', () => {
  it('draws an axis that has not been measured yet', () => {
    // The measurement lands after the reply has streamed, so every wheel is drawn empty first.
    const { poles } = layoutPoles([definition('mit_toxic')], [], 0);
    expect(poles.map((pole) => pole.key)).toEqual(['mit_toxic:minus', 'mit_toxic:plus']);
    expect(poles.every((pole) => !pole.measured)).toBe(true);
    expect(poles.every((pole) => pole.value === 0)).toBe(true);
  });

  it('does not draw a readout it has no definition for', () => {
    // Otherwise a cached row measured for some other axis would add a wedge, and the wheel would
    // change shape between turns of one conversation.
    const { poles } = layoutPoles([definition('mit_toxic')], [readout('lu_assistant-axis', 0.5)], 0);
    expect(poles.map((pole) => pole.key)).toEqual(['mit_toxic:minus', 'mit_toxic:plus']);
  });

  it('keeps the order it was given', () => {
    const axes = [definition('mit_toxic'), definition('mit_empathy'), definition('mit_erudite')];
    const { poles } = layoutPoles(axes, [], 0);
    // Signed axes take the wedges nearest twelve o'clock, each with its opposite mirrored across
    // the vertical, and the stylistic one straddles the bottom.
    expect(poles.map((pole) => [pole.key, pole.startAngle / (Math.PI / 3)])).toEqual([
      ['mit_toxic:minus', 0],
      ['mit_toxic:plus', 5],
      ['mit_empathy:plus', 1],
      ['mit_empathy:minus', 4],
      ['mit_erudite:plus', 2],
      ['mit_erudite:minus', 3],
    ]);
  });

  it('names each pole from its own end of the axis', () => {
    // The swap that would not look like a bug: both names are present, both are plausible, and
    // every reading is then reported against the opposite trait.
    const { poles } = layoutPoles([definition('mit_toxic')], [], 0);
    const plus = poleBy(poles, 'mit_toxic:plus');
    const minus = poleBy(poles, 'mit_toxic:minus');
    expect([plus.name, plus.definition]).toEqual(['Toxic', 'harmful or offensive']);
    expect([minus.name, minus.definition]).toEqual(['Respectful', 'considerate']);
  });

  it('gives each pole the replies its own end was fitted on', () => {
    // As quiet a failure as the name swap above, and the same swap: three plausible replies filed
    // under the opposite trait, read as what that trait looks like.
    const { poles } = layoutPoles([definition('mit_toxic')], [], 0);
    expect(poleBy(poles, 'mit_toxic:plus').examples).toEqual(TRAIT_EXAMPLES.mit_toxic.plus);
    expect(poleBy(poles, 'mit_toxic:minus').examples).toEqual(TRAIT_EXAMPLES.mit_toxic.minus);
    expect(TRAIT_EXAMPLES.mit_toxic.plus).toHaveLength(3);
  });

  it('shows no examples for an axis it holds none for', () => {
    // An axis someone else brings has no rollouts here, and the tooltip drops the section rather
    // than borrowing another axis's.
    const { poles } = layoutPoles([definition('someone_else_trait')], [], 0);
    expect(poleBy(poles, 'someone_else_trait:plus').examples).toEqual([]);
  });

  it('carries the layer and the caveat the row states', () => {
    const axes = [definition('mit_sycophantic', { layer: 15, caveat: 'Turn bias 0.40' })];
    const plus = poleBy(layoutPoles(axes, [], 0).poles, 'mit_sycophantic:plus');
    expect([plus.layer, plus.caveat]).toEqual([15, 'Turn bias 0.40']);
  });

  it('leaves a pole undescribed rather than inventing a description', () => {
    const axes = [definition('mit_toxic', { polePositiveDescription: null })];
    expect(poleBy(layoutPoles(axes, [], 0).poles, 'mit_toxic:plus').definition).toBeNull();
  });
});

describe('a reading goes to the pole it favours', () => {
  it('gives a positive reading to the positive pole and zero to the other', () => {
    const { poles } = layoutPoles([definition('mit_toxic')], [readout('mit_toxic', 0.4)], 0);
    expect(poleBy(poles, 'mit_toxic:plus').value).toBeCloseTo(0.4);
    expect(poleBy(poles, 'mit_toxic:minus').value).toBe(0);
    expect(poleBy(poles, 'mit_toxic:minus').measured).toBe(true);
  });

  it('gives a negative reading to the negative pole', () => {
    const { poles } = layoutPoles([definition('mit_toxic')], [readout('mit_toxic', -0.4)], 0);
    expect(poleBy(poles, 'mit_toxic:minus').value).toBeCloseTo(0.4);
    expect(poleBy(poles, 'mit_toxic:plus').value).toBe(0);
  });

  it('draws the percentile rather than the measurement behind it', () => {
    // The wheel is read as a dial, so what it draws cannot exceed 1 however far past the corpus
    // the raw measurement went.
    const { poles } = layoutPoles([definition('mit_toxic')], [readout('mit_toxic', 4.0, 1.0)], 0);
    const plus = poleBy(poles, 'mit_toxic:plus');
    expect([plus.value, plus.ranked]).toEqual([1.0, true]);
  });

  it('falls back to the measurement for an axis with no percentile', () => {
    const plus = poleBy(layoutPoles([definition('mit_toxic')], [readout('mit_toxic', 0.7)], 0).poles, 'mit_toxic:plus');
    expect([plus.value, plus.ranked]).toEqual([0.7, false]);
  });

  it('reads the turn it was asked for', () => {
    const measured: PublicAxisReadout = {
      ...readout('mit_toxic', 0.1),
      turns: [{ value: 0.1 }, { value: 0.9 }],
    };
    expect(poleBy(layoutPoles([definition('mit_toxic')], [measured], 1).poles, 'mit_toxic:plus').value).toBeCloseTo(
      0.9,
    );
  });
});

describe('which pole to prefer is this demo\u2019s own call', () => {
  it('colours the named axes by that judgement', () => {
    const axes = [definition('mit_toxic'), definition('mit_empathy')];
    const { poles } = layoutPoles(axes, [], 0);
    // `toxic` is the pole you would rather not see; `empathetic` is the one you would.
    expect(poleBy(poles, 'mit_toxic:plus').category).toBe('undesirable');
    expect(poleBy(poles, 'mit_toxic:minus').category).toBe('desirable');
    expect(poleBy(poles, 'mit_empathy:plus').category).toBe('desirable');
    expect(poleBy(poles, 'mit_empathy:minus').category).toBe('undesirable');
  });

  it('treats an axis it has no opinion about as stylistic', () => {
    // The safe default for an axis this demo does not know: grey, rather than calling a trait
    // good or bad on a guess. Nothing in the table or the API says which pole is preferable.
    const { poles, sectors } = layoutPoles([definition('someone_else_trait')], [], 0);
    expect(poles.every((pole) => pole.category === 'neutral')).toBe(true);
    expect(sectors.map((sector) => sector.category)).toEqual(['neutral']);
  });

  it('sizes the inner ring by how many poles each category holds', () => {
    const axes = [definition('mit_toxic'), definition('mit_empathy'), definition('mit_erudite')];
    const { sectors } = layoutPoles(axes, [], 0);
    const width = (category: string) => {
      const sector = sectors.find((candidate) => candidate.category === category);
      return sector ? (sector.endAngle - sector.startAngle) / (Math.PI / 3) : 0;
    };
    // Two signed axes put one pole each side; the stylistic one puts both across the bottom.
    expect(width('desirable')).toBeCloseTo(2);
    expect(width('neutral')).toBeCloseTo(2);
    expect(width('undesirable')).toBeCloseTo(2);
  });
});

describe('which traits a turn is named by', () => {
  const axes = [definition('mit_toxic'), definition('mit_empathy'), definition('mit_erudite')];
  // The percentile carries the sign, as it does from inference: it is a signed position on the axis,
  // so which pole a reading belongs to is read off it rather than off the ratio beside it.
  const readouts = [
    readout('mit_toxic', 0.2, 0.3),
    readout('mit_empathy', 0.9, 0.8),
    readout('mit_erudite', -0.5, -0.6),
  ];

  it('ranks the measured poles strongest first', () => {
    // The first of these is also the trait the readout opens its tooltip on, so this order is what
    // a reader is told the turn was about.
    const { poles } = layoutPoles(axes, readouts, 0);
    expect(namedPoles(poles).map((pole) => pole.key)).toEqual([
      'mit_empathy:plus',
      'mit_erudite:minus',
      'mit_toxic:plus',
    ]);
  });

  it('names nothing before the turn is measured', () => {
    // Every pole reads zero then, so ranking them would hand the column and the hint an arbitrary
    // winner rather than no winner.
    expect(namedPoles(layoutPoles(axes, [], 0).poles)).toEqual([]);
  });

  it('names no more poles than there are slots', () => {
    const many = [...axes, definition('mit_sycophantic'), definition('mit_romantic')];
    const measured = [...readouts, readout('mit_sycophantic', 0.4, 0.7), readout('mit_romantic', 0.4, 0.75)];
    expect(namedPoles(layoutPoles(many, measured, 0).poles)).toHaveLength(3);
  });
});

describe('an empty wheel', () => {
  it('draws nothing when there are no definitions', () => {
    // What the page shows if the rows are missing: no wedges at all, rather than a frame of
    // placeholder axes that nothing can measure.
    expect(layoutPoles([], [readout('mit_toxic', 0.5)], 0)).toEqual({ poles: [], sectors: [] });
  });
});

/** One axis over several turns, as signed percentiles: positive reads on `plus`, negative on `minus`. */
function overTurns(name: string, percentiles: (number | null)[]): PublicAxisReadout {
  return {
    id: name,
    author: 'mit',
    title: name,
    layer: 13,
    turns: percentiles.map((p) => (p === null ? { value: Number.NaN } : { value: p, percentile: p })),
  };
}

describe('behavioral swings', () => {
  const toxic = definition('mit_toxic');

  it('names a change of pole by at least half', () => {
    // Respectful 40% at the reply before, toxic 30% at this one: 70% apart, on opposite poles.
    const swing = findBehaviorSwing([toxic], [overTurns('mit_toxic', [-0.4, 0.3])], 1);
    expect(swing?.from.key).toBe('mit_toxic:minus');
    expect(swing?.to.key).toBe('mit_toxic:plus');
    expect(swing?.separation).toBeCloseTo(0.7);
  });

  it('counts a swing in either direction', () => {
    const swing = findBehaviorSwing([toxic], [overTurns('mit_toxic', [0.3, -0.4])], 1);
    expect(swing?.from.name).toBe('Toxic');
    expect(swing?.to.name).toBe('Respectful');
  });

  it('ignores a change of pole smaller than the threshold', () => {
    expect(findBehaviorSwing([toxic], [overTurns('mit_toxic', [-0.2, 0.2])], 1)).toBeNull();
  });

  it('ignores a large change that stays on one pole', () => {
    expect(findBehaviorSwing([toxic], [overTurns('mit_toxic', [-0.1, -0.9])], 1)).toBeNull();
  });

  it('never swings into the first reply, or from a reply not measured', () => {
    expect(findBehaviorSwing([toxic], [overTurns('mit_toxic', [0.9])], 0)).toBeNull();
    expect(findBehaviorSwing([toxic], [overTurns('mit_toxic', [null, 0.9])], 1)).toBeNull();
  });

  it('names the largest swing when more than one axis swings', () => {
    const empathy = definition('mit_empathy', { polePositive: 'empathetic', poleNegative: 'unempathetic' });
    const swing = findBehaviorSwing(
      [toxic, empathy],
      [overTurns('mit_toxic', [-0.3, 0.3]), overTurns('mit_empathy', [0.5, -0.4])],
      1,
    );
    expect(swing?.from.name).toBe('Empathetic');
    expect(swing?.to.name).toBe('Unempathetic');
    expect(swing?.separation).toBeCloseTo(0.9);
  });
});
