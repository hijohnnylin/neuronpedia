/**
 * Tailwind needs literal pixel values in its classes, so the layout keeps some widths twice: as a
 * number for arithmetic, and as a class. These tests make sure the two agree.
 */

import { describe, expect, it } from 'vitest';
import { LAYOUT_CHECK } from './neural-transparency-messages';
import { BADGES_CLASS, TRAIT_BADGE_WIDTH, TRAIT_BADGE_WIDTH_PHONE } from './persona-sunburst';

/** The pixel values in a class list, in order: phone first, then `sm`. */
function pixels(classes: string): number[] {
  return Array.from(classes.matchAll(/(\d+)px/g), (match) => Number(match[1]));
}

describe('layout classes', () => {
  it('give the gutter the computed widths', () => {
    expect(pixels(LAYOUT_CHECK.gutterClass)).toEqual(LAYOUT_CHECK.gutter);
  });

  it('give the monitor column the computed widths', () => {
    expect(pixels(LAYOUT_CHECK.monitorColumnClass)).toEqual(LAYOUT_CHECK.monitorColumn);
  });

  it('hold a question clear of the readouts', () => {
    expect([...pixels(LAYOUT_CHECK.questionRowClass), 0]).toEqual(LAYOUT_CHECK.questionRow);
  });

  it('start the system prompt where the reply text starts', () => {
    expect(pixels(LAYOUT_CHECK.systemRowClass)).toEqual(LAYOUT_CHECK.systemRow);
  });

  it('give the badge column the badge widths', () => {
    expect(pixels(BADGES_CLASS)).toEqual([TRAIT_BADGE_WIDTH_PHONE, TRAIT_BADGE_WIDTH]);
  });
});
