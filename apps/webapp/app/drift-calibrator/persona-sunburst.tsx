'use client';

/**
 * The sunburst: one conversation turn's reading on every trait axis, drawn as a two-ring polar
 * chart. Ported from `interface/js/persona-sunburst.js` in the multiturn-neural-transparency
 * repository, which is D3 and drives an imperative container. The geometry and the colors are that
 * file's; the arc paths are computed here rather than with `d3.arc` so this stays a plain React
 * component like `persona-chart.tsx` beside it.
 *
 * Two things the radial layout is for, both from that paper. A wheel has no top row, so no trait
 * is read first the way it would be in a bar list. And because each axis' two poles sit at equal
 * offsets either side of twelve o'clock, a shift along one axis tilts the whole silhouette --
 * the outline over a conversation is the message, more than any single bar.
 *
 * Our readouts are one signed value per axis, so a value is split into the pole it favours (which
 * gets its magnitude) and the pole it does not (which gets zero). That is the same split the
 * original consumed, done here rather than upstream.
 */

import type { PersonaAxisDefinition } from '@/lib/utils/persona-axis';
import type { PublicAxisReadout } from '@/lib/utils/steer-axis-public';
import { X } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { TRAIT_EXAMPLES, TraitExample } from './neural-transparency-snippets';
import { NT_SHOW_TRAIT_EXAMPLES } from './shared';

// Proportions of `radius`. The inner circles are deliberately small so the outer bars have room to
// show differences in activation, and these are a little tighter than the original's 0.2 and 0.33:
// the centre is a legend rather than a reading, so what it costs is band the bars could have had.
// The gap between the two is the category ring's thickness, and it only has to be thick enough to
// be a color.
const INNER_RATIO = 0.17;
const MIDDLE_RATIO = 0.28;
const MAX_OUTER_RATIO = 0.85;

// How much of the box the wheel itself takes. Labelled, it has to leave room for the spokes of
// text; bare, it can nearly fill the box.
const RADIUS_DIVISOR_LABELLED = 3.1;
const RADIUS_DIVISOR_BARE = 2.15;

// How far past `MAX_OUTER_RATIO` the rim sits, as a share of the band. From the original, where it
// was what let a bar reach the outer ring at a reading of 0.8; here the rim is where a full reading
// ends, so it only sets how large the wheel comes out.
const GROWTH_MULTIPLIER = 1.25;

// Radius grows as `value ** SCALE_EXPONENT`, not with the value itself: the first half of a
// reading's range takes about two thirds of the way out to the rim and the second half the
// remaining third. Readings cluster low, so a linear radius spends most of the wheel on a range
// nothing occupies, and the distance that decides whether a wheel looks alarming is the one
// between 0.05 and 0.3 rather than the one between 0.8 and 1.0.
const SCALE_EXPONENT = 0.6;

// Hover: the wedge under the pointer grows about the wheel's centre and drops a shade of lightness,
// both eased so the wheel reads as answering the pointer rather than jumping. Kept small because
// twelve wedges meet edge to edge, and a bar that grew much more would look like the whole wheel
// had moved.
const HOVER_SCALE = 1.04;
const HOVER_DARKEN = 8;

/**
 * What a bar reads at when nothing is under the pointer, and the share of that it keeps while some
 * other slice is.
 *
 * Pointing at one of twelve wedges is a question about that one, and the answer is easier to see
 * with the other eleven out of the way. Faded rather than hidden, because the reading being asked
 * about is a length compared with its neighbours: an empty wheel with one bar on it has no scale.
 */
const BAR_REST_OPACITY = 0.9;
const DIM_FACTOR = 0.4;

/**
 * How much lighter a bar is than the category ring it grows out of.
 *
 * One step, not a ramp: the bars have to part from the ring, since a bar in the ring's own color
 * reads as the ring having thickened rather than as a measurement, but a step much larger than
 * this and they stop reading as the same family of color at all.
 */
const BAR_LIGHTEN = 10;

/**
 * Gap between one slice and the next, as a share of `radius`, so it comes out the same size
 * against the wheel whatever size the wheel is drawn at.
 *
 * The bars meet edge to edge without it, and one shade of lightness is not enough to divide two
 * neighbours of the same category that read alike: at the size a wheel sits beside a reply they
 * merge into a single wider bar, which is a different reading. The gap says where one ends.
 */
const SLICE_GAP_RATIO = 0.02;

/** Long enough to read as a fade, short enough that the wheel does not lag the pointer. */
const HOVER_EASE = '120ms ease-out';

/**
 * How a readout arrives: the bars grow out of the category ring, the names fade in beside them.
 *
 * A measurement lands in one step at the end of a turn, so without this the whole wheel switches
 * on between two frames -- which is read as the interface glitching rather than as an answer, and
 * gives no sense of which bars are the long ones. Slower than the hover ease because this is the
 * page telling the reader something arrived, not the page answering the pointer. The curve is the
 * same one the tooltips use, which ends slowly and so settles rather than stopping.
 */
const REVEAL_EASE = '550ms cubic-bezier(0.16, 1, 0.3, 1)';

/**
 * Fade for a trait name, over the same moment the bars grow. Written as an arbitrary property
 * because the keyframe is one this page defines, in `app/globals.css`.
 */
const BADGE_REVEAL = '[animation:persona-badge-in_400ms_ease-out_both]';

/** The loading sweep, and the shade a slice takes. slate-400 and -300. */
const PULSE_FILL = '#94a3b8';
const SLICE_HOVER_FILL = '#cbd5e1';

/**
 * How much of `SLICE_HOVER_FILL` a slice shows when the pointer is elsewhere.
 *
 * Half, so a slice reads as a track the bar in it is measured against rather than as a shape of its
 * own, and the one under the pointer still stands out at full strength. The bars are drawn over
 * these, so this also has to stay light enough to leave a low reading legible.
 */
const SLICE_REST_OPACITY = 0.5;

/** The tint the left of the page is drawn on, wherever it is drawn. */
export const GUTTER_TINT = 'bg-slate-200/50';

/**
 * The padding around a readout. It has no card of its own: the gutter tint runs the whole height
 * of the transcript behind it, so the left of the page reads as one instrument.
 *
 * All of the padding is on the right. The chart is on the left, and a dot on it and the wheel
 * level with it are one reading drawn twice, so they sit close enough to read as a pair. The
 * padding sets the gutter's width, so it must agree with `GUTTER_WIDTH`.
 */
const CARD_CLASS = 'pb-0 pl-0 pr-1 pt-0.5 sm:pr-4';

/** Side of a readout's wheel, in viewBox units. It is drawn at this size from `sm` up. */
const WHEEL_SIZE = 120;
/** Its size on screen: smaller on a phone. */
const WHEEL_CLASS = 'h-[80px] w-[80px] sm:h-[120px] sm:w-[120px]';

/**
 * `GUTTER_TINT` flattened against the panel it is laid over: slate-200 at half over slate-50.
 *
 * The hub is there to hide the bars behind it, so it cannot be translucent, and it has to be the
 * exact shade of what surrounds it or it reads as a disc in the middle of the wheel. One value
 * covers both layouts because the card and the gutter are the same tint over the same panel. A
 * readout drawn on anything else would need a tinted element under it rather than a matched fill.
 */
export const CARD_FILL = '#edf1f6';

/** One turn of the loading pulse described at the sweep below. */
const PULSE_DURATION = '1.4s';

export type Category = 'desirable' | 'neutral' | 'undesirable';

/**
 * Inner-ring colors, as HSL so the bars and the pills can be stated as steps of lightness off
 * them rather than as their own values.
 *
 * The green and the red are the original's. The stylistic ring is not: the original's was a true
 * grey, `#9E9E9E`, which beside the slate card and the slate bubbles of this page reads as a
 * different family of color rather than as the quiet one. Slate's hue and saturation at the same
 * lightness, so it still sits between the other two the way a neutral should.
 */
export const CATEGORY_HSL: Record<Category, { h: number; s: number; l: number }> = {
  desirable: { h: 122, s: 39, l: 49 }, // #4CAF50
  neutral: { h: 214, s: 28, l: 62 }, // slate, in place of the original's #9E9E9E
  undesirable: { h: 4, s: 90, l: 58 }, // #F44336
};

const CATEGORY_LABEL: Record<Category, string> = {
  desirable: 'Desirable',
  neutral: 'Stylistic',
  undesirable: 'Harmful',
};

/**
 * Which pole of an axis you would rather see, or that neither is.
 *
 * The one thing about an axis this page decides for itself. It is an editorial call about a trait
 * rather than a property of a fit -- an axis someone else brings has no reason to carry ours -- so
 * neither the table nor the inference API has any notion of a desirable pole, and this demo keeps
 * its judgement here, keyed by the axis name.
 *
 * The original decided it by matching pole names against word lists, which put `romantic` among
 * the harmful traits and `platonic` among the desirable ones by way of a substring. Naming the
 * axes keeps the same judgement without it being a property of how a pole happens to be spelled.
 * An axis not listed here is drawn as stylistic, which is the safe default: grey, rather than
 * calling a trait good or bad on a guess.
 */
const DESIRABLE_POLE: Record<string, 'plus' | 'minus' | 'neither'> = {
  mit_empathy: 'plus',
  mit_erudite: 'neither',
  mit_robotic: 'neither',
  mit_romantic: 'minus',
  mit_sycophantic: 'minus',
  mit_toxic: 'minus',
};

function titleCase(name: string): string {
  return name.replace(/_/g, ' ').replace(/(^|[\s-])([a-z])/g, (_, boundary, letter) => boundary + letter.toUpperCase());
}

/** One pole of one axis: a single wedge of the outer ring. */
export type Pole = {
  key: string;
  name: string;
  /** What this pole means, as the axis row states it, where the row states it at all. */
  definition: string | null;
  /**
   * Replies the fit was taken from, at this end of the axis. Empty for an axis this page has no
   * rollouts for, which is what an axis someone else brings will be.
   */
  examples: TraitExample[];
  /**
   * Magnitude of the reading on this pole, 0 when the axis favours the other one, and what
   * everything on screen is drawn and labelled from.
   *
   * The percentile where the axis ships one, so it cannot exceed 1: a wheel is read as a dial and
   * "102%" reads as a broken one, however true it is. An axis with no quantile tables falls back to
   * the raw measurement -- how many of its own spreads from centre the turn sat -- which is then
   * the only reading it has.
   */
  value: number;
  /** Whether `value` is a percentile, rather than the raw measurement an unranked axis reports. */
  ranked: boolean;
  /** False before this turn has been measured, which is drawn as an empty wedge. */
  measured: boolean;
  category: Category;
  layer: number | null;
  caveat: string | null;
  /** Wedge boundaries, radians clockwise from twelve o'clock. */
  startAngle: number;
  endAngle: number;
};

type Sector = { category: Category; startAngle: number; endAngle: number };

/**
 * What the pointer is on, which turn it belongs to, and where on screen to hang the tooltip.
 *
 * One hover for the whole transcript rather than one per readout. Three things draw the same
 * poles -- the badge column, the wheel, and the trait monitor running down the left of the page
 * -- and pointing at any of them has to light up the other two, so none of the three can own
 * this and the transcript that holds them all does. `turnIndex` is what makes that reachable: it
 * says which reply a reading belongs to, so a dot on the monitor can name a wedge in a wheel
 * without either knowing about the other.
 */
export type TooltipAlign = 'center' | 'right';

export type TraitHover = {
  turnIndex: number;
  pole: Pole;
  x: number;
  y: number;
  /**
   * Shown by the page rather than pointed at, which is the hint a readout opens with. Tracked
   * because a hint has to be taken down again, and a real hover takes itself down on mouseleave.
   */
  auto?: boolean;
  /** From a dot on the monitor, which sits away from its reply, so the tooltip quotes the reply. */
  withReply?: boolean;
  /** The wheel's top and bottom edges, for a hover on a wheel. The tooltip then keeps clear of it. */
  chart?: { top: number; bottom: number };
  /** Centred on `x`, or with its right edge on `x`, and no notch: every slice puts it in one place. */
  align?: TooltipAlign;
} | null;

/**
 * Whether this turn's readout has arrived yet.
 *
 * The measurement is taken on the finished turn, so a reply sits on screen for a moment with
 * nothing measured about it. The wheel and the badges both key their loading state off this, so
 * the pulse and the skeletons appear and clear together. A turn whose measurement never arrives
 * therefore reads as still pending, which is what it is.
 */
export function isTurnMeasured(readouts: PublicAxisReadout[], turnIndex: number): boolean {
  if (turnIndex < 0) {
    return false;
  }
  return readouts.some((readout) => {
    const value = readout.turns?.[turnIndex]?.value;
    return typeof value === 'number' && Number.isFinite(value);
  });
}

/**
 * Place every pole on the wheel, mirrored about the vertical axis.
 *
 * Each axis contributes two wedges of equal width, its desirable pole on the right half and the
 * pole opposite it at the mirrored angle on the left. Axes where neither pole is preferable
 * straddle the bottom instead, so the two halves stay reflections of each other and the inner
 * ring comes out as three arcs sized by how many poles each holds.
 *
 * The frame is `definitions`, not the readouts: a turn that has not been measured yet still has to
 * draw its wheel, so the empty one beside a streaming reply is the same shape as the one that
 * replaces it, and a wheel ordered by however inference happened to answer would rearrange itself
 * between turns -- which is exactly the comparison a radial layout exists to make. A readout with
 * no definition beside it is therefore not drawn, and an axis with no readout yet is.
 *
 * Everything an axis says about itself -- both pole names, what each means, the layer, the caveat
 * -- comes from its definition rather than from the readout, which reports the same facts. One
 * source means a label cannot change under a reader when the numbers land.
 */
export function layoutPoles(
  definitions: PersonaAxisDefinition[],
  readouts: PublicAxisReadout[],
  turnIndex: number,
): { poles: Pole[]; sectors: Sector[] } {
  const axes = definitions.map((definition) => {
    const readout = readouts.find((candidate) => candidate.id === definition.name);
    const turn = turnIndex >= 0 ? readout?.turns?.[turnIndex] : undefined;
    const measured = typeof turn?.value === 'number' && Number.isFinite(turn.value);
    // The bounded reading when the axis ships the tables for one, and the raw measurement
    // otherwise. `ranked` is what tells the tooltip which of the two it is looking at, since the
    // two do not mean the same thing and only one of them is a percentage of anything.
    const ranked = typeof turn?.percentile === 'number' && Number.isFinite(turn.percentile);
    // A row need not name its poles -- a probe or a plain steering vector reads to the same number
    // with no named ends -- so both fall back to the axis's own title, which is what every other
    // display of one does. A wedge with no label at all would leave a nameless bar on the wheel.
    const title = definition.displayName ?? definition.name;
    return {
      id: definition.name,
      minus: definition.poleNegative ?? title,
      plus: definition.polePositive ?? title,
      minusDefinition: definition.poleNegativeDescription,
      plusDefinition: definition.polePositiveDescription,
      desirable: DESIRABLE_POLE[definition.name] ?? 'neither',
      value: ranked ? (turn?.percentile as number) : measured ? (turn?.value as number) : 0,
      ranked,
      measured,
      layer: definition.layer,
      caveat: definition.caveat,
    };
  });

  // Signed axes first so they take the wedges nearest twelve o'clock, stylistic ones after.
  const signed = axes.filter((axis) => axis.desirable !== 'neither');
  const stylistic = axes.filter((axis) => axis.desirable === 'neither');
  const nWedges = 2 * axes.length;
  if (nWedges === 0) {
    return { poles: [], sectors: [] };
  }
  const wedge = (2 * Math.PI) / nWedges;

  const poles: Pole[] = [];
  const push = (axis: (typeof axes)[number], which: 'plus' | 'minus', category: Category, wedgeIndex: number): void => {
    const isPlus = which === 'plus';
    poles.push({
      key: `${axis.id}:${which}`,
      name: titleCase(isPlus ? axis.plus : axis.minus),
      definition: (isPlus ? axis.plusDefinition : axis.minusDefinition) ?? null,
      examples: TRAIT_EXAMPLES[axis.id]?.[which] ?? [],
      value: isPlus ? Math.max(axis.value, 0) : Math.max(-axis.value, 0),
      ranked: axis.ranked,
      measured: axis.measured,
      category,
      layer: axis.layer,
      caveat: axis.caveat,
      startAngle: wedgeIndex * wedge,
      endAngle: (wedgeIndex + 1) * wedge,
    });
  };

  signed.forEach((axis, i) => {
    const desirableIsPlus = axis.desirable === 'plus';
    // Wedge i on the right, and its reflection across the vertical axis on the left.
    push(axis, desirableIsPlus ? 'plus' : 'minus', 'desirable', i);
    push(axis, desirableIsPlus ? 'minus' : 'plus', 'undesirable', nWedges - 1 - i);
  });
  stylistic.forEach((axis, j) => {
    // The plus pole takes the right of the bottom, arbitrarily but consistently: with neither
    // pole preferable there is nothing to align it to, and a stable side keeps the wheel
    // comparable between turns.
    push(axis, 'plus', 'neutral', signed.length + j);
    push(axis, 'minus', 'neutral', nWedges - 1 - signed.length - j);
  });

  // Inner ring: the run of wedges each category ended up occupying.
  const sectors: Sector[] = [];
  if (signed.length > 0) {
    sectors.push({ category: 'desirable', startAngle: 0, endAngle: signed.length * wedge });
    sectors.push({
      category: 'undesirable',
      startAngle: (nWedges - signed.length) * wedge,
      endAngle: nWedges * wedge,
    });
  }
  if (stylistic.length > 0) {
    sectors.push({
      category: 'neutral',
      startAngle: signed.length * wedge,
      endAngle: (nWedges - signed.length) * wedge,
    });
  }
  return { poles, sectors };
}

/** Least separation, as a fraction, between two readings on opposite poles to call a swing. */
export const SWING_THRESHOLD = 0.5;

/** One axis that moved from one pole to the other between two replies. */
export type BehaviorSwing = {
  /** The pole the axis read on at the reply before, with that reading. */
  from: Pole;
  /** The pole it reads on at this reply, with this reading. */
  to: Pole;
  /** How far apart the two readings are: `from.value + to.value`. */
  separation: number;
};

/**
 * The largest swing into this reply, or null if no axis changed pole by `SWING_THRESHOLD` or more.
 *
 * An axis reads on one pole at a time, so it swings when its leading pole at this reply is not the
 * one it led with at the reply before. The first reply has nothing before it, so it never swings.
 */
export function findBehaviorSwing(
  definitions: PersonaAxisDefinition[],
  readouts: PublicAxisReadout[],
  turnIndex: number,
): BehaviorSwing | null {
  if (turnIndex < 1) {
    return null;
  }
  const before = layoutPoles(definitions, readouts, turnIndex - 1).poles;
  const after = layoutPoles(definitions, readouts, turnIndex).poles;
  const leading = (poles: Pole[], axisId: string): Pole | null => {
    const both = poles.filter((pole) => pole.key.startsWith(`${axisId}:`));
    const top = both.find((pole) => pole.measured && pole.value > 0);
    return top ?? null;
  };
  let best: BehaviorSwing | null = null;
  definitions.forEach((definition) => {
    const from = leading(before, definition.name);
    const to = leading(after, definition.name);
    if (!from || !to || from.key === to.key) {
      return;
    }
    const separation = from.value + to.value;
    if (separation >= SWING_THRESHOLD && (!best || separation > best.separation)) {
      best = { from, to, separation };
    }
  });
  return best;
}

/**
 * The pole a reply's hint names, out of `poles` at that turn: the swing into the reply if
 * `preferSwing` and there is one, else the headline trait.
 */
export function hintPole(
  definitions: PersonaAxisDefinition[],
  readouts: PublicAxisReadout[],
  turnIndex: number,
  poles: Pole[],
  preferSwing: boolean,
): Pole | undefined {
  const swing = preferSwing ? findBehaviorSwing(definitions, readouts, turnIndex) : null;
  return (swing && poles.find((pole) => pole.key === swing.to.key)) || namedPoles(poles)[0];
}

/** The radii the wheel is drawn from, in viewBox units. */
function wheelGeometry(size: number, showLabels: boolean) {
  const radius = size / (showLabels ? RADIUS_DIVISOR_LABELLED : RADIUS_DIVISOR_BARE);
  const middleRadius = radius * MIDDLE_RATIO;
  const band = radius * MAX_OUTER_RATIO - middleRadius;
  return {
    center: size / 2,
    radius,
    innerRadius: radius * INNER_RATIO,
    middleRadius,
    ringAt: (value: number) => middleRadius + value ** SCALE_EXPONENT * GROWTH_MULTIPLIER * band,
  };
}
/**
 * `measured`, but false for the frame a readout first appears in, so the bars have somewhere to
 * grow from.
 *
 * The growth is a CSS transition, which needs the short state painted before the long one is set.
 * A streamed reply gives that for free -- the wheel is on screen empty while the text arrives --
 * but a reply served from the cache does not: its text and its measurement are one update, so the
 * wheel mounts already measured and a transition driven by `measured` alone has nothing to run
 * from. The bars then appear at full length, which is the one thing the reveal exists to avoid.
 *
 * Two frames rather than one: the first only guarantees the commit, and it is the paint of that
 * commit the transition eases away from.
 */
function useRevealed(measured: boolean): boolean {
  const [revealed, setRevealed] = useState(false);
  useEffect(() => {
    if (!measured) {
      setRevealed(false);
      return undefined;
    }
    let inner = 0;
    const outer = requestAnimationFrame(() => {
      inner = requestAnimationFrame(() => setRevealed(true));
    });
    return () => {
      cancelAnimationFrame(outer);
      cancelAnimationFrame(inner);
    };
  }, [measured]);
  return revealed;
}

/**
 * The wheel alone.
 *
 * Hover is a prop rather than state here because the badge column beside it points at the same
 * poles: one owner means hovering either lights up both, and the tooltip is rendered once by
 * whoever owns it. :func:`PersonaReadout` below is that owner.
 */
export default function PersonaSunburst({
  axes,
  readouts,
  turnIndex,
  /**
   * Side of the square the chart is drawn into, and the units the label font sizes below are
   * written in. Set it to the size the wheel will actually occupy and those come out as pixels;
   * the SVG still scales to its container from there.
   */
  size = 600,
  showLabels = true,
  labelsInside = false,
  className,
  hoveredKey = null,
  onHover,
}: {
  /** The axes the wheel is made of, in the order they are placed around it. */
  axes: PersonaAxisDefinition[];
  readouts: PublicAxisReadout[];
  /** Which assistant turn to show. Defaults to the most recent one measured. */
  turnIndex?: number;
  size?: number;
  showLabels?: boolean;
  /**
   * Name and percent along each slice, from the hub out past the rim. For a wheel too narrow for
   * `showLabels`, which needs room outside the ring. Use with `showLabels={false}`.
   */
  labelsInside?: boolean;
  className?: string;
  /** The pole lit up, wherever the pointer that named it happens to be. */
  hoveredKey?: string | null;
  onHover?: (pole: Pole | null) => void;
}) {
  const turns = Math.max(0, ...readouts.map((readout) => readout.turns?.length ?? 0));
  const shown = turnIndex ?? turns - 1;

  const { poles, sectors } = useMemo(() => layoutPoles(axes, readouts, shown), [axes, readouts, shown]);
  const setHovered = (pole: Pole | null) => onHover?.(pole);
  const measured = isTurnMeasured(readouts, shown);
  const revealed = useRevealed(measured);

  // The hovered wedge is drawn last. SVG has no z-index, so a wedge grown into the one beside it
  // would otherwise be painted over along that edge and look clipped on one side rather than
  // raised. Keyed on `pole.key` below, so this reorders the nodes without remounting them and the
  // transition still plays.
  const drawOrder = [
    ...poles.filter((pole) => pole.key !== hoveredKey),
    ...poles.filter((pole) => pole.key === hoveredKey),
  ];

  const { center, radius, innerRadius, middleRadius, ringAt } = wheelGeometry(size, showLabels);

  const pointAt = (r: number, angle: number): [number, number] => [
    center + r * Math.sin(angle),
    center - r * Math.cos(angle),
  ];

  /**
   * The gap, as an angle rather than a width, so a slice keeps radial edges: taken off both ends
   * of every slice, which is what makes it the same gap between any two neighbours.
   *
   * Set from the middle of the band, where a boundary is read, so it measures about
   * `SLICE_GAP_RATIO` of the radius there -- tighter towards the hub and wider towards the rim,
   * which is the shape a fixed angle gives and is the right way round: the bars are narrow near
   * the hub, where too wide a gap would eat them.
   */
  const padAngle = (radius * SLICE_GAP_RATIO) / ((middleRadius + ringAt(1)) / 2);

  /**
   * Path for a slice of an annulus, angles clockwise from twelve o'clock. `pad` is taken off each
   * end, and clamped so a narrow slice thins rather than turning inside out.
   */
  const annulusSector = (
    inner: number,
    outer: number,
    startAngle: number,
    endAngle: number,
    pad: number = 0,
  ): string => {
    const half = Math.min(pad, (endAngle - startAngle) * 0.8) / 2;
    const [start, end] = [startAngle + half, endAngle - half];
    const large = end - start > Math.PI ? 1 : 0;
    const [x0, y0] = pointAt(outer, start);
    const [x1, y1] = pointAt(outer, end);
    const [x2, y2] = pointAt(inner, end);
    const [x3, y3] = pointAt(inner, start);
    return [
      `M ${x0} ${y0}`,
      `A ${outer} ${outer} 0 ${large} 1 ${x1} ${y1}`,
      `L ${x2} ${y2}`,
      `A ${inner} ${inner} 0 ${large} 0 ${x3} ${y3}`,
      'Z',
    ].join(' ');
  };

  if (poles.length === 0) {
    return null;
  }

  return (
    <svg
      viewBox={`0 0 ${size} ${size}`}
      preserveAspectRatio="xMidYMid meet"
      className={className}
      role="img"
      aria-label="Trait activations for this turn"
    >
      {/* Each pole's full range, as a track the bar is read against: how long this bar could get,
          drawn whether or not the turn favours this pole. Faint at rest and at full strength under
          the pointer, which is one element at two opacities rather than a shade that appears on
          hover -- a path mounted at the moment of hover has no previous opacity to ease from, so it
          would snap in.

          Padded like the bars are, so the twelve stay twelve: at rest a full circle of tracks with
          no gaps would read as another ring of the chart rather than as the slices it is. */}
      {poles.map((pole) => (
        <path
          key={`${pole.key}:range`}
          d={annulusSector(middleRadius, ringAt(1), pole.startAngle, pole.endAngle, padAngle)}
          fill={SLICE_HOVER_FILL}
          opacity={pole.key === hoveredKey ? 1 : SLICE_REST_OPACITY * (hoveredKey ? DIM_FACTOR : 1)}
          style={{ transition: `opacity ${HOVER_EASE}` }}
        />
      ))}

      {/* The turn being measured, as a sweep out to where a full reading would reach: it grows
          from the outside of the category ring to the rim and fades, on repeat. It covers the band
          the bars will occupy, which is what the wheel is waiting on -- starting from the centre
          instead spent the first third of every pulse inside the hub, where there is nothing to
          say. Animated with SMIL rather than CSS because `r` is the thing that has to move, and
          animating the radius keeps the pulse concentric with the rest of the wheel -- a scaled
          div could not be. Opacity reaches 0 exactly as the radius wraps, so the restart is
          invisible.

          Over the tracks rather than under them, and a shade darker than they are. Under them it
          read as almost nothing: a track is slate at half, so it takes most of the sweep out of
          whatever passes beneath, and the pulse showed mainly in the gaps between slices.

          Drawn only while waiting. Nothing is under the pointer or measured yet, so it has the
          band to itself. */}
      {!measured && (
        <circle cx={center} cy={center} r={ringAt(1)} fill={PULSE_FILL}>
          <animate
            attributeName="r"
            values={`${middleRadius};${ringAt(1)}`}
            dur={PULSE_DURATION}
            repeatCount="indefinite"
          />
          <animate
            attributeName="opacity"
            values="0.55;0.55;0"
            keyTimes="0;0.5;1"
            dur={PULSE_DURATION}
            repeatCount="indefinite"
          />
        </circle>
      )}

      {/* The measurement, as growth out of the ring rather than as twelve bars appearing between
          two frames.

          The whole set is drawn at full length and scaled about the wheel's centre, from small
          enough that the longest possible bar ends at the ring -- so what a reader sees is each
          bar coming out from under the ring and stopping where its own reading does. The disc
          below is what makes that true rather than nearly true: scaled down, a bar's inner end
          falls inside the hub, where it would otherwise show as a stub in the middle of the
          wheel.

          A scale on a group, rather than a clip that opens: an element inside `clipPath` is in a
          subtree the browser does not render, where a CSS transition is not guaranteed to run at
          all. This is the same transform the wedges use for hover, one level up. */}
      <g
        style={{
          transformBox: 'view-box',
          transformOrigin: '50% 50%',
          transform: `scale(${revealed ? 1 : middleRadius / ringAt(1)})`,
          transition: `transform ${REVEAL_EASE}`,
        }}
      >
        {drawOrder.map((pole) => {
          const base = CATEGORY_HSL[pole.category];
          // Length alone carries the reading, at one shade above the category ring whatever it
          // reads. Lightness used to ramp with the value as well, which said the same thing twice
          // and cost more than it added: it put the low readings -- where most turns sit -- at a
          // pale shade barely separable from the card, so a real but modest trait looked like
          // nothing at all. `capped` still bounds the radius, which bites only for an axis with no
          // quantile tables, whose reading is a ratio and can genuinely run past the rim.
          const capped = Math.min(pole.value, 1);
          const isHovered = pole.key === hoveredKey;
          const lightness = base.l + BAR_LIGHTEN - (isHovered ? HOVER_DARKEN : 0);
          return (
            <path
              key={pole.key}
              d={annulusSector(middleRadius, ringAt(capped), pole.startAngle, pole.endAngle, padAngle)}
              fill={`hsl(${base.h} ${base.s}% ${lightness}%)`}
              opacity={isHovered ? 1 : BAR_REST_OPACITY * (hoveredKey ? DIM_FACTOR : 1)}
              style={{
                // `view-box` so the origin below is the middle of the wheel, which is what makes
                // the wedge grow along its own axis instead of away from its own bounding box.
                transformBox: 'view-box',
                transformOrigin: '50% 50%',
                transform: `scale(${isHovered ? HOVER_SCALE : 1})`,
                transition: `transform ${HOVER_EASE}, fill ${HOVER_EASE}, opacity ${HOVER_EASE}`,
              }}
            />
          );
        })}
      </g>

      {/* The hub, in the card's own shade and so invisible either way. Out to the category ring
          once the turn is measured, where its job is to hide the bars while they are scaled down
          -- which is why it is drawn after them. While waiting it covers the middle only: the
          pulse behind it is a disc rather than a ring, so without this the hole in the centre of
          the wheel would pulse along with the band. */}
      <circle cx={center} cy={center} r={measured ? middleRadius : innerRadius} fill={CARD_FILL} />

      {/* The category ring, over the bars for the same reason -- it is the thing they grow out
          from, so it has to be the thing in front. Neither overlaps the other at rest. */}
      {sectors.map((sector) => {
        const { h, s, l } = CATEGORY_HSL[sector.category];
        return (
          <path
            key={sector.category}
            d={annulusSector(innerRadius, middleRadius, sector.startAngle, sector.endAngle)}
            fill={`hsl(${h} ${s}% ${l}%)`}
            opacity={0.9}
          >
            <title>{CATEGORY_LABEL[sector.category]}</title>
          </path>
        );
      })}

      {/* Hit areas, spanning the full ring whatever a pole reads. A pole at 0% draws a wedge
          with no thickness and so nothing to point at, which is most of a wheel: every axis has
          one pole at zero by construction, and an unmeasured turn has twelve of them. Drawn
          over the wedges rather than under, so the target is the same size everywhere. Without
          the gap the wedges are drawn with, so there is no dead strip between two neighbours. */}
      {poles.map((pole) => (
        <path
          key={`${pole.key}:hit`}
          d={annulusSector(middleRadius, ringAt(1), pole.startAngle, pole.endAngle)}
          fill="transparent"
          className="cursor-pointer"
          onMouseEnter={() => setHovered(pole)}
          onMouseLeave={() => setHovered(null)}
        />
      ))}

      {showLabels &&
        poles.map((pole) => {
          const mid = (pole.startAngle + pole.endAngle) / 2;
          const [x, y] = pointAt(ringAt(1) + radius * 0.045, mid);
          let rotation = (mid * 180) / Math.PI - 90;
          let anchor: 'start' | 'end' = 'start';
          // A label on the left half would come out upside down, so turn it over and let it
          // read inward from its far end instead.
          if (rotation > 90 && rotation < 270) {
            rotation += 180;
            anchor = 'end';
          }
          return (
            <g
              key={pole.key}
              transform={`translate(${x}, ${y}) rotate(${rotation})`}
              onMouseEnter={() => setHovered(pole)}
              onMouseLeave={() => setHovered(null)}
            >
              <text textAnchor={anchor} dominantBaseline="middle" style={{ userSelect: 'none' }}>
                <tspan fontSize={Math.max(10, radius * 0.028)} fontWeight={500} fill="#334155">
                  {pole.name}
                  {pole.caveat ? '*' : ''}
                </tspan>
                <tspan fontSize={Math.max(8, radius * 0.024)} fontWeight={500} fill="#64748b">
                  {pole.measured ? ` ${(pole.value * 100).toFixed(0)}%` : ''}
                </tspan>
              </text>
            </g>
          );
        })}

      {labelsInside &&
        poles.map((pole) => {
          const mid = (pole.startAngle + pole.endAngle) / 2;
          const [x, y] = pointAt((middleRadius + center) / 2, mid);
          let rotation = (mid * 180) / Math.PI - 90;
          // As above: turned over on the left half so no label reads upside down.
          if (rotation > 90 && rotation < 270) {
            rotation += 180;
          }
          const fontSize = Math.max(8, radius * 0.08);
          return (
            <text
              key={`${pole.key}:label`}
              transform={`translate(${x}, ${y}) rotate(${rotation})`}
              textAnchor="middle"
              style={{ userSelect: 'none', pointerEvents: 'none' }}
            >
              <tspan x={0} dy={-0.15 * fontSize} fontSize={fontSize} fontWeight={600} fill="#334155">
                {pole.name}
                {pole.caveat ? '*' : ''}
              </tspan>
              <tspan x={0} dy={1.1 * fontSize} fontSize={fontSize * 0.9} fontWeight={500} fill="#64748b">
                {pole.measured ? `${(pole.value * 100).toFixed(0)}%` : ''}
              </tspan>
            </text>
          );
        })}
    </svg>
  );
}

/**
 * How many traits a turn is named by, strongest first. Always this many, with no floor on how
 * strongly a trait has to read to be named: the same slots filled every turn is what makes the
 * column scannable down the transcript, and a turn whose top trait is weak says so with a low
 * number rather than by going blank.
 */
const TRAIT_BADGE_LIMIT = 3;

/** Pixels. Fixed, so the column is the same shape whatever the trait names turn out to be. */
export const TRAIT_BADGE_WIDTH = 120;
/** The same, on a phone, where the pills show no percent. Keep in step with `BADGES_CLASS`. */
export const TRAIT_BADGE_WIDTH_PHONE = 80;
export const BADGES_CLASS = 'w-[80px] sm:w-[120px]';

/**
 * The poles a turn is named by, strongest first.
 *
 * Ranked by the percentile rather than the ratio, which also makes the ordering fairer: a ratio is
 * in units of its own axis's spread, so comparing two of them ranks the narrowly calibrated axis
 * above the rest. A percentile is the same question of every axis.
 *
 * One function because two things read this order -- the column, and the hint a readout opens with,
 * which names the first of these. Ranked separately they could disagree about which trait is the
 * turn's headline while both being right.
 */
export function namedPoles(poles: Pole[]): Pole[] {
  return poles
    .filter((pole) => pole.measured)
    .sort((a, b) => b.value - a.value)
    .slice(0, TRAIT_BADGE_LIMIT);
}

/**
 * One class list for all three states a slot can be in -- named, loading, empty -- so a slot is
 * the same height in each and nothing beside the column moves when a readout lands.
 *
 * Not text to be read out of: dragging across the column highlights nothing. The cursor is left to
 * the two callers, since a filled pill is a control and an empty slot waiting for a reading is not.
 */
const BADGE_CLASS =
  'flex w-full select-none flex-row items-center justify-between gap-x-1 rounded-full px-1.5 py-0.5 sm:px-2.5 text-[10px] font-medium leading-tight';

/**
 * Pill fill lightness per category, at rest and under the pointer.
 *
 * One ramp for all three does not work: at equal lightness the red carries much further than the
 * green or the slate, so a single number leaves it shouting while the other two wash out against
 * the page. The red is where it should be, and the other two are brought down to meet it.
 */
const BADGE_LIGHTNESS: Record<Category, { rest: number; active: number }> = {
  desirable: { rest: 86, active: 78 },
  neutral: { rest: 86, active: 78 },
  undesirable: { rest: 93, active: 84 },
};

/**
 * One named pole, as a pill. Used both in the column and as the tooltip's own heading, so the
 * thing the pointer is on and the thing the tooltip is about are the same object twice.
 */
export function TraitPill({
  pole,
  active = false,
  className,
  width = TRAIT_BADGE_WIDTH,
  phoneValue = true,
  onMouseEnter,
  onMouseLeave,
}: {
  pole: Pole;
  /** Under the pointer, here or on the wheel. Deepens the fill, matching what the wedge does. */
  active?: boolean;
  className?: string;
  /** In pixels, `'auto'` to fit the name and percent, or `'100%'` to fill the parent. */
  width?: number | 'auto' | '100%';
  /** Show the percent on a phone too. */
  phoneValue?: boolean;
  onMouseEnter?: (event: React.MouseEvent) => void;
  onMouseLeave?: () => void;
}) {
  const { h, s } = CATEGORY_HSL[pole.category];
  const fill = BADGE_LIGHTNESS[pole.category];
  return (
    <div
      className={`${BADGE_CLASS} cursor-pointer ${className ?? ''}`}
      style={{
        width,
        backgroundColor: `hsl(${h} ${s}% ${active ? fill.active : fill.rest}%)`,
        color: `hsl(${h} ${s}% 28%)`,
        transition: `background-color ${HOVER_EASE}`,
      }}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      {/* `min-w-0` so a long pole name ellipsises inside the pill; a flex item will not shrink
          below its own text without it, however narrow the pill is. */}
      <span className="min-w-0 truncate">{pole.name}</span>
      <span className={phoneValue ? 'shrink-0' : 'hidden shrink-0 sm:inline'}>
        {pole.measured ? `${(pole.value * 100).toFixed(0)}%` : ''}
      </span>
    </div>
  );
}

/**
 * The poles worth naming for a turn, strongest first, as a column of pills.
 *
 * The wheel says which traits are up at a glance but cannot say their names at the size it is
 * drawn beside a reply: a legible label needs about seventy pixels of room outside the ring, more
 * than a wheel this size has in any direction. So the names live here instead, and the wheel keeps
 * carrying the shape.
 *
 * Always exactly `TRAIT_BADGE_LIMIT` slots tall. A turn is read against the turns above and below
 * it, which it cannot be if a quiet turn and a loud one put their readouts at different heights.
 * Before the readout lands the slots are skeletons, and a wheel with fewer axes than slots leaves
 * the remainder as empty space, rather than either collapsing.
 */
function TraitBadges({
  poles,
  measured,
  hoveredKey,
  onHover,
}: {
  poles: Pole[];
  measured: boolean;
  hoveredKey: string | null;
  /**
   * The pole under the pointer, not a point: a badge's tooltip is placed against the wedge it
   * names rather than against the badge, and only the owner of both knows where that wedge is.
   */
  onHover: (pole: Pole | null) => void;
}) {
  const named = namedPoles(poles);

  return (
    <div className={`flex shrink-0 flex-col items-center gap-y-1 ${BADGES_CLASS}`}>
      {Array.from({ length: TRAIT_BADGE_LIMIT }, (_, slot) => {
        const pole = named[slot];
        if (!pole) {
          return (
            <div
              key={slot}
              className={`${BADGE_CLASS} cursor-default ${measured ? 'invisible' : 'animate-pulse bg-slate-200'}`}
            >
              {'\u00a0'}
            </div>
          );
        }
        return (
          <TraitPill
            key={pole.key}
            pole={pole}
            active={pole.key === hoveredKey}
            // A keyframe rather than a transition: this pill takes the place of a skeleton, so it
            // is mounting, and a mounting element has no previous opacity to ease from.
            className={BADGE_REVEAL}
            width="100%"
            phoneValue={false}
            onMouseEnter={() => onHover(pole)}
            onMouseLeave={() => onHover(null)}
          />
        );
      })}
    </div>
  );
}

/**
 * One turn's readout: the traits worth naming, with the wheel under them.
 *
 * The two halves are one component because they are two views of the same poles: hovering a badge
 * lights up its wedge, and hovering a wedge lights up its badge. What the pointer is on lives a
 * level up, in the transcript, because the monitor down the left of the page is a third view of
 * these same poles and has to light them up too.
 */
export function PersonaReadout({
  axes,
  readouts,
  turnIndex,
  className,
  hovered,
  onHover,
  autoHint = true,
  hintSwing = false,
  onZoom,
  wheelId,
}: {
  /** The axes the wheel is made of, in the order they are placed around it. */
  axes: PersonaAxisDefinition[];
  readouts: PublicAxisReadout[];
  turnIndex: number;
  className?: string;
  /** The transcript's hover, which may name a pole on some other turn than this one. */
  hovered: TraitHover;
  onHover: (hover: TraitHover) => void;
  /** Off when the trait monitor is on screen, which names the headline at its dot instead. */
  autoHint?: boolean;
  /** The hint names the behavioral swing into this reply, if any, before the headline trait. */
  hintSwing?: boolean;
  /** A click on the wheel, with the wheel's box, so the owner can open it larger. */
  onZoom?: (wheelBox: DOMRect) => void;
  /** An id for the wheel itself, not the pills above it. */
  wheelId?: string;
}) {
  const { poles } = useMemo(() => layoutPoles(axes, readouts, turnIndex), [axes, readouts, turnIndex]);
  const wheel = useRef<HTMLDivElement>(null);
  const measured = isTurnMeasured(readouts, turnIndex);

  // Only a hover on this turn lights anything here up. Pointing at another reply's wheel, or at
  // another reply's dot on the monitor, leaves this one at rest.
  const mine = hovered?.turnIndex === turnIndex ? hovered : null;

  /**
   * Naming a pole shows the tooltip out on the wheel, beside the slice that name belongs to, rather
   * than beside the badge. The badge already says the trait and its number; what it cannot say is
   * which of twelve wedges is the one being named, and a tooltip that opens there says it without a
   * leader line or a second highlight. Measured off the wheel's own box, since that is what places
   * the SVG on screen -- the fixed tooltip has no other frame to sit in.
   *
   * Used both by a hover on a badge and by the hint below, which is the same act by the page.
   */
  const showPole = useCallback(
    (pole: Pole | null, auto = false) => {
      const box = wheel.current?.getBoundingClientRect();
      if (!pole || !box) {
        onHover(null);
        return;
      }
      // Under the wheel for every slice, ending at its right edge so it does not cover the reply.
      onHover({
        turnIndex,
        pole,
        x: box.right,
        y: box.bottom,
        auto,
        chart: { top: box.top, bottom: box.bottom },
        align: 'right',
      });
    },
    [turnIndex, onHover],
  );

  /**
   * The turn's swing or headline trait, named without being asked for, as its readout lands.
   *
   * A wheel is twelve wedges and three numbers, and the first question it raises is what the top
   * one means -- which until now took a hover to answer, on a page whose whole subject is the
   * reading.
   *
   * Fires on a readout that arrives already measured as well as on one that becomes measured,
   * because a cached reply is the first kind: it comes back with its measurement in the same
   * update, so there is no transition to watch for. What keeps that from opening a whole transcript
   * full of tooltips is the last condition -- the hint belongs to the newest measurement, and a
   * turn with a measured turn after it is not it.
   */
  const wasMeasured = useRef(false);
  useEffect(() => {
    // Nothing to name is nothing to say, rather than a reason to take down whatever the reader is
    // pointing at: the hover is the whole page's now, so clearing it from here would reach into
    // another turn's readout.
    if (autoHint && measured && !wasMeasured.current && !isTurnMeasured(readouts, turnIndex + 1)) {
      const pole = hintPole(axes, readouts, turnIndex, poles, hintSwing);
      if (pole) {
        showPole(pole, true);
      }
    }
    wasMeasured.current = measured;
  }, [autoHint, hintSwing, axes, measured, poles, readouts, turnIndex, showPole]);

  /**
   * The hint stands until the reader does anything at all.
   *
   * It is placed against the wheel's box at one moment and then fixed to the window, so anything
   * that moves the transcript under it leaves it pointing at nothing -- and one per turn, never
   * dismissed, would stack up down the page. Sending the next message is a click or a key, and
   * rereading an earlier turn is a scroll, so those three cover it. A hover on the wheel replaces
   * the hint on its own, being a hover.
   */
  useEffect(() => {
    if (!mine?.auto) {
      return undefined;
    }
    const dismiss = () => onHover(null);
    window.addEventListener('scroll', dismiss, { capture: true, passive: true });
    window.addEventListener('pointerdown', dismiss);
    window.addEventListener('keydown', dismiss);
    return () => {
      window.removeEventListener('scroll', dismiss, { capture: true });
      window.removeEventListener('pointerdown', dismiss);
      window.removeEventListener('keydown', dismiss);
    };
  }, [mine?.auto, onHover]);

  // The wheel is drawn a few pixels inside its own box, so this is not the whole distance between
  // the last badge and the rim -- it is what is added to that. Enough that the two halves read as
  // a list and a chart rather than as one stack: the pills are names, the wheel is the shape, and
  // butted together the bottom pill looked like a label on the wedge under it.
  return (
    <div className={`flex shrink-0 flex-col items-center gap-y-2 ${CARD_CLASS} ${className ?? ''}`}>
      <TraitBadges poles={poles} measured={measured} hoveredKey={mine?.pole.key ?? null} onHover={showPole} />
      <div
        ref={wheel}
        id={wheelId}
        data-nt-wheel
        className={`${WHEEL_CLASS} ${
          onZoom && measured
            ? 'cursor-pointer rounded-2xl transition-[transform,box-shadow] duration-200 ease-out hover:scale-[1.03] hover:shadow-lg'
            : ''
        }`}
        onClick={() => {
          const box = wheel.current?.getBoundingClientRect();
          if (onZoom && measured && box) {
            onZoom(box);
          }
        }}
      >
        <PersonaSunburst
          axes={axes}
          readouts={readouts}
          turnIndex={turnIndex}
          size={WHEEL_SIZE}
          showLabels={false}
          className="h-full w-full"
          hoveredKey={mine?.pole.key ?? null}
          onHover={showPole}
        />
      </div>
    </div>
  );
}

/** About how tall a tooltip is, in pixels. Below this much room, it goes on top. */
const TOOLTIP_HEIGHT_ESTIMATE = NT_SHOW_TRAIT_EXAMPLES ? 220 : 140;
/** About how much a quoted reply adds to it. */
const REPLY_HEIGHT_ESTIMATE = 60;

/** Space between the zoomed card's edge and its wheel, in pixels. */
const ZOOM_PADDING = 8;

/**
 * One turn's wheel, opened larger over a dimmed page, with every slice labelled.
 *
 * In a portal on top of everything, so nothing under it moves. Fixed in place while open: the
 * backdrop takes the pointer, so the transcript cannot scroll out from under it. A click outside
 * the card, the close button, Escape or a resize closes it.
 */
export function WheelZoom({
  axes,
  readouts,
  turnIndex,
  left,
  width,
  centerY,
  hovered,
  onHover,
  onClose,
  id,
}: {
  axes: PersonaAxisDefinition[];
  readouts: PublicAxisReadout[];
  turnIndex: number;
  /** An id for the card. */
  id?: string;
  /** The card's left edge and width, in window pixels. */
  left: number;
  width: number;
  /** Where the card is centred vertically, before it is kept on screen. */
  centerY: number;
  hovered: TraitHover;
  onHover: (hover: TraitHover) => void;
  onClose: () => void;
}) {
  const size = width - 2 * ZOOM_PADDING;
  const wheel = useRef<HTMLDivElement>(null);
  const mine = hovered?.turnIndex === turnIndex ? hovered : null;

  const showPole = useCallback(
    (pole: Pole | null) => {
      const box = wheel.current?.getBoundingClientRect();
      if (!pole || !box) {
        onHover(null);
        return;
      }
      // Under the middle of the wheel for every slice, not at the slice.
      onHover({
        turnIndex,
        pole,
        x: box.left + box.width / 2,
        y: box.bottom,
        chart: { top: box.top, bottom: box.bottom },
        align: 'center',
      });
    },
    [turnIndex, onHover],
  );

  /**
   * The card opens under the pointer, which fires a hover with no move. Until the pointer really
   * moves, keep the slice under it here and show nothing.
   */
  const moved = useRef(false);
  const pending = useRef<Pole | null>(null);
  const onSliceHover = useCallback(
    (pole: Pole | null) => {
      if (moved.current) {
        showPole(pole);
      } else {
        pending.current = pole;
      }
    },
    [showPole],
  );
  const onPointerMove = (event: React.PointerEvent) => {
    if (moved.current || (event.movementX === 0 && event.movementY === 0)) {
      return;
    }
    moved.current = true;
    showPole(pending.current);
  };

  // Clear a hover held over from the small wheel.
  useEffect(() => onHover(null), [onHover]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', onKey);
    window.addEventListener('resize', onClose);
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('resize', onClose);
    };
  }, [onClose]);

  // A small grow with a slight overshoot.
  const card = useRef<HTMLDivElement>(null);
  useEffect(() => {
    card.current?.animate(
      [
        { transform: 'scale(0.85)', opacity: 0 },
        { transform: 'scale(1.03)', opacity: 1, offset: 0.7 },
        { transform: 'scale(1)', opacity: 1 },
      ],
      { duration: 240, easing: 'ease-out' },
    );
  }, []);

  const height = size + 2 * ZOOM_PADDING;
  const top = Math.max(ZOOM_PADDING, Math.min(centerY - height / 2, window.innerHeight - height - ZOOM_PADDING));

  return createPortal(
    <div className="fixed inset-0 z-[9000]" onClick={onClose}>
      <div className="absolute inset-0 animate-fade bg-slate-900/30 backdrop-blur-[2px] animate-duration-150" />
      <div
        id={id}
        role="dialog"
        aria-label="Trait activations for this turn, enlarged"
        ref={card}
        className="absolute rounded-2xl shadow-2xl"
        style={{ left, top, width, padding: ZOOM_PADDING, backgroundColor: CARD_FILL }}
        onClick={(event) => event.stopPropagation()}
        onPointerMove={onPointerMove}
      >
        {/* In the card's corner, which the round wheel leaves empty. */}
        <button
          type="button"
          aria-label="Close"
          onClick={onClose}
          className="absolute right-2 top-2 z-10 flex h-6 w-6 items-center justify-center rounded-full text-slate-500 hover:bg-slate-300/60 hover:text-slate-700"
        >
          <X className="h-3.5 w-3.5" />
        </button>
        <div ref={wheel} style={{ width: size, height: size }}>
          <PersonaSunburst
            axes={axes}
            readouts={readouts}
            turnIndex={turnIndex}
            size={size}
            showLabels={false}
            labelsInside
            className="h-full w-full"
            hoveredKey={mine?.pole.key ?? null}
            onHover={onSliceHover}
          />
        </div>
      </div>
    </div>,
    document.body,
  );
}

/**
 * Fixed rather than positioned within the chart, so it is not clipped by the scrolling transcript
 * a readout sits inside.
 *
 * Headed by the same pill the column shows, so the thing you pointed at is still on screen and
 * still says its number, rather than being restated as a sentence you have to match up.
 *
 * Hung off the point it describes by a notch, rather than merely placed beside it. Twelve wedges
 * meet at a wheel a hundred pixels across, so "the one this is about" is a claim worth making
 * exactly, and the notch's tip is on that point wherever the box has had to go to stay on screen.
 *
 * Down and to the left, which is where there is room: a readout sits at the left of the transcript
 * with the reply it measures to its right, so a box hung the other way covers the words the reading
 * is about.
 */
export function PoleTooltip({
  pole,
  x,
  y,
  reply,
  chart,
  align,
}: {
  pole: Pole;
  x: number;
  y: number;
  reply?: string;
  /** The hovered wheel's top and bottom edges. When set, the box goes under the wheel, not on it. */
  chart?: { top: number; bottom: number };
  /** Centred on `x`, or with its right edge on `x`, and no notch. */
  align?: TooltipAlign;
}) {
  const width = 320;
  /** Least the box may come to the edge of the window, and least the notch may come to its own. */
  const margin = 8;
  /** How far in from its right corner the box hangs off the point, and so where the notch sits. */
  const notch = 12;
  /** Clearance under the point, enough for the notch to stand in and no more. */
  const drop = 6;
  /**
   * Hung to the left of the point, then slid back inside the window if that would take it off an
   * edge.
   *
   * Slid rather than flipped to the other side. Flipping puts the same reading in one of two
   * places depending on where on the page it was hovered, so a reader working down a column of
   * wheels gets the box on the left for most of them and on the right for the ones near an edge.
   * The notch is placed from wherever the box ended up, so what stays fixed across all of it is
   * the tip, which is the part that says which of twelve wedges this is about.
   */
  const viewport = typeof window === 'undefined' ? Infinity : window.innerWidth;
  const left = Math.min(
    Math.max(align === 'center' ? x - width / 2 : align === 'right' ? x - width : x + notch - width, margin),
    Math.max(margin, viewport - width - margin),
  );
  const notchX = Math.min(Math.max(x - left, margin + 4), width - margin - 4);
  /**
   * Below the point, or for a wheel below the whole wheel so no slice is covered. Above only when
   * there is less room below than the box needs and more room above. An estimate, not a measure,
   * so the box never shows one frame in the wrong place.
   */
  const viewportHeight = typeof window === 'undefined' ? Infinity : window.innerHeight;
  const belowEdge = chart ? chart.bottom : y;
  const aboveEdge = chart ? chart.top : y;
  const roomBelow = viewportHeight - belowEdge;
  const need = TOOLTIP_HEIGHT_ESTIMATE + (reply ? REPLY_HEIGHT_ESTIMATE : 0);
  const above = roomBelow < need && aboveEdge > roomBelow;
  return (
    <div
      className="pointer-events-none fixed z-[10001] rounded-md border border-slate-300 bg-white px-0 py-2.5 text-[11px] font-normal leading-snug text-slate-600 shadow-lg"
      style={{
        left,
        top: above ? undefined : belowEdge + drop,
        bottom: above ? viewportHeight - aboveEdge + drop : undefined,
        width,
      }}
    >
      {/* A square turned on its corner, bordered on the two sides that end up facing out and
          filled to hide the box's own border behind it. Half of it stands outside the box, which is
          the part that reads as the notch, on whichever edge faces the point. */}
      {!align && (
        <div
          className={`absolute h-2 w-2 rotate-45 border-slate-300 bg-white ${
            above ? 'border-b border-r' : 'border-l border-t'
          }`}
          style={{
            top: above ? undefined : -4,
            bottom: above ? -4 : undefined,
            left: notchX - 4,
          }}
        />
      )}
      {/* The pill, with where in the model this was read from out at the other edge. The layer is
          a property of the axis rather than of the reading, so it belongs beside the name it
          qualifies and not at the end of the prose about this turn. */}
      <div className="flex flex-row items-center justify-between gap-x-2 px-2.5">
        <TraitPill pole={pole} active />
        {pole.layer !== null && (
          <span className="shrink-0 text-[9px] font-bold uppercase text-slate-300">{`Layer ${pole.layer}`}</span>
        )}
      </div>
      {!pole.measured && <div className="mt-1.5 px-2 text-slate-500">Not yet measured.</div>}
      {pole.definition && <div className="mt-2 px-2">{pole.definition}</div>}
      {reply && (
        <div className="mx-2 mt-2 line-clamp-3 border-l-2 border-slate-200 pl-2 text-[10px] text-slate-500">
          {reply}
        </div>
      )}
      {pole.measured && pole.ranked && (
        // What the number on the pill means: a position in the corpus this axis was fitted on, not
        // a fraction of some maximum. Lighter than the definition above it, and ruled off from it,
        // because the two are different kinds of statement: that one is what the axis says about
        // itself, this is a note on how to read a number already on screen.
        <div className="mt-2 border-t border-slate-200 px-2 pt-2 text-[10px] text-slate-500">
          {`Stronger than ${(pole.value * 100).toFixed(0)}% of the turns this axis was fitted on.`}
        </div>
      )}
      {NT_SHOW_TRAIT_EXAMPLES && pole.examples.length > 0 && (
        /* What the trait looked like, rather than another sentence about what it is: the replies
           the direction was read off. Each is shown under the message it answered, because a reply
           on its own is not obviously empathetic or curt -- it is that against what was asked.
           Clamped to two lines, so three of them fit and the column of them stays scannable. */
        <div className="mt-2 border-slate-200 px-2 pt-0">
          {/* <div className="text-left text-[9px] font-semibold uppercase tracking-wide text-slate-300">Fitted on</div> */}
          {/* The heading is a label on what follows rather than a section of its own, so the first
              example sits closer under it than the examples sit under each other. */}
          {pole.examples.map((example, index) => (
            <div key={example.response} className={index === 0 ? 'mt-1' : 'mt-1'}>
              <div className="mb-0 truncate text-[9px] text-slate-400">{example.user}</div>
              <div className="line-clamp-1 text-[9px]">{example.response}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
