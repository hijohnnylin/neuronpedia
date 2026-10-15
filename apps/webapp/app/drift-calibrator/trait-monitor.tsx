'use client';

/**
 * The monitor: every trait's reading across the whole conversation, as five lines running down the
 * left of the transcript.
 *
 * The wheels beside each reply answer "what is this turn like"; this answers "what has been
 * happening", which no column of wheels can -- comparing turn to turn means holding two wheels in
 * your head at once, and a drift that builds over five replies is invisible that way. It is the
 * vertical chart from the assistant-axis page with a line per trait instead of a line per steer
 * setting.
 *
 * It is also a scrubber. The chart is pinned at the height of the visible transcript and always
 * shows the whole conversation, one slot per turn. The turns on screen get a larger share of the
 * height, like a fisheye lens, and a light band marks what is on screen. Clicking a slot scrolls
 * the transcript to that turn. When the whole chat fits, each dot sits level with its wheel instead.
 */

import type { PersonaAxisDefinition } from '@/lib/utils/persona-axis';
import type { PublicAxisReadout } from '@/lib/utils/steer-axis-public';
import { type RefObject, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { CATEGORY_HSL, hintPole, layoutPoles, type Pole, type TraitHover } from './persona-sunburst';

/** A trait line lit from outside the monitor, with the turn whose dot is ringed. */
export type LineHighlight = { axisId: string; turnIndex: number } | null;

/** Width of the chart itself, in pixels, not counting the inset that holds it off the left edge. */
export const MONITOR_WIDTH = 120;
/** The same, on a phone. */
export const MONITOR_WIDTH_PHONE = 48;

/** How far the chart sits in from the left edge of the gutter. */
export const MONITOR_INSET = 12;
/** The same, on a phone. */
export const MONITOR_INSET_PHONE = 4;

/**
 * Room either side of the band the lines move in.
 *
 * Enough that a dot at a full reading sits clear of the edge, so the extreme value is a position
 * rather than something pinned against a wall. Only the dot: its hover ring is wider than this and
 * reaches past the chart, which is why the SVG does not clip. What it reaches into is the inset
 * holding the chart off the left of the page, and empty tint on the right.
 */
const PADDING_X = 6;

/**
 * Where the lines start, above the first reply: a hair below the top of the chart, so the fan-out
 * from centre is visible rather than clipped by the edge of the SVG.
 */
const ORIGIN_Y = 6;

/** Where the first turn's slot starts, below the origin dot. */
const SLOTS_TOP = ORIGIN_Y + 10;

/** Room under the last slot, so its dot and hover ring stay inside the chart. */
const SLOTS_BOTTOM_PAD = 10;

/**
 * Where the chart sticks, in pixels down from the top of the scroller. It must equal the
 * transcript's top padding (`pt-4`), so the chart does not jump when it starts to stick.
 */
export const CHART_TOP = 16;

/** Space between the foot of the chart and the input row. */
const CHART_BOTTOM_GAP = 12;

/**
 * The tallest a slot gets when every turn fits. A short chat stays near the top of the chart,
 * close to its replies, instead of spreading two dots over the whole height.
 */
const SLOT_MAX = 140;

/**
 * The share of the slot height given to the turns on screen. The rest is shared evenly by all
 * turns, so a turn far off screen still has a slot to click.
 */
const FOCUS_SHARE = 0.5;

/** How far above a turn the transcript stops when a slot is clicked. */
const SCROLL_MARGIN = 8;

/** Fill of the band that marks the part of the conversation on screen. */
const VIEW_BAND_FILL = 'rgb(255 255 255 / 0.6)';

/** A turn's stretch of the transcript, from the end of the reply before it to the end of its own. */
export type TurnSpan = { start: number; end: number };

type Slot = { top: number; bottom: number };

/**
 * The scroller and what is in it, as last read. `chartTop` is the chart's top below the scroller's;
 * `wheelYs` is the centre of each turn's wheel, in pixels down from the chart's top. `wheelsAtRest`
 * is true while every turn's row starts below the readouts' sticky `top`: no turn has begun to
 * scroll off, and no readout can be stuck.
 */
type View = { scrollTop: number; clientHeight: number; chartTop: number; wheelYs: number[]; wheelsAtRest: boolean };

const EMPTY_VIEW: View = { scrollTop: 0, clientHeight: 0, chartTop: 0, wheelYs: [], wheelsAtRest: false };

function sameView(a: View, b: View): boolean {
  return (
    a.scrollTop === b.scrollTop &&
    a.clientHeight === b.clientHeight &&
    a.chartTop === b.chartTop &&
    a.wheelsAtRest === b.wheelsAtRest &&
    a.wheelYs.length === b.wheelYs.length &&
    a.wheelYs.every((y, i) => y === b.wheelYs[i])
  );
}

/**
 * One slot per turn. Each turn gets an even share of the height, plus a share of `FOCUS_SHARE`
 * in proportion to how much of it is on screen. Fully visible turns get equal focus.
 */
function layoutSlots(spans: TurnSpan[], viewTop: number, viewBottom: number, available: number): Slot[] {
  const count = spans.length;
  if (count === 0) {
    return [];
  }
  const viewHeight = Math.max(1, viewBottom - viewTop);
  const visibility = spans.map((span) => {
    const overlap = Math.max(0, Math.min(span.end, viewBottom) - Math.max(span.start, viewTop));
    return Math.min(1, overlap / Math.max(1, Math.min(span.end - span.start, viewHeight)));
  });
  const totalVisibility = visibility.reduce((sum, v) => sum + v, 0);
  const used = Math.max(0, Math.min(available, count * SLOT_MAX));
  let top = SLOTS_TOP;
  return visibility.map((v) => {
    const share = totalVisibility > 0 ? (1 - FOCUS_SHARE) / count + (FOCUS_SHARE * v) / totalVisibility : 1 / count;
    const slot = { top, bottom: top + share * used };
    top = slot.bottom;
    return slot;
  });
}

/** A point in the transcript, mapped to the chart through the slots. */
function mapToChart(y: number, spans: TurnSpan[], slots: Slot[]): number {
  for (let i = 0; i < spans.length; i += 1) {
    if (y < spans[i].end) {
      const length = Math.max(1, spans[i].end - spans[i].start);
      const t = Math.min(1, Math.max(0, (y - spans[i].start) / length));
      return slots[i].top + t * (slots[i].bottom - slots[i].top);
    }
  }
  return slots[slots.length - 1]?.bottom ?? SLOTS_TOP;
}

/**
 * The same exponent the wheel scales its bars with, and for the same reason: readings cluster low,
 * so a linear axis spends most of its width on a range nothing occupies. Sharing the number is
 * what keeps the two views agreeing -- a bar that reaches halfway out on the wheel is halfway to
 * the edge here, rather than being a third of the way out on one and two thirds on the other.
 */
const SCALE_EXPONENT = 0.6;

/** Radius of a dot, and of the invisible circle that catches the pointer for it. */
const DOT_RADIUS = 3.5;
const HIT_RADIUS = 9;

/** What a line that is not the one under the pointer drops to. */
const DIM_FACTOR = 0.22;

/** Width of the invisible stroke that catches the pointer for a line. */
const LINE_HIT_WIDTH = 10;

/** The line under the pointer: full colour, no fade to centre, thicker, and darker. */
const LIT_STROKE_WIDTH = 3;
const LIT_FILTER = 'brightness(0.8) saturate(1.2)';

/** Long enough to read as a fade, short enough that the monitor does not lag the pointer. */
const HOVER_EASE = '120ms ease-out';

/**
 * How long a new dot takes to travel from the last one, in milliseconds.
 *
 * Longer than the 550ms the wheel's bars grow over. Matching the wheel was the obvious thing to
 * do and the wrong one: a bar growing in place is read at a glance, while a dot has to be followed
 * from one row to the next, and at the wheel's pace the eye arrived after the dot had. This is
 * slow enough to watch and still short against the seconds a reply takes.
 */
const TRAVEL_MS = 1400;

/** One turn of the pulse that stands where the next reading will land. */
const PULSE_DURATION = '1.2s';

/** How far the pulse swells past a dot before it fades. */
const PULSE_RADIUS = 11;

/**
 * The ramp the four signed traits are drawn with, and the one thing on this chart that says which
 * way is which.
 *
 * Every line shares it, so a line's colour is its position rather than its identity -- drift
 * towards the harmful pole turns red whichever trait is drifting. That is what lets the monitor go
 * without a legend: the colour under a dot is the colour of the pill that names it, so a red dot
 * and a red pill are the same reading twice rather than two things to match up.
 *
 * Slate in the middle, which is the wheel's own colour for a pole that is neither: it is the
 * honest colour for a turn sitting near centre, and red interpolated straight to green passes
 * through an olive that means nothing.
 */
const RAMP_ID = 'nt-trait-ramp';

/**
 * The same two ramps again, faded towards the middle: what the lines and their dots are actually
 * drawn in.
 *
 * A reading near centre is the monitor reporting that it found nothing, and at full strength five
 * of those are a knot of line down the middle that draws the eye to the one place with nothing to
 * see. Fading by distance from centre makes weight mean size-of-reading, so a chart of a
 * conversation that never drifted is quiet and the one turn that did stands out of it.
 *
 * Only the fill and the stroke of the marks. The ring that lights under the pointer keeps the
 * ramps above at full strength, because that one is answering "which dot" rather than reporting a
 * number, and a highlight that fades exactly where the dots are hardest to tell apart is no
 * highlight.
 */
const FADED_RAMP_ID = 'nt-trait-ramp-faded';
const FADED_NEUTRAL_ID = 'nt-trait-neutral-faded';

/**
 * What the fade bottoms out at, dead on centre.
 *
 * Low enough that a reading of nothing is very nearly nothing on the chart. Five lines leaving one
 * origin all cross the middle whatever they go on to do, so this is also what those crossings cost:
 * near zero the lines thin out to a suggestion rather than staying joined.
 */
const CENTRE_OPACITY = 0.05;

/**
 * What the fade has climbed back to at the halfway mark, a quarter of the chart's width out.
 *
 * This is the whole shape of the curve. Straight from `CENTRE_OPACITY` to full at the rim, opacity
 * rose with distance, which meant only a reading close enough to centre to be uninteresting anyway
 * ever looked faint -- the middling ones, which is most of them, came out over half strength. A
 * knee here bends the ramp: the inner half of each side stays near the floor and the climb to full
 * happens over the outer half, so a line has to be genuinely out towards a pole to be drawn at
 * full weight. Below the halfway mark rather than at it, which is what makes it a curve.
 */
const KNEE_OPACITY = 0.3;

/** A category's colour exactly as the wheel draws its ring, so the two cannot drift apart. */
function categoryColor(category: keyof typeof CATEGORY_HSL): string {
  const { h, s, l } = CATEGORY_HSL[category];
  return `hsl(${h} ${s}% ${l}%)`;
}

/** A category's colour as sRGB channels, so two of them can be mixed the way a gradient mixes. */
function categoryRgb(category: keyof typeof CATEGORY_HSL): number[] {
  const { h, s, l } = CATEGORY_HSL[category];
  const sat = s / 100;
  const light = l / 100;
  const amplitude = sat * Math.min(light, 1 - light);
  const channel = (n: number) => {
    const k = (n + h / 30) % 12;
    return Math.round(255 * (light - amplitude * Math.max(-1, Math.min(k - 3, 9 - k, 1))));
  };
  return [channel(0), channel(8), channel(4)];
}

/**
 * The colour halfway between two categories, in sRGB, which is the space a gradient interpolates
 * in.
 *
 * The knee needs a stop partway along the ramp, and a stop has to carry a colour as well as an
 * opacity. Taking the midpoint the browser was already going to paint there means the extra stop
 * lands exactly on the line it interrupts, so the ramp gains a bend in opacity and none in colour.
 * Mixing in HSL instead would have swung the hue through a teal that was never on the ramp.
 */
function categoryMix(a: keyof typeof CATEGORY_HSL, b: keyof typeof CATEGORY_HSL): string {
  const from = categoryRgb(a);
  const to = categoryRgb(b);
  return `rgb(${from.map((channel, i) => Math.round((channel + to[i]) / 2)).join(' ')})`;
}

/**
 * The line for a trait with no preferable pole, a shade lighter than the wheel's neutral ring.
 *
 * The one place the monitor departs from the wheel's colours on purpose. On the wheel a neutral
 * ring sits against a plain card, where the shared lightness reads as quiet; here it sits among
 * four lines drawn in a ramp that is saturated at both ends, and at that same lightness it was the
 * heaviest thing on the chart rather than the one asking for least attention. Still well clear of
 * the centre rule, which is lighter again.
 */
const NEUTRAL_LINE = `hsl(${CATEGORY_HSL.neutral.h} ${CATEGORY_HSL.neutral.s}% 72%)`;

/**
 * The rule down the centre: between slate-300 and slate-200, rather than either.
 *
 * It is scenery, and the lines crossing it are now drawn at a sixth of their weight, so it wants
 * to be lighter than the slate-300 it was. Not as far as slate-200, which is the gutter's own
 * tint over slate-50 and would have left the axis all but invisible on its own background.
 */
const CENTRE_RULE = '#d7dfe9';

/** One trait, across every turn measured so far. */
type Series = {
  axisId: string;
  /** Which pole this trait is showing at each turn, so a dot can name the wedge it belongs to. */
  points: { turnIndex: number; x: number; y: number; pole: Pole }[];
  /** True for an axis with no preferable pole, which is drawn slate rather than on the ramp. */
  neutral: boolean;
};

type Point = { x: number; y: number };

/** How far along a segment, as a fraction of its height, each control point sits. */
const CURVE_TENSION = 0.4;

/**
 * A soft S-curve from each reading to the next.
 *
 * Each segment leaves and enters its dots straight down, so the line has no corners. The control
 * points share their dot's x, so the curve never leans past either reading. A segment depends on
 * its own two ends and nothing else, so a reading landing extends the line and leaves everything
 * above it exactly where it was.
 */
function linePath(points: Point[]): string {
  if (points.length < 2) {
    return '';
  }
  return points
    .map((point, i) => {
      if (i === 0) {
        return `M ${point.x} ${point.y}`;
      }
      const prev = points[i - 1];
      const bend = (point.y - prev.y) * CURVE_TENSION;
      return `C ${prev.x} ${prev.y + bend} ${point.x} ${point.y - bend} ${point.x} ${point.y}`;
    })
    .join(' ');
}

/**
 * The two poles of one axis, told apart by which side of the wheel each sits on.
 *
 * The wheel puts the pole you would rather see on the right half and its opposite on the left, and
 * this chart is that same arrangement unrolled: right is the desirable end whatever sign the fit
 * happened to give it. So the split is read off the categories the wheel already assigned rather
 * than restated here, and an axis where neither pole is preferable keeps its own sign, with the
 * plus pole on the right -- the same side the wheel gives it.
 */
function polesOf(poles: Pole[], axisId: string): { right: Pole; left: Pole } | null {
  const both = poles.filter((pole) => pole.key.startsWith(`${axisId}:`));
  const right = both.find((pole) => pole.category === 'desirable') ?? both.find((pole) => pole.key.endsWith(':plus'));
  const left = both.find((pole) => pole !== right);
  return right && left ? { right, left } : null;
}

/**
 * Where the next reading will land, while it is being taken.
 *
 * A ring swelling out of the dot the line currently ends at, which is the assistant-axis chart's
 * own way of saying the same thing. It answers a question the transcript cannot: a reply finishes
 * streaming well before it has been measured, so without this the chart sits still for a second or
 * two looking finished while the reading is still on its way.
 *
 * SMIL rather than CSS because `r` is what has to move, and growing the radius keeps the ring
 * concentric with the dot -- a scaled element would not be, since it is positioned by its centre.
 * The same reasoning, and the same shape, as the wheel's own loading sweep.
 */
function Pulse({ x, y, fill }: { x: number; y: number; fill: string }) {
  return (
    <circle cx={x} cy={y} r={DOT_RADIUS} fill={fill} opacity={0.6} style={{ pointerEvents: 'none' }}>
      <animate attributeName="r" from={DOT_RADIUS} to={PULSE_RADIUS} dur={PULSE_DURATION} repeatCount="indefinite" />
      <animate attributeName="opacity" from="0.6" to="0" dur={PULSE_DURATION} repeatCount="indefinite" />
    </circle>
  );
}

export default function TraitMonitor({
  axes,
  readouts,
  /**
   * Each assistant turn's stretch of the transcript, in pixels down from the top of the
   * transcript. One entry per turn, including a turn still being generated, which has no dot yet.
   */
  spans,
  /** Where the transcript starts inside the scroller's content, in pixels. */
  columnTop,
  /** The scroller the transcript lives in, which this chart reads and moves. */
  scrollContainerRef,
  /** How much of the foot of the scroller is behind the input row, in pixels. */
  bottomInset,
  hovered,
  onHover,
  /** A line to light while nothing on the monitor is hovered, such as from a behavioral swing. */
  highlight = null,
  /** True while a reply is being generated, whether or not any of it has arrived yet. */
  loading,
  /** Name the newest turn's headline trait at its dot when the reading lands. */
  autoHint,
  /** The hint names the behavioral swing into the reply, if any, before the headline trait. */
  hintSwing = false,
  /** Width of the chart, in pixels. */
  width = MONITOR_WIDTH,
}: {
  axes: PersonaAxisDefinition[];
  readouts: PublicAxisReadout[];
  spans: TurnSpan[];
  columnTop: number;
  scrollContainerRef: RefObject<HTMLDivElement | null>;
  bottomInset: number;
  hovered: TraitHover;
  onHover: (hover: TraitHover) => void;
  highlight?: LineHighlight;
  loading: boolean;
  autoHint: boolean;
  hintSwing?: boolean;
  width?: number;
}) {
  const centerX = width / 2;
  const halfWidth = width / 2 - PADDING_X;

  const svgRef = useRef<SVGSVGElement>(null);

  /**
   * The scroll position, kept here so that scrolling re-renders only this chart, not the transcript.
   * With it, where the chart and each turn's wheel are on screen: both are sticky, so where they
   * are is read off the page rather than worked out from the scroll.
   */
  const [view, setView] = useState<View>(EMPTY_VIEW);
  const [hoveredSlot, setHoveredSlot] = useState<number | null>(null);

  const read = useCallback(() => {
    const scroller = scrollContainerRef.current;
    const svg = svgRef.current;
    if (!scroller) {
      return;
    }
    const svgTop = svg?.getBoundingClientRect().top ?? 0;
    const scrollerTop = scroller.getBoundingClientRect().top;
    const wheels = svg ? Array.from(scroller.querySelectorAll('[data-nt-wheel]')) : [];
    const wheelsAtRest = wheels.every((wheel) => {
      const row = wheel.closest('[data-turn-row]');
      const readout = row?.firstElementChild;
      if (!row || !readout) {
        return false;
      }
      const stickyTop = parseFloat(getComputedStyle(readout).top) || 0;
      return row.getBoundingClientRect().top - scrollerTop >= stickyTop;
    });
    const next: View = {
      scrollTop: scroller.scrollTop,
      clientHeight: scroller.clientHeight,
      chartTop: svgTop - scrollerTop,
      wheelYs: wheels.map((wheel) => {
        const box = wheel.getBoundingClientRect();
        return box.top + box.height / 2 - svgTop;
      }),
      wheelsAtRest,
    };
    setView((previous) => (sameView(previous, next) ? previous : next));
  }, [scrollContainerRef]);

  // An effect, not a layout effect: the scroller is an ancestor, so its ref is not set yet when
  // this component's layout effects run on mount.
  useEffect(() => {
    const scroller = scrollContainerRef.current;
    if (!scroller) {
      return undefined;
    }
    let raf = 0;
    const schedule = () => {
      if (!raf) {
        raf = requestAnimationFrame(() => {
          raf = 0;
          read();
        });
      }
    };
    read();
    scroller.addEventListener('scroll', schedule, { passive: true });
    const observer = new ResizeObserver(schedule);
    observer.observe(scroller);
    return () => {
      scroller.removeEventListener('scroll', schedule);
      observer.disconnect();
      cancelAnimationFrame(raf);
    };
  }, [scrollContainerRef, read]);

  const visibleHeight = Math.max(0, view.clientHeight - bottomInset);
  const height = Math.max(0, visibleHeight - CHART_TOP - CHART_BOTTOM_GAP);
  const viewTop = view.scrollTop - columnTop;
  const viewBottom = viewTop + visibleHeight;

  // The wheels move without a scroll when a turn arrives, text streams in, or the chart first
  // appears, so they are read again whenever the transcript or the chart changes.
  useLayoutEffect(read, [read, spans, columnTop, height]);

  // The band only helps when part of the chat is off screen.
  const allVisible = spans.every((span) => span.start >= viewTop && span.end <= viewBottom);

  // When every wheel is at rest inside the chart, each dot sits level with its wheel, and the
  // chart follows the transcript one to one instead of through slots.
  const wheelChartYs = useMemo(
    () =>
      spans.length > 0 &&
      view.wheelsAtRest &&
      view.wheelYs.length === spans.length &&
      view.wheelYs.every((y) => y >= SLOTS_TOP && y <= height - SLOTS_BOTTOM_PAD)
        ? view.wheelYs
        : null,
    [spans.length, view.wheelsAtRest, view.wheelYs, height],
  );

  const slots = useMemo(() => {
    if (!wheelChartYs) {
      return layoutSlots(spans, viewTop, viewBottom, height - SLOTS_TOP - SLOTS_BOTTOM_PAD);
    }
    const toChart = (y: number) => columnTop + y - view.scrollTop - view.chartTop;
    return spans.map((span) => ({
      top: Math.max(SLOTS_TOP, toChart(span.start)),
      bottom: Math.min(height, toChart(span.end)),
    }));
  }, [wheelChartYs, spans, columnTop, view.scrollTop, view.chartTop, viewTop, viewBottom, height]);
  const turnYs = useMemo(
    () => wheelChartYs ?? slots.map((slot) => (slot.top + slot.bottom) / 2),
    [wheelChartYs, slots],
  );

  // Laid out once per turn, and not on scroll: the wheel's layout is a property of the turn, and
  // the dots move on every scroll frame.
  const turnCount = spans.length;
  const turnPoles = useMemo(
    () => Array.from({ length: turnCount }, (_, turnIndex) => layoutPoles(axes, readouts, turnIndex).poles),
    [axes, readouts, turnCount],
  );

  const bandTop = slots.length > 0 ? mapToChart(viewTop, spans, slots) : 0;
  const bandBottom = slots.length > 0 ? mapToChart(viewBottom, spans, slots) : 0;

  const scrollToTurn = useCallback(
    (turnIndex: number) => {
      const scroller = scrollContainerRef.current;
      const span = spans[turnIndex];
      if (scroller && span) {
        scroller.scrollTo({ top: Math.max(0, columnTop + span.start - SCROLL_MARGIN), behavior: 'smooth' });
      }
    },
    [scrollContainerRef, spans, columnTop],
  );

  const series = useMemo<Series[]>(() => {
    // A reading's distance from centre, signed so the desirable pole is positive. `layoutPoles`
    // has already put the magnitude on whichever pole the turn favours and zero on the other, so
    // the sign falls out of which of the two is non-zero rather than being recovered here.
    const xOf = (value: number) =>
      centerX + Math.sign(value) * Math.min(Math.abs(value), 1) ** SCALE_EXPONENT * halfWidth;

    const byTurn = turnPoles.map((poles, turnIndex) => ({ y: turnYs[turnIndex], turnIndex, poles }));

    return axes.map((axis) => {
      const points: Series['points'] = [];
      let neutral = false;
      byTurn.forEach((turn) => {
        const pair = polesOf(turn.poles, axis.name);
        if (!pair || !pair.right.measured) {
          return;
        }
        neutral = pair.right.category === 'neutral';
        const value = pair.right.value > 0 ? pair.right.value : -pair.left.value;
        points.push({ turnIndex: turn.turnIndex, x: xOf(value), y: turn.y, pole: value >= 0 ? pair.right : pair.left });
      });
      return { axisId: axis.name, points, neutral };
    });
  }, [axes, turnPoles, turnYs, centerX, halfWidth]);

  // The elements the travel below moves, and how many points each line had the last time it was
  // painted. Refs rather than state: this runs a frame at a time and re-rendering React sixty
  // times a second to move one dot would be paying for a diff nothing else uses.
  const pathEls = useRef(new Map<string, SVGPathElement | null>());
  const litPathEls = useRef(new Map<string, SVGPathElement | null>());
  const hitPathEls = useRef(new Map<string, SVGPathElement | null>());
  const headEls = useRef(new Map<string, SVGGElement | null>());
  const painted = useRef(new Map<string, number>());
  const frame = useRef(0);

  // What a travel already in flight draws from, so it follows the transcript if the text under it
  // reflows mid-move rather than easing towards where the dot used to be going.
  const latest = useRef(series);
  useLayoutEffect(() => {
    latest.current = series;
  }, [series]);

  useLayoutEffect(() => () => cancelAnimationFrame(frame.current), []);

  /** The whole line, with its newest point pulled back by an offset. */
  const pathFor = useCallback(
    (line: Series, dx: number, dy: number) =>
      linePath([
        { x: centerX, y: ORIGIN_Y },
        ...line.points.map((point, i) => (i === line.points.length - 1 ? { x: point.x + dx, y: point.y + dy } : point)),
      ]),
    [centerX],
  );

  /**
   * A new reading arrives as its dot travelling from the one before it, rather than as a dot
   * appearing somewhere new.
   *
   * The travel is the point of the chart: what a reader is being shown is a change, and a dot that
   * simply appears leaves them to work out which of five lines moved and by how much. So the dot
   * and the end of its line are drawn where the previous reading was and eased to where this one
   * is, out of a layout effect -- before the browser paints, so the start of the move is the first
   * frame on screen rather than a flash at the destination.
   *
   * Written to the DOM rather than through React for the same reason the wheel's reveal is a CSS
   * transition: the position is a property of the frame, not of the data, and the data has not
   * changed since the effect began.
   */
  useLayoutEffect(() => {
    const offsets = new Map<string, { dx: number; dy: number }>();
    series.forEach((line) => {
      const before = painted.current.get(line.axisId) ?? 0;
      painted.current.set(line.axisId, line.points.length);
      const to = line.points[line.points.length - 1];
      if (!to || line.points.length <= before) {
        return;
      }
      // Everything leaves the origin the first time, which is what makes the opening turn read as
      // a fan-out from centre rather than as five dots switching on.
      const from = line.points[line.points.length - 2] ?? { x: centerX, y: ORIGIN_Y };
      offsets.set(line.axisId, { dx: from.x - to.x, dy: from.y - to.y });
    });

    // Every line every frame, including the ones not moving: React skips an attribute it already
    // believes is correct, so a line left offset by an interrupted travel would stay offset until
    // its own data changed.
    const paint = (fraction: number) => {
      latest.current.forEach((line) => {
        const offset = offsets.get(line.axisId);
        const dx = offset ? offset.dx * fraction : 0;
        const dy = offset ? offset.dy * fraction : 0;
        headEls.current.get(line.axisId)?.setAttribute('transform', `translate(${dx} ${dy})`);
        const d = pathFor(line, dx, dy);
        pathEls.current.get(line.axisId)?.setAttribute('d', d);
        litPathEls.current.get(line.axisId)?.setAttribute('d', d);
        hitPathEls.current.get(line.axisId)?.setAttribute('d', d);
      });
    };

    if (offsets.size === 0) {
      // A travel already running keeps running. This effect fires on every re-measure of the
      // transcript, and a reply's own text streaming is enough to cause one, so resetting here
      // unconditionally would cut most travels off in their first few frames.
      if (!frame.current) {
        paint(0);
      }
      return;
    }

    cancelAnimationFrame(frame.current);
    paint(1);
    const started = performance.now();
    const step = (now: number) => {
      const progress = Math.min((now - started) / TRAVEL_MS, 1);
      // Out of the previous reading quickly and into the new one slowly, so the dot settles onto
      // its value rather than arriving at speed.
      paint((1 - progress) ** 3);
      frame.current = progress < 1 ? requestAnimationFrame(step) : 0;
    };
    frame.current = requestAnimationFrame(step);
  }, [series, centerX, pathFor]);

  // How many readings have landed, and so whether one is still being taken. `loading` alone is not
  // enough: the stream stays open past the reading for the write that follows it, and a pulse that
  // outlasted the dot it was waiting for would say a measurement was still coming when it had
  // already been drawn.
  const measured = Math.max(0, ...series.map((line) => line.points.length));
  const awaitingReading = loading && turnYs.length > measured;

  /**
   * When a new reading lands, name the turn's swing or headline trait at its dot, as a hover would.
   *
   * Waits for the travel above, so the notch points at where the dot stops. Placed from `latest`
   * at that moment, since the transcript can move in the wait. Also fires on mount, so a saved
   * chat opens with its last turn named. An undo lowers the count, so the next reading names again.
   */
  const hintedCount = useRef(0);
  // Read when the hint fires, so a new readout does not cancel the wait.
  const hintSource = useRef({ axes, readouts });
  hintSource.current = { axes, readouts };
  useEffect(() => {
    const isNew = measured > hintedCount.current;
    hintedCount.current = measured;
    if (!autoHint || !isNew) {
      return undefined;
    }
    const turnIndex = measured - 1;
    const timer = window.setTimeout(() => {
      const points = latest.current.flatMap((line) => line.points.filter((point) => point.turnIndex === turnIndex));
      const { axes: definitions, readouts: measuredReadouts } = hintSource.current;
      const named = hintPole(
        definitions,
        measuredReadouts,
        turnIndex,
        points.map((point) => point.pole),
        hintSwing,
      );
      const point = points.find((candidate) => candidate.pole.key === named?.key);
      const box = svgRef.current?.getBoundingClientRect();
      if (!point || !box || box.width === 0) {
        return;
      }
      onHover({
        turnIndex,
        pole: point.pole,
        x: box.left + point.x,
        y: box.top + point.y,
        auto: true,
        withReply: true,
      });
    }, TRAVEL_MS);
    return () => window.clearTimeout(timer);
  }, [measured, autoHint, hintSwing, onHover]);

  if (height <= 0 || spans.length === 0) {
    return null;
  }

  const hoveredAxis = hovered ? hovered.pole.key.split(':')[0] : (highlight?.axisId ?? null);
  // The line under the pointer is drawn last. SVG has no z-index, so a lit line crossing a dimmed
  // one would otherwise be painted over at the crossing.
  const drawOrder = [
    ...series.filter((line) => line.axisId !== hoveredAxis),
    ...series.filter((line) => line.axisId === hoveredAxis),
  ];

  /** The dot on a line nearest the pointer in height. */
  const nearestPoint = (line: Series, clientY: number) => {
    const box = svgRef.current?.getBoundingClientRect();
    if (!box || line.points.length === 0) {
      return null;
    }
    const localY = clientY - box.top;
    const point = line.points.reduce((best, candidate) =>
      Math.abs(candidate.y - localY) < Math.abs(best.y - localY) ? candidate : best,
    );
    return { point, box };
  };

  /** A hover on a line shows its nearest dot, as a hover on that dot would. */
  const hoverLine = (line: Series, clientY: number) => {
    const nearest = nearestPoint(line, clientY);
    if (!nearest) {
      return;
    }
    const { point, box } = nearest;
    if (!hovered?.auto && hovered?.pole.key === point.pole.key && hovered.turnIndex === point.turnIndex) {
      return;
    }
    setHoveredSlot(point.turnIndex);
    onHover({
      turnIndex: point.turnIndex,
      pole: point.pole,
      x: box.left + point.x,
      y: box.top + point.y,
      withReply: true,
    });
  };

  return (
    <svg ref={svgRef} width={width} height={height} className="overflow-visible">
      <defs>
        <linearGradient id={RAMP_ID} gradientUnits="userSpaceOnUse" x1={PADDING_X} x2={width - PADDING_X}>
          <stop offset="0%" stopColor={categoryColor('undesirable')} />
          <stop offset="50%" stopColor={categoryColor('neutral')} />
          <stop offset="100%" stopColor={categoryColor('desirable')} />
        </linearGradient>

        {/* Both of these span the same width as the ramp above, which is the width a reading can
            reach, so a mark's opacity is read off its own x the way its colour already is. No
            second pass over the data and nothing to keep in step as a dot animates: the fade is a
            property of where a thing is drawn, not of what it is. */}
        <linearGradient id={FADED_RAMP_ID} gradientUnits="userSpaceOnUse" x1={PADDING_X} x2={width - PADDING_X}>
          <stop offset="0%" stopColor={categoryColor('undesirable')} />
          <stop offset="25%" stopColor={categoryMix('undesirable', 'neutral')} stopOpacity={KNEE_OPACITY} />
          <stop offset="50%" stopColor={categoryColor('neutral')} stopOpacity={CENTRE_OPACITY} />
          <stop offset="75%" stopColor={categoryMix('neutral', 'desirable')} stopOpacity={KNEE_OPACITY} />
          <stop offset="100%" stopColor={categoryColor('desirable')} />
        </linearGradient>

        <linearGradient id={FADED_NEUTRAL_ID} gradientUnits="userSpaceOnUse" x1={PADDING_X} x2={width - PADDING_X}>
          <stop offset="0%" stopColor={NEUTRAL_LINE} />
          <stop offset="25%" stopColor={NEUTRAL_LINE} stopOpacity={KNEE_OPACITY} />
          <stop offset="50%" stopColor={NEUTRAL_LINE} stopOpacity={CENTRE_OPACITY} />
          <stop offset="75%" stopColor={NEUTRAL_LINE} stopOpacity={KNEE_OPACITY} />
          <stop offset="100%" stopColor={NEUTRAL_LINE} />
        </linearGradient>
      </defs>

      {/* Centre, where a turn favours neither pole. The only rule on the chart: without it the
          lines are five wiggles with no zero to be either side of.

          Drawn the full height from the first message on, before there is anything to plot against
          it. It is the axis rather than a reading, so waiting for data to draw it left the gutter
          empty for exactly as long as a reader was waiting to be told something, and then made the
          first reading arrive as a whole chart rather than as a measurement. */}
      {!allVisible && !wheelChartYs && (
        <rect
          x={1}
          y={bandTop}
          width={width - 2}
          height={Math.max(0, bandBottom - bandTop)}
          rx={8}
          fill={VIEW_BAND_FILL}
          style={{ pointerEvents: 'none' }}
        />
      )}

      {/* One click target per turn, the full width of the chart. The dots above are targets of
          their own, for the tooltip, and scroll to the same place. */}
      {slots.map((slot, turnIndex) => (
        <rect
          key={turnIndex}
          x={1}
          y={slot.top}
          width={width - 2}
          height={Math.max(0, slot.bottom - slot.top)}
          rx={6}
          fill={hoveredSlot === turnIndex ? 'rgb(255 255 255 / 0.5)' : 'transparent'}
          className="cursor-pointer"
          style={{ transition: `fill ${HOVER_EASE}` }}
          onMouseEnter={() => setHoveredSlot(turnIndex)}
          onMouseLeave={() => setHoveredSlot(null)}
          onClick={() => scrollToTurn(turnIndex)}
        />
      ))}

      <line
        x1={centerX}
        y1={ORIGIN_Y}
        x2={centerX}
        y2={height}
        stroke={CENTRE_RULE}
        strokeWidth={1}
        style={{ pointerEvents: 'none' }}
      />

      {/* The start of the conversation, before anything had been read off it. One dot for all five
          lines, since they all leave from here and five stacked at a point is one dot with extra
          steps. It pulses while the first reading is being taken, which is the only time there is
          nothing else on the chart to pulse. */}
      <g>
        {awaitingReading && measured === 0 && <Pulse x={centerX} y={ORIGIN_Y} fill={categoryColor('neutral')} />}
        <circle cx={centerX} cy={ORIGIN_Y} r={DOT_RADIUS} fill={categoryColor('neutral')} />
      </g>

      {drawOrder.map((line) => {
        // A trait with no preferable pole has no side to drift towards, so it takes slate rather
        // than a place on the ramp: the ramp is a claim about better and worse, and making one
        // about a stylistic trait would be the wrong reading at both ends.
        const paint = line.neutral ? `url(#${FADED_NEUTRAL_ID})` : `url(#${FADED_RAMP_ID})`;
        const ringPaint = line.neutral ? NEUTRAL_LINE : `url(#${RAMP_ID})`;
        const isLit = hoveredAxis === line.axisId;
        const opacity = hoveredAxis && !isLit ? DIM_FACTOR : 1;
        const litOpacity = isLit ? 1 : 0;
        // Every line leaves the same point above the first reply, which is the conversation before
        // anything was said of it: with no reading yet there is nowhere else honest to start, and
        // it makes the opening turn read as a move away from centre rather than as five dots.
        return (
          <g
            key={line.axisId}
            style={{
              opacity,
              filter: isLit ? LIT_FILTER : 'none',
              transition: `opacity ${HOVER_EASE}, filter ${HOVER_EASE}`,
            }}
          >
            <path
              ref={(el) => {
                pathEls.current.set(line.axisId, el);
              }}
              d={pathFor(line, 0, 0)}
              fill="none"
              stroke={paint}
              strokeWidth={2}
              strokeLinecap="round"
              // A dot covers every joint, so this is only insurance: two segments meeting at a
              // sharp angle -- a reading swinging across the chart and straight back -- would
              // otherwise throw a mitre spike out past the dot.
              strokeLinejoin="round"
            />
            {/* The same line in full colour, shown on hover. */}
            <path
              ref={(el) => {
                litPathEls.current.set(line.axisId, el);
              }}
              d={pathFor(line, 0, 0)}
              fill="none"
              stroke={ringPaint}
              strokeWidth={LIT_STROKE_WIDTH}
              strokeLinecap="round"
              strokeLinejoin="round"
              style={{ opacity: litOpacity, transition: `opacity ${HOVER_EASE}`, pointerEvents: 'none' }}
            />
            {/* The target for the line, wider than what is drawn. Below the dots, which keep their own. */}
            <path
              ref={(el) => {
                hitPathEls.current.set(line.axisId, el);
              }}
              d={pathFor(line, 0, 0)}
              fill="none"
              stroke="transparent"
              strokeWidth={LINE_HIT_WIDTH}
              strokeLinecap="round"
              strokeLinejoin="round"
              className="cursor-pointer"
              style={{ pointerEvents: 'stroke' }}
              onMouseMove={(event) => hoverLine(line, event.clientY)}
              onMouseLeave={() => {
                setHoveredSlot(null);
                onHover(null);
              }}
              onClick={(event) => {
                const nearest = nearestPoint(line, event.clientY);
                if (nearest) {
                  scrollToTurn(nearest.point.turnIndex);
                }
              }}
            />
            {line.points.map((point, index) => {
              const lit = hovered
                ? hovered.pole.key === point.pole.key && hovered.turnIndex === point.turnIndex
                : highlight?.axisId === line.axisId && highlight.turnIndex === point.turnIndex;
              // The newest dot is the one that travels, and the one the next reading is awaited
              // at, so it is the only one that needs to be reachable from the animation.
              const head = index === line.points.length - 1;
              return (
                <g
                  key={point.turnIndex}
                  ref={
                    head
                      ? (el) => {
                          headEls.current.set(line.axisId, el);
                        }
                      : undefined
                  }
                  className="cursor-pointer"
                  // For the tour, which hovers a dot.
                  data-axis={line.axisId}
                  data-turn={point.turnIndex}
                  onClick={() => scrollToTurn(point.turnIndex)}
                  onMouseEnter={(event) => {
                    setHoveredSlot(point.turnIndex);
                    const box = event.currentTarget.getBoundingClientRect();
                    onHover({
                      turnIndex: point.turnIndex,
                      pole: point.pole,
                      x: box.left + box.width / 2,
                      y: box.top + box.height / 2,
                      withReply: true,
                    });
                  }}
                  onMouseLeave={() => {
                    setHoveredSlot(null);
                    onHover(null);
                  }}
                >
                  {head && awaitingReading && <Pulse x={point.x} y={point.y} fill={paint} />}
                  <circle
                    cx={point.x}
                    cy={point.y}
                    r={HIT_RADIUS}
                    fill="none"
                    stroke={ringPaint}
                    strokeWidth={2}
                    className={lit ? 'opacity-100' : 'opacity-0'}
                    style={{ transition: `opacity ${HOVER_EASE}` }}
                  />
                  <circle cx={point.x} cy={point.y} r={DOT_RADIUS} fill={paint} />
                  <circle
                    cx={point.x}
                    cy={point.y}
                    r={DOT_RADIUS}
                    fill={ringPaint}
                    style={{ opacity: litOpacity, transition: `opacity ${HOVER_EASE}` }}
                  />
                  {/* The target, the same size whatever the dot is doing. A dot is three pixels
                      across and five of them share a row, so pointing at one has to be easier than
                      pointing at what is drawn. */}
                  <circle cx={point.x} cy={point.y} r={HIT_RADIUS} fill="transparent" />
                </g>
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}
