'use client';

/**
 * The transcript, with each assistant turn's readout drawn beside the turn it describes, and every
 * turn's readings drawn together on the monitor down the left.
 *
 * This does not use the shared `SteerChatMessage`, which lays a conversation out as bubbles alone
 * and is used by the assistant-axis page. Putting a wheel next to a reply is the whole point of
 * this page, so the row is the unit here rather than the bubble.
 *
 * A wheel appears as soon as its turn does, empty, and fills in when the readout arrives -- the
 * measurement is taken on the finished turn, so it lands in one step after the text has streamed.
 * Drawing the frame early keeps the row from changing height when that happens.
 *
 * Two things live here rather than in the pieces below, because more than one piece needs them.
 * What the pointer is on: a wheel, its pills and a dot on the monitor are three views of the same
 * pole, and pointing at any of them lights up the other two. And where each turn sits down the
 * page: the monitor maps the whole transcript onto one pinned chart, so it needs the position of
 * every reply at once, and only the element that lays them out has that.
 */

import { LoadingSquare } from '@/components/svg/loading-square';
import type { PersonaAxisDefinition } from '@/lib/utils/persona-axis';
import { ChatMessage } from '@/lib/utils/steer';
import type { PublicAxisReadout } from '@/lib/utils/steer-axis-public';
import { ArrowRight } from 'lucide-react';
import { type RefObject, useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  NT_FIRST_QUESTION_ID,
  NT_FIRST_READOUT_ID,
  NT_FIRST_REPLY_ID,
  NT_FIRST_WHEEL_ID,
  NT_GUTTER_CLASS,
  NT_MONITOR_CLASS,
  NT_NO_SWING_CLASS,
  NT_TOOLTIP_CLASS,
  NT_TRAIT_MONITOR_ID,
  NT_WHEEL_ZOOM_ID,
} from './nt-tour-constants';
import {
  type BehaviorSwing,
  findBehaviorSwing,
  isTurnMeasured,
  PersonaReadout,
  PoleTooltip,
  TRAIT_BADGE_WIDTH,
  TRAIT_BADGE_WIDTH_PHONE,
  type TraitHover,
  TraitPill,
  WheelZoom,
} from './persona-sunburst';
import { NT_MODEL_ID, NT_SHOW_BEHAVIOR_SWINGS } from './shared';
import TraitMonitor, {
  CHART_TOP,
  type LineHighlight,
  MONITOR_INSET,
  MONITOR_INSET_PHONE,
  MONITOR_WIDTH,
  MONITOR_WIDTH_PHONE,
  type TurnSpan,
} from './trait-monitor';

/** How far inside the gutter's edges the enlarged wheel's card sits, in pixels. */
const ZOOM_INSET = 8;

/** `gap-x-2` in pixels, needed as a number because the gutter below is measured through it. */
const ROW_GAP = 8;
/** On a phone, the gap between a readout and its reply is `gap-x-1`. */
const ROW_GAP_PHONE = 4;
/** A reply's left padding: `px-3`, or `px-2` on a phone. */
const REPLY_PAD = 12;
const REPLY_PAD_PHONE = 8;
/** On a phone, the transcript's left padding is `pl-1`, so the readouts sit close to the chart. */
const COLUMN_PAD_PHONE = 4;

/** `gap-x-1` in pixels: the chart and the readouts beside it, closer than anything else on a row. */
const MONITOR_GAP = 4;

/**
 * A readout: the badge column, which is the wider of its two halves, plus its padding on the right.
 * Less padding on a phone.
 */
const READOUT_WIDTH = TRAIT_BADGE_WIDTH + 2 * ROW_GAP;
const READOUT_WIDTH_PHONE = TRAIT_BADGE_WIDTH_PHONE + ROW_GAP_PHONE;

/**
 * How far the tinted left of the page reaches, in pixels.
 *
 * The two columns this page appears to have are one column of rows with a stripe of colour behind
 * their left end -- which is the whole trick, and the reason this number exists. Genuinely
 * splitting the monitor from the transcript would mean keeping two scrollers in step and computing
 * where each reply sits in the other one; leaving them in the same flow means a readout is laid out
 * beside its reply by the browser, and the only thing anyone has to get right is where the colour
 * stops.
 *
 * It stops a full gap past the readouts, which is where a reply's own box begins. The reply carries
 * padding of its own, so the edge still falls a clear dozen pixels short of the first word rather
 * than against it -- and since the replies lost their bubbles, the box is the only edge there is to
 * stop at.
 */
export const GUTTER_WIDTH = MONITOR_INSET + MONITOR_WIDTH + MONITOR_GAP + READOUT_WIDTH + ROW_GAP;
const GUTTER_WIDTH_PHONE =
  MONITOR_INSET_PHONE + MONITOR_WIDTH_PHONE + MONITOR_GAP + READOUT_WIDTH_PHONE + COLUMN_PAD_PHONE;

/**
 * The gutter width for CSS, set on the panel and read through `GUTTER`. Tailwind needs literal
 * pixels, so these must equal `GUTTER_WIDTH_PHONE` and `GUTTER_WIDTH`; a test checks this.
 */
export const GUTTER_CLASS = '[--nt-gutter:144px] sm:[--nt-gutter:280px]';
export const GUTTER = 'var(--nt-gutter)';

/** The monitor column: inset plus chart. Must equal the constants it is made of; a test checks. */
const MONITOR_COLUMN_CLASS = 'w-[52px] sm:w-[132px]';

/**
 * On a phone, holds a question's bubble clear of the readouts, so it starts no further left than
 * the reply text does. Must equal the constants it is made of; a test checks.
 */
const QUESTION_ROW_CLASS = 'pl-[96px] sm:pl-0';

/** Starts the system prompt where the reply text starts. Must equal its constants; a test checks. */
const SYSTEM_ROW_CLASS = 'pl-[96px] sm:pl-[156px]';

/** For the test that keeps the literal classes above in step with the numbers. */
export const LAYOUT_CHECK = {
  gutterClass: GUTTER_CLASS,
  gutter: [GUTTER_WIDTH_PHONE, GUTTER_WIDTH],
  monitorColumnClass: MONITOR_COLUMN_CLASS,
  monitorColumn: [MONITOR_INSET_PHONE + MONITOR_WIDTH_PHONE, MONITOR_INSET + MONITOR_WIDTH],
  questionRowClass: QUESTION_ROW_CLASS,
  questionRow: [READOUT_WIDTH_PHONE + ROW_GAP_PHONE + REPLY_PAD_PHONE, 0],
  systemRowClass: SYSTEM_ROW_CLASS,
  systemRow: [READOUT_WIDTH_PHONE + ROW_GAP_PHONE + REPLY_PAD_PHONE, READOUT_WIDTH + ROW_GAP + REPLY_PAD],
};

/**
 * How far the transcript reaches past the foot of the scroller it lives in, in pixels: the height
 * of the input row, plus the panel's own bottom padding.
 *
 * The transcript used to stop where the input row began, so scrolling cut the messages and the
 * monitor's axis off along a hard line with tinted gutter and bare panel below it -- the page
 * looked like it ended twice. Running it to the foot of the panel instead means content slides
 * under the box you type in and out of sight at the panel's own rounded edge, which is the only
 * place a reader expects it to end.
 *
 * The input row is a fixed height because the textarea is: it never grows with what is typed. A
 * phone has a shorter row and no panel padding, so the value changes at `sm`. CSS reads it from
 * `UNDERLAP`, set on the panel by `UNDERLAP_CLASS`. The numbers are for the monitor.
 */
const TRANSCRIPT_UNDERLAP = 116 + 40;
const TRANSCRIPT_UNDERLAP_PHONE = 106;
export const UNDERLAP_CLASS = '[--nt-underlap:106px] sm:[--nt-underlap:156px]';
export const UNDERLAP = 'var(--nt-underlap)';

/**
 * Blank space held at the foot of the transcript, past the underlap.
 *
 * The underlap above puts the last reply behind the input rather than above it once the reader
 * scrolls to the end, so the transcript has to carry that much scrolling room of its own to give
 * back. The extra is the gap the last reply used to clear the input row by.
 */
const TAIL_EXTRA = 24;

/**
 * How far above the input row the blur behind it starts, in pixels.
 *
 * The transcript now runs under the input rather than stopping at it, which left the last of a
 * reply legible in the gaps around the box you type in. Blurring the band it passes through turns
 * that from text that ran out of room into text that has gone behind something, and the lead-in is
 * what keeps the blur itself from being an edge -- it arrives over this distance rather than
 * starting along a line of its own.
 */
export const INPUT_BLUR_LEAD = 56;

/**
 * Room under the last turn's readout, so its wheel clears the blur when scrolled to the end. It
 * adds height only when the readout is taller than the reply beside it.
 */
const LAST_READOUT_CLEARANCE = INPUT_BLUR_LEAD - TAIL_EXTRA + 8;

/**
 * How much of the row a bubble may take. The readout beside it is a fixed width, so this is a
 * share of the transcript rather than of what is left, and a long reply wraps well short of the
 * right edge instead of running the full width of the page.
 */
const BUBBLE_MAX_WIDTH = 'max-w-full sm:max-w-[70%]';

/**
 * Who a reply is from, above it. Small and low contrast: which side of the transcript something
 * sits on already says this, so the label is here for a reader arriving mid-page or at a
 * screenshot of one exchange, not as something to read every turn. Naming the checkpoint rather
 * than "Assistant" is the part that earns the row -- what the wheel beside it measures is a
 * property of that particular model.
 *
 * Carries its own colour, since only the assistant is labelled and the two places that render it
 * were keeping the same one in step by hand.
 */
const SPEAKER_CLASS = 'text-[9px] font-semibold uppercase leading-none tracking-wide text-slate-400/70';

/** The behavioral swing state of one reply: still being measured, none, or the swing found. */
type SwingState = { status: 'detecting' } | { status: 'none' } | { status: 'swing'; swing: BehaviorSwing };

const SWING_LABEL_CLASS = 'shrink-0 text-[9px] font-semibold uppercase leading-none tracking-wide';

/** The box a swing sits in. On a phone the label and its content are stacked and centred. */
const SWING_BOX_CLASS =
  'flex h-10 min-w-0 flex-none flex-col items-center justify-center gap-y-1 sm:h-5 sm:flex-1 sm:flex-row';

/** The wait for a swing, or the lack of one. Not on a phone, where only a swing gets the space. */
const SWING_STATUS_CLASS = 'hidden min-w-0 sm:flex sm:h-5 sm:flex-1 sm:items-center';

/**
 * The model's name above a reply, and the behavioral swing into that reply: to its right, or
 * below it on a phone.
 *
 * Spans the whole width of the chat, not only the reply's. One height in every state, so the
 * reply under it does not move when the measurement lands.
 */
function ReplyHeader({
  swing,
  turnIndex,
  onHoverSwing,
}: {
  swing: SwingState | null;
  turnIndex: number;
  onHoverSwing?: (highlight: LineHighlight) => void;
}) {
  return (
    <div
      data-reply-turn={turnIndex}
      data-swing-status={swing?.status}
      className="flex w-full min-w-0 flex-col gap-y-1 px-2 sm:h-5 sm:flex-row sm:items-center sm:gap-x-2 sm:px-3"
    >
      <div className={`${SPEAKER_CLASS} shrink-0`}>{NT_MODEL_ID}</div>
      {swing?.status === 'detecting' && (
        <div className={`${SWING_STATUS_CLASS} gap-x-3 text-slate-400/80`}>
          <span className={SWING_LABEL_CLASS}>Behavioral Swing</span>
          <span className="animate-pulse text-[10px] font-medium">Detecting...</span>
        </div>
      )}
      {swing?.status === 'none' && (
        <div className={`${NT_NO_SWING_CLASS} ${SWING_STATUS_CLASS} text-slate-300`}>
          <span className={SWING_LABEL_CLASS}>No Behavioral Swing</span>
        </div>
      )}
      {swing?.status === 'swing' && (
        <div
          className={`${SWING_BOX_CLASS} cursor-pointer gap-x-2 overflow-hidden rounded-lg border border-transparent bg-slate-200/80 px-2.5 text-slate-600 transition-colors hover:border-slate-400`}
          onMouseEnter={() => onHoverSwing?.({ axisId: swing.swing.to.key.split(':')[0], turnIndex })}
          onMouseLeave={() => onHoverSwing?.(null)}
        >
          <span className={`${SWING_LABEL_CLASS} max-sm:text-[8px]`}>Behavioral Swing</span>
          <div className="flex min-w-0 max-w-full flex-row items-center gap-x-1 sm:gap-x-2">
            <TraitPill pole={swing.swing.from} width="auto" phoneValue={false} className="min-w-0 cursor-pointer" />
            <ArrowRight className="h-3 w-3 shrink-0 text-slate-500" />
            <TraitPill pole={swing.swing.to} width="auto" phoneValue={false} className="min-w-0 cursor-pointer" />
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * The assistant's reply, shared by a finished one and by one being waited for, so what a reader is
 * watching is the same object that fills with text rather than something that appears in its place.
 *
 * No bubble: only one side of this transcript is bubbled now. A reply is the long half of an
 * exchange and it already has a readout beside it saying it is the model's, so the bubble was a
 * slab of slate carrying nothing the position did not. The padding stays, which is what keeps the
 * first line level with the first trait pill and the dot on the monitor level with both.
 *
 * The column holds the header, which spans the full width. The reply under it is capped at 70% of
 * the turn row (`cqw`, from the row's container), the same share a question gets.
 */
const ASSISTANT_COLUMN_CLASS = 'flex min-w-0 flex-1 flex-col pt-2';
const ASSISTANT_BUBBLE_CLASS =
  'min-w-0 max-w-[70cqw] break-words px-2 pb-2 text-[12px] leading-snug text-slate-700 sm:px-3';

// `!` beats the blog's `.prose p`, `.prose li` and `.prose h1` rules in globals.css.
const ASSISTANT_MARKDOWN_CLASS =
  'prose prose-slate max-w-none text-[12px] leading-snug text-slate-700 prose-headings:my-2 prose-headings:!text-[13px] prose-headings:!leading-snug prose-p:my-2 prose-p:!text-[12px] prose-p:!leading-snug prose-blockquote:my-2 prose-strong:text-slate-800 prose-code:text-[11px] prose-code:before:content-none prose-code:after:content-none prose-pre:my-2 prose-ol:my-2 prose-ul:my-2 prose-li:my-0.5 prose-li:!text-[12px] prose-li:!leading-snug';

/** A reply as simple markdown. Models break lines with a single newline, so each one is kept. */
function AssistantMarkdown({ text }: { text: string }) {
  return (
    <div className={ASSISTANT_MARKDOWN_CLASS}>
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children, ...props }) => (
            <a {...props} href={href} target="_blank" rel="noopener noreferrer">
              {children}
            </a>
          ),
        }}
      >
        {text.replace(/([^\n])\n(?=[^\n])/g, '$1  \n')}
      </Markdown>
    </div>
  );
}

/** A reply as one run of plain text, without markdown marks, for a short quote in a tooltip. */
function replySnippet(text: string | undefined) {
  return text
    ?.replace(/^\s*(?:[-*+]|\d+\.|#+|>)\s+/gm, '')
    .replace(/[*_`]+/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * A reply and the readout that measures it, shared by a finished turn and by one being waited for.
 *
 * The margin makes the column's spacing uneven on purpose. Only the gap above a question was meant
 * to close, so the column gap is the tighter of the two and this puts back what a reply needs to
 * stay clear of the question it answers.
 */
const TURN_ROW_CLASS = 'mt-1 flex w-full flex-row items-start gap-x-1 [container-type:inline-size] sm:gap-x-2';

/**
 * A readout sticks at the chart's top while its reply scrolls past. Sticky stays inside the parent
 * row, which ends with the reply, so it stops there and never covers the next turn's readout.
 */
const READOUT_CLASS = `${NT_GUTTER_CLASS} sticky shrink-0`;

/**
 * Each turn's stretch of the transcript, and where the transcript starts in the scroller -- what the
 * monitor maps the conversation onto its chart from.
 */
type Layout = { columnTop: number; spans: TurnSpan[] };

const EMPTY_LAYOUT: Layout = { columnTop: 0, spans: [] };

function sameLayout(a: Layout, b: Layout): boolean {
  return (
    a.columnTop === b.columnTop &&
    a.spans.length === b.spans.length &&
    a.spans.every((span, i) => span.start === b.spans[i].start && span.end === b.spans[i].end)
  );
}

/** Tailwind's `sm`, where the layout leaves its phone sizes. */
const WIDE_MEDIA = '(min-width: 640px)';

/**
 * Whether the screen is `sm` or wider. Only the monitor reads it, and the monitor draws nothing on
 * its first render, so hydration cannot disagree.
 */
function useWide() {
  const [wide, setWide] = useState(() => typeof window !== 'undefined' && window.matchMedia(WIDE_MEDIA).matches);
  useEffect(() => {
    const query = window.matchMedia(WIDE_MEDIA);
    const update = () => setWide(query.matches);
    update();
    query.addEventListener('change', update);
    return () => query.removeEventListener('change', update);
  }, []);
  return wide;
}

export default function NeuralTransparencyMessages({
  chatMessages,
  axes,
  readouts,
  awaitingResponse,
  isGenerating,
  scrollContainerRef,
  showSwings = NT_SHOW_BEHAVIOR_SWINGS,
}: {
  /** The scroller this transcript lives in, which the monitor reads and moves. */
  scrollContainerRef: RefObject<HTMLDivElement | null>;
  /** Show the behavioral swing into each reply, beside the model's name. */
  showSwings?: boolean;
  chatMessages: ChatMessage[];
  /** The axes every wheel is drawn from, measured or not, in the order they are placed around it. */
  axes: PersonaAxisDefinition[];
  readouts: PublicAxisReadout[];
  /** True while the next assistant turn has been asked for but has not started arriving. */
  awaitingResponse: boolean;
  /** True for the whole of a turn, including while its text streams and its reading is taken. */
  isGenerating: boolean;
}) {
  const [hovered, setHovered] = useState<TraitHover>(null);
  // Stable, because every readout runs an effect that depends on it: a new identity each render
  // would re-fire the hint each render. A hint does not replace a real hover.
  const onHover = useCallback(
    (hover: TraitHover) => setHovered((current) => (hover?.auto && current && !current.auto ? current : hover)),
    [],
  );
  /** The trait line a hovered behavioral swing lights on the monitor. No tooltip. */
  const [swingHover, setSwingHover] = useState<LineHighlight>(null);
  const onHoverSwing = useCallback((highlight: LineHighlight) => {
    setSwingHover(highlight);
    // A hint would light its own line instead.
    if (highlight) setHovered((current) => (current?.auto ? null : current));
  }, []);
  const wide = useWide();

  // The wheel opened larger, if any: which turn, and where its small wheel was on screen.
  const rootRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState<{ turnIndex: number; left: number; width: number; centerY: number } | null>(null);
  const openZoom = useCallback((turnIndex: number, wheelBox: DOMRect) => {
    const rootLeft = rootRef.current?.getBoundingClientRect().left ?? 0;
    const width = Math.min(GUTTER_WIDTH - 2 * ZOOM_INSET, window.innerWidth - 2 * ZOOM_INSET);
    setHovered(null);
    setZoom({ turnIndex, left: rootLeft + ZOOM_INSET, width, centerY: wheelBox.top + wheelBox.height / 2 });
  }, []);
  const closeZoom = useCallback(() => {
    setHovered(null);
    setZoom(null);
  }, []);

  const swingAt = (turnIndex: number): SwingState | null => {
    if (!showSwings) {
      return null;
    }
    if (!isTurnMeasured(readouts, turnIndex)) {
      return { status: 'detecting' };
    }
    const swing = findBehaviorSwing(axes, readouts, turnIndex);
    return swing ? { status: 'swing', swing } : { status: 'none' };
  };

  const columnRef = useRef<HTMLDivElement>(null);
  const rowRefs = useRef<(HTMLDivElement | null)[]>([]);
  const [layout, setLayout] = useState<Layout>(EMPTY_LAYOUT);

  // Which assistant turn each message is, counting only assistant messages, because that is what
  // indexes a readout's `turns`. A system message or a non-alternating conversation would
  // otherwise slide every wheel onto the wrong reply.
  let assistantTurns = 0;
  const turnOf = chatMessages.map((message) =>
    message.role === 'assistant' || message.role === 'model' ? assistantTurns++ : -1,
  );
  // The turn being generated has a row of its own, so it gets a slot on the monitor too -- empty,
  // but holding the height, so nothing below it moves when its dot lands.
  const turnCount = assistantTurns + (awaitingResponse ? 1 : 0);
  const firstQuestion = chatMessages.findIndex((message) => message.role === 'user');
  rowRefs.current.length = turnCount;

  /**
   * Measure where each turn ended up, so the monitor can map the transcript onto its chart.
   *
   * Off the DOM rather than computed, because the thing being matched is the height of wrapped
   * prose: a reply's row is as tall as its text, which nothing here knows in advance. A layout
   * effect rather than an effect, so the measurement is taken and applied in the same frame the
   * text was laid out in and the dots never lag the bubbles by one paint.
   *
   * Watched with a `ResizeObserver` as well as on each render because the text streams in: the
   * column grows a line at a time under React updates that do not change any of the props read
   * here. The observer cannot loop, since the monitor is positioned out of flow and so cannot
   * change what is being measured, and the guard below stops a re-render when nothing moved.
   */
  const measure = useCallback(() => {
    const column = columnRef.current;
    const scroller = scrollContainerRef.current;
    if (!column || !scroller) {
      return;
    }
    const top = column.getBoundingClientRect().top;
    const columnTop = top - scroller.getBoundingClientRect().top + scroller.scrollTop;
    // A turn runs from the end of the reply before it, so it holds the question it answers.
    let previousEnd = 0;
    const spans = rowRefs.current.map((row) => {
      const end = row ? row.getBoundingClientRect().bottom - top : previousEnd;
      const span = { start: previousEnd, end: Math.max(end, previousEnd) };
      previousEnd = span.end;
      return span;
    });
    setLayout((previous) => {
      const next = { columnTop, spans };
      return sameLayout(previous, next) ? previous : next;
    });
  }, [scrollContainerRef]);

  // Every render, since an undo or a readout landing moves things without changing any height.
  useLayoutEffect(measure);

  useLayoutEffect(() => {
    const column = columnRef.current;
    if (!column) {
      return undefined;
    }
    const observer = new ResizeObserver(measure);
    observer.observe(column);
    // Rewrapping at a new window width moves the rows whether or not the column ends up any
    // taller, which is the one way they can move without either trigger above firing.
    window.addEventListener('resize', measure);
    return () => {
      observer.disconnect();
      window.removeEventListener('resize', measure);
    };
  }, [measure]);

  // The tooltip is fixed to the window, so a scroll moves its point out from under it.
  useEffect(() => {
    const scroller = scrollContainerRef.current;
    if (!scroller) {
      return undefined;
    }
    const clear = () => setHovered(null);
    scroller.addEventListener('scroll', clear, { passive: true });
    return () => scroller.removeEventListener('scroll', clear);
  }, [scrollContainerRef]);

  return (
    // Grown to whatever height the scroller has left, so the pinned chart has room in a short chat.
    <div ref={rootRef} className="flex flex-1 flex-row items-stretch gap-x-1">
      {/* The monitor's column. The chart in it sticks to the top of the scroller, so it stays in
          view while the transcript scrolls. Narrower on a phone.

          The box is wider than the chart by the inset that holds it off the left edge of the page,
          and the chart is pinned to the right of it, so that inset is empty tint rather than empty
          SVG. */}
      <div className={`${NT_GUTTER_CLASS} ${NT_MONITOR_CLASS} relative shrink-0 ${MONITOR_COLUMN_CLASS}`}>
        <div id={NT_TRAIT_MONITOR_ID} className="sticky flex flex-row justify-end" style={{ top: CHART_TOP }}>
          <TraitMonitor
            axes={axes}
            readouts={readouts}
            spans={layout.spans}
            columnTop={layout.columnTop}
            scrollContainerRef={scrollContainerRef}
            bottomInset={wide ? TRANSCRIPT_UNDERLAP : TRANSCRIPT_UNDERLAP_PHONE}
            hovered={hovered}
            onHover={onHover}
            highlight={swingHover}
            loading={isGenerating}
            autoHint={wide}
            hintSwing={showSwings}
            width={wide ? MONITOR_WIDTH : MONITOR_WIDTH_PHONE}
          />
        </div>
      </div>

      {/* The transcript owns the horizontal padding now, not the panel: the panel has none left,
          so that the tint behind this row can run all the way to its edge. */}
      <div ref={columnRef} className="flex min-w-0 flex-1 flex-col gap-y-1 pl-1 pr-2 sm:pl-2 sm:pr-5">
        {chatMessages.map((message, i) => {
          if (message.role === 'system') {
            return (
              <div className={`mb-1 flex w-full ${SYSTEM_ROW_CLASS}`} key={i}>
                <div className="flex min-w-0 flex-col gap-y-0.5 rounded-xl bg-slate-200/80 px-3 py-1.5 text-slate-500">
                  <div className={SWING_LABEL_CLASS}>System Prompt</div>
                  <div className="whitespace-pre-wrap break-words text-[10px] leading-snug">{message.content}</div>
                </div>
              </div>
            );
          }
          if (message.role === 'user') {
            return (
              <div className={`flex w-full justify-end ${QUESTION_ROW_CLASS}`} key={i}>
                <div
                  id={i === firstQuestion ? NT_FIRST_QUESTION_ID : undefined}
                  className={`${BUBBLE_MAX_WIDTH} whitespace-pre-wrap break-words rounded-xl bg-slate-700 px-3 py-2 text-[12px] text-white sm:text-[13px]`}
                >
                  {message.content}
                </div>
              </div>
            );
          }
          if (turnOf[i] < 0) {
            return null;
          }
          return (
            <div
              className={TURN_ROW_CLASS}
              data-turn-row
              key={i}
              ref={(el) => {
                rowRefs.current[turnOf[i]] = el;
              }}
            >
              <div
                id={turnOf[i] === 0 ? NT_FIRST_READOUT_ID : undefined}
                className={READOUT_CLASS}
                style={{
                  top: CHART_TOP,
                  marginBottom: turnOf[i] === turnCount - 1 ? LAST_READOUT_CLEARANCE : undefined,
                }}
              >
                <PersonaReadout
                  axes={axes}
                  readouts={readouts}
                  turnIndex={turnOf[i]}
                  hovered={hovered}
                  onHover={onHover}
                  autoHint={false}
                  hintSwing={showSwings}
                  onZoom={(box) => openZoom(turnOf[i], box)}
                  wheelId={turnOf[i] === 0 ? NT_FIRST_WHEEL_ID : undefined}
                />
              </div>
              <div id={turnOf[i] === 0 ? NT_FIRST_REPLY_ID : undefined} className={ASSISTANT_COLUMN_CLASS}>
                <ReplyHeader swing={swingAt(turnOf[i])} turnIndex={turnOf[i]} onHoverSwing={onHoverSwing} />
                <div className={ASSISTANT_BUBBLE_CLASS}>
                  <AssistantMarkdown text={message.content} />
                </div>
              </div>
            </div>
          );
        })}

        {/* The turn that has been asked for, as the bubble it is about to be: named, with the
            spinner where its text will go. A bare spinner said the same thing, but the reply then
            arrived as a new object in a slightly different place, so the moment of answering read
            as a small jump rather than as the bubble beginning to fill. */}
        {awaitingResponse && (
          <div
            className={TURN_ROW_CLASS}
            ref={(el) => {
              rowRefs.current[assistantTurns] = el;
            }}
          >
            <div className={READOUT_CLASS} style={{ top: CHART_TOP, marginBottom: LAST_READOUT_CLEARANCE }}>
              <PersonaReadout
                axes={axes}
                readouts={readouts}
                turnIndex={assistantTurns}
                hovered={hovered}
                onHover={onHover}
                autoHint={false}
                hintSwing={showSwings}
              />
            </div>
            <div className={ASSISTANT_COLUMN_CLASS}>
              <ReplyHeader swing={showSwings ? { status: 'detecting' } : null} turnIndex={assistantTurns} />
              <div className={ASSISTANT_BUBBLE_CLASS}>
                <LoadingSquare size={16} className="py-0.5" />
              </div>
            </div>
          </div>
        )}

        {/* Only once there is something to scroll. With nothing said yet the panel is as tall as
            the greeting inside it, and this would be a screen of empty space under it. */}
        {chatMessages.length > 0 && (
          <div aria-hidden className="shrink-0" style={{ height: `calc(${UNDERLAP} + ${TAIL_EXTRA}px)` }} />
        )}
      </div>

      {/* One tooltip for the page, rather than one per readout: what it describes can be named from
          a wheel, from a pill or from a dot on the monitor, and only the thing that owns the hover
          can be sure exactly one of them is showing. */}
      {zoom && (
        <WheelZoom
          axes={axes}
          readouts={readouts}
          turnIndex={zoom.turnIndex}
          left={zoom.left}
          width={zoom.width}
          centerY={zoom.centerY}
          hovered={hovered}
          onHover={onHover}
          onClose={closeZoom}
          id={NT_WHEEL_ZOOM_ID}
        />
      )}

      {/* Part of the left bar, so the tour hides it too. `contents`, so it adds no flex item. */}
      {hovered && (
        <div className={`${NT_GUTTER_CLASS} ${NT_TOOLTIP_CLASS} contents`}>
          <PoleTooltip
            pole={hovered.pole}
            x={hovered.x}
            y={hovered.y}
            chart={hovered.chart}
            align={hovered.align}
            reply={
              hovered.withReply ? replySnippet(chatMessages[turnOf.indexOf(hovered.turnIndex)]?.content) : undefined
            }
          />
        </div>
      )}
    </div>
  );
}
