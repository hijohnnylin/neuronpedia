'use client';

import { LensMode, LensTokenMessage, LensType, LensTypeSlice } from '@/lib/utils/lens';
import * as PopoverPrimitive from '@radix-ui/react-popover';
import {
  createContext,
  memo,
  ReactNode,
  RefObject,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { LensModeContext } from './jlens-lens-mode';
import { START_LAYER_FRACTION } from './jlens-panel';
import JlensTokenPopup, { displayToken, LayerRange, PopupSteerContext, PopupSteerHandler } from './jlens-token-popup';

// ---------------------------------------------------------------------------
// Shared single popup coordinator
// ---------------------------------------------------------------------------
// Historically every chip owned its own HoverCard, so sweeping the cursor across
// tokens closed one card and opened the next — a visible flash (made worse by the
// open/close delay gap where nothing was shown). Instead, a single popup is
// mounted once per `JlensPopupHost` and stays open while the cursor moves between
// tokens; only its anchor + contents swap. A per-chip `setOpen` is driven by the
// coordinator so just the two involved chips re-render their active outline.

type ChipSetOpen = (open: boolean) => void;

type JlensPopupActions = {
  // Request this token's popup (after a short open delay when nothing is open;
  // immediately — a content swap — when a popup is already showing).
  show: (token: LensTokenMessage, anchor: HTMLElement, setOpen: ChipSetOpen) => void;
  // Toggle this token's popup open/closed immediately. Used for tap (mobile)
  // where there's no hover: tapping the active token closes it, tapping a
  // different token swaps to it.
  toggle: (token: LensTokenMessage, anchor: HTMLElement, setOpen: ChipSetOpen) => void;
  // Schedule a close (cancelled if another token or the popup itself is entered).
  hideSoon: () => void;
  // Cancel a pending close (called when the cursor moves onto the popup).
  cancelHide: () => void;
  // Drop the popup if `setOpen` is the currently-active chip (unmount safety).
  release: (setOpen: ChipSetOpen) => void;
};

// True on viewports narrower than Tailwind's `sm` breakpoint (640px). On these
// (touch) screens there's no hover, so token popups open on tap instead of
// hover and chat-token position selection is disabled.
export function useIsLensMobile(): boolean {
  const [isMobile, setIsMobile] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia('(max-width: 639px)');
    const update = () => setIsMobile(mq.matches);
    update();
    mq.addEventListener('change', update);
    return () => mq.removeEventListener('change', update);
  }, []);
  return isMobile;
}

// Scroll ONLY the given container (never any ancestors — which
// `Element.scrollIntoView` would) so the run of `positions` is brought into
// view. Used on share load to reveal a restored selection without disturbing
// the page's outer scroll position. Centers the selection when it fits the
// viewport; otherwise pins its start (with a little breathing room).
export function scrollContainerToTokenPositions(container: HTMLElement, positions: number[]) {
  if (positions.length === 0) {
    return;
  }
  let minPos = Infinity;
  let maxPos = -Infinity;
  for (const p of positions) {
    if (p < minPos) minPos = p;
    if (p > maxPos) maxPos = p;
  }
  const firstEl = container.querySelector<HTMLElement>(`[data-token-position="${minPos}"]`);
  if (!firstEl) {
    return;
  }
  const lastEl = container.querySelector<HTMLElement>(`[data-token-position="${maxPos}"]`) ?? firstEl;
  const containerRect = container.getBoundingClientRect();
  // Offsets of the selection relative to the container's scrollable content.
  const selectionTop = firstEl.getBoundingClientRect().top - containerRect.top + container.scrollTop;
  const selectionBottom = lastEl.getBoundingClientRect().bottom - containerRect.top + container.scrollTop;
  const selectionHeight = selectionBottom - selectionTop;
  const viewport = container.clientHeight;
  const PADDING = 16;
  const target = selectionHeight <= viewport ? selectionTop - (viewport - selectionHeight) / 2 : selectionTop - PADDING;
  const maxScroll = container.scrollHeight - viewport;
  container.scrollTop = Math.max(0, Math.min(target, maxScroll));
}

const JlensPopupActionsContext = createContext<JlensPopupActions | null>(null);

const OPEN_DELAY_MS = 150;
const HIDE_DELAY_MS = 80;

// While true (e.g. during a chat position drag-select), token popups are
// suppressed so they don't open under the cursor mid-drag. Setting it true also
// force-closes whichever popup is currently open (via the registered hosts).
let popupsSuppressed = false;
const suppressClosers = new Set<() => void>();
export function setJlensPopupsSuppressed(suppressed: boolean) {
  popupsSuppressed = suppressed;
  if (suppressed) {
    suppressClosers.forEach((close) => close());
  }
}

type JlensPopupActive = { token: LensTokenMessage; anchor: HTMLElement };

// Mounts exactly one shared lens popup for the tokens rendered inside it, and
// provides the coordinator the chips talk to. Each interface region (chat,
// completion, and their steered transcripts) wraps its tokens in its own host so
// the popup reads from the matching analysis state.
export function JlensPopupHost({
  layersByType,
  layerRange,
  onTokenHover,
  children,
}: {
  layersByType: Record<string, number[]>;
  layerRange: LayerRange | null;
  // Notified with the popup's current token so the sidebar can show that
  // position's readout while it's open (and clear it when the popup closes).
  onTokenHover?: (token: LensTokenMessage, open: boolean) => void;
  children: ReactNode;
}) {
  const mode = useContext(LensModeContext);
  const isMobile = useIsLensMobile();
  // Ref to the popup's content so an outside tap (mobile) can be distinguished
  // from a tap inside the popup (which must not dismiss it).
  const contentRef = useRef<HTMLDivElement | null>(null);
  // The popup shows two lens columns side by side in DIFF mode, so it needs
  // double the width.
  const popupWidthClass =
    mode === LensMode.DIFF
      ? 'w-[320px] min-w-[320px] max-w-[320px] sm:w-[960px] sm:min-w-[960px] sm:max-w-[960px]'
      : 'w-[280px] min-w-[280px] max-w-[280px] sm:w-[540px] sm:min-w-[540px] sm:max-w-[540px]';

  const [active, setActive] = useState<JlensPopupActive | null>(null);
  const activeRef = useRef<JlensPopupActive | null>(null);
  const activeSetOpenRef = useRef<ChipSetOpen | null>(null);
  const openTimer = useRef<number | null>(null);
  const hideTimer = useRef<number | null>(null);
  // A virtual anchor for Radix: Popper re-reads `.current` every render, so
  // swapping `active` repositions the (already-mounted) popup with no flash.
  const anchorRef = useRef<{ getBoundingClientRect: () => DOMRect } | null>(null);
  anchorRef.current = active ? active.anchor : null;

  const clearOpenTimer = useCallback(() => {
    if (openTimer.current != null) {
      clearTimeout(openTimer.current);
      openTimer.current = null;
    }
  }, []);
  const clearHideTimer = useCallback(() => {
    if (hideTimer.current != null) {
      clearTimeout(hideTimer.current);
      hideTimer.current = null;
    }
  }, []);

  const closeNow = useCallback(() => {
    clearOpenTimer();
    clearHideTimer();
    if (activeSetOpenRef.current) {
      activeSetOpenRef.current(false);
      activeSetOpenRef.current = null;
    }
    activeRef.current = null;
    setActive(null);
  }, [clearOpenTimer, clearHideTimer]);

  const show = useCallback(
    (token: LensTokenMessage, anchor: HTMLElement, setOpen: ChipSetOpen) => {
      if (popupsSuppressed) {
        return;
      }
      clearHideTimer();
      const doOpen = () => {
        openTimer.current = null;
        if (popupsSuppressed) {
          return;
        }
        // Close the previously-active chip's outline, then become active.
        if (activeSetOpenRef.current && activeSetOpenRef.current !== setOpen) {
          activeSetOpenRef.current(false);
        }
        activeSetOpenRef.current = setOpen;
        setOpen(true);
        activeRef.current = { token, anchor };
        setActive({ token, anchor });
      };
      clearOpenTimer();
      if (activeRef.current) {
        // A popup is already open: swap contents immediately (no flash).
        doOpen();
      } else {
        openTimer.current = window.setTimeout(doOpen, OPEN_DELAY_MS);
      }
    },
    [clearHideTimer, clearOpenTimer],
  );

  const hideSoon = useCallback(() => {
    clearOpenTimer();
    clearHideTimer();
    hideTimer.current = window.setTimeout(() => {
      hideTimer.current = null;
      closeNow();
    }, HIDE_DELAY_MS);
  }, [clearOpenTimer, clearHideTimer, closeNow]);

  const toggle = useCallback(
    (token: LensTokenMessage, anchor: HTMLElement, setOpen: ChipSetOpen) => {
      if (popupsSuppressed) {
        return;
      }
      clearOpenTimer();
      clearHideTimer();
      // Tapping the already-active token closes it.
      if (activeSetOpenRef.current === setOpen) {
        closeNow();
        return;
      }
      // Otherwise open (or swap to) this token immediately — no hover delay.
      if (activeSetOpenRef.current && activeSetOpenRef.current !== setOpen) {
        activeSetOpenRef.current(false);
      }
      activeSetOpenRef.current = setOpen;
      setOpen(true);
      activeRef.current = { token, anchor };
      setActive({ token, anchor });
    },
    [clearOpenTimer, clearHideTimer, closeNow],
  );

  const release = useCallback(
    (setOpen: ChipSetOpen) => {
      if (activeSetOpenRef.current === setOpen) {
        closeNow();
      }
    },
    [closeNow],
  );

  // Force-close on drag-select suppression.
  useEffect(() => {
    suppressClosers.add(closeNow);
    return () => {
      suppressClosers.delete(closeNow);
    };
  }, [closeNow]);

  // On mobile the popup opens on tap, so there's no pointer-leave to close it.
  // Dismiss it when the user taps anywhere outside the popup and outside a
  // token (token taps are handled by the chip's own toggle).
  useEffect(() => {
    if (!isMobile || !active) {
      return undefined;
    }
    const onDocPointerDown = (e: PointerEvent) => {
      const target = e.target as HTMLElement | null;
      if (!target) {
        return;
      }
      if (target.closest('[data-token-position]')) {
        return;
      }
      if (contentRef.current && contentRef.current.contains(target)) {
        return;
      }
      closeNow();
    };
    document.addEventListener('pointerdown', onDocPointerDown, true);
    return () => document.removeEventListener('pointerdown', onDocPointerDown, true);
  }, [isMobile, active, closeNow]);

  // Keep the sidebar's position readout in sync with the open popup.
  const prevTokenRef = useRef<LensTokenMessage | null>(null);
  useEffect(() => {
    const token = active?.token ?? null;
    if (token) {
      onTokenHover?.(token, true);
    } else if (prevTokenRef.current) {
      onTokenHover?.(prevTokenRef.current, false);
    }
    prevTokenRef.current = token;
  }, [active, onTokenHover]);

  const actions = useMemo<JlensPopupActions>(
    () => ({ show, toggle, hideSoon, cancelHide: clearHideTimer, release }),
    [show, toggle, hideSoon, clearHideTimer, release],
  );

  // Steering from a popup row should close the popup immediately so it doesn't
  // cover the sidebar it just updated. Wrap the upstream handler with `closeNow`.
  const upstreamSteer = useContext(PopupSteerContext);
  const handlePopupSteer = useCallback<PopupSteerHandler>(
    (key, type, steerMode) => {
      closeNow();
      upstreamSteer?.(key, type, steerMode);
    },
    [closeNow, upstreamSteer],
  );

  return (
    <JlensPopupActionsContext.Provider value={actions}>
      {children}
      <PopoverPrimitive.Root open={active !== null}>
        <PopoverPrimitive.Anchor virtualRef={anchorRef as RefObject<{ getBoundingClientRect: () => DOMRect }>} />
        <PopoverPrimitive.Portal>
          <PopoverPrimitive.Content
            side="bottom"
            align="start"
            sideOffset={0}
            collisionPadding={8}
            onOpenAutoFocus={(e) => e.preventDefault()}
            onCloseAutoFocus={(e) => e.preventDefault()}
            onPointerEnter={clearHideTimer}
            onPointerLeave={hideSoon}
            className={`shadow-0 z-50 rounded-2xl border-none bg-transparent p-0 pt-0 shadow-none outline-none data-[state=closed]:!duration-0 data-[state=open]:!duration-0 ${popupWidthClass}`}
          >
            <div
              ref={contentRef}
              className="-ml-4 h-full w-full overflow-hidden rounded-xl border border-slate-300 bg-white shadow-lg"
            >
              {active && (
                <PopupSteerContext.Provider value={upstreamSteer ? handlePopupSteer : null}>
                  <JlensTokenPopup token={active.token} layersByType={layersByType} layerRange={layerRange} />
                </PopupSteerContext.Provider>
              )}
            </div>
          </PopoverPrimitive.Content>
        </PopoverPrimitive.Portal>
      </PopoverPrimitive.Root>
    </JlensPopupActionsContext.Provider>
  );
}

// Render a single token's glyph, making whitespace/newlines visible the same
// way the NLA chip does (middle dots for spaces, ↵ for newlines).
function renderGlyph(token: string) {
  const newlineCount = (token.match(/\n/g) || []).length;
  if (token.trim() === '') {
    if (newlineCount > 0) {
      return <span className="opacity-40">{'\u21B5'.repeat(newlineCount)}</span>;
    }
    return <span className="text-slate-300">{'\u00B7'}</span>;
  }
  return (
    <>
      {token.startsWith(' ') && ' '}
      {token.replaceAll('\n', '\u21B5').trim()}
      {token.endsWith(' ') && <span className="text-slate-300">{'\u00B7'}</span>}
    </>
  );
}

// The top-K most frequent top-1 tokens across the (visible) layers of a lens
// slice. "Most common" = the token that is the layer's top-1 prediction in the
// most layers. The first 1/START_LAYER_FRACTION of layers are skipped to match
// the hover popup's layer window.
function topTokensByFrequency(slice: LensTypeSlice, k: number): string[] {
  const startIdx = Math.floor(slice.top_tokens.length / START_LAYER_FRACTION);
  const counts = new Map<string, number>();
  for (let i = startIdx; i < slice.top_tokens.length; i += 1) {
    const tok = slice.top_tokens[i]?.[0];
    if (tok === undefined) {
      continue;
    }
    counts.set(tok, (counts.get(tok) ?? 0) + 1);
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, k)
    .map(([tok]) => tok);
}

// Cap label length so the diagonal labels stay short.
function clipLabel(s: string): string {
  return s.length > 12 ? `${s.slice(0, 12)}…` : s;
}

// One horizontal band of a token's highlight background. When `border` is set,
// the band is drawn as the chip's border color instead of a background fill —
// used to preview a NON-selected sidebar token on hover (so it reads as a
// transient outline rather than a committed solid highlight).
export type TokenBand = { color: string; opacity: number; border?: boolean };

// Build a vertical-stripe background gradient from up to a handful of bands
// (one per selected sidebar token), stacked top→bottom in equal heights.
// Border bands also fill, but at 40% opacity (their full opacity goes to the
// border so the preview reads as an outline + a subtler background).
function bandsGradient(bands: TokenBand[]): string | undefined {
  if (bands.length === 0) {
    return undefined;
  }
  const n = bands.length;
  const stops = bands
    .map((b, i) => {
      const c = `rgba(${b.color}, ${b.border ? b.opacity * 0.4 : b.opacity})`;
      const start = ((i / n) * 100).toFixed(3);
      const end = (((i + 1) / n) * 100).toFixed(3);
      return `${c} ${start}%, ${c} ${end}%`;
    })
    .join(', ');
  return `linear-gradient(to bottom, ${stops})`;
}

// Always-visible summary rendered above a token: the top-3 most frequent
// Jacobian tokens then top-3 Logit tokens, laid out left → right, each drawn as
// a short label rotated 45° (bottom-left → top-right) so they sit parallel and
// can be glanced without hovering.
function LensAnnotations({ token }: { token: LensTokenMessage }) {
  const jac = token.results.find((r) => r.type === LensType.JACOBIAN_LENS);
  // const log = token.results.find((r) => r.type === LensType.LOGIT_LENS);

  const items: { token: string; className: string }[] = [];
  if (jac) {
    topTokensByFrequency(jac, 3).forEach((t) => items.push({ token: t, className: 'text-sky-600' }));
  }
  // if (log) {
  //   topTokensByFrequency(log, 3).forEach((t) => items.push({ token: t, className: 'text-rose-500' }));
  // }

  if (items.length === 0) {
    return null;
  }

  // Each cell anchors one diagonal label; we shift the row left by half the
  // total anchor span so the anchors' midpoint sits on the token's center.
  const CELL_W = 19;
  const centerOffset = ((items.length - 1) * CELL_W) / 2;

  return (
    <span aria-hidden className="pointer-events-none relative block hidden h-[32px] w-0 self-center">
      {/* Bottom-anchored row, horizontally centered on the token. Each
          fixed-width cell anchors one diagonal label so the labels sit
          parallel and spread out horizontally. */}
      <span
        className="absolute -bottom-1 left-0 flex flex-row items-center"
        style={{ transform: `translateX(-${centerOffset}px)` }}
      >
        {items.map((it, i) => (
          <span key={`${i}-${it.token}`} className="block shrink-0" style={{ width: CELL_W }}>
            <span
              className={`inline-block origin-bottom-left -rotate-[28deg] whitespace-nowrap font-sans text-[8.5px] font-medium leading-none tracking-tight ${it.className}`}
            >
              {clipLabel(displayToken(it.token.replace(/^\s+/, '')))}
            </span>
          </span>
        ))}
      </span>
    </span>
  );
}

// A hoverable token chip. The lens popup is a single shared element owned by the
// enclosing `JlensPopupHost`; the chip only reports hover intent to the
// coordinator (so a fast sweep swaps the popup's contents in place rather than
// flashing a new card). Content/generated tokens also render an always-visible
// top-3 summary above the glyph (see `LensAnnotations`).
function JlensTokenChipInner({
  token,
  variant = 'content',
  bands = [],
  positionSelected = false,
  prevSelected = false,
  nextSelected = false,
  highlighted = false,
  prevEndsWithLineBreak = false,
  nextStartsNewLine = false,
}: {
  token: LensTokenMessage;
  variant?: 'content' | 'special' | 'generated' | 'think';
  // Highlight background bands (one per selected sidebar token), driven by the
  // sidebar selection/hover.
  bands?: TokenBand[];
  // Accepted for call-site compatibility but consumed by the enclosing
  // `JlensPopupHost` (which owns the single shared popup), not the chip itself.
  layersByType?: Record<string, number[]>;
  layerRange?: LayerRange | null;
  onHoverChange?: (token: LensTokenMessage, open: boolean) => void;
  // Whether this token POSITION is selected (chat position filter). Rendered as
  // a rose outline that merges across consecutive selected tokens; `prev`/`next`
  // say whether the adjacent positions are selected so the interior edges are
  // dropped (and only the run's outer caps get a side + rounded corner).
  positionSelected?: boolean;
  prevSelected?: boolean;
  nextSelected?: boolean;
  // When true, draws a full red box-shadow around just this token (used when its
  // row is hovered in the "show positions" popup).
  highlighted?: boolean;
  // Whether the previous token ends with a line break. When it does, this token
  // wraps onto a new line, so its left cap is drawn even if the previous
  // position is also selected (otherwise the outline looks disconnected).
  prevEndsWithLineBreak?: boolean;
  // Whether the next token starts on a new line/block (e.g. this is the last
  // content token before the footer block). When true, this token's right cap
  // is drawn even if the next position is also selected.
  nextStartsNewLine?: boolean;
}) {
  const actions = useContext(JlensPopupActionsContext);
  const isMobile = useIsLensMobile();
  const hasLens = token.results.length > 0;
  const showAnnotations = hasLens && variant !== 'special' && variant !== 'think';
  // Whether this chip's popup is the active one — drives its active outline. The
  // coordinator flips this (and the previously-active chip's) so only the two
  // involved chips re-render on a hover swap.
  const [open, setOpen] = useState(false);

  // Drop the shared popup if this chip unmounts while it's the active one.
  useEffect(
    () => () => {
      if (hasLens) {
        actions?.release(setOpen);
      }
    },
    [actions, hasLens],
  );

  // `think` tokens (<think>/</think>) are rendered inline with the content but
  // share the dim, monospaced look of the structural special tokens.
  const colorClass =
    variant === 'special'
      ? 'text-slate-400 text-[7px] sm:text-[9px]'
      : variant === 'think'
        ? 'font-mono text-slate-400 text-[7px]'
        : 'text-slate-700 text-[10px] sm:text-[13px]';

  const bg = bandsGradient(bands);
  // Hovering a NON-selected sidebar token previews it as a full-opacity border
  // plus a 40%-opacity background fill (same color), so it stands out while
  // still reading as transient vs. a committed (selected) solid highlight.
  const borderBand = bands.find((b) => b.border);
  const borderColor = borderBand ? `rgba(${borderBand.color}, ${borderBand.opacity})` : undefined;
  // A token that ends with a line break wraps the following token onto a new
  // line, so the run is visually broken to this token's right. Likewise, when
  // the previous token ended with a line break, this token starts a new line
  // and the run is broken to its left. In both cases we draw the side cap even
  // if the adjacent position is selected, so the outline doesn't look
  // disconnected across the wrap.
  const endsWithLineBreak = /\n/.test(token.token);
  const leftCap = !prevSelected || prevEndsWithLineBreak;
  const rightCap = !nextSelected || endsWithLineBreak || nextStartsNewLine;
  // Position-selection outline: inset box-shadows (no layout space, so no shift
  // vs. a border). Top/bottom always; the left/right caps are only drawn at the
  // run's ends so consecutive selected tokens merge into one continuous box.
  const selectionShadow = (() => {
    if (!positionSelected) {
      return undefined;
    }
    const rose = 'rgb(251,113,133)'; // rose-400
    const parts = [`inset 0 2px 0 0 ${rose}`, `inset 0 -2px 0 0 ${rose}`];
    if (leftCap) {
      parts.push(`inset 2px 0 0 0 ${rose}`);
    }
    if (rightCap) {
      parts.push(`inset -2px 0 0 0 ${rose}`);
    }
    return parts.join(', ');
  })();
  // A full red box around just this token, drawn when its row is hovered in the
  // "show positions" popup. Takes precedence over the (rose) selection outline.
  const highlightShadow = highlighted
    ? (() => {
        const red = 'rgb(239,68,68)'; // red-500
        return [
          `inset 0 2px 0 0 ${red}`,
          `inset 0 -2px 0 0 ${red}`,
          `inset 2px 0 0 0 ${red}`,
          `inset -2px 0 0 0 ${red}`,
        ].join(', ');
      })()
    : undefined;
  const outlineShadow = highlightShadow ?? selectionShadow;
  // Round only the outer caps of a run. The box-shadow outline follows the
  // element's border-radius, so a larger radius here gives the selection box
  // noticeably rounder corners than the token's own (e.g. the hover outline).
  const capRounding = highlighted
    ? 'rounded-[7px]'
    : positionSelected
      ? `${leftCap ? 'rounded-l-[7px]' : ''} ${rightCap ? 'rounded-r-[7px]' : ''}`
      : '';
  const glyph = (
    <span
      data-token-position={token.position}
      onPointerEnter={!isMobile && hasLens ? (e) => actions?.show(token, e.currentTarget, setOpen) : undefined}
      onPointerLeave={!isMobile && hasLens ? () => actions?.hideSoon() : undefined}
      onClick={isMobile && hasLens ? (e) => actions?.toggle(token, e.currentTarget, setOpen) : undefined}
      style={{
        ...(bg ? { backgroundImage: bg } : {}),
        ...(borderColor ? { borderColor } : {}),
        ...(outlineShadow ? { boxShadow: outlineShadow } : {}),
      }}
      className={`relative -mx-px cursor-pointer select-none whitespace-pre-wrap border border-transparent bg-clip-padding px-[0.5px] py-[1px] leading-normal sm:py-[3px] ${colorClass} ${capRounding} ${
        open ? 'rounded outline outline-2 outline-slate-500' : ''
      }`}
    >
      {renderGlyph(token.token)}
    </span>
  );

  // Content/generated tokens are always wrapped in the same inline-flex box
  // (the annotations row is added only once lens results exist) so their line
  // height stays constant whether or not read-outs have streamed in yet —
  // otherwise the line spacing visibly grows per token as results arrive.
  // Structural tokens (special/think) render bare so the dim, small header/
  // footer rows stay compact.
  const isStructural = variant === 'special' || variant === 'think';
  return isStructural ? (
    glyph
  ) : (
    <span className="inline-flex flex-col items-center align-bottom">
      {showAnnotations && <LensAnnotations token={token} />}
      {glyph}
    </span>
  );
}

const JlensTokenChip = memo(JlensTokenChipInner);
export default JlensTokenChip;
