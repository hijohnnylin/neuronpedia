'use client';

import { useChineseTranslation } from '@/lib/utils/chinese-translations';
import { LensMode, LensTokenMessage, LensType, LensTypeSlice } from '@/lib/utils/lens';
import { createContext, CSSProperties, UIEvent, useContext, useMemo, useRef, useState } from 'react';
import { LensModeContext, lensTypesForMode } from './jlens-lens-mode';
import { START_LAYER_FRACTION } from './jlens-panel';

// "r, g, b" for the per-row occurrence-by-layer mini heatmap stripes.
const POPUP_BAR_RGB = '100, 116, 139';

// Minimum opacity for a layer cell in the top layer-summary strip when the
// selected token IS present at that layer (weight > 0). Without a floor, a
// token whose probability is near 0.00% normalizes to an alpha so close to 0
// that its color is effectively invisible, even though it occurs there. We keep
// weight 0 fully transparent (token genuinely absent) and map any nonzero
// weight into [MIN_BAND_ALPHA, 1] so it always shows while preserving gradient.
const MIN_BAND_ALPHA = 0.18;

// Opacity for a layer cell in the top layer-summary strip: 0 stays transparent
// (token absent at that layer); any present-but-tiny weight is lifted to a
// visible floor so a ~0.00% token's color still shows.
function bandCellAlpha(weight: number): number {
  return weight > 0 ? MIN_BAND_ALPHA + (1 - MIN_BAND_ALPHA) * weight : 0;
}

// Inclusive [minLayer, maxLayer] range of layer NUMBERS to display.
export type LayerRange = [number, number];

// Fraction of the model's layers that the default layer selection skips from
// the start (kept in sync with `defaultRange` in use-jlens-analysis). Layers
// below this point (the first ~1/3 of the model) are where the J-Lens tends to
// be degenerate, so its readouts there are unreliable.
export const DEGENERATE_LAYER_FRACTION = 0.29;

// Shown above J-Lens readouts for layers before the default layer selection.
export const JLENS_DEGENERATE_DISCLAIMER = (
  <span>
    J-Lens is <span className="hidden sm:inline">typically </span> degenerate in the{' '}
    <a
      href="https://transformer-circuits.pub/2026/workspace/#struct-layers"
      className="underline"
      target="_blank"
      rel="noopener noreferrer"
    >
      first 1/3 layers
    </a>
    , leading to unreliable readouts.
  </span>
);

// The lowest layer NUMBER included in the default layer selection. Layers below
// this (the first ~1/3 of the model) are considered degenerate for the J-Lens.
export function degenerateLayerCutoff(layers: number[]): number {
  if (layers.length === 0) {
    return 0;
  }
  return layers[0] + Math.floor(layers.length * DEGENERATE_LAYER_FRACTION);
}

// Whether `layerNumber` sits before the default layer selection (i.e. in the
// first ~1/3 of the model, where the J-Lens is typically degenerate).
export function isDegenerateLayer(layers: number[], layerNumber: number): boolean {
  return layerNumber < degenerateLayerCutoff(layers);
}

// Indices into `layers` to display: those whose layer number falls within
// `range` (inclusive). When `range` is null, fall back to the
// START_LAYER_FRACTION window (skip the lowest 1/N of layers).
export function visibleLayerIndices(layers: number[], range: LayerRange | null): number[] {
  const all = layers.map((_, i) => i);
  if (range) {
    const [lo, hi] = range;
    return all.filter((i) => layers[i] >= lo && layers[i] <= hi);
  }
  const startIdx = Math.floor(layers.length / START_LAYER_FRACTION);
  return all.filter((i) => i >= startIdx);
}

// Make whitespace-only / leading-space tokens visible via replacement glyphs.
// Still used by the sidebar / in-text annotations; the popup itself uses
// `TokenPill` instead.
export function displayToken(token: string): string {
  if (token === '') {
    return '∅';
  }
  return token.replace(/ /g, '␣').replace(/\n/g, '⏎');
}

// A token is "word-like" when, after trimming, it is non-empty, not a special
// token, and every (unicode) character is alphanumeric — with `'`, `-`, `’`
// allowed only in interior positions. Used to hide noise
// (punctuation/whitespace/symbol/special) tokens from the sidebar + popup lists.
export function isWordLikeToken(token: string): boolean {
  const stripped = token.trim();
  if (stripped === '') {
    return false;
  }
  if (stripped.includes('<|') || (stripped.startsWith('<') && stripped.endsWith('>'))) {
    return false;
  }
  // Iterate by unicode code point (matches Python's per-character iteration).
  const chars = [...stripped];
  return chars.every((ch, pos) => {
    if (/[\p{L}\p{N}]/u.test(ch)) {
      return true;
    }
    return pos > 0 && pos < chars.length - 1 && (ch === "'" || ch === '-' || ch === '’');
  });
}

// TEMPORARY TOGGLE: when false, sidebar tokens are NOT merged at all — each
// exact decoded token string is its own row (so " france" and "france" are
// distinct). When true, tokens are merged by trimmed-lowercase + simple-plural.
export const DEDUPE_SIMILAR_TOKENS = false;

// Key used to bucket tokens everywhere (sidebar lists, transcript highlight, and
// the hover popup), so the same token always maps to the same key in all views.
// When DEDUPE_SIMILAR_TOKENS is off, the key is the EXACT token
// (leading/trailing whitespace preserved); otherwise surrounding whitespace is
// trimmed and the token is lowercased.
export function normKey(t: string): string {
  return DEDUPE_SIMILAR_TOKENS ? t.trim().toLowerCase() : t;
}

// Whether to hide non-word tokens from the popup lists (mirrors the sidebar's
// "Show Non-Word Tokens" toggle). Provided by the chat.
export const HideNonWordContext = createContext<boolean>(true);

// Resolver provided by the chat: given a predicted token + lens type, returns
// the "r, g, b" of its assigned sidebar-selection color (or null if it's not a
// selected token). Lets the popup tint pills that match selected tokens.
// Returns the exact sidebar chip colors (`bg`/`ring` as "r, g, b" strings) for
// a token selected in the sidebar, or null when it isn't selected.
export type PillColorResolver = (token: string, type: LensType) => { bg: string; ring: string } | null;
export const PillColorContext = createContext<PillColorResolver | null>(null);

// A sidebar-selected token, with the heatmap color + canonicalizer used to match
// it within a token's per-layer top-n.
export type LensSliderSelectedToken = {
  key: string;
  type: LensType;
  colorRgb: string;
  canonicalOf: Map<string, string>;
};

// Provided by the chat so the sidebar slider stays in sync (same bounds + value
// + setter). Still consumed by the sidebar; the popup no longer renders a slider.
export type LensSliderControls = {
  bounds: LayerRange | null;
  range: LayerRange | null;
  setRange: (r: LayerRange) => void;
  selected: LensSliderSelectedToken[];
  // "r, g, b" of the anticipated next selection color (red when at the cap).
  nextColorRgb: string;
  // Toggle a token's selection (same behavior as the sidebar rows).
  onToggle: (key: string, type: LensType) => void;
  // Preview a token on hover (drives the chat-token highlight, same as hovering
  // the sidebar list rows). `null` clears the preview.
  onHoverPreview?: (token: { key: string; type: LensType } | null) => void;
  // The layer INDEX locked in the popup's selector strip, or null for the
  // "All Layers" overview. Lives in the shared analysis state (not the popup)
  // so a locked layer persists as the cursor sweeps between token popups, and
  // is cleared when the chat/completion is reset.
  lockedLayer: number | null;
  setLockedLayer: (i: number | null) => void;
};
export const LensSliderContext = createContext<LensSliderControls | null>(null);

// Resolver provided by the chat: given a lens type + token, returns that token's
// OVERALL prominence-by-layer (0..1 per layer, across ALL token positions —
// the sidebar's data), or [] if unknown. Used for the per-layer readout stripes.
export type LayerStatsResolver = (type: LensType, token: string) => number[];
export const LayerStatsContext = createContext<LayerStatsResolver | null>(null);

// The token currently being steered/swapped (null when not steering). While set,
// the popup's per-layer selector strip shows ONLY this token's band (full
// height) instead of one stacked band per selected sidebar token.
export const SteerActiveContext = createContext<{ key: string; type: LensType } | null>(null);

// Steer/swap entry point provided by the chat (same `beginSteer` the sidebar
// rows use). When present, popup readout rows show hover "Steer"/"Swap" buttons.
export type PopupSteerHandler = (key: string, type: LensType, mode: 'steer' | 'swap') => void;
export const PopupSteerContext = createContext<PopupSteerHandler | null>(null);

// The hover "Steer"/"Swap" buttons shown on a readout row (mirrors the sidebar
// rows). Absolutely positioned at the right edge of the token-name area; appears
// on row hover via the parent's `group-hover`. `colorRgb` is the row's selected
// (or anticipated next) "r, g, b".
function PopupSteerButtons({
  tokenKey,
  token,
  type,
  colorRgb,
  onSteer,
}: {
  tokenKey: string;
  token: string;
  type: LensType;
  colorRgb: string;
  onSteer: PopupSteerHandler;
}) {
  return (
    <span className="absolute right-0 top-1/2 z-20 flex -translate-y-1/2 flex-row items-center gap-x-1 opacity-0 transition group-hover:opacity-100">
      {(['steer', 'swap'] as const).map((m) => (
        <span
          key={m}
          role="button"
          tabIndex={0}
          title={`${m === 'swap' ? 'Swap' : 'Steer'} '${token}'`}
          aria-label={`${m === 'swap' ? 'Swap' : 'Steer'} '${token}'`}
          onClick={(e) => {
            e.stopPropagation();
            onSteer(tokenKey, type, m);
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.stopPropagation();
              e.preventDefault();
              onSteer(tokenKey, type, m);
            }
          }}
          style={
            {
              '--steer-bg': 'rgb(255, 255, 255)',
              '--steer-bg-hover': `rgb(${colorRgb})`,
              '--steer-text': `rgb(${colorRgb})`,
              borderColor: `rgb(${colorRgb})`,
            } as CSSProperties
          }
          className="flex items-center justify-center whitespace-nowrap rounded-full border bg-[var(--steer-bg)] px-1.5 py-0.5 text-[9px] font-medium uppercase leading-none text-[color:var(--steer-text)] shadow-sm transition hover:bg-[var(--steer-bg-hover)] hover:text-white"
        >
          {m}
        </span>
      ))}
    </span>
  );
}

// A token's top-n occurrences across this position's layers, plus its per-layer
// occurrence counts (for the inline mini heatmap).
export type PositionToken = {
  key: string;
  token: string;
  count: number;
  perLayer: number[];
  // Whether the token appears in top-n at any layer inside the selected bounds.
  inRange: boolean;
  // Whether this is the model's output (final-layer top-1) token.
  isOutput: boolean;
};

// Aggregate one lens slice's per-layer top-n into a list of tokens (this
// position only). Ranked by: the highest layer at which the token is the top-1
// (top-logit) prediction, then frequency within the selected layer bounds, then
// overall frequency. Whitespace-only tokens are skipped from the list (they
// still show in the per-layer readout).
export function positionTopTokens(
  slice: LensTypeSlice,
  layers: number[],
  hideNonWord: boolean,
  range: LayerRange | null,
): PositionToken[] {
  const counts = new Map<string, number>();
  // Occurrences restricted to layers inside the selected bounds.
  const inRangeCounts = new Map<string, number>();
  // Occurrences weighted by layer index (i + 1) so later layers count more;
  // used as the final tiebreaker.
  const weightedCounts = new Map<string, number>();
  const display = new Map<string, string>();
  const perLayer = new Map<string, number[]>();
  // The model's actual next token (top-1 at the final/output layer). Always
  // shown, even when non-word, since it's "the token that comes next", and
  // pinned to the top of the list.
  const topLogitKey = normKey(slice.top_tokens[layers.length - 1]?.[0] ?? '');
  const inRange = (layer: number) => range == null || (layer >= range[0] && layer <= range[1]);
  layers.forEach((layerNum, i) => {
    const within = inRange(layerNum);
    const row = slice.top_tokens[i] ?? [];
    row.forEach((t) => {
      // Key tokens identically to the sidebar (`normKey`), so leading/trailing
      // whitespace and case keep tokens distinct (e.g. " Hello" vs "Hello").
      if (t.trim() === '') {
        return;
      }
      const k = normKey(t);
      // Respect the "Show Non-Word Tokens" setting, but never hide the next token.
      if (hideNonWord && k !== topLogitKey && !isWordLikeToken(k)) {
        return;
      }
      counts.set(k, (counts.get(k) ?? 0) + 1);
      weightedCounts.set(k, (weightedCounts.get(k) ?? 0) + (i + 1));
      if (within) {
        inRangeCounts.set(k, (inRangeCounts.get(k) ?? 0) + 1);
      }
      if (!display.has(k)) {
        display.set(k, t);
      }
      let arr = perLayer.get(k);
      if (!arr) {
        arr = layers.map(() => 0);
        perLayer.set(k, arr);
      }
      arr[i] += 1;
    });
  });
  return [...counts.entries()]
    .sort((a, b) => {
      // The output (final-layer top-1) token first.
      const ta = a[0] === topLogitKey ? 1 : 0;
      const tb = b[0] === topLogitKey ? 1 : 0;
      if (tb !== ta) {
        return tb - ta;
      }
      const ra = inRangeCounts.get(a[0]) ?? 0;
      const rb = inRangeCounts.get(b[0]) ?? 0;
      if (rb !== ra) {
        return rb - ra;
      }
      // Overall occurrence, weighted toward later layers.
      return (weightedCounts.get(b[0]) ?? 0) - (weightedCounts.get(a[0]) ?? 0);
    })
    .map(([key, count]) => ({
      key,
      token: display.get(key) ?? key,
      count,
      perLayer: perLayer.get(key) ?? [],
      inRange: (inRangeCounts.get(key) ?? 0) > 0,
      isOutput: key === topLogitKey,
    }));
}

// Build the per-layer occurrence mini-heatmap as a single CSS gradient with one
// hard-edged stop per layer (equal widths), instead of one DOM node per layer.
// `colorFor(i)` returns the "r, g, b" for layer index i; alpha is that layer's
// normalized weight.
function layerBarGradient(perLayer: number[], max: number, colorFor: (i: number) => string): string | undefined {
  const n = perLayer.length;
  if (n === 0) {
    return undefined;
  }
  const stops = perLayer
    .map((c, i) => {
      const alpha = max > 0 ? c / max : 0;
      const col = `rgba(${colorFor(i)}, ${alpha})`;
      const start = ((i / n) * 100).toFixed(3);
      const end = (((i + 1) / n) * 100).toFixed(3);
      return `${col} ${start}%, ${col} ${end}%`;
    })
    .join(', ');
  return `linear-gradient(to right, ${stops})`;
}

// One row of the position's top-token list: a selection square, the token pill,
// and a per-layer occurrence mini heatmap (this token position only). Clicking
// the row toggles the token's selection (same behavior as the sidebar).
function PositionTokenRow({
  tokenKey,
  token,
  count,
  perLayer,
  inRange,
  isOutput,
  layers,
  range,
  type,
  selectedColorRgb,
  nextColorRgb,
  prob,
  showCount,
  onToggle,
  onSteer,
  onHoverPreview,
}: PositionToken & {
  // The token's normalized key (used to steer/swap it).
  tokenKey: string;
  layers: number[];
  range: LayerRange | null;
  // The lens type this row belongs to (used to steer/swap it).
  type: LensType;
  // Assigned color "r, g, b" when this token is selected, else null.
  selectedColorRgb: string | null;
  // Anticipated next-selection color "r, g, b".
  nextColorRgb: string;
  // Optional probability (0..1) shown left of the stripes (per-layer readout).
  prob?: number;
  // Whether to show the total-count column (the position's top-token list).
  showCount?: boolean;
  onToggle: () => void;
  // Open the steer/swap flow for this token (hover buttons); omitted when steering unavailable.
  onSteer?: PopupSteerHandler;
  // Notified on hover enter/leave so the chat tokens preview this token.
  onHoverPreview?: (hovering: boolean) => void;
}) {
  const [hover, setHover] = useState(false);
  const translate = useChineseTranslation();
  const translation = translate(token) ?? null;
  const max = Math.max(0, ...perLayer);
  // Dim tokens that don't appear within the selected layer range, unless this is
  // the output (final-layer top-1) token.
  const dimmed = !inRange && !isOutput;
  const selected = selectedColorRgb !== null;
  // Whole-row tint + ring, mirroring the sidebar: assigned color when selected,
  // anticipated next color (lighter) on hover.
  const rowStyle: CSSProperties | undefined = selected
    ? { backgroundColor: `rgba(${selectedColorRgb}, 0.12)`, boxShadow: `inset 0 0 0 1px rgb(${selectedColorRgb})` }
    : hover
      ? { backgroundColor: `rgba(${nextColorRgb}, 0.1)`, boxShadow: `inset 0 0 0 1px rgba(${nextColorRgb}, 0.6)` }
      : undefined;
  const inRangeLayer = (i: number) => range == null || (layers[i] >= range[0] && layers[i] <= range[1]);
  // Out-of-range layers always render slate; in-range use the token's color.
  // Rendered as a single gradient element per layer instead of one node each.
  const baseRgb = selected ? selectedColorRgb : POPUP_BAR_RGB;
  const baseGradient = layerBarGradient(perLayer, max, (i) => (inRangeLayer(i) ? baseRgb : POPUP_BAR_RGB));
  const hoverGradient = selected
    ? undefined
    : layerBarGradient(perLayer, max, (i) => (inRangeLayer(i) ? nextColorRgb : POPUP_BAR_RGB));
  return (
    <button
      type="button"
      onClick={onToggle}
      onMouseEnter={() => {
        setHover(true);
        onHoverPreview?.(true);
      }}
      onMouseLeave={() => {
        setHover(false);
        onHoverPreview?.(false);
      }}
      style={rowStyle}
      className={`group mx-0 flex h-5 max-h-5 min-h-5 w-full flex-row items-center gap-x-1.5 rounded px-2 py-0.5 text-left font-mono text-[10px] transition-colors ${
        dimmed ? 'opacity-50' : ''
      }`}
    >
      {selected ? (
        <span
          className="h-2.5 w-2.5 shrink-0 rounded-sm"
          style={{ backgroundColor: `rgb(${selectedColorRgb})` }}
          aria-hidden
        />
      ) : (
        <span className="relative h-2.5 w-2.5 shrink-0" aria-hidden>
          {/* Faint placeholder outline when neither hovered nor selected. */}
          <span className="absolute inset-0 rounded-sm border border-slate-400 opacity-70 transition-opacity group-hover:opacity-0" />
          {/* Anticipated next-color outline that fades in on hover. */}
          <span
            className="absolute inset-0 rounded-sm border opacity-0 transition-opacity group-hover:opacity-100"
            style={{ borderColor: `rgb(${nextColorRgb})` }}
          />
        </span>
      )}
      <span className="relative flex min-w-0 flex-1 flex-row items-center gap-x-1.5">
        <span className="min-w-0 truncate">{displayToken(token)}</span>
        {translation && (
          <span
            className="hidden min-w-0 flex-1 truncate font-sans text-[10px] text-slate-400 sm:block"
            title={translation}
          >
            {translation}
          </span>
        )}
        {onSteer && (
          <PopupSteerButtons
            tokenKey={tokenKey}
            token={token}
            type={type}
            colorRgb={selectedColorRgb ?? nextColorRgb}
            onSteer={onSteer}
          />
        )}
      </span>
      {showCount && <span className="w-10 shrink-0 text-right tabular-nums text-slate-400">{count}</span>}
      {prob !== undefined && (
        <span className="w-12 shrink-0 text-right tabular-nums text-slate-400">{(prob * 100).toFixed(2)}%</span>
      )}
      <span
        aria-hidden
        className="relative flex h-3.5 w-1/2 shrink-0 overflow-hidden rounded-sm border border-slate-200 bg-slate-50"
      >
        <span
          className={`absolute inset-0 ${selected ? '' : 'transition-opacity group-hover:opacity-0'}`}
          style={baseGradient ? { backgroundImage: baseGradient } : undefined}
        />
        {!selected && hoverGradient && (
          <span
            className="absolute inset-0 opacity-0 transition-opacity group-hover:opacity-100"
            style={{ backgroundImage: hoverGradient }}
          />
        )}
      </span>
    </button>
  );
}

// Per-layer prominence (0..1) of `key` within THIS slice (this token position),
// normalized against its busiest layer. Uses the predicted probability so the
// heatmap gradates smoothly across layers (matches the old slider bands).
//
// Presence (the token being in a layer's top_tokens) is tracked SEPARATELY from
// its probability: probabilities can be truncated/rounded to 0.00% upstream, and
// we still want a present token's band to show. A present layer therefore always
// gets a strictly-positive weight (normalized prob when we have signal, else a
// small floor), while an absent layer stays 0 so the band is transparent there.
function sliceLayerWeights(
  slice: LensTypeSlice,
  layers: number[],
  key: string,
  canonicalOf: Map<string, string>,
): number[] {
  const present: boolean[] = [];
  const probs = layers.map((_, i) => {
    const row = slice.top_tokens[i] ?? [];
    const rowProbs = slice.top_probs[i] ?? [];
    let best = 0;
    let found = false;
    for (let j = 0; j < row.length; j += 1) {
      const k = normKey(row[j]);
      if ((canonicalOf.get(k) ?? k) === key) {
        found = true;
        best = Math.max(best, rowProbs[j] ?? 0);
      }
    }
    present.push(found);
    return best;
  });
  const max = Math.max(0, ...probs);
  return probs.map((p, i) => {
    if (!present[i]) {
      return 0;
    }
    const norm = max > 0 ? p / max : 0;
    // Keep present layers strictly positive so `bandCellAlpha`'s floor renders
    // them at minimum opacity even when the probability is (or truncates to) 0.
    return Math.max(norm, Number.MIN_VALUE);
  });
}

// A selected sidebar token's heatmap band: a horizontal stripe colored by its
// per-layer prominence at this position.
type StripBand = { key: string; type: LensType; colorRgb: string; weights: number[] };

// The per-layer selector strip: one horizontal stripe per selected sidebar
// token, plus a per-layer hover overlay. Hovering a layer PREVIEWS the readout
// (transient — cleared on leave); clicking LOCKS it so you can move down and
// select a token (the lock persists across token hovers until cleared). An
// "All Layers" button (left) clears the lock and returns to the overview; it's
// shown as selected whenever no layer is locked.
function LayerSelectorStrip({
  layers,
  bands,
  hoveredLayer,
  lockedLayer,
  onHover,
  onLock,
  onShowAll,
  caption,
}: {
  layers: number[];
  bands: StripBand[];
  hoveredLayer: number | null;
  lockedLayer: number | null;
  onHover: (i: number | null) => void;
  onLock: (i: number) => void;
  // Clear the locked layer (return to the "All Layers" / by-layer overview).
  onShowAll: () => void;
  caption: string;
}) {
  const firstLayer = layers[0];
  const lastLayer = layers[layers.length - 1];
  const n = layers.length;
  const step = n > 0 ? 100 / n : 0;
  // "All Layers" is the selected state whenever no specific layer is locked.
  const allSelected = lockedLayer == null;
  return (
    <div className="hidden shrink-0 px-3 pb-2 sm:block">
      {lockedLayer != null ? (
        <div className="mb-1 flex flex-row items-center justify-center gap-x-2 text-[10px] text-slate-600">
          <span>
            Layer <span className="font-semibold text-slate-700">{layers[lockedLayer]}</span> locked
          </span>
          <button
            type="button"
            onClick={onShowAll}
            className="rounded border border-slate-300 bg-slate-100 px-1.5 py-0.5 text-[9px] font-semibold uppercase leading-none tracking-wide text-slate-600 transition-colors hover:bg-slate-200 hover:text-slate-800"
          >
            Unlock
          </button>
        </div>
      ) : (
        <div className="mb-1 text-center text-[10px] text-slate-500">{caption}</div>
      )}
      <div className="flex flex-row items-stretch gap-x-3">
        <button
          type="button"
          onClick={onShowAll}
          className={`flex w-14 shrink-0 items-center justify-center rounded-md border px-1 text-center text-[9px] font-semibold uppercase leading-tight tracking-wide transition-colors ${
            allSelected
              ? 'border-slate-500 bg-slate-200 text-slate-800'
              : 'border-slate-300 bg-slate-100 text-slate-600 hover:bg-slate-200 hover:text-slate-800'
          }`}
        >
          All
          <br />
          Layers
        </button>
        <div className="relative flex-1">
          <div className="relative h-8 w-full overflow-hidden rounded-md border border-slate-200 bg-slate-50">
            {bands.length > 0 ? (
              <div className="absolute inset-0 flex flex-col">
                {bands.map((band) => (
                  <div key={`${band.type}-${band.key}`} className="flex min-h-0 flex-1 flex-row">
                    {band.weights.map((w, i) => (
                      <span
                        key={i}
                        className="h-full flex-1"
                        style={{ backgroundColor: `rgba(${band.colorRgb}, ${bandCellAlpha(w)})` }}
                      />
                    ))}
                  </div>
                ))}
              </div>
            ) : (
              <div className="absolute inset-0 flex flex-row">
                {layers.map((layer) => (
                  <span key={layer} className="h-full flex-1 bg-slate-200" />
                ))}
              </div>
            )}
            {/* Per-layer hover/click interaction + highlight. Hover previews,
                click locks; leaving clears only the (transient) hover. */}
            <div className="absolute inset-0 flex flex-row" onMouseLeave={() => onHover(null)}>
              {layers.map((layer, i) => (
                <span
                  key={layer}
                  onMouseEnter={() => onHover(i)}
                  onClick={() => onLock(i)}
                  className={`h-full flex-1 cursor-pointer ${i > 0 ? 'border-l border-slate-400/10' : ''} ${
                    lockedLayer === i
                      ? 'bg-slate-500/25 ring-2 ring-inset ring-slate-600'
                      : hoveredLayer === i
                        ? 'bg-slate-500/20 ring-1 ring-inset ring-slate-500'
                        : 'hover:bg-slate-900/10'
                  }`}
                />
              ))}
            </div>
          </div>
          {/* Lightweight "Layer N" tooltip over the hovered slot. */}
          {hoveredLayer != null && (
            <div
              className="pointer-events-none absolute -top-4 z-10 -translate-x-1/2 whitespace-nowrap rounded border border-slate-500 bg-white px-1.5 py-0.5 text-[8px] font-bold tabular-nums leading-none text-slate-700 shadow-sm"
              style={{ left: `${(hoveredLayer + 0.5) * step}%` }}
            >
              Layer {layers[hoveredLayer]}
            </div>
          )}
          <div className="mt-0.5 flex flex-row justify-between text-[8px] uppercase tracking-wide text-slate-400">
            <div>Layer {firstLayer}</div>
            <div>Layer {lastLayer}</div>
          </div>
        </div>
      </div>
    </div>
  );
}

// One row of the "Top-1 by layer" overview: the layer number, that layer's
// top-1 token, and the token's per-layer occurrence mini heatmap. Clicking
// toggles the token's sidebar selection.
function TopByLayerRow({
  layer,
  tokenKey,
  token,
  perLayer,
  layers,
  range,
  type,
  selectedColorRgb,
  nextColorRgb,
  onToggle,
  onSteer,
  onHoverPreview,
}: {
  layer: number;
  // The token's normalized key (used to steer/swap it).
  tokenKey: string;
  token: string;
  perLayer: number[];
  layers: number[];
  range: LayerRange | null;
  // The lens type this row belongs to (used to steer/swap it).
  type: LensType;
  selectedColorRgb: string | null;
  nextColorRgb: string;
  onToggle: () => void;
  // Open the steer/swap flow for this token (hover buttons); omitted when steering unavailable.
  onSteer?: PopupSteerHandler;
  // Notified on hover enter/leave so the chat tokens preview this token.
  onHoverPreview?: (hovering: boolean) => void;
}) {
  const [hover, setHover] = useState(false);
  const translate = useChineseTranslation();
  const translation = translate(token) ?? null;
  const max = Math.max(0, ...perLayer);
  const selected = selectedColorRgb !== null;
  const inRangeLayer = (i: number) => range == null || (layers[i] >= range[0] && layers[i] <= range[1]);
  const baseRgb = selected ? selectedColorRgb : POPUP_BAR_RGB;
  const baseGradient = layerBarGradient(perLayer, max, (i) => (inRangeLayer(i) ? baseRgb : POPUP_BAR_RGB));
  const hoverGradient = selected
    ? undefined
    : layerBarGradient(perLayer, max, (i) => (inRangeLayer(i) ? nextColorRgb : POPUP_BAR_RGB));
  const rowStyle: CSSProperties | undefined = selected
    ? { backgroundColor: `rgba(${selectedColorRgb}, 0.12)`, boxShadow: `inset 0 0 0 1px rgb(${selectedColorRgb})` }
    : hover
      ? { backgroundColor: `rgba(${nextColorRgb}, 0.1)`, boxShadow: `inset 0 0 0 1px rgba(${nextColorRgb}, 0.6)` }
      : undefined;
  return (
    <button
      type="button"
      onClick={onToggle}
      onMouseEnter={() => {
        setHover(true);
        onHoverPreview?.(true);
      }}
      onMouseLeave={() => {
        setHover(false);
        onHoverPreview?.(false);
      }}
      style={rowStyle}
      className="group mx-0 flex h-5 max-h-5 min-h-5 w-full flex-row items-center gap-x-1.5 rounded px-2 py-0.5 text-left font-mono text-[10px] transition-colors"
    >
      <span className="w-10 shrink-0 text-[9px] tabular-nums text-slate-400">{layer}</span>
      {selected ? (
        <span
          className="h-2.5 w-2.5 shrink-0 rounded-sm"
          style={{ backgroundColor: `rgb(${selectedColorRgb})` }}
          aria-hidden
        />
      ) : (
        <span className="relative h-2.5 w-2.5 shrink-0" aria-hidden>
          {/* Faint placeholder outline when neither hovered nor selected. */}
          <span className="absolute inset-0 rounded-sm border border-slate-400 opacity-70 transition-opacity group-hover:opacity-0" />
          {/* Anticipated next-color outline that fades in on hover. */}
          <span
            className="absolute inset-0 rounded-sm border opacity-0 transition-opacity group-hover:opacity-100"
            style={{ borderColor: `rgb(${nextColorRgb})` }}
          />
        </span>
      )}
      <span className="relative flex min-w-0 flex-1 flex-row items-center gap-x-1.5">
        <span className="min-w-0 truncate">{displayToken(token)}</span>
        {translation && (
          <span
            className="hidden min-w-0 flex-1 truncate font-sans text-[10px] text-slate-400 sm:block"
            title={translation}
          >
            {translation}
          </span>
        )}
        {onSteer && (
          <PopupSteerButtons
            tokenKey={tokenKey}
            token={token}
            type={type}
            colorRgb={selectedColorRgb ?? nextColorRgb}
            onSteer={onSteer}
          />
        )}
      </span>
      <span
        aria-hidden
        className="relative hidden h-3.5 w-1/2 shrink-0 overflow-hidden rounded-sm border border-slate-200 bg-slate-50 sm:flex"
      >
        <span
          className={`absolute inset-0 ${selected ? '' : 'transition-opacity group-hover:opacity-0'}`}
          style={baseGradient ? { backgroundImage: baseGradient } : undefined}
        />
        {!selected && hoverGradient && (
          <span
            className="absolute inset-0 opacity-0 transition-opacity group-hover:opacity-100"
            style={{ backgroundImage: hoverGradient }}
          />
        )}
      </span>
    </button>
  );
}

// One lens type's readout. When `activeLayer` is set (hovered or locked), shows
// that layer's top-n tokens. Otherwise shows the "Top-1 by layer" overview:
// each in-range layer (descending) with its top-1 token + count-by-layer bar.
function LayerReadout({
  slice,
  layers,
  activeLayer,
  range,
  scrollRef,
  onScroll,
}: {
  slice: LensTypeSlice;
  layers: number[];
  activeLayer: number | null;
  range: LayerRange | null;
  // Optional scroll-sync hooks (DIFF mode): attach to the scroll container so
  // the two side-by-side readouts can mirror each other's scroll position.
  scrollRef?: (el: HTMLDivElement | null) => void;
  onScroll?: (e: UIEvent<HTMLDivElement>) => void;
}) {
  const sliderCtx = useContext(LensSliderContext);
  const layerStats = useContext(LayerStatsContext);
  const onSteer = useContext(PopupSteerContext);
  // Selection plumbing for the readout rows (mirrors the sidebar squares).
  const selectedColorByKey = useMemo(() => {
    const sel = (sliderCtx?.selected ?? []).filter((s) => s.type === slice.type);
    return new Map(sel.map((s) => [s.key, s.colorRgb]));
  }, [sliderCtx, slice.type]);
  const nextColorRgb = sliderCtx?.nextColorRgb ?? POPUP_BAR_RGB;
  const lensLabel = slice.type === LensType.JACOBIAN_LENS ? 'J-Lens Readout' : 'Logit Lens';
  // For the J-Lens, warn when the active layer sits before the default layer
  // selection (the first ~1/3 of the model), where readouts are unreliable.
  const showDegenerateWarning =
    slice.type === LensType.JACOBIAN_LENS && activeLayer != null && isDegenerateLayer(layers, layers[activeLayer]);

  // The active layer's top-n readout, each token carrying its OVERALL
  // prominence-by-layer (across all positions) for the right-side stripes.
  const layerResultRows = useMemo<(PositionToken & { prob: number })[]>(() => {
    if (activeLayer == null) {
      return [];
    }
    const row = slice.top_tokens[activeLayer] ?? [];
    const probs = slice.top_probs[activeLayer] ?? [];
    return row.map((t, j) => {
      const w = layerStats?.(slice.type, t) ?? [];
      const perLayer = layers.map((_, idx) => w[idx] ?? 0);
      return { key: normKey(t), token: t, count: 0, perLayer, inRange: true, isOutput: false, prob: probs[j] ?? 0 };
    });
  }, [activeLayer, slice, layers, layerStats]);

  // The "Top-1 by layer" overview: each in-range layer (descending) with its
  // top-1 token + that token's overall per-layer mini heatmap.
  const topByLayerRows = useMemo(() => {
    if (activeLayer != null) {
      return [];
    }
    const idxs = visibleLayerIndices(layers, range);
    return [...idxs].reverse().map((idx) => {
      const display = slice.top_tokens[idx]?.[0] ?? '';
      const key = normKey(display);
      const w = layerStats?.(slice.type, key) ?? [];
      const perLayer = layers.map((_, k) => w[k] ?? 0);
      return { layer: layers[idx], key, token: display, perLayer };
    });
  }, [activeLayer, slice, layers, range, layerStats]);

  return (
    <div className="flex max-h-[378px] min-h-[378px] w-full flex-col border-t-4 border-slate-200">
      {activeLayer != null ? (
        <>
          <div className="flex shrink-0 flex-row items-center justify-between border-b border-slate-200 px-5 py-1.5 text-[9px] uppercase tracking-wide text-slate-500">
            <span>
              Layer <span className="font-bold text-slate-700">{layers[activeLayer]}</span>{' '}
              {slice.type === LensType.JACOBIAN_LENS ? 'J-Lens Readout' : 'Logit Lens'}
            </span>
            <span className="text-slate-400">Count by layer</span>
          </div>
          <div
            ref={scrollRef}
            onScroll={onScroll}
            className="flex min-h-0 flex-1 flex-col gap-y-0.5 overflow-y-auto px-5 py-2"
          >
            {showDegenerateWarning && (
              <div className="mb-1 shrink-0 rounded border border-amber-200 bg-amber-50 px-2 py-1 text-[10px] leading-snug text-amber-700">
                {JLENS_DEGENERATE_DISCLAIMER}
              </div>
            )}
            {layerResultRows.map((it, j) => (
              <PositionTokenRow
                key={`${j}-${it.key}`}
                tokenKey={it.key}
                token={it.token}
                count={it.count}
                perLayer={it.perLayer}
                inRange={it.inRange}
                isOutput={it.isOutput}
                layers={layers}
                range={range}
                type={slice.type}
                selectedColorRgb={selectedColorByKey.get(it.key) ?? null}
                nextColorRgb={nextColorRgb}
                prob={it.prob}
                onToggle={() => sliderCtx?.onToggle(it.key, slice.type)}
                onSteer={onSteer ?? undefined}
                onHoverPreview={(h) => sliderCtx?.onHoverPreview?.(h ? { key: it.key, type: slice.type } : null)}
              />
            ))}
          </div>
        </>
      ) : (
        <>
          <div className="flex shrink-0 flex-row items-center gap-x-1.5 border-b border-slate-200 px-2 py-1.5 text-[9px] uppercase tracking-wide text-slate-500 sm:px-5">
            <span className="w-8 shrink-0 sm:w-10">Layer</span>
            <span className="min-w-0 flex-1 whitespace-nowrap">{lensLabel}</span>
            <span className="hidden w-1/2 shrink-0 text-slate-400 sm:block">Count by layer</span>
          </div>
          <div
            ref={scrollRef}
            onScroll={onScroll}
            className="flex min-h-0 flex-1 flex-col gap-y-0.5 overflow-y-auto px-2 py-2 sm:px-5"
          >
            {topByLayerRows.map((r) => (
              <TopByLayerRow
                key={r.layer}
                layer={r.layer}
                tokenKey={r.key}
                token={r.token}
                perLayer={r.perLayer}
                layers={layers}
                range={range}
                type={slice.type}
                selectedColorRgb={selectedColorByKey.get(r.key) ?? null}
                nextColorRgb={nextColorRgb}
                onToggle={() => sliderCtx?.onToggle(r.key, slice.type)}
                onSteer={onSteer ?? undefined}
                onHoverPreview={(h) => sliderCtx?.onHoverPreview?.(h ? { key: r.key, type: slice.type } : null)}
              />
            ))}
          </div>
        </>
      )}
    </div>
  );
}

// One lens type's column in the popup: a layer selector strip on top and the
// active layer's readout below. Used in single-lens modes.
function LensColumn({ slice, layersByType }: { slice: LensTypeSlice; layersByType: Record<string, number[]> }) {
  const layers = useMemo(() => layersByType[slice.type] ?? [], [layersByType, slice.type]);
  // Transient hover preview (local) vs. a locked (clicked) layer that lives in
  // the shared analysis state so it persists across token hovers; hover takes
  // priority for the readout shown.
  const [hoveredLayer, setHoveredLayer] = useState<number | null>(null);
  const sliderCtx = useContext(LensSliderContext);
  const lockedLayer = sliderCtx?.lockedLayer ?? null;
  const setLockedLayer = sliderCtx?.setLockedLayer;
  const activeLayer = hoveredLayer ?? lockedLayer;
  const steerActive = useContext(SteerActiveContext);
  const range = sliderCtx?.range ?? null;

  // One heatmap row per selected sidebar token (of this lens type). While
  // steering, restrict to the steered/swapped token so its band fills the strip.
  const bands = useMemo<StripBand[]>(() => {
    let sel = (sliderCtx?.selected ?? []).filter((s) => s.type === slice.type);
    if (steerActive) {
      sel = sel.filter((s) => s.key === steerActive.key && s.type === steerActive.type);
    }
    return sel.map((s) => ({
      key: s.key,
      type: s.type,
      colorRgb: s.colorRgb,
      weights: sliceLayerWeights(slice, layers, s.key, s.canonicalOf),
    }));
  }, [sliderCtx, slice, layers, steerActive]);

  return (
    <div className="flex w-full flex-col gap-y-0 sm:pt-3">
      <LayerSelectorStrip
        layers={layers}
        bands={bands}
        hoveredLayer={hoveredLayer}
        lockedLayer={lockedLayer}
        onHover={setHoveredLayer}
        onLock={(i) => setLockedLayer?.(lockedLayer === i ? null : i)}
        onShowAll={() => setLockedLayer?.(null)}
        caption={`Hover a layer to preview, click to lock its ${
          slice.type === LensType.JACOBIAN_LENS ? 'J-Lens readouts' : 'Logit Lens tokens'
        }.`}
      />
      <LayerReadout slice={slice} layers={layers} activeLayer={activeLayer} range={range} />
    </div>
  );
}

// DIFF mode: a SINGLE combined layer selector strip (striped horizontally by
// every selected sidebar token across both lenses) drives two readouts below,
// kept split left/right by lens type.
function CombinedLensColumns({
  slices,
  layersByType,
}: {
  slices: LensTypeSlice[];
  layersByType: Record<string, number[]>;
}) {
  const sliderCtx = useContext(LensSliderContext);
  const steerActive = useContext(SteerActiveContext);
  const range = sliderCtx?.range ?? null;
  const [hoveredLayer, setHoveredLayer] = useState<number | null>(null);
  const lockedLayer = sliderCtx?.lockedLayer ?? null;
  const setLockedLayer = sliderCtx?.setLockedLayer;
  const activeLayer = hoveredLayer ?? lockedLayer;
  // Layers are shared across lens types (the model's layers); use the first.
  const layers = layersByType[slices[0]?.type] ?? [];

  // Keep the two readouts' scroll positions mirrored: scrolling one scrolls the
  // other to the same offset. The equality guard avoids a programmatic-scroll
  // feedback loop between the paired containers.
  const scrollEls = useRef<(HTMLDivElement | null)[]>([]);
  const syncScroll = (idx: number) => (e: UIEvent<HTMLDivElement>) => {
    const top = e.currentTarget.scrollTop;
    scrollEls.current.forEach((el, i) => {
      if (i !== idx && el && el.scrollTop !== top) {
        el.scrollTop = top;
      }
    });
  };

  // One horizontal stripe per selected token, across both lenses, each colored
  // by its per-layer prominence within its own lens slice. While steering,
  // restrict to the steered/swapped token so its band fills the strip.
  const bands = useMemo<StripBand[]>(() => {
    const sel = sliderCtx?.selected ?? [];
    const out: StripBand[] = [];
    for (const slice of slices) {
      const layersT = layersByType[slice.type] ?? [];
      let selT = sel.filter((x) => x.type === slice.type);
      if (steerActive) {
        selT = selT.filter((s) => s.key === steerActive.key && s.type === steerActive.type);
      }
      for (const s of selT) {
        out.push({
          key: s.key,
          type: s.type,
          colorRgb: s.colorRgb,
          weights: sliceLayerWeights(slice, layersT, s.key, s.canonicalOf),
        });
      }
    }
    return out;
  }, [sliderCtx, slices, layersByType, steerActive]);

  return (
    <div className="flex w-full flex-col gap-y-0 sm:pt-3">
      <LayerSelectorStrip
        layers={layers}
        bands={bands}
        hoveredLayer={hoveredLayer}
        lockedLayer={lockedLayer}
        onHover={setHoveredLayer}
        onLock={(i) => setLockedLayer?.(lockedLayer === i ? null : i)}
        onShowAll={() => setLockedLayer?.(null)}
        caption="Hover a layer to preview, click to lock its J-Lens & Logit Lens readouts."
      />
      <div className="flex flex-row divide-x-2 divide-slate-200">
        {slices.map((slice, idx) => (
          <LayerReadout
            key={slice.type}
            slice={slice}
            layers={layersByType[slice.type] ?? []}
            activeLayer={activeLayer}
            range={range}
            scrollRef={(el) => {
              scrollEls.current[idx] = el;
            }}
            onScroll={syncScroll(idx)}
          />
        ))}
      </div>
    </div>
  );
}

// The hovered-token popup body: one column per displayed lens type (two
// side-by-side when mode is DIFF). Mounted only while hovered.
export default function JlensTokenPopup({
  token,
  layersByType,
  layerRange = null,
}: {
  token: LensTokenMessage;
  layersByType: Record<string, number[]>;
  // The selected layer range, shown in the static header.
  layerRange?: LayerRange | null;
}) {
  const mode = useContext(LensModeContext);
  const byType = useMemo(() => new Map(token.results.map((r) => [r.type, r] as const)), [token.results]);
  const displayedTypes = useMemo(() => lensTypesForMode(mode).filter((t) => byType.has(t)), [mode, byType]);

  if (displayedTypes.length === 0) {
    return null;
  }

  const twoColumn = mode === LensMode.DIFF;
  const lensLabel = twoColumn ? 'J-Space & Logit Lens' : mode === LensMode.JACOBIAN_LENS ? 'J-Space' : 'Logit Lens';

  return (
    <div className="flex w-full flex-col">
      {/* Static 3-column header: lens / layer(s) / position (no controls). */}
      <div className="flex h-10 w-full flex-row items-stretch border-b border-slate-200 bg-white text-[12px] leading-none text-slate-600">
        <div className="flex flex-1 items-center justify-center gap-x-2 px-3 text-center">
          <span className="rounded bg-slate-200 px-1.5 py-1 font-mono text-slate-700">{displayToken(token.token)}</span>
          <span>
            Position <span className="font-semibold text-slate-700">{token.position}</span>
          </span>
        </div>
        <div className="flex hidden flex-1 items-center justify-center border-l border-slate-200 px-3 text-center">
          <span>
            {layerRange ? (
              layerRange[0] === layerRange[1] ? (
                <>
                  Layer <span className="font-semibold text-slate-700">{layerRange[0]}</span>
                </>
              ) : (
                <>
                  Layers <span className="font-semibold text-slate-700">{layerRange[0]}</span> to{' '}
                  <span className="font-semibold text-slate-700">{layerRange[1]}</span>
                </>
              )
            ) : (
              'All layers'
            )}
          </span>
        </div>
        <div className="hidden flex-1 items-center justify-center border-l border-slate-200 px-3 text-center sm:flex">
          <strong className="text-slate-600">{lensLabel}</strong>
        </div>
      </div>
      {twoColumn ? (
        <CombinedLensColumns
          slices={displayedTypes.map((t) => byType.get(t)).filter((s): s is LensTypeSlice => s != null)}
          layersByType={layersByType}
        />
      ) : (
        <div className="flex w-full flex-row gap-x-0 divide-x divide-slate-200 px-0 py-0">
          {displayedTypes.map((t) => {
            const slice = byType.get(t);
            return slice ? <LensColumn key={t} slice={slice} layersByType={layersByType} /> : null;
          })}
        </div>
      )}
    </div>
  );
}
