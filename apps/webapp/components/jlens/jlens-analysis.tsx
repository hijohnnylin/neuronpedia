'use client';

// Shared analysis primitives for the jlens interfaces (chat + completion):
// the sidebar aggregation helpers, highlight/opacity math, color tables, and
// the presentational sidebar column/row components. These are pure (or purely
// presentational) so both the chat and completion panels can reuse them.

import { JLENS_JACOBIAN_SPACE_ID, JLENS_STEER_SPIDER_ID } from '@/app/[modelId]/jlens/jlens-tour-constants';
import { useJlensTourStep } from '@/app/[modelId]/jlens/jlens-tour-context';
import { useChineseTranslation } from '@/lib/utils/chinese-translations';
import { LENS_TYPE_ORDER, LensMode, LensTokenMessage, LensType } from '@/lib/utils/lens';
import { Search, X } from 'lucide-react';
import { useContext, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { LayerWeight } from './jlens-layer-slider';
import { LensModeContext } from './jlens-lens-mode';
import {
  DEDUPE_SIMILAR_TOKENS,
  displayToken,
  isWordLikeToken,
  LayerRange,
  normKey,
  visibleLayerIndices,
} from './jlens-token-popup';

// Up to 5 sidebar tokens can be selected for analysis; each gets a color by
// selection order. "red" is the over-cap warning color.
export const MAX_SELECT = 4;
export const SELECT_COLORS = ['sky', 'amber', 'emerald', 'indigo', 'stone'] as const;
export type SelectColor = (typeof SELECT_COLORS)[number] | 'red';
export type SelectedToken = { key: string; type: LensType };
export type TokenViz = { color: SelectColor; opacityByPosition: Map<number, number>; layerWeights: LayerWeight[] };
// `border` marks a band that should be drawn as the chip's border (a transient
// outline preview) instead of a solid background fill.
export type DisplayBand = TokenViz & { emphasis: number; border?: boolean };

export const COLOR_RGB: Record<SelectColor, string> = {
  sky: '14, 165, 233',
  amber: '245, 158, 11',
  emerald: '16, 185, 129',
  stone: '120, 113, 108',
  indigo: '99, 102, 241',
  red: '239, 68, 68',
};
// Two Tailwind shades darker (-700) than COLOR_RGB (-500), as "r, g, b" strings.
export const COLOR_RGB_DARK: Record<SelectColor, string> = {
  sky: '3, 105, 161',
  amber: '180, 83, 9',
  emerald: '4, 120, 87',
  stone: '68, 64, 60',
  indigo: '67, 56, 202',
  red: '185, 28, 28',
};
// Per-band opacity multipliers for the hover states.
export const EMPH_NORMAL = 1; // baseline (and the hovered selected band)

// Exact sidebar chip colors (bg-*-100 fill, ring-*-400 border) as "r, g, b"
// strings, so popup pills can match the sidebar selection exactly.
export const COLOR_PILL: Record<SelectColor, { bg: string; ring: string }> = {
  sky: { bg: '224, 242, 254', ring: '56, 189, 248' },
  amber: { bg: '254, 243, 199', ring: '251, 191, 36' },
  emerald: { bg: '209, 250, 229', ring: '52, 211, 153' },
  stone: { bg: '245, 245, 244', ring: '168, 162, 158' },
  indigo: { bg: '224, 231, 255', ring: '129, 140, 248' },
  red: { bg: '254, 226, 226', ring: '248, 113, 113' },
};

// Sidebar selected-token chip styling per color (static classes for Tailwind).
export const COLOR_SIDEBAR: Record<SelectColor, string> = {
  sky: 'bg-sky-100 ring-1 ring-inset ring-sky-400',
  amber: 'bg-amber-100 ring-1 ring-inset ring-amber-400',
  emerald: 'bg-emerald-100 ring-1 ring-inset ring-emerald-400',
  stone: 'bg-stone-100 ring-1 ring-inset ring-stone-400',
  indigo: 'bg-indigo-100 ring-1 ring-inset ring-indigo-400',
  red: 'bg-red-100 ring-1 ring-inset ring-red-400',
};

// Lighter hover preview (anticipated next color) for non-selected rows.
export const COLOR_SIDEBAR_HOVER: Record<SelectColor, string> = {
  sky: 'hover:bg-sky-50 hover:ring-1 hover:ring-inset hover:ring-sky-300',
  amber: 'hover:bg-amber-50 hover:ring-1 hover:ring-inset hover:ring-amber-300',
  emerald: 'hover:bg-emerald-50 hover:ring-1 hover:ring-inset hover:ring-emerald-300',
  stone: 'hover:bg-stone-50 hover:ring-1 hover:ring-inset hover:ring-stone-300',
  indigo: 'hover:bg-indigo-50 hover:ring-1 hover:ring-inset hover:ring-indigo-300',
  red: 'hover:bg-red-50 hover:ring-1 hover:ring-inset hover:ring-red-300',
};

export type CommonToken = { key: string; token: string; count: number };
// Per-layer prominence stat for one token: the layer number, its normalized
// weight (0..1, against that token's busiest layer) and the raw occurrence count.
export type LayerStat = { layer: number; weight: number; count: number };
export type TypeSidebar = {
  // The capped (top-K) list that drives the default render.
  items: CommonToken[];
  // The FULL merged list (every predicted token in range), sorted by count.
  allItems: CommonToken[];
  // Range-aware occurrence count for EVERY key (not just the top-K).
  countByKey: Map<string, number>;
  canonicalOf: Map<string, string>;
  // Per-key prominence-by-layer over ALL layers, used for the row mini heatmap.
  layerStatsByKey: Map<string, LayerStat[]>;
  // DIFF mode only: the "thisLens:otherLens" count label shown for EVERY key in
  // place of a single count (e.g. "8:0"). Undefined in the other modes.
  countLabelByKey?: Map<string, string>;
  // DIFF mode only: the FULL search pool for this column — every token known to
  // EITHER lens (not just the top-diff `allItems`), so search can surface any
  // known result. Undefined in single-lens modes (which search `allItems`).
  searchItems?: CommonToken[];
};

// Additive smoothing (pseudocount) for the DIFF-mode ratio score. Added to both
// sides before dividing so all-or-nothing tokens with tiny support (e.g. 1 vs 0)
// stay near the neutral ratio of 1 instead of dominating the top.
export const DIFF_RATIO_ALPHA = 2;

// Base "r, g, b" for the per-row prominence stripes before a token is assigned
// a color (matches the slider's unselected slate).
export const PROMINENCE_SLATE_RGB = '100, 116, 139';

// Max number of common tokens to list per lens type.
export const SIDEBAR_TOP_K = 100;

// Max number of rows to render for an active token search.
export const SIDEBAR_SEARCH_CAP = 100;

// Fixed row height (px) for sidebar token rows.
export const SIDEBAR_ROW_H = 32;

// Height (px) of the sticky column header row.
export const SIDEBAR_HEADER_H = 24;

// Extra gap (px) kept between the bottom-pinned sticky rows and the bottom edge.
export const SIDEBAR_STICKY_BOTTOM = 16;

// `normKey` / `DEDUPE_SIMILAR_TOKENS` are defined in `jlens-token-popup` (the
// lower-level module) so the popup can key tokens identically to the sidebar.
// Re-exported here for callers that import them from this module.
export { DEDUPE_SIMILAR_TOKENS, normKey };

// Resolve the token position under a pointer event target (the token chip
// glyphs carry `data-token-position`), or null when the pointer isn't on a token.
export function tokenPositionFromTarget(target: EventTarget | null): number | null {
  if (!(target instanceof HTMLElement)) {
    return null;
  }
  const el = target.closest('[data-token-position]');
  if (!el) {
    return null;
  }
  const pos = Number(el.getAttribute('data-token-position'));
  return Number.isFinite(pos) ? pos : null;
}

// Non-word filtering happens server-side, but the server always preserves each
// layer's TRUE top-1 (the model's output token), so special/markup tokens like
// "<|im_end|>" can still dominate the SIDEBAR LIST aggregation. When the
// non-word filter is on we additionally drop tokens starting with "<" from the
// sidebar list ONLY — the hover popup and the highlight/stats resolvers keep
// them (they read the per-key maps, not the list). This is the ONLY remaining
// client-side token filtering.
function startsWithAngleBracket(s: string): boolean {
  return s.trimStart().startsWith('<');
}
function dropAngleBracketItems(items: CommonToken[]): CommonToken[] {
  return items.filter((it) => !startsWithAngleBracket(it.token) && !startsWithAngleBracket(it.key));
}

// Build per-type sidebar data: counts of every predicted token within the layer
// range, plus a canonicalizer reused for chat highlighting. Returns the top-K.
// `hideAngleBracketTokens` drops "<"-prefixed tokens from the returned list
// (items/allItems) only; the per-key maps are left intact.
export function buildSidebar(
  tokens: LensTokenMessage[],
  layersByType: Record<string, number[]>,
  range: LayerRange | null,
  hideNonWord: boolean,
  hideAngleBracketTokens = false,
): Record<string, TypeSidebar> {
  const out: Record<string, TypeSidebar> = {};
  for (const type of LENS_TYPE_ORDER) {
    const layers = layersByType[type] ?? [];
    const counts = new Map<string, number>();
    // normKey -> (display string -> sub-count), to pick the best display form.
    const displays = new Map<string, Map<string, number>>();
    // normKey -> per-layer occurrence counts over ALL layers (for the mini heatmap).
    const layerCounts = new Map<string, number[]>();

    for (const tok of tokens) {
      const slice = tok.results.find((r) => r.type === type);
      if (!slice) {
        continue;
      }
      for (const i of visibleLayerIndices(layers, range)) {
        const row = slice.top_tokens[i];
        if (!row) {
          continue;
        }
        for (const t of row) {
          if (hideNonWord && !isWordLikeToken(t)) {
            continue;
          }
          const k = normKey(t);
          if (k.trim() === '') {
            continue;
          }
          counts.set(k, (counts.get(k) ?? 0) + 1);
          // When deduping is off, the key IS the exact token, so display it
          // verbatim (whitespace preserved). When deduping, strip whitespace.
          const disp = DEDUPE_SIMILAR_TOKENS ? t.replace(/^\s+/, '').replace(/\s+$/, '') : t;
          let dm = displays.get(k);
          if (!dm) {
            dm = new Map();
            displays.set(k, dm);
          }
          dm.set(disp, (dm.get(disp) ?? 0) + 1);
        }
      }
      // Prominence-by-layer over ALL layers (independent of the selected range).
      layers.forEach((_, i) => {
        const row = slice.top_tokens[i];
        if (!row) {
          return;
        }
        for (const t of row) {
          if (hideNonWord && !isWordLikeToken(t)) {
            continue;
          }
          const k = normKey(t);
          if (k.trim() === '') {
            continue;
          }
          let arr = layerCounts.get(k);
          if (!arr) {
            arr = layers.map(() => 0);
            layerCounts.set(k, arr);
          }
          arr[i] += 1;
        }
      });
    }

    // Plural merge: fold a trailing-"s" key into its singular when present.
    const canonicalOf = new Map<string, string>();
    for (const k of counts.keys()) {
      if (DEDUPE_SIMILAR_TOKENS) {
        const singular = k.length > 1 && k.endsWith('s') ? k.slice(0, -1) : k;
        canonicalOf.set(k, singular !== k && counts.has(singular) ? singular : k);
      } else {
        canonicalOf.set(k, k);
      }
    }

    const merged = new Map<string, number>();
    const mergedDisplays = new Map<string, Map<string, number>>();
    for (const [k, c] of counts) {
      const ck = canonicalOf.get(k) ?? k;
      merged.set(ck, (merged.get(ck) ?? 0) + c);
      let mdm = mergedDisplays.get(ck);
      if (!mdm) {
        mdm = new Map();
        mergedDisplays.set(ck, mdm);
      }
      for (const [disp, dc] of displays.get(k) ?? []) {
        mdm.set(disp, (mdm.get(disp) ?? 0) + dc);
      }
    }

    const bestDisplay = (key: string): string => {
      let best = key;
      let bestCount = -1;
      for (const [disp, dc] of mergedDisplays.get(key) ?? []) {
        if (dc > bestCount) {
          bestCount = dc;
          best = disp;
        }
      }
      return best;
    };
    const allItemsUnfiltered: CommonToken[] = [...merged.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([key, count]) => ({ key, token: bestDisplay(key), count }));
    const allItems = hideAngleBracketTokens ? dropAngleBracketItems(allItemsUnfiltered) : allItemsUnfiltered;
    const items = allItems.slice(0, SIDEBAR_TOP_K);

    // Merge per-layer counts into canonical keys, then normalize each key 0..1
    // against its own busiest layer.
    const mergedLayerCounts = new Map<string, number[]>();
    for (const [k, arr] of layerCounts) {
      const ck = canonicalOf.get(k) ?? k;
      let m = mergedLayerCounts.get(ck);
      if (!m) {
        m = layers.map(() => 0);
        mergedLayerCounts.set(ck, m);
      }
      for (let i = 0; i < arr.length; i += 1) {
        m[i] += arr[i];
      }
    }
    const layerStatsByKey = new Map<string, LayerStat[]>();
    for (const [ck, arr] of mergedLayerCounts) {
      const maxLayer = Math.max(0, ...arr);
      layerStatsByKey.set(
        ck,
        layers.map((layer, i) => ({ layer, weight: maxLayer > 0 ? arr[i] / maxLayer : 0, count: arr[i] })),
      );
    }

    out[type] = { items, allItems, countByKey: merged, canonicalOf, layerStatsByKey };
  }
  return out;
}

// Like `buildSidebar`, but always surfaces each position's model output token
// (the last-layer top-1) per lens type — even when it's outside the selected
// layer range or a non-word token.
export function buildSidebarWithOutputTokens(
  subset: LensTokenMessage[],
  layersByType: Record<string, number[]>,
  range: LayerRange | null,
  hideNonWord: boolean,
  hideAngleBracketTokens = false,
): Record<string, TypeSidebar> {
  const base = buildSidebar(subset, layersByType, range, hideNonWord, hideAngleBracketTokens);
  for (const type of LENS_TYPE_ORDER) {
    const layers = layersByType[type] ?? [];
    const sb = base[type];
    if (!sb || layers.length === 0) {
      continue;
    }
    const lastIdx = layers.length - 1;
    const outputs = new Map<string, string>();
    for (const tok of subset) {
      const slice = tok.results.find((r) => r.type === type);
      const raw = slice?.top_tokens[lastIdx]?.[0];
      if (raw == null) {
        continue;
      }
      const key = normKey(raw);
      if (key.trim() !== '' && !outputs.has(key)) {
        outputs.set(key, DEDUPE_SIMILAR_TOKENS ? raw.trim() : raw);
      }
    }
    const existing = new Set(sb.items.map((it) => it.key));
    const inject: CommonToken[] = [];
    for (const [key, display] of outputs) {
      if (existing.has(key)) {
        continue;
      }
      let count = 0;
      const perLayer = layers.map(() => 0);
      for (const tok of subset) {
        const slice = tok.results.find((r) => r.type === type);
        if (!slice) {
          continue;
        }
        for (const i of visibleLayerIndices(layers, range)) {
          for (const t of slice.top_tokens[i] ?? []) {
            if (normKey(t) === key) {
              count += 1;
            }
          }
        }
        layers.forEach((_, i) => {
          for (const t of slice.top_tokens[i] ?? []) {
            if (normKey(t) === key) {
              perLayer[i] += 1;
            }
          }
        });
      }
      const maxLayer = Math.max(0, ...perLayer);
      sb.layerStatsByKey.set(
        key,
        layers.map((layer, i) => ({ layer, weight: maxLayer > 0 ? perLayer[i] / maxLayer : 0, count: perLayer[i] })),
      );
      sb.canonicalOf.set(key, key);
      if (!sb.countByKey.has(key)) {
        sb.countByKey.set(key, count);
        sb.allItems.push({ key, token: display, count });
      }
      inject.push({ key, token: display, count });
    }
    if (inject.length > 0) {
      sb.items = [...sb.items, ...inject].sort((a, b) => b.count - a.count);
      sb.allItems = sb.allItems.sort((a, b) => b.count - a.count);
    }
    // Re-apply the list filter: the injected output tokens (e.g. "<|im_end|>")
    // bypass `buildSidebar`'s filter, so drop them from the displayed list here
    // (the per-key maps set above keep them for the popup/resolvers).
    if (hideAngleBracketTokens) {
      sb.items = dropAngleBracketItems(sb.items);
      sb.allItems = dropAngleBracketItems(sb.allItems);
    }
  }
  return base;
}

// Transform a per-type sidebar (built by `buildSidebar` /
// `buildSidebarWithOutputTokens`) into the DIFF view. Each lens-type column
// surfaces the tokens its lens leads on, ranked by how DISPROPORTIONATELY this
// lens predicts them vs the other — a smoothed ratio
// `(this count + α) / (other count + α)` (α = DIFF_RATIO_ALPHA). The smoothing
// keeps low-support edge cases (e.g. 1 vs 0) near the neutral ratio of 1 so
// genuinely lopsided, well-supported tokens rise to the top. The displayed
// value is the raw `thisLens:otherLens` count pair. Per-layer stripes are
// passed through UNCHANGED from the per-type sidebar (this lens's own counts).
export function buildDiffSidebar(
  base: Record<string, TypeSidebar>,
  hideAngleBracketTokens = false,
): Record<string, TypeSidebar> {
  const jSb = base[LensType.JACOBIAN_LENS];
  const logitSb = base[LensType.LOGIT_LENS];
  if (!jSb || !logitSb) {
    return base;
  }

  const out: Record<string, TypeSidebar> = {};
  for (const type of LENS_TYPE_ORDER) {
    const self = base[type];
    const other = type === LensType.JACOBIAN_LENS ? logitSb : jSb;

    // Token display strings, preferring this column's form then the other's.
    const tokenOf = new Map<string, string>();
    for (const it of self.allItems) {
      tokenOf.set(it.key, it.token);
    }
    for (const it of other.allItems) {
      if (!tokenOf.has(it.key)) {
        tokenOf.set(it.key, it.token);
      }
    }

    const allKeys = new Set<string>([...self.countByKey.keys(), ...other.countByKey.keys()]);
    // `countByKey` keeps this lens's own count (used for fallback rows);
    // `countLabelByKey` carries the displayed "self:other" pair for every key.
    const selfCountByKey = new Map<string, number>();
    const countLabelByKey = new Map<string, string>();
    for (const k of allKeys) {
      const cSelf = self.countByKey.get(k) ?? 0;
      const cOther = other.countByKey.get(k) ?? 0;
      selfCountByKey.set(k, cSelf);
      countLabelByKey.set(k, `${cSelf}:${cOther}`);
    }

    const favoredUnfiltered: CommonToken[] = [...allKeys]
      .map((k) => {
        const cSelf = self.countByKey.get(k) ?? 0;
        const cOther = other.countByKey.get(k) ?? 0;
        const score = (cSelf + DIFF_RATIO_ALPHA) / (cOther + DIFF_RATIO_ALPHA);
        return { key: k, token: tokenOf.get(k) ?? k, count: cSelf, cOther, score };
      })
      // Only tokens this lens leads on, ranked purely by the smoothed ratio.
      .filter((it) => it.count > it.cOther)
      .sort((a, b) => b.score - a.score)
      .map(({ key, token, count }) => ({ key, token, count }));
    const favored = hideAngleBracketTokens ? dropAngleBracketItems(favoredUnfiltered) : favoredUnfiltered;

    // Full search pool: every key known to EITHER lens (with this lens's own
    // count), so search isn't limited to the top-diff `favored` list.
    const searchItemsUnfiltered: CommonToken[] = [...allKeys]
      .map((k) => ({ key: k, token: tokenOf.get(k) ?? k, count: self.countByKey.get(k) ?? 0 }))
      .sort((a, b) => b.count - a.count);
    const searchItems = hideAngleBracketTokens ? dropAngleBracketItems(searchItemsUnfiltered) : searchItemsUnfiltered;

    out[type] = {
      items: favored.slice(0, SIDEBAR_TOP_K),
      allItems: favored,
      countByKey: selfCountByKey,
      canonicalOf: self.canonicalOf,
      // Stripes show this lens's own per-layer counts, for every key.
      layerStatsByKey: self.layerStatsByKey,
      countLabelByKey,
      searchItems,
    };
  }
  return out;
}

// For the active sidebar token, count how many times it (canonicalized) appears
// in each position's top-n grid within the layer range, then scale to opacity.
export function computeHighlightOpacities(
  tokens: LensTokenMessage[],
  layers: number[],
  range: LayerRange | null,
  key: string,
  type: LensType,
  canonicalOf: Map<string, string>,
): Map<number, number> {
  const opacities = new Map<number, number>();
  const rawCounts = new Map<number, number>();
  let max = 0;
  for (const tok of tokens) {
    const slice = tok.results.find((r) => r.type === type);
    if (!slice) {
      continue;
    }
    let count = 0;
    for (const i of visibleLayerIndices(layers, range)) {
      const row = slice.top_tokens[i];
      if (!row) {
        continue;
      }
      for (const t of row) {
        const k = normKey(t);
        if ((canonicalOf.get(k) ?? k) === key) {
          count += 1;
        }
      }
    }
    if (count > 0) {
      rawCounts.set(tok.position, count);
      if (count > max) {
        max = count;
      }
    }
  }
  const MIN_OPACITY = 0.25;
  for (const [position, count] of rawCounts) {
    opacities.set(position, max > 0 ? Math.max(MIN_OPACITY, (count / max) * 0.7) : 0);
  }
  return opacities;
}

// For the active token, count its (canonicalized) occurrences at EVERY layer
// (summed over all positions), normalized 0..1 against the busiest layer.
export function computeLayerWeights(
  tokens: LensTokenMessage[],
  layers: number[],
  key: string,
  type: LensType,
  canonicalOf: Map<string, string>,
): LayerWeight[] {
  const counts = layers.map(() => 0);
  for (const tok of tokens) {
    const slice = tok.results.find((r) => r.type === type);
    if (!slice) {
      continue;
    }
    layers.forEach((_, i) => {
      const row = slice.top_tokens[i];
      if (!row) {
        return;
      }
      for (const t of row) {
        const k = normKey(t);
        if ((canonicalOf.get(k) ?? k) === key) {
          counts[i] += 1;
        }
      }
    });
  }
  const max = Math.max(0, ...counts);
  return layers.map((layer, i) => ({ layer, weight: max > 0 ? counts[i] / max : 0 }));
}

// A compact prominence-by-layer heatmap: one vertical stripe per layer, each
// stripe's opacity set by that layer's weight (0..1).
export function LayerProminenceBar({
  stats,
  tokenLabel,
  colorRgb,
  hoverColorRgb,
  range,
  scopeLabel,
  lensLabel,
  onLayerHover,
  className,
  disableHover = false,
}: {
  stats: LayerStat[];
  tokenLabel: string;
  colorRgb: string;
  hoverColorRgb?: string;
  range: LayerRange | null;
  scopeLabel: string;
  lensLabel: string;
  onLayerHover?: (layer: number | null) => void;
  className?: string;
  // Suppress all hover affordances (ring, per-layer highlight, tooltip, and
  // `onLayerHover` reporting). Used while the tour spotlights this row so the
  // bar stays static.
  disableHover?: boolean;
}) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const [tip, setTip] = useState<{ x: number; y: number } | null>(null);
  const lastReportedLayer = useRef<number | null>(null);
  const n = stats.length;
  const step = n > 0 ? 100 / n : 0;
  const inRange = (layer: number) => range == null || (layer >= range[0] && layer <= range[1]);

  const gradient = (rgb: string): string | undefined => {
    if (n === 0) {
      return undefined;
    }
    const stops: string[] = [];
    stats.forEach((s, i) => {
      const c = `rgba(${inRange(s.layer) ? rgb : PROMINENCE_SLATE_RGB}, ${s.weight})`;
      stops.push(`${c} ${(i * step).toFixed(4)}%`, `${c} ${((i + 1) * step).toFixed(4)}%`);
    });
    return `linear-gradient(to right, ${stops.join(', ')})`;
  };

  const clear = () => {
    setHoverIdx(null);
    setTip(null);
    if (lastReportedLayer.current !== null) {
      lastReportedLayer.current = null;
      onLayerHover?.(null);
    }
  };

  const handleMove = (e: React.MouseEvent<HTMLSpanElement>) => {
    if (n === 0 || disableHover) {
      return;
    }
    const rect = e.currentTarget.getBoundingClientRect();
    const idx = Math.min(n - 1, Math.max(0, Math.floor(((e.clientX - rect.left) / rect.width) * n)));
    if (idx !== hoverIdx) {
      setHoverIdx(idx);
    }
    setTip({ x: e.clientX, y: e.clientY });
    const layer = stats[idx].layer;
    if (lastReportedLayer.current !== layer) {
      lastReportedLayer.current = layer;
      onLayerHover?.(layer);
    }
  };

  return (
    <span
      aria-hidden
      onMouseLeave={clear}
      className={`relative hidden flex-row items-stretch rounded-sm border border-slate-200 bg-slate-50 sm:flex ${
        !disableHover && hoverIdx == null
          ? 'group-hover:ring-2 group-hover:ring-slate-500 group-hover:ring-offset-1'
          : ''
      } ${className ?? ''}`}
    >
      <span
        className={`absolute inset-0 ${hoverColorRgb && !disableHover ? 'transition-opacity group-hover:opacity-0' : ''}`}
        style={{ backgroundImage: gradient(colorRgb) }}
      />
      {hoverColorRgb && !disableHover && (
        <span
          className="absolute inset-0 opacity-0 transition-opacity group-hover:opacity-100"
          style={{ backgroundImage: gradient(hoverColorRgb) }}
        />
      )}
      {hoverIdx != null && (
        <span
          className="pointer-events-none absolute inset-y-0 bg-slate-500/20 ring-2 ring-slate-500 ring-offset-1"
          style={{ left: `${hoverIdx * step}%`, width: `${step}%` }}
        />
      )}
      {!disableHover && <span className="absolute inset-0" onMouseMove={handleMove} />}
      {hoverIdx != null &&
        tip != null &&
        typeof document !== 'undefined' &&
        createPortal(
          <div
            className="pointer-events-none fixed z-[60] -translate-x-1/2 -translate-y-full whitespace-nowrap rounded-md border border-slate-200 bg-white px-2 py-1 text-[10px] leading-tight text-slate-600 shadow-lg"
            style={{ left: tip.x, top: tip.y - 10 }}
          >
            <div className="font-semibold text-slate-700">
              Layer {stats[hoverIdx].layer} - {lensLabel}
            </div>
            <div>
              <span className="font-mono font-semibold text-slate-700">{displayToken(tokenLabel)}</span> appears{' '}
              <span className="font-semibold tabular-nums">{stats[hoverIdx].count}</span> time
              {stats[hoverIdx].count === 1 ? '' : 's'} {scopeLabel}
            </div>
          </div>,
          document.body,
        )}
    </span>
  );
}

// Render a token's text with whitespace shown as a visible marker ("␣"/"⏎") in
// a lighter slate color, so leading/trailing spaces are visible (and clearly
// distinct from real glyphs) without blending into the token text.
function TokenText({ token }: { token: string }) {
  if (token === '') {
    return <span className="text-slate-300">∅</span>;
  }
  const parts = token.match(/\s+|\S+/g) ?? [];
  return (
    <>
      {parts.map((part, i) =>
        /\s/.test(part) ? (
          <span key={i} className="text-slate-300">
            {part.replace(/ /g, '␣').replace(/\n/g, '⏎')}
          </span>
        ) : (
          <span key={i}>{part}</span>
        ),
      )}
    </>
  );
}

function SidebarTokenRow({
  id,
  tokenKey,
  token,
  translation,
  count,
  barWidthClass = 'w-1/2',
  color,
  nextColor,
  sticky,
  layerStats,
  showBar,
  layerRange,
  scopeLabel,
  lensLabel,
  onHover,
  onLayerHover,
  onToggle,
  onSteer,
}: {
  id?: string;
  tokenKey: string;
  token: string;
  translation: string | null;
  count: number;
  // Tailwind width class for the "Count by Layer" bar (narrower in two-column
  // modes to leave more room for the token/count columns).
  barWidthClass?: string;
  color: SelectColor | null;
  nextColor: SelectColor;
  sticky: { top: number; bottom: number } | null;
  layerStats: LayerStat[];
  showBar: boolean;
  layerRange: LayerRange | null;
  scopeLabel: string;
  lensLabel: string;
  onHover: (key: string | null) => void;
  onLayerHover: (layer: number | null) => void;
  onToggle: (key: string) => void;
  // Open the steer/swap flow for this readout in the given mode, anchored at the
  // click coordinates.
  onSteer?: (key: string, mode: 'steer' | 'swap', anchor: { x: number; y: number }) => void;
}) {
  const selected = color !== null;
  const blockColor = selected ? color : nextColor;
  const pinned = sticky !== null;
  // While the tour spotlights the spider readout, this row is constrained to a
  // single "swap" action: the steer button is hidden, the swap button is always
  // visible (no hover needed), the row's general select is rerouted to swap, and
  // the layer bar's hover affordances are disabled.
  const tourStep = useJlensTourStep();
  const isSpiderTourStep =
    id === JLENS_STEER_SPIDER_ID &&
    typeof tourStep?.element === 'string' &&
    tourStep.element === `#${JLENS_STEER_SPIDER_ID}`;
  return (
    <button
      id={id}
      type="button"
      onMouseEnter={() => onHover(tokenKey)}
      onClick={(e) => {
        // On the spider tour step the row's general select is repurposed as the
        // "swap" action, so clicking anywhere on the row opens the swap flow
        // rather than toggling selection.
        if (isSpiderTourStep && onSteer) {
          onSteer(tokenKey, 'swap', { x: e.clientX, y: e.clientY });
          return;
        }
        onToggle(tokenKey);
      }}
      style={{ height: SIDEBAR_ROW_H, ...(sticky ? { top: sticky.top, bottom: sticky.bottom } : {}) }}
      className={`group relative flex h-7 max-h-7 min-h-7 shrink-0 flex-row items-center gap-x-2.5 rounded border-slate-100 px-2 text-left font-mono transition-colors last:border-b-0 ${
        selected
          ? `sticky z-10 ${COLOR_SIDEBAR[color]}`
          : pinned
            ? `sticky z-10 bg-white ${COLOR_SIDEBAR_HOVER[nextColor]}`
            : `z-0 ${COLOR_SIDEBAR_HOVER[nextColor]}`
      }`}
    >
      <span className="relative flex min-w-0 flex-1 flex-row items-center justify-between gap-x-1">
        <span className="flex min-w-0 flex-1 flex-row items-center gap-x-1.5">
          {selected ? (
            <span
              className="h-2.5 w-2.5 shrink-0 rounded-sm"
              style={{ backgroundColor: `rgb(${COLOR_RGB[blockColor]})` }}
              aria-hidden
            />
          ) : (
            <span className="relative h-2.5 w-2.5 shrink-0" aria-hidden>
              <span className="absolute inset-0 rounded-sm border border-slate-400 opacity-70 transition-opacity group-hover:opacity-0" />
              <span
                className="absolute inset-0 rounded-sm border opacity-0 transition-opacity group-hover:opacity-100"
                style={{ borderColor: `rgb(${COLOR_RGB[blockColor]})` }}
              />
            </span>
          )}
          <span className="shrink-0 truncate text-[11px] text-slate-700" title={token}>
            <TokenText token={token} />
          </span>
          {translation && (
            <span
              className="min-w-0 flex-1 truncate whitespace-nowrap font-sans text-[10px] text-slate-400"
              title={translation}
            >
              {translation}
            </span>
          )}
        </span>
        {onSteer && (
          <span
            className={`absolute right-0 top-1/2 z-20 flex -translate-y-1/2 flex-row items-center gap-x-1 transition ${
              isSpiderTourStep ? 'opacity-100' : 'opacity-100 sm:opacity-0 sm:group-hover:opacity-100'
            }`}
          >
            {(isSpiderTourStep ? (['swap'] as const) : (['steer', 'swap'] as const)).map((m) => (
              <span
                key={m}
                role="button"
                tabIndex={0}
                title={`${m === 'swap' ? 'Swap' : 'Steer'} '${token}'`}
                aria-label={`${m === 'swap' ? 'Swap' : 'Steer'} '${token}'`}
                onClick={(e) => {
                  e.stopPropagation();
                  onSteer(tokenKey, m, { x: e.clientX, y: e.clientY });
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.stopPropagation();
                    e.preventDefault();
                    const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
                    onSteer(tokenKey, m, { x: r.left, y: r.bottom });
                  }
                }}
                style={
                  {
                    '--steer-bg': 'rgb(255, 255, 255)',
                    '--steer-bg-hover': `rgb(${COLOR_RGB[blockColor]})`,
                    '--steer-text': `rgb(${COLOR_RGB[blockColor]})`,
                    '--steer-border': `rgb(${COLOR_PILL[blockColor].ring})`,
                  } as React.CSSProperties
                }
                className={`flex items-center justify-center whitespace-nowrap rounded-full border bg-[var(--steer-bg)] px-1.5 py-0.5 text-[9px] font-medium uppercase leading-none shadow-sm transition hover:bg-[var(--steer-bg-hover)] hover:text-white ${
                  selected
                    ? 'border-[color:var(--steer-border)] text-[color:var(--steer-text)]'
                    : 'border-slate-300 text-slate-500 sm:border-[color:var(--steer-border)] sm:text-[color:var(--steer-text)]'
                }`}
              >
                {m}
              </span>
            ))}
          </span>
        )}
      </span>
      <span className={`w-5 shrink-0 text-right text-[10px] tabular-nums text-slate-400`}>{count}</span>
      {showBar ? (
        // Always render the bar box (even for 0-count rows with no per-layer
        // stats): `LayerProminenceBar` handles empty stats by showing the empty
        // bordered box with no filled stripes, and its `hidden sm:flex` keeps it
        // off mobile so no phantom column is reserved there.
        <LayerProminenceBar
          stats={layerStats}
          tokenLabel={token}
          colorRgb={color !== null ? COLOR_RGB[color] : PROMINENCE_SLATE_RGB}
          hoverColorRgb={color !== null ? undefined : COLOR_RGB[nextColor]}
          range={layerRange}
          scopeLabel={scopeLabel}
          lensLabel={lensLabel}
          onLayerHover={onLayerHover}
          disableHover={isSpiderTourStep}
          className={`h-4 shrink-0 ${barWidthClass}`}
        />
      ) : (
        <span className={`hidden shrink-0 sm:block ${barWidthClass}`} aria-hidden />
      )}
    </button>
  );
}

// The search icon + label/input control shown in a sidebar header. Toggling the
// icon opens an inline query input in place of the label. Extracted so the
// per-column header (single-lens modes) and the combined "Search Both" row
// (DIFF mode) share identical styling and behavior.
export function SidebarSearchControl({
  open,
  query,
  label,
  onToggle,
  onQueryChange,
  onOpen,
}: {
  open: boolean;
  query: string;
  label: React.ReactNode;
  onToggle: () => void;
  onQueryChange: (value: string) => void;
  onOpen: () => void;
}) {
  return (
    <span className="flex min-w-0 flex-1 flex-row items-center gap-x-1">
      <button
        type="button"
        onClick={onToggle}
        title={open ? 'Close search' : 'Search tokens'}
        aria-label={open ? 'Close search' : 'Search tokens'}
        className="flex h-4 w-4 shrink-0 items-center justify-center rounded bg-slate-200 text-slate-500 transition-colors hover:bg-slate-300 hover:text-slate-600"
      >
        {open ? <X className="h-2.5 w-2.5" /> : <Search className="h-2.5 w-2.5" />}
      </button>
      {open ? (
        <input
          type="text"
          autoFocus
          value={query}
          onChange={(e) => onQueryChange(e.target.value)}
          placeholder="Search query"
          className="min-w-0 flex-1 border-0 bg-transparent p-0 text-[10px] font-normal leading-none text-slate-600 outline-none placeholder:text-slate-300 focus:ring-0"
        />
      ) : (
        <button
          type="button"
          onClick={onOpen}
          title="Search tokens"
          className="min-w-0 flex-1 truncate text-left transition-colors hover:text-slate-600"
        >
          {label}
        </button>
      )}
    </span>
  );
}

// One sidebar column (a single lens type). Hovering a token previews it in the
// transcript / slider; clicking toggles its selection (up to MAX_SELECT).
export function LensCommonColumn({
  type,
  items,
  allItems,
  countByKey,
  pinnedKeyOrder,
  selectedKeys,
  crossPinnedKeys,
  colorOf,
  nextColor,
  layerStatsByKey,
  layerRange,
  scopeLabel,
  combinedQuery,
  searchItems,
  searchOpen,
  searchQuery,
  onSearchToggle,
  onSearchQueryChange,
  onSearchOpen,
  onHover,
  onLayerHover,
  onToggle,
  onSteer,
}: {
  type: LensType;
  items: CommonToken[];
  allItems: CommonToken[];
  // DIFF mode only: the full search pool (all tokens known to either lens). When
  // set, search filters this instead of `allItems` (the top-diff list).
  searchItems?: CommonToken[];
  // Single-lens modes only: the per-column search box state, controlled by the
  // panel (so it lives in the shared analysis state and can be reset on clear).
  // Ignored in DIFF mode, which uses the panel's combined "Search Both" row.
  searchOpen: boolean;
  searchQuery: string;
  onSearchToggle: () => void;
  onSearchQueryChange: (value: string) => void;
  onSearchOpen: () => void;
  countByKey: Map<string, number>;
  // DIFF mode only: a shared, column-independent order of all selected keys so
  // both columns pin them in the same rows (side-by-side comparable).
  pinnedKeyOrder?: string[];
  selectedKeys: string[];
  crossPinnedKeys: string[];
  colorOf: (key: string) => SelectColor | null;
  nextColor: SelectColor;
  layerStatsByKey: Map<string, LayerStat[]>;
  layerRange: LayerRange | null;
  scopeLabel: string;
  // DIFF mode only: the shared "Search Both" query, applied to both columns at
  // once. When set (DIFF), the column hides its own header/search and filters by
  // this instead. Ignored in single-lens modes, which use their own search.
  combinedQuery?: string;
  onHover: (key: string | null) => void;
  onLayerHover: (key: string, layer: number | null) => void;
  onToggle: (key: string) => void;
  // Open the steer/swap flow for a readout in this column (carries the lens type
  // and the chosen mode).
  onSteer?: (key: string, type: LensType, mode: 'steer' | 'swap', anchor: { x: number; y: number }) => void;
}) {
  const translate = useChineseTranslation();
  const mode = useContext(LensModeContext);
  // DIFF is the two-column mode; its search + count headers are replaced by a
  // single shared "Search Both" row rendered above both columns by the panel.
  const diffMode = mode === LensMode.DIFF;
  const showBar = true;
  // Narrower Count-by-Layer bar in the cramped two-column DIFF mode, which frees
  // up room for the wider Diff count cell and the token translation.
  const barWidthClass = diffMode ? 'w-1/4' : 'w-1/2';

  // Toggling a token reorders the list so the selection jumps to the top. That
  // reflow slides a DIFFERENT row under a stationary cursor, and the browser
  // fires a synthetic `mouseenter` on it (:hover is re-evaluated on layout
  // change, no movement needed) — which would steal the transcript preview onto
  // some unrelated token. We pin the hover to the just-clicked token and ignore
  // pointer-driven hover changes until the user genuinely moves the mouse again.
  const suppressHoverRef = useRef(false);
  const handleToggle = (key: string) => {
    suppressHoverRef.current = true;
    onToggle(key);
    onHover(key);
  };
  const handleHover = (key: string | null) => {
    if (suppressHoverRef.current && key !== null) {
      return;
    }
    onHover(key);
  };

  // The pool searched when a query is active: the full both-lens set in DIFF
  // mode, or this column's own full list otherwise.
  const searchPool = searchItems ?? allItems;

  const selectedSet = new Set(selectedKeys);
  const itemByKey = new Map(items.map((it) => [it.key, it] as const));
  // Include `searchPool` so rows surfaced by search (outside the top-diff list)
  // still resolve their display token/count.
  const allItemByKey = new Map([...allItems, ...searchPool].map((it) => [it.key, it] as const));
  const rowFor = (k: string): CommonToken =>
    itemByKey.get(k) ?? allItemByKey.get(k) ?? { key: k, token: k, count: countByKey.get(k) ?? 0 };
  const selectedInList = items.filter((it) => selectedSet.has(it.key)).map((it) => it.key);
  const selectedOutOfList = selectedKeys.filter((k) => !itemByKey.has(k));

  const headerOffset = diffMode ? 0 : SIDEBAR_HEADER_H;

  const crossKeys = diffMode ? crossPinnedKeys.filter((k) => !selectedSet.has(k)) : [];
  // DIFF: both columns pin the SAME selected keys in the SAME (shared) order so
  // each selected token sits in the same row across columns, side by side. The
  // shared order already includes cross-pinned keys, so `crossKeys` isn't
  // appended separately here.
  const pinnedOrder = diffMode ? (pinnedKeyOrder ?? []).filter((k) => selectedSet.has(k) || crossKeys.includes(k)) : [];
  const pinnedSet = new Set(pinnedOrder);
  const restItems = diffMode ? items.filter((it) => !pinnedSet.has(it.key)) : [];
  const selectedDomOrder = [...selectedInList, ...selectedOutOfList];

  const orderedKeys = diffMode
    ? [...pinnedOrder, ...restItems.map((it) => it.key)]
    : [...selectedDomOrder, ...items.filter((it) => !selectedSet.has(it.key)).map((it) => it.key)];

  const numPinned = pinnedOrder.length;
  const numSelected = selectedDomOrder.length;
  const stickyFor = (key: string): { top: number; bottom: number } | null => {
    if (diffMode) {
      const i = pinnedOrder.indexOf(key);
      if (i < 0) {
        return null;
      }
      return {
        top: headerOffset + i * SIDEBAR_ROW_H,
        bottom: (numPinned - 1 - i) * SIDEBAR_ROW_H + SIDEBAR_STICKY_BOTTOM,
      };
    }
    const i = selectedDomOrder.indexOf(key);
    if (i < 0) {
      return null;
    }
    return {
      top: headerOffset + i * SIDEBAR_ROW_H,
      bottom: (numSelected - 1 - i) * SIDEBAR_ROW_H + SIDEBAR_STICKY_BOTTOM,
    };
  };

  const headerStats = items.length > 0 ? layerStatsByKey.get(items[0].key) : undefined;
  const firstLayer = headerStats?.[0]?.layer;
  const lastLayer = headerStats?.[headerStats.length - 1]?.layer;

  const query = diffMode
    ? (combinedQuery ?? '').trim().toLowerCase()
    : searchOpen
      ? searchQuery.trim().toLowerCase()
      : '';
  const filteredKeys =
    query === ''
      ? orderedKeys
      : searchPool
          .filter((it) => {
            const translation = translate(it.token) ?? '';
            return (
              it.token.toLowerCase().includes(query) ||
              it.key.toLowerCase().includes(query) ||
              translation.toLowerCase().includes(query)
            );
          })
          .slice(0, SIDEBAR_SEARCH_CAP)
          .map((it) => it.key);

  const labelText = diffMode
    ? type === LensType.JACOBIAN_LENS
      ? 'J-Lens Top'
      : 'Logit Lens Top'
    : type === LensType.JACOBIAN_LENS
      ? 'J-Lens Readout'
      : 'Logit Lens Token';
  const headerLabel = (
    <SidebarSearchControl
      open={searchOpen}
      query={searchQuery}
      label={labelText}
      onToggle={onSearchToggle}
      onQueryChange={onSearchQueryChange}
      onOpen={onSearchOpen}
    />
  );

  return (
    <div
      id={JLENS_JACOBIAN_SPACE_ID}
      className={`flex min-h-0 min-w-0 flex-1 flex-col text-[11px] ${diffMode ? 'px-1' : 'px-2 sm:px-3'}`}
    >
      <div
        className={`flex max-h-full min-h-0 flex-1 flex-col gap-y-0.5 overflow-y-auto`}
        onMouseLeave={() => onHover(null)}
        onMouseMove={() => {
          suppressHoverRef.current = false;
        }}
      >
        {orderedKeys.length === 0 ? (
          <div className="flex max-w-[180px] flex-1 flex-col items-center justify-center self-center px-1 py-2 text-center text-xl font-semibold leading-normal text-slate-400">
            {type === LensType.JACOBIAN_LENS ? (
              <div className="-mt-32 flex flex-row items-center justify-center gap-x-1 whitespace-nowrap">
                Jacobian Space
              </div>
            ) : (
              <div className="-mt-32 flex flex-row items-center justify-center gap-x-1 whitespace-nowrap">
                Logit Lens
              </div>
            )}
          </div>
        ) : (
          <>
            {!diffMode && (
              <div
                style={{ height: SIDEBAR_HEADER_H }}
                className="sticky top-0 z-20 mb-1 flex w-full shrink-0 flex-row items-center gap-x-2.5 border-b border-slate-100 bg-white pb-0.5 text-[10px] text-slate-400"
              >
                <div className="flex flex-1 flex-row items-center justify-between">{headerLabel}</div>
                <div className="w-8 shrink-0 text-right text-[10px] text-slate-400">Count</div>
                <div className="hidden w-[50%] min-w-[50%] max-w-[50%] flex-row items-center justify-between sm:flex">
                  <div className="text-[8px] uppercase text-slate-300">
                    {firstLayer != null ? `Layer ${firstLayer}` : ''}
                  </div>{' '}
                  <div className="text-[10px] text-slate-400">Count by Layer</div>
                  <div className="text-[8px] uppercase text-slate-300">
                    {lastLayer != null ? `Layer ${lastLayer}` : ''}
                  </div>
                </div>
              </div>
            )}
            {filteredKeys.map((k, i) => {
              const it = rowFor(k);
              return (
                <SidebarTokenRow
                  key={k}
                  id={i === 0 ? JLENS_STEER_SPIDER_ID : undefined}
                  tokenKey={k}
                  token={it.token}
                  translation={translate(it.token) ?? null}
                  count={it.count}
                  barWidthClass={barWidthClass}
                  color={colorOf(k)}
                  nextColor={nextColor}
                  sticky={query === '' ? stickyFor(k) : null}
                  layerStats={layerStatsByKey.get(k) ?? []}
                  showBar={showBar}
                  layerRange={layerRange}
                  scopeLabel={scopeLabel}
                  lensLabel={type === LensType.JACOBIAN_LENS ? 'J-Space' : 'Logit Lens'}
                  onHover={handleHover}
                  onLayerHover={(layer) => onLayerHover(k, layer)}
                  onToggle={handleToggle}
                  onSteer={onSteer ? (key, mode, anchor) => onSteer(key, type, mode, anchor) : undefined}
                />
              );
            })}
          </>
        )}
      </div>
    </div>
  );
}
