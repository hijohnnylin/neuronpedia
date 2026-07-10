'use client';

// The shared analysis "brain" for both jlens interfaces (chat + completion).
// Owns the sidebar token selection, the chat-token position selection
// (click/drag/shift-click), the layer-range state, hover/band visualization,
// and the context-resolver values. Both interfaces feed it the same token
// stream + meta and render the same `JlensAnalysisPanel` from its output; only
// the transcript layout + composer differ between them.

import { useGlobalContext } from '@/components/provider/global-provider';
import { JlensShareSteer } from '@/lib/utils/jlens-share';
import { DEFAULT_LENS_STEER_STRENGTH, LensMetaMessage, LensMode, LensTokenMessage, LensType } from '@/lib/utils/lens';
import { useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import {
  buildDiffSidebar,
  buildSidebar,
  buildSidebarWithOutputTokens,
  COLOR_PILL,
  COLOR_RGB,
  computeHighlightOpacities,
  computeLayerWeights,
  DisplayBand,
  EMPH_NORMAL,
  MAX_SELECT,
  normKey,
  SELECT_COLORS,
  SelectColor,
  SelectedToken,
  tokenPositionFromTarget,
  TokenViz,
  TypeSidebar,
} from './jlens-analysis';
import { LensModeContext, lensTypesForMode } from './jlens-lens-mode';
import { setJlensPopupsSuppressed, TokenBand, useIsLensMobile } from './jlens-token';
import { LayerRange, LayerStatsResolver, LensSliderControls, PillColorResolver } from './jlens-token-popup';

// Whether the intervention adds/suppresses a readout (`steer`) or replaces the
// source readout with a target one (`swap`).
export type SteerMode = 'steer' | 'swap';

// The active steering configuration. `token` is the EXACT decoded string (e.g.
// " cat"); `layers` are the model layer numbers to inject at (user-selectable,
// possibly non-contiguous); `strength` is signed (negative suppresses).
// `ablate` projects the readout direction out of the residual instead of
// additively steering — mutually exclusive with `strength` (which is ignored
// when `ablate` is true). When `mode` is 'swap', the source readout (`token`)
// is replaced by `swapToken` (a free-typed target token); `strength`/`ablate`
// are ignored in that case.
export type SteerConfig = {
  token: string;
  type: LensType;
  layers: number[];
  strength: number;
  ablate: boolean;
  mode: SteerMode;
  swapToken: string;
  // Apply the intervention to generated tokens too (default false = prompt only).
  steerGenerated: boolean;
};

// Per-layer occurrence count of the steered token (in the current scope), used
// to render the count-by-layer bar + the layer selector and to pick the
// default layer (the one where the token reads out the most).
export type SteerLayerCount = { layer: number; count: number };
export type SteerInfo = { layerCounts: SteerLayerCount[]; defaultLayers: number[] };

// Pick the default layer selection for a steer config. Additive "steer" defaults
// to the single peak layer (where the token reads out the most; the last layer
// if it never appears). "swap" instead defaults to the layers currently selected
// in the sidebar (the effective layer range), so a swap spans the same window the
// user is inspecting; it falls back to the peak when no range is available.
function computeDefaultSteerLayers(
  mode: SteerMode,
  layerCounts: SteerLayerCount[],
  sidebarRange: LayerRange | null,
): number[] {
  if (mode === 'swap' && sidebarRange) {
    const [lo, hi] = sidebarRange;
    const inRange = layerCounts.filter((lc) => lc.layer >= lo && lc.layer <= hi).map((lc) => lc.layer);
    if (inRange.length > 0) {
      return inRange;
    }
  }
  const maxCount = Math.max(0, ...layerCounts.map((l) => l.count));
  if (maxCount > 0) {
    const peak = layerCounts.reduce((best, l) => (l.count > best.count ? l : best), layerCounts[0]);
    return [peak.layer];
  }
  if (layerCounts.length > 0) {
    return [layerCounts[layerCounts.length - 1].layer];
  }
  return [];
}

// Runs a steered lens stream into the interface's SEPARATE steered-results
// state (or, with `null`, clears those results / aborts). Registered by the
// chat/completion interface, which owns `runLensStream`. Returns a promise that
// resolves when the steered run finishes (or is aborted).
export type SteerRunner = (config: SteerConfig | null) => Promise<void> | void;

export type JlensAnalysis = ReturnType<typeof useJlensAnalysis>;

export function useJlensAnalysis({
  tokens,
  meta,
  modelId,
  busy,
  inferenceAvailable = true,
  onInferenceUnavailable,
  onSidebarSelectionChange,
}: {
  tokens: LensTokenMessage[];
  meta: LensMetaMessage | null;
  modelId: string;
  // True while a run is in flight (streaming / awaiting first response). The
  // one-time auto-select waits until this clears.
  busy: boolean;
  // Whether an inference host currently serves this model. When false, steer /
  // swap entry points are gated and call `onInferenceUnavailable` instead of
  // starting a run (so cached/shared results stay viewable).
  inferenceAvailable?: boolean;
  onInferenceUnavailable?: () => void;
  // Fired whenever the user locks/unlocks a sidebar readout token. A shared run
  // whose sidebar selection is edited has diverged from the shared snapshot, so
  // the shared view uses this to drop the `?shareId=` (URL + tracked state).
  onSidebarSelectionChange?: () => void;
}) {
  const { globalModels, showToastMessage } = useGlobalContext();
  // Kept in a ref so `toggleSelect` (below) can call the latest handler without
  // listing it as a dependency / recreating the callback on every render.
  const onSidebarSelectionChangeRef = useRef(onSidebarSelectionChange);
  onSidebarSelectionChangeRef.current = onSidebarSelectionChange;
  const lensMode = useContext(LensModeContext);
  const sidebarTypes = lensTypesForMode(lensMode);
  const lensModeLabel =
    lensMode === LensMode.DIFF
      ? 'J-Space vs Logit Lens'
      : lensMode === LensMode.JACOBIAN_LENS
        ? 'J-Space'
        : 'Logit Lens';
  const modelLayers = globalModels[modelId]?.layers ?? null;
  const layersByType = useMemo(() => meta?.layers_by_type ?? {}, [meta]);

  // ---- Chat position selection (click / drag / shift-click) --------------
  // Disabled on mobile (<sm): there a token tap shows its lens popup instead.
  const isMobile = useIsLensMobile();
  const isMobileRef = useRef(isMobile);
  isMobileRef.current = isMobile;
  const [selectedPositions, setSelectedPositions] = useState<Set<number>>(() => new Set());
  const [dragging, setDragging] = useState(false);
  const selectedPositionsRef = useRef(selectedPositions);
  useEffect(() => {
    selectedPositionsRef.current = selectedPositions;
  }, [selectedPositions]);
  const positionsSetRef = useRef<Set<number>>(new Set());
  useEffect(() => {
    positionsSetRef.current = new Set(tokens.map((t) => t.position));
  }, [tokens]);
  const gestureRef = useRef<{ anchor: number; base: Set<number>; mode: 'select' | 'deselect'; moved: boolean } | null>(
    null,
  );
  const lastAnchorRef = useRef<number | null>(null);

  const clearSelectedPositions = useCallback(() => setSelectedPositions(new Set()), []);

  const [positionsPopupOpen, setPositionsPopupOpen] = useState(false);
  const [positionsPopupAnchor, setPositionsPopupAnchor] = useState<{ x: number; y: number } | null>(null);
  const [highlightedPosition, setHighlightedPosition] = useState<number | null>(null);
  const removeSelectedPosition = useCallback(
    (pos: number) =>
      setSelectedPositions((cur) => {
        const next = new Set(cur);
        next.delete(pos);
        return next;
      }),
    [],
  );

  const positionsInRange = useCallback((a: number, b: number): number[] => {
    const lo = Math.min(a, b);
    const hi = Math.max(a, b);
    const out: number[] = [];
    for (let i = lo; i <= hi; i += 1) {
      if (positionsSetRef.current.has(i)) {
        out.push(i);
      }
    }
    return out;
  }, []);

  const onChatPointerDown = useCallback(
    (e: React.PointerEvent) => {
      if (isMobileRef.current) {
        return;
      }
      if (e.button !== 0) {
        return;
      }
      const pos = tokenPositionFromTarget(e.target);
      if (pos == null) {
        return;
      }
      if (e.shiftKey && lastAnchorRef.current != null) {
        const anchor = lastAnchorRef.current;
        setSelectedPositions((cur) => {
          const next = new Set(cur);
          for (const i of positionsInRange(anchor, pos)) {
            next.add(i);
          }
          return next;
        });
        lastAnchorRef.current = pos;
        return;
      }
      gestureRef.current = {
        anchor: pos,
        base: new Set(selectedPositionsRef.current),
        mode: selectedPositionsRef.current.has(pos) ? 'deselect' : 'select',
        moved: false,
      };
    },
    [positionsInRange],
  );

  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      const g = gestureRef.current;
      if (!g) {
        return;
      }
      const pos = tokenPositionFromTarget(e.target);
      if (pos == null) {
        return;
      }
      if (!g.moved && pos !== g.anchor) {
        g.moved = true;
        setDragging(true);
        setJlensPopupsSuppressed(true);
        window.getSelection()?.removeAllRanges();
      }
      if (g.moved) {
        const next = new Set(g.base);
        for (const i of positionsInRange(g.anchor, pos)) {
          if (g.mode === 'select') {
            next.add(i);
          } else {
            next.delete(i);
          }
        }
        setSelectedPositions(next);
      }
    };
    const onUp = () => {
      const g = gestureRef.current;
      gestureRef.current = null;
      if (!g) {
        return;
      }
      if (!g.moved) {
        setSelectedPositions((cur) => {
          const next = new Set(cur);
          if (next.has(g.anchor)) {
            next.delete(g.anchor);
          } else {
            next.add(g.anchor);
          }
          return next;
        });
      }
      lastAnchorRef.current = g.anchor;
      setDragging(false);
      setJlensPopupsSuppressed(false);
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    return () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
  }, [positionsInRange]);

  // The token currently hovered (its popup is open). While set, the sidebar
  // shows that position's top tokens instead of the global aggregation.
  const [hoveredChatToken, setHoveredChatToken] = useState<LensTokenMessage | null>(null);
  const handleTokenHover = useCallback((t: LensTokenMessage, open: boolean) => {
    setHoveredChatToken((cur) => (open ? t : cur && cur.position === t.position ? null : cur));
  }, []);

  // ---- Layer range -------------------------------------------------------
  const layerBounds = useMemo<LayerRange | null>(() => {
    const all = new Set<number>();
    for (const arr of Object.values(layersByType)) {
      for (const l of arr) {
        all.add(l);
      }
    }
    if (all.size > 0) {
      const sorted = [...all].sort((a, b) => a - b);
      return [sorted[0], sorted[sorted.length - 1]];
    }
    if (modelLayers && modelLayers > 0) {
      return [0, modelLayers - 1];
    }
    return null;
  }, [layersByType, modelLayers]);

  const defaultRange = useMemo<LayerRange | null>(() => {
    if (!layerBounds) {
      return null;
    }
    const [min, max] = layerBounds;
    const n = max - min + 1;
    const start = min + Math.floor(n * 0.29);
    const end = max;
    return [start, Math.max(start, end)];
  }, [layerBounds]);

  const [layerRange, setLayerRange] = useState<LayerRange | null>(null);
  const boundsKey = layerBounds ? `${layerBounds[0]}-${layerBounds[1]}` : '';
  useEffect(() => {
    setLayerRange(null);
  }, [boundsKey]);

  const effectiveRange = useMemo<LayerRange | null>(() => layerRange ?? defaultRange, [layerRange, defaultRange]);

  const [hideNonWordTokens, setHideNonWordTokens] = useState(true);
  const [sliderHoverLayer, setSliderHoverLayer] = useState<number | null>(null);

  // ---- Sidebar token search ----------------------------------------------
  // Held here (rather than inside the sidebar column / panel) so it can be
  // reset when the chat/completion is cleared or a new run is loaded. A single
  // shared state backs both the per-column search (single-lens modes) and the
  // combined "Search Both" row (DIFF mode), so the query survives a mode swap.
  const [sidebarSearchOpen, setSidebarSearchOpen] = useState(false);
  const [sidebarSearchQuery, setSidebarSearchQuery] = useState('');
  const clearSidebarSearch = useCallback(() => {
    setSidebarSearchOpen(false);
    setSidebarSearchQuery('');
  }, []);

  // The layer INDEX locked in the hover popup's per-layer selector strip (null =
  // the "All Layers" overview). Held here (rather than inside the popup) so a
  // locked layer persists as the cursor sweeps between token popups, and so it
  // can be cleared when the chat/completion is reset.
  const [lockedLayer, setLockedLayer] = useState<number | null>(null);

  const listRange = useMemo<LayerRange | null>(
    () => (sliderHoverLayer != null ? [sliderHoverLayer, sliderHoverLayer] : effectiveRange),
    [sliderHoverLayer, effectiveRange],
  );

  // Non-word filtering now happens SERVER-SIDE (the read-out's top-n is already
  // filtered when `hideNonWordTokens` is on), so the builders no longer filter
  // by word-likeness — they display what the server returned, and the toggle
  // re-runs the request (see the interface's filter re-run). The ONE exception
  // is client-side: when the non-word filter is on, "<"-prefixed tokens (e.g.
  // the always-preserved output token "<|im_end|>") are dropped from the
  // SIDEBAR LIST only (not the hover popup) via `hideNonWordTokens` below.
  const sidebar = useMemo(
    () => buildSidebar(tokens, layersByType, effectiveRange, false, hideNonWordTokens),
    [tokens, layersByType, effectiveRange, hideNonWordTokens],
  );

  const listSidebar = useMemo(
    () =>
      sliderHoverLayer == null ? sidebar : buildSidebar(tokens, layersByType, listRange, false, hideNonWordTokens),
    [sliderHoverLayer, sidebar, tokens, layersByType, listRange, hideNonWordTokens],
  );

  const positionSidebar = useMemo<Record<string, TypeSidebar> | null>(() => {
    if (!hoveredChatToken) {
      return null;
    }
    return buildSidebarWithOutputTokens([hoveredChatToken], layersByType, listRange, false, hideNonWordTokens);
  }, [hoveredChatToken, layersByType, listRange, hideNonWordTokens]);

  const filteredSidebar = useMemo(() => {
    if (selectedPositions.size === 0) {
      return null;
    }
    const filtered = tokens.filter((t) => selectedPositions.has(t.position));
    return buildSidebarWithOutputTokens(filtered, layersByType, listRange, false, hideNonWordTokens);
  }, [selectedPositions, tokens, layersByType, listRange, hideNonWordTokens]);

  const activeSidebarBase = filteredSidebar ?? positionSidebar ?? listSidebar;
  // In DIFF mode each column shows its lens's net advantage over the other; the
  // per-type viz/highlighting below still uses the untransformed per-type data.
  const activeSidebar = useMemo(
    () => (lensMode === LensMode.DIFF ? buildDiffSidebar(activeSidebarBase, hideNonWordTokens) : activeSidebarBase),
    [lensMode, activeSidebarBase, hideNonWordTokens],
  );

  const positionScopeLabel = useMemo(() => {
    if (selectedPositions.size > 0) {
      const sorted = [...selectedPositions].sort((a, b) => a - b);
      return sorted.length === 1
        ? `at position ${sorted[0]}`
        : `at positions ${sorted[0]} to ${sorted[sorted.length - 1]}`;
    }
    if (hoveredChatToken) {
      return `at position ${hoveredChatToken.position}`;
    }
    return 'at all positions';
  }, [selectedPositions, hoveredChatToken]);

  // ---- Sidebar token selection (color-coded) -----------------------------
  const [selected, setSelected] = useState<SelectedToken[]>([]);
  const [hover, setHover] = useState<SelectedToken | null>(null);
  const [maxSelectError, setMaxSelectError] = useState(false);
  const [barHover, setBarHover] = useState<{ key: string; type: LensType; layer: number } | null>(null);

  // Migrate the existing selection when the lens mode changes so the chosen
  // tokens follow the user across modes.
  useEffect(() => {
    setSelected((cur) => {
      if (cur.length === 0) {
        return cur;
      }
      const keysInOrder: string[] = [];
      const seen = new Set<string>();
      for (const s of cur) {
        if (!seen.has(s.key)) {
          seen.add(s.key);
          keysInOrder.push(s.key);
        }
      }
      if (lensMode === LensMode.DIFF) {
        // Preserve each token's existing column (its lens type) instead of
        // mirroring the selection into both columns — keep only the column that
        // was selected last (coming from a single-lens mode, that's that mode).
        const seenPair = new Set<string>();
        const kept: SelectedToken[] = [];
        for (const s of cur) {
          const pairKey = `${s.type}:${s.key}`;
          if (!seenPair.has(pairKey)) {
            seenPair.add(pairKey);
            kept.push(s);
          }
        }
        return kept.slice(0, MAX_SELECT);
      }
      return keysInOrder.map((key) => ({ key, type: lensMode }));
    });
  }, [lensMode]);

  const toggleSelect = useCallback(
    (key: string, type: LensType) => {
      const idx = selected.findIndex((s) => s.key === key && s.type === type);
      if (idx >= 0) {
        // Deselect: the sidebar selection changed → notify (drops any shareId).
        onSidebarSelectionChangeRef.current?.();
        setSelected((cur) => cur.filter((s) => !(s.key === key && s.type === type)));
        showToastMessage(
          <div className="text-sm">
            Unlocked token <span className="font-semibold text-slate-700">&lsquo;{key}&rsquo;</span>.
          </div>,
        );
        return;
      }
      if (selected.length >= MAX_SELECT) {
        setMaxSelectError(true);
        return;
      }
      // Select: the sidebar selection changed → notify (drops any shareId).
      onSidebarSelectionChangeRef.current?.();
      const color = SELECT_COLORS[selected.length];
      setSelected((cur) => [...cur, { key, type }]);
      showToastMessage(
        <>
          Locked token <span className="font-semibold text-slate-700">&lsquo;{key}&rsquo;</span> to color{' '}
          <span className="font-semibold capitalize" style={{ color: `rgb(${COLOR_RGB[color]})` }}>
            {color}
          </span>
          .
          <br />
          Click again to unlock it.
        </>,
      );
    },
    [selected, showToastMessage],
  );

  // Active steer config (null = not steering). Declared here (ahead of the rest
  // of the steering block below) so the highlight bands can restrict themselves
  // to just the steered readout while a steer/swap is being set up.
  const [steer, setSteer] = useState<SteerConfig | null>(null);

  const selectedViz = useMemo<TokenViz[]>(
    () =>
      selected.map((s, i) => {
        const canonicalOf = sidebar[s.type]?.canonicalOf ?? new Map<string, string>();
        const layers = layersByType[s.type] ?? [];
        return {
          color: SELECT_COLORS[i],
          opacityByPosition: computeHighlightOpacities(tokens, layers, effectiveRange, s.key, s.type, canonicalOf),
          layerWeights: computeLayerWeights(tokens, layers, s.key, s.type, canonicalOf),
        };
      }),
    [selected, sidebar, tokens, layersByType, effectiveRange],
  );

  const hoverViz = useMemo<TokenViz | null>(() => {
    if (!hover) {
      return null;
    }
    const idx = selected.findIndex((s) => s.key === hover.key && s.type === hover.type);
    const color: SelectColor =
      idx >= 0 ? SELECT_COLORS[idx] : selected.length < MAX_SELECT ? SELECT_COLORS[selected.length] : 'red';
    const canonicalOf = sidebar[hover.type]?.canonicalOf ?? new Map<string, string>();
    const layers = layersByType[hover.type] ?? [];
    return {
      color,
      opacityByPosition: computeHighlightOpacities(tokens, layers, effectiveRange, hover.key, hover.type, canonicalOf),
      layerWeights: computeLayerWeights(tokens, layers, hover.key, hover.type, canonicalOf),
    };
  }, [hover, selected, sidebar, tokens, layersByType, effectiveRange]);

  const displayBands = useMemo<DisplayBand[]>(() => {
    // While steering/swapping, the transcript should highlight ONLY the readout
    // being steered (it's locked into the sidebar selection), regardless of any
    // other selected tokens or hover state.
    if (steer) {
      const idx = selected.findIndex((s) => s.key === steer.token && s.type === steer.type);
      return idx >= 0 ? [{ ...selectedViz[idx], emphasis: EMPH_NORMAL }] : [];
    }

    if (barHover) {
      const canonicalOf = sidebar[barHover.type]?.canonicalOf ?? new Map<string, string>();
      const layers = layersByType[barHover.type] ?? [];
      const idx = selected.findIndex((s) => s.key === barHover.key && s.type === barHover.type);
      const color: SelectColor =
        idx >= 0 ? SELECT_COLORS[idx] : selected.length < MAX_SELECT ? SELECT_COLORS[selected.length] : 'red';
      const singleLayer: LayerRange = [barHover.layer, barHover.layer];
      return [
        {
          color,
          opacityByPosition: computeHighlightOpacities(
            tokens,
            layers,
            singleLayer,
            barHover.key,
            barHover.type,
            canonicalOf,
          ),
          layerWeights: computeLayerWeights(tokens, layers, barHover.key, barHover.type, canonicalOf),
          emphasis: EMPH_NORMAL,
        },
      ];
    }

    const hoverIdx = hover ? selected.findIndex((s) => s.key === hover.key && s.type === hover.type) : -1;

    if (hoverIdx >= 0) {
      return [{ ...selectedViz[hoverIdx], emphasis: EMPH_NORMAL }];
    }
    if (hover && hoverViz) {
      // Hovering a NON-selected sidebar token: preview it as a border outline
      // rather than a solid background, to distinguish it from a committed
      // (selected) highlight.
      return [{ ...hoverViz, emphasis: EMPH_NORMAL, border: true }];
    }
    return selectedViz.map((v) => ({ ...v, emphasis: EMPH_NORMAL }));
  }, [steer, barHover, hover, hoverViz, selected, selectedViz, sidebar, tokens, layersByType]);

  const hoverInfo = useMemo(() => {
    if (!hover) {
      return null;
    }
    const idx = selected.findIndex((s) => s.key === hover.key && s.type === hover.type);
    const color: SelectColor =
      idx >= 0 ? SELECT_COLORS[idx] : selected.length < MAX_SELECT ? SELECT_COLORS[selected.length] : 'red';
    const layer = barHover && barHover.key === hover.key && barHover.type === hover.type ? barHover.layer : null;
    return { key: hover.key, color, layer, type: hover.type };
  }, [hover, barHover, selected]);

  const bandsByPosition = useMemo<Map<number, TokenBand[]>>(() => {
    const m = new Map<number, TokenBand[]>();
    if (displayBands.length === 0) {
      return m;
    }
    const positions = new Set<number>();
    for (const b of displayBands) {
      for (const p of b.opacityByPosition.keys()) {
        positions.add(p);
      }
    }
    for (const p of positions) {
      const bands = displayBands
        .filter((b) => (b.opacityByPosition.get(p) ?? 0) > 0)
        .map((b) => ({
          color: COLOR_PILL[b.color].ring,
          opacity: (b.opacityByPosition.get(p) ?? 0) * b.emphasis,
          border: b.border,
        }));
      if (bands.length > 0) {
        m.set(p, bands);
      }
    }
    return m;
  }, [displayBands]);

  const pillColorResolver = useCallback<PillColorResolver>(
    (token, type) => {
      const canonicalOf = sidebar[type]?.canonicalOf;
      const nk = normKey(token);
      const ck = canonicalOf?.get(nk) ?? nk;
      const idx = selected.findIndex((s) => s.type === type && s.key === ck);
      return idx >= 0 ? COLOR_PILL[SELECT_COLORS[idx]] : null;
    },
    [selected, sidebar],
  );

  const layerStatsResolver = useCallback<LayerStatsResolver>(
    (type, token) => {
      const sb = sidebar[type];
      if (!sb) {
        return [];
      }
      const nk = normKey(token);
      const ck = sb.canonicalOf.get(nk) ?? nk;
      const stats = sb.layerStatsByKey.get(ck);
      return stats ? stats.map((s) => s.weight) : [];
    },
    [sidebar],
  );

  const sliderControls = useMemo<LensSliderControls>(
    () => ({
      bounds: layerBounds,
      range: effectiveRange,
      setRange: setLayerRange,
      selected: selected.map((s, i) => ({
        key: s.key,
        type: s.type,
        colorRgb: COLOR_RGB[SELECT_COLORS[i]],
        canonicalOf: sidebar[s.type]?.canonicalOf ?? new Map<string, string>(),
      })),
      nextColorRgb: COLOR_RGB[selected.length < MAX_SELECT ? SELECT_COLORS[selected.length] : 'red'],
      onToggle: toggleSelect,
      onHoverPreview: setHover,
      lockedLayer,
      setLockedLayer,
    }),
    [layerBounds, effectiveRange, selected, sidebar, toggleSelect, lockedLayer],
  );

  // ---- Steering -----------------------------------------------------------
  // The chat/completion interface registers a runner + stopper here (it owns the
  // stream and the separate steered-results state).
  const steerRunnerRef = useRef<SteerRunner | null>(null);
  const registerSteerRunner = useCallback((fn: SteerRunner | null) => {
    steerRunnerRef.current = fn;
  }, []);
  const steerStopperRef = useRef<(() => void) | null>(null);
  const registerSteerStopper = useCallback((fn: (() => void) | null) => {
    steerStopperRef.current = fn;
  }, []);
  // True while a steered run is streaming (drives the Steer/Stop button + locks
  // the steer settings).
  const [steerStreaming, setSteerStreaming] = useState(false);

  // Active steer config's layer info (the config itself is declared earlier so
  // the highlight bands can read it).
  const steerRef = useRef<SteerConfig | null>(null);
  useEffect(() => {
    steerRef.current = steer;
  }, [steer]);
  const [steerInfo, setSteerInfo] = useState<SteerInfo | null>(null);

  // The current scope subset of tokens (selected positions / hovered / all),
  // used for per-layer counting so it matches the sidebar.
  const steerScopeSubset = useCallback(
    (): LensTokenMessage[] =>
      selectedPositions.size > 0
        ? tokens.filter((t) => selectedPositions.has(t.position))
        : hoveredChatToken
          ? [hoveredChatToken]
          : tokens,
    [tokens, selectedPositions, hoveredChatToken],
  );

  // Per-layer occurrence count of an exact token string across ALL layers of
  // its lens type (independent of the selected range), within the current scope.
  const collectSteerLayerCounts = useCallback(
    (token: string, type: LensType): SteerLayerCount[] => {
      const layers = layersByType[type] ?? [];
      const counts = layers.map(() => 0);
      for (const tok of steerScopeSubset()) {
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
            if (t === token) {
              counts[i] += 1;
            }
          }
        });
      }
      return layers.map((layer, i) => ({ layer, count: counts[i] }));
    },
    [layersByType, steerScopeSubset],
  );

  const startSteer = useCallback(
    (token: string, type: LensType, strength: number, mode: SteerMode = 'steer') => {
      const layerCounts = collectSteerLayerCounts(token, type);
      const defaultLayers = computeDefaultSteerLayers(mode, layerCounts, effectiveRange);
      const config: SteerConfig = {
        token,
        type,
        layers: defaultLayers,
        strength,
        ablate: false,
        mode,
        swapToken: '',
        steerGenerated: true,
      };
      // Enter steer mode with this config, but DON'T run yet — the user presses
      // "Steer" to send the request.
      setSteer(config);
      setSteerInfo({ layerCounts, defaultLayers });
    },
    [collectSteerLayerCounts, effectiveRange],
  );

  // Re-enter steer mode with a SAVED config (e.g. restored from a shared link),
  // keeping its exact layers + strength rather than recomputing defaults. The
  // per-layer counts (for the selector tint) + the default-layer reference are
  // recomputed from the current (loaded) token scope.
  const restoreSteer = useCallback(
    (config: JlensShareSteer) => {
      const layerCounts = collectSteerLayerCounts(config.token, config.type);
      const mode = config.mode ?? 'steer';
      const defaultLayers = computeDefaultSteerLayers(mode, layerCounts, effectiveRange);
      // Normalize older saved configs that predate swap / generated-token steering.
      setSteer({
        ...config,
        mode,
        swapToken: config.swapToken ?? '',
        steerGenerated: config.steerGenerated ?? false,
      });
      setSteerInfo({ layerCounts, defaultLayers });
    },
    [collectSteerLayerCounts, effectiveRange],
  );

  // Clear any existing steered/swapped output results (without leaving steer
  // mode), so stale results don't linger after a settings change. Every steer
  // setting setter below calls this so the displayed output never disagrees
  // with the current settings until the user re-runs.
  const clearSteerResults = useCallback(() => {
    steerRunnerRef.current?.(null);
  }, []);

  const setSteerLayers = useCallback(
    (layers: number[]) => {
      clearSteerResults();
      setSteer((cur) => (cur ? { ...cur, layers } : cur));
    },
    [clearSteerResults],
  );

  const resetSteerLayers = useCallback(() => {
    if (steerInfo) {
      setSteerLayers(steerInfo.defaultLayers);
    }
  }, [steerInfo, setSteerLayers]);

  // Entry point from a readout row's "Steer"/"Swap" buttons. Each readout key is
  // an exact token, so steer/swap on it directly (no variant disambiguation).
  // `mode` selects which mode the panel opens in.
  const beginSteer = useCallback(
    (key: string, type: LensType, mode: SteerMode = 'steer') => {
      // Don't start a steer/swap while the primary run is still streaming —
      // it would kick off a second inference run mid-generation.
      if (busy) {
        return;
      }
      // Steering/swapping requires live inference. When unavailable (e.g. a
      // shared run whose model is no longer served), surface the notice instead.
      if (!inferenceAvailable) {
        onInferenceUnavailable?.();
        return;
      }
      const alreadySelected = selected.some((s) => s.key === key && s.type === type);
      if (!alreadySelected) {
        // Don't open steer/swap on an unselected token when at the cap — show the
        // same selection-limit error as locking would.
        if (selected.length >= MAX_SELECT) {
          setMaxSelectError(true);
          return;
        }
        // Otherwise lock it in the sidebar so it gets a stable selection color.
        setSelected((cur) => [...cur, { key, type }]);
      }
      startSteer(key, type, DEFAULT_LENS_STEER_STRENGTH, mode);
    },
    [busy, selected, startSteer, inferenceAvailable, onInferenceUnavailable],
  );

  // Setting a strength is mutually exclusive with ablation: it always clears the
  // ablate flag (the slider "wins").
  const setSteerStrength = useCallback(
    (strength: number) => {
      clearSteerResults();
      setSteer((cur) => (cur ? { ...cur, strength, ablate: false } : cur));
    },
    [clearSteerResults],
  );

  // Toggle ablation on/off. Turning it on leaves `strength` untouched (it's just
  // ignored while ablating); turning it off restores additive steering.
  const setSteerAblate = useCallback(
    (ablate: boolean) => {
      clearSteerResults();
      setSteer((cur) => (cur ? { ...cur, ablate } : cur));
    },
    [clearSteerResults],
  );

  // Switch between additive steering and the source->target swap intervention.
  // Each mode has its own default layer selection (peak for steer, the sidebar's
  // selected layers for swap), so toggling resets the layers to the new mode's
  // default.
  const setSteerMode = useCallback(
    (mode: SteerMode) => {
      clearSteerResults();
      const info = steerInfo;
      if (!info) {
        setSteer((cur) => (cur ? { ...cur, mode } : cur));
        return;
      }
      const defaultLayers = computeDefaultSteerLayers(mode, info.layerCounts, effectiveRange);
      setSteer((cur) => (cur ? { ...cur, mode, layers: defaultLayers } : cur));
      setSteerInfo({ ...info, defaultLayers });
    },
    [steerInfo, effectiveRange, clearSteerResults],
  );

  // Set the free-typed target token for a swap.
  const setSwapToken = useCallback(
    (swapToken: string) => {
      clearSteerResults();
      setSteer((cur) => (cur ? { ...cur, swapToken } : cur));
    },
    [clearSteerResults],
  );

  // Toggle whether the intervention also applies to generated tokens.
  const setSteerGenerated = useCallback(
    (steerGenerated: boolean) => {
      clearSteerResults();
      setSteer((cur) => (cur ? { ...cur, steerGenerated } : cur));
    },
    [clearSteerResults],
  );

  // Send the steering request for the current config (manual trigger). Tracks
  // streaming so the button can flip to "Stop" and the settings can lock.
  const runSteer = useCallback(async () => {
    if (!inferenceAvailable) {
      onInferenceUnavailable?.();
      return;
    }
    const fn = steerRunnerRef.current;
    const config = steerRef.current;
    if (!fn || !config || config.layers.length === 0) {
      return;
    }
    // A swap requires a target token.
    if (config.mode === 'swap' && !config.swapToken.trim()) {
      return;
    }
    setSteerStreaming(true);
    try {
      await fn(config);
    } finally {
      setSteerStreaming(false);
    }
  }, [inferenceAvailable, onInferenceUnavailable]);

  const stopSteer = useCallback(() => {
    steerStopperRef.current?.();
  }, []);

  const exitSteer = useCallback(() => {
    steerStopperRef.current?.();
    setSteer(null);
    setSteerInfo(null);
    setSteerStreaming(false);
    steerRunnerRef.current?.(null);
  }, []);

  return {
    // True while the primary run is in flight (streaming / awaiting first
    // response). Used to gate steer/swap entry points so they can't start a
    // second inference run mid-generation.
    busy,
    // lens mode
    lensMode,
    sidebarTypes,
    lensModeLabel,
    layersByType,
    // layer range
    layerBounds,
    defaultRange,
    effectiveRange,
    layerRange,
    setLayerRange,
    sliderHoverLayer,
    setSliderHoverLayer,
    lockedLayer,
    setLockedLayer,
    hideNonWordTokens,
    setHideNonWordTokens,
    // sidebar search
    sidebarSearchOpen,
    setSidebarSearchOpen,
    sidebarSearchQuery,
    setSidebarSearchQuery,
    clearSidebarSearch,
    // sidebar data
    activeSidebar,
    sidebar,
    positionScopeLabel,
    // sidebar selection
    selected,
    setSelected,
    hover,
    setHover,
    barHover,
    setBarHover,
    toggleSelect,
    maxSelectError,
    setMaxSelectError,
    // position selection
    selectedPositions,
    setSelectedPositions,
    clearSelectedPositions,
    removeSelectedPosition,
    onChatPointerDown,
    dragging,
    positionsPopupOpen,
    setPositionsPopupOpen,
    positionsPopupAnchor,
    setPositionsPopupAnchor,
    highlightedPosition,
    setHighlightedPosition,
    // hover
    hoveredChatToken,
    handleTokenHover,
    hoverInfo,
    bandsByPosition,
    // resolvers / providers
    pillColorResolver,
    layerStatsResolver,
    sliderControls,
    // steering
    steer,
    steerInfo,
    steerStreaming,
    beginSteer,
    restoreSteer,
    setSteerStrength,
    setSteerAblate,
    setSteerMode,
    setSwapToken,
    setSteerGenerated,
    setSteerLayers,
    resetSteerLayers,
    runSteer,
    stopSteer,
    exitSteer,
    registerSteerRunner,
    registerSteerStopper,
  };
}
