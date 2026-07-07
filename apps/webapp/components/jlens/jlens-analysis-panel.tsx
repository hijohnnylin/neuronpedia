'use client';

// The shared left-hand analysis panel for both jlens interfaces: the lens-mode
// toggle, the layer-range slider, the status row (lens / layer / position
// filter), and the per-lens "most common token" columns. Driven entirely by a
// `JlensAnalysis` (see `useJlensAnalysis`); both chat and completion render this
// identically.

import { LensMode, LensTokenMessage, LensType } from '@/lib/utils/lens';
import { Share2 } from 'lucide-react';
import { useContext, useMemo } from 'react';
import {
  COLOR_PILL,
  COLOR_RGB,
  LensCommonColumn,
  MAX_SELECT,
  SELECT_COLORS,
  SidebarSearchControl,
} from './jlens-analysis';
import { LayerRangeSlider } from './jlens-layer-slider';
import { LensModeSetContext, LensModeToggle } from './jlens-lens-mode';
import { JlensPositionFilter } from './jlens-position-filter';
import { JlensSteerPanel, SteerModeBanner } from './jlens-steer-panel';
import { JlensPopupHost } from './jlens-token';
import {
  displayToken,
  HideNonWordContext,
  isDegenerateLayer,
  JLENS_DEGENERATE_DISCLAIMER,
  LayerStatsContext,
  LensSliderContext,
  LensSliderControls,
  PillColorContext,
  PopupSteerContext,
  SteerActiveContext,
} from './jlens-token-popup';
import { JlensAnalysis } from './use-jlens-analysis';

// Wraps children in the four lens contexts that the token chips + popups read
// from, sourced from the shared analysis state. Also mounts the single shared
// lens popup (`JlensPopupHost`) so the tokens inside swap one popup's contents
// in place rather than flashing a per-token card on a fast hover sweep.
export function JlensProviders({
  analysis,
  steering: steeringProp,
  children,
}: {
  analysis: JlensAnalysis;
  // Whether the interface is in steering mode. Defaults to this analysis's own
  // steer state, but can be forced on (e.g. for the steered-output transcript,
  // which runs on a SEPARATE analysis instance whose `steer` is always null but
  // is only ever shown while the main interface is steering). While steering,
  // popup readout rows can't toggle selection and their Steer/Swap buttons are
  // hidden.
  steering?: boolean;
  children: React.ReactNode;
}) {
  const steering = steeringProp ?? analysis.steer != null;
  const steerActive = useMemo(
    () => (analysis.steer ? { key: analysis.steer.token, type: analysis.steer.type } : null),
    [analysis.steer],
  );
  // While steering, block the popup rows' selection toggle (so hovering a chat
  // token can't add/remove sidebar selections mid-steer) and explain why.
  const sliderControls = useMemo<LensSliderControls>(
    () =>
      steering
        ? {
            ...analysis.sliderControls,
            // eslint-disable-next-line no-alert
            onToggle: () => window.alert("You can't select tokens while in steering mode. Exit steering mode first."),
          }
        : analysis.sliderControls,
    [steering, analysis.sliderControls],
  );
  return (
    <PillColorContext.Provider value={analysis.pillColorResolver}>
      <LensSliderContext.Provider value={sliderControls}>
        <LayerStatsContext.Provider value={analysis.layerStatsResolver}>
          {/* Non-word filtering is now server-side; the popup displays every
              token the server returned (so this is always false). */}
          <HideNonWordContext.Provider value={false}>
            <SteerActiveContext.Provider value={steerActive}>
              <PopupSteerContext.Provider value={steering ? null : analysis.beginSteer}>
                <JlensPopupHost
                  layersByType={analysis.layersByType}
                  layerRange={analysis.effectiveRange}
                  onTokenHover={analysis.handleTokenHover}
                >
                  {children}
                </JlensPopupHost>
              </PopupSteerContext.Provider>
            </SteerActiveContext.Provider>
          </HideNonWordContext.Provider>
        </LayerStatsContext.Provider>
      </LensSliderContext.Provider>
    </PillColorContext.Provider>
  );
}

// The "readout" chip describing the hovered sidebar token row (its color + the
// specific layer if a stripe is hovered). Rendered inline on the left of the
// share bar above the transcript pane.
export function JlensHoverInfo({ analysis }: { analysis: JlensAnalysis }) {
  const { hoverInfo } = analysis;
  if (!hoverInfo) {
    return null;
  }
  return (
    <div
      className="pointer-events-none flex min-w-[360px] max-w-[360px] flex-col items-start gap-x-1 gap-y-0.5 rounded-md px-3.5 py-2 text-[11px] leading-tight text-slate-600 shadow-sm backdrop-blur"
      style={{
        backgroundColor: `rgba(${COLOR_PILL[hoverInfo.color].bg})`,
        border: `1px solid rgb(${COLOR_PILL[hoverInfo.color].ring})`,
      }}
    >
      <div className="mb-1 text-[13px] font-bold text-slate-600">
        {hoverInfo.type === LensType.JACOBIAN_LENS ? 'J-Lens Readout' : 'Logit Lens'}
      </div>
      <div className="flex flex-row items-center justify-center gap-x-1.5 text-base">
        <div className="flex flex-row items-center gap-x-1.5">
          <span
            className="h-4 w-4 shrink-0 rounded text-base"
            style={{ backgroundColor: `rgb(${COLOR_RGB[hoverInfo.color]})` }}
            aria-hidden
          />
          <span
            className="rounded-md px-2.5 py-0 font-mono text-base font-semibold"
            style={{ backgroundColor: `rgb(255,255,255,0.9)` }}
          >
            {displayToken(hoverInfo.key)}
          </span>
        </div>
        at
        <div className="text-base font-bold text-slate-700">
          {hoverInfo.layer != null ? `Layer ${hoverInfo.layer}` : 'All Layers'}
        </div>
      </div>
    </div>
  );
}

export function JlensAnalysisPanel({
  analysis,
  tokens,
  numCompletionTokens,
  setNumCompletionTokens,
  maxCompletionTokens,
  onShare,
  canShare,
  shareDisabled,
  shareLabel,
  onExport,
  exportDisabled,
  exportLabel,
}: {
  analysis: JlensAnalysis;
  tokens: LensTokenMessage[];
  // The shared "generated tokens" setting (same value the advanced panel
  // controls), surfaced as a slider in the steer panel.
  numCompletionTokens?: number;
  setNumCompletionTokens?: (n: number) => void;
  maxCompletionTokens?: number;
  // Share entry point (same as the action bar's share button), surfaced in the
  // steer panel's header so it stays reachable while steering.
  onShare?: () => void;
  canShare?: boolean;
  shareDisabled?: boolean;
  shareLabel?: string;
  // Export/download entry point, surfaced alongside share in the steer panel's
  // header so it stays reachable while steering.
  onExport?: () => void;
  exportDisabled?: boolean;
  exportLabel?: string;
}) {
  const setLensMode = useContext(LensModeSetContext);
  const {
    lensMode,
    sidebarTypes,
    layersByType,
    layerBounds,
    effectiveRange,
    defaultRange,
    setLayerRange,
    sliderHoverLayer,
    setSliderHoverLayer,
    activeSidebar,
    positionScopeLabel,
    selected,
    setHover,
    setBarHover,
    toggleSelect,
    selectedPositions,
    sidebarSearchOpen,
    setSidebarSearchOpen,
    sidebarSearchQuery,
    setSidebarSearchQuery,
  } = analysis;

  // DIFF mode replaces the two per-column searches with a single "Search Both"
  // row that filters both columns at once. The search box state is shared for
  // both (lives in the analysis state) so it survives a mode swap and can be
  // reset on clear.
  const diffMode = lensMode === LensMode.DIFF;
  const onSearchToggle = () => {
    if (sidebarSearchOpen) {
      setSidebarSearchQuery('');
    }
    setSidebarSearchOpen(!sidebarSearchOpen);
  };
  const onSearchOpen = () => setSidebarSearchOpen(true);
  const combinedQuery = sidebarSearchOpen ? sidebarSearchQuery : '';

  // Warn (above the token list) when the J-Lens is displayed and the layer being
  // hovered in the layer selector — or the start of the currently selected range
  // — sits before the default selection (the first ~1/3 of the model), where the
  // J-Lens is typically degenerate.
  const jLayers = layersByType[LensType.JACOBIAN_LENS] ?? [];
  const showJlensDegenerateWarning =
    tokens.length > 0 &&
    sidebarTypes.includes(LensType.JACOBIAN_LENS) &&
    jLayers.length > 0 &&
    ((sliderHoverLayer != null && isDegenerateLayer(jLayers, sliderHoverLayer)) ||
      (sliderHoverLayer == null && effectiveRange != null && isDegenerateLayer(jLayers, effectiveRange[0])));

  return (
    // On mobile the sidebar shares the stacked height with the chat at a 2:3
    // ratio (40% sidebar / 60% chat) so a long token list scrolls inside the
    // card instead of squeezing the chat. The split is driven by flex-grow with
    // an absolute `basis-0`: the parent's height is indefinite, so a *percentage*
    // basis (or max-height) falls back to `auto`/content and the panel balloons.
    <div className="relative flex min-h-0 w-full grow-[3] basis-0 flex-col gap-y-0 sm:w-auto sm:min-w-[40%] sm:max-w-[40%] sm:flex-none sm:shrink-0 sm:basis-auto">
      {/* Analysis content stays mounted while steering so the steer panel can
          float over a blurred copy of it (see the overlay below). */}
      <div
        className={`flex min-h-0 flex-1 flex-col gap-y-0 overflow-hidden border border-slate-200 bg-white pb-0 pt-0 transition-opacity sm:rounded-xl sm:shadow-lg ${
          tokens.length === 0 ? 'opacity-40' : 'opacity-100'
        }`}
      >
        <div className="flex flex-col gap-y-0 px-0 pb-1 pt-0">
          {/* Mode toggle + layer range slider — only once there are results. */}
          {tokens.length > 0 && (
            <div className="flex flex-row items-center justify-center gap-x-2 px-2.5 pb-0 pt-2.5 sm:px-4 sm:pt-3.5">
              <LensModeToggle mode={lensMode} setMode={setLensMode} />
              {layerBounds && effectiveRange ? (
                <LayerRangeSlider
                  bounds={layerBounds}
                  value={effectiveRange}
                  onChange={setLayerRange}
                  onLayerHover={setSliderHoverLayer}
                  defaultValue={defaultRange ?? undefined}
                  onReset={() => setLayerRange(null)}
                  label={<div className="whitespace-nowrap text-center leading-snug">LAYERS</div>}
                />
              ) : (
                <div className="flex flex-col gap-y-1 rounded-md border border-slate-200 bg-white px-2 py-1.5">
                  <div className="text-[10px] font-medium uppercase tracking-wide text-slate-400">Layers</div>
                  <div className="flex h-4 items-center justify-center rounded-md border border-dashed border-slate-200 bg-slate-50 text-[9px] text-slate-400">
                    layer range filter
                  </div>
                </div>
              )}
              {onShare && (
                <button
                  type="button"
                  onClick={onShare}
                  disabled={!canShare || shareDisabled}
                  title={shareLabel ?? 'Share'}
                  aria-label={shareLabel ?? 'Share'}
                  className="ml-1.5 mr-0 mt-3 flex h-7 w-7 shrink-0 items-center justify-center rounded-md border border-slate-300 bg-white text-slate-400 transition-colors hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  <Share2 className="h-4 w-4" />
                </button>
              )}
            </div>
          )}

          {/* Status row: the active filter, the hovered token's position, or default. */}
          {tokens.length > 0 && selectedPositions.size > 0 ? (
            <div className="mt-2 flex h-5 max-h-5 min-h-5 flex-row items-stretch overflow-hidden bg-white px-5 text-[10px] leading-none text-slate-500 sm:h-10 sm:max-h-10 sm:min-h-10 sm:text-[12px]">
              <JlensPositionFilter analysis={analysis} tokens={tokens} />
            </div>
          ) : (
            <div className="h-1.5 min-h-1.5 sm:h-3 sm:min-h-3"></div>
          )}
        </div>

        {/* Most-common token list. */}
        <div className="flex min-h-0 flex-1 flex-col gap-y-2">
          {/* DIFF mode: a single search spanning both columns, filtering them at
              once (replaces the per-column search + count headers). */}
          {diffMode && tokens.length > 0 && (
            <div className="flex flex-row items-center justify-between gap-x-2 border-b border-slate-100 px-3 pb-1 pt-1 text-[10px] font-normal text-slate-400">
              <span className="shrink-0 font-medium text-slate-300">J-Lens Readouts</span>
              <span className="-ml-3 flex min-w-0 max-w-36 flex-1 flex-row items-center justify-center">
                <SidebarSearchControl
                  open={sidebarSearchOpen}
                  query={sidebarSearchQuery}
                  label="Search J-lens and Logits"
                  onToggle={onSearchToggle}
                  onQueryChange={setSidebarSearchQuery}
                  onOpen={onSearchOpen}
                />
              </span>
              <span className="shrink-0 font-medium text-slate-300">Logit Lens</span>
            </div>
          )}
          {showJlensDegenerateWarning && (
            <div className="mx-2 -mb-1 shrink-0 rounded border border-amber-200 bg-amber-50 px-2 py-1 text-[10px] leading-snug text-amber-700 sm:mx-3">
              {JLENS_DEGENERATE_DISCLAIMER}
            </div>
          )}
          <div className={`flex min-h-0 flex-1 flex-row px-0 sm:px-1 ${diffMode ? 'divide-x divide-slate-100' : ''}`}>
            {sidebarTypes.map((type) => {
              const t = type as LensType;
              const thisSelectedKeys = selected.filter((s) => s.type === t).map((s) => s.key);
              // Shared (column-independent) order of every selected key, so DIFF
              // pins them in the same rows across both columns for easy compare.
              const pinnedKeyOrder = Array.from(new Set(selected.map((s) => s.key)));
              const otherType: LensType = t === LensType.JACOBIAN_LENS ? LensType.LOGIT_LENS : LensType.JACOBIAN_LENS;
              const crossPinnedKeys =
                lensMode === LensMode.DIFF
                  ? Array.from(
                      new Set(
                        selected
                          .filter((s) => s.type === otherType && !thisSelectedKeys.includes(s.key))
                          .map((s) => s.key),
                      ),
                    )
                  : [];
              return (
                <LensCommonColumn
                  key={type}
                  type={t}
                  items={activeSidebar[t]?.items ?? []}
                  allItems={activeSidebar[t]?.allItems ?? []}
                  searchItems={activeSidebar[t]?.searchItems}
                  countByKey={activeSidebar[t]?.countByKey ?? new Map()}
                  pinnedKeyOrder={pinnedKeyOrder}
                  selectedKeys={thisSelectedKeys}
                  crossPinnedKeys={crossPinnedKeys}
                  searchOpen={sidebarSearchOpen}
                  searchQuery={sidebarSearchQuery}
                  onSearchToggle={onSearchToggle}
                  onSearchQueryChange={setSidebarSearchQuery}
                  onSearchOpen={onSearchOpen}
                  colorOf={(key) => {
                    const idx = selected.findIndex((s) => s.key === key && s.type === t);
                    return idx >= 0 ? SELECT_COLORS[idx] : null;
                  }}
                  nextColor={selected.length < MAX_SELECT ? SELECT_COLORS[selected.length] : 'red'}
                  layerStatsByKey={activeSidebar[t]?.layerStatsByKey ?? new Map()}
                  layerRange={effectiveRange}
                  scopeLabel={positionScopeLabel}
                  combinedQuery={diffMode ? combinedQuery : undefined}
                  onHover={(key) => setHover(key ? { key, type: t } : null)}
                  onLayerHover={(key, layer) => setBarHover(layer == null ? null : { key, type: t, layer })}
                  onToggle={(key) => toggleSelect(key, t)}
                  onSteer={analysis.busy ? undefined : analysis.beginSteer}
                />
              );
            })}
          </div>
        </div>
      </div>

      {/* Steering: blur the whole column and float the steer card at the top,
          sized to its content rather than filling the column. */}
      {analysis.steer && (
        <div
          className="absolute inset-0 z-20 flex min-h-0 flex-col gap-y-2 overflow-y-auto bg-slate-300/60 px-3 pt-2.5 backdrop-blur-[1.5px] sm:gap-y-4 sm:rounded-xl sm:pt-4"
          // Clicking the blurred backdrop (but not the panel content inside it)
          // dismisses the steer modal.
          onClick={(e) => {
            if (e.target === e.currentTarget) {
              analysis.exitSteer();
            }
          }}
        >
          <SteerModeBanner
            onExit={analysis.exitSteer}
            onShare={onShare}
            canShare={canShare}
            shareDisabled={shareDisabled}
            shareLabel={shareLabel}
            onExport={onExport}
            exportDisabled={exportDisabled}
            exportLabel={exportLabel}
          />
          <JlensSteerPanel
            analysis={analysis}
            numCompletionTokens={numCompletionTokens}
            setNumCompletionTokens={setNumCompletionTokens}
            maxCompletionTokens={maxCompletionTokens}
          />
        </div>
      )}
    </div>
  );
}
