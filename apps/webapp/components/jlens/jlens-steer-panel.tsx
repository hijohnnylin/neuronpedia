'use client';

// The steering UI for the jlens interfaces: the full-panel steer interface that
// replaces the left analysis panel while a readout is being steered, plus its
// interactive per-layer selector. Driven entirely by a `JlensAnalysis` (see
// `useJlensAnalysis`).

import {
  JLENS_DEMOS_BAR_ID,
  JLENS_RIGHT_BUTTONS_ELEMENT_ID,
  JLENS_STEER_COLUMNS_ID,
  JLENS_STEER_PANEL_ID,
} from '@/app/[modelId]/jlens/jlens-tour-constants';
import { useJlensTourStep } from '@/app/[modelId]/jlens/jlens-tour-context';
import { LENS_STEER_STRENGTH_STEP, LensType, MAX_LENS_STEER_STRENGTH } from '@/lib/utils/lens';
import * as Slider from '@radix-ui/react-slider';
import { ArrowLeft, ArrowRight, Download, PlayIcon, Settings, Share2, X } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { COLOR_PILL, COLOR_RGB, COLOR_RGB_DARK, MAX_SELECT, SELECT_COLORS, SelectColor } from './jlens-analysis';
import { displayToken } from './jlens-token-popup';
import { JlensAnalysis, SteerConfig } from './use-jlens-analysis';

// Per-color "text-*-700" classes for the selected-layer number labels (kept as
// literal strings so Tailwind doesn't purge them).
/* eslint-disable @typescript-eslint/no-unused-vars */
const COLOR_TEXT_700: Record<SelectColor, string> = {
  sky: 'text-sky-700',
  amber: 'text-amber-700',
  emerald: 'text-emerald-700',
  stone: 'text-stone-700',
  indigo: 'text-indigo-700',
  red: 'text-red-700',
};

// Slate "r, g, b" used to tint unselected layer cells by occurrence count.
const SLATE_RGB = '100, 116, 139';

// Compact summary of the selected layer numbers: runs of 3+ consecutive layers
// collapse to a "start-end" range, shorter runs are listed individually
// (e.g. [12,13,14,15,16,18,32,33,34,35] -> "12-16, 18, 32-35").
function formatLayerSelection(layers: number[]): string {
  if (layers.length === 0) {
    return 'none';
  }
  const sorted = [...layers].sort((a, b) => a - b);
  const runs: number[][] = [];
  let run: number[] = [sorted[0]];
  for (let i = 1; i < sorted.length; i += 1) {
    if (sorted[i] === sorted[i - 1] + 1) {
      run.push(sorted[i]);
    } else {
      runs.push(run);
      run = [sorted[i]];
    }
  }
  runs.push(run);
  const parts: string[] = [];
  for (const r of runs) {
    if (r.length >= 3) {
      parts.push(`${r[0]}-${r[r.length - 1]}`);
    } else {
      parts.push(...r.map(String));
    }
  }
  return parts.join(', ');
}

// Human-readable label + one-line description of the active intervention, shown
// above the steered/swapped transcript (which renders inline beneath the default
// output in the chat/completion interfaces). Mirrors the wording the steer panel
// uses: e.g. `Swap J-Lens Readout "cat" with "dog"` or `Steer positively "cat"
// in J-Space`.
export function steerOutputSummary(steer: SteerConfig): { label: string; description: React.ReactNode } {
  const isSwap = steer.mode === 'swap';
  const isJacobian = steer.type === LensType.JACOBIAN_LENS;
  const readoutName = isJacobian ? 'J-Lens Readout' : 'Logit Lens Readout';
  const spaceName = isJacobian ? 'J-Space' : 'Logit Lens';
  if (isSwap) {
    return {
      label: 'Swapped Output',
      description: (
        <div className="flex flex-row items-center gap-x-1.5 whitespace-nowrap font-semibold">
          <div className="flex hidden flex-row items-center gap-x-1.5 whitespace-nowrap sm:flex">
            Swap {readoutName}{' '}
            <div className="rounded border border-slate-500 bg-slate-300 px-2 py-0.5 font-mono text-[9px] font-semibold text-slate-700 sm:px-1.5 sm:py-0.5 sm:text-[10px]">
              {displayToken(steer.token)}
            </div>{' '}
            with{' '}
            <div className="rounded border border-slate-500 bg-slate-300 px-2 py-0.5 font-mono text-[9px] font-semibold text-slate-700 sm:px-1.5 sm:py-0.5 sm:text-[10px]">
              {displayToken(steer.swapToken)}
            </div>
          </div>
          <div className="sm:hidden">Swap {readoutName}</div>
        </div>
      ),
    };
  }
  const action = steer.ablate ? 'Ablate' : steer.strength < 0 ? 'Steer negatively' : 'Steer positively';
  return {
    label: 'Steered Output',
    description: (
      <div className="flex flex-row items-center gap-x-1.5 font-semibold">
        <div className="hidden flex-row items-center gap-x-1.5 sm:flex">
          {action}{' '}
          <div className="rounded border border-slate-500 bg-slate-300 px-2 py-0.5 font-mono text-[9px] font-semibold text-slate-800 sm:px-1.5 sm:py-0.5 sm:text-[10px]">
            {displayToken(steer.token)}
          </div>{' '}
          in {spaceName}
        </div>
        {/* Mobile simplified version */}
        <div className="sm:hidden">Steered Output</div>
      </div>
    ),
  };
}

// The "you're steering" banner shown above the steer panel (atop the analysis
// sidebar) while a readout is being steered. Holds the "Back to J-Space" exit
// button on the left and the share/export buttons on the right (moved here from
// the steer panel header, so they sit across from "Back to J-Space").
export function SteerModeBanner({
  onExit,
  onShare,
  canShare = false,
  shareDisabled = false,
  shareLabel = 'Share',
  onExport,
  exportDisabled = false,
  exportLabel = 'Export',
}: {
  onExit: () => void;
  onShare?: () => void;
  canShare?: boolean;
  shareDisabled?: boolean;
  shareLabel?: string;
  onExport?: () => void;
  exportDisabled?: boolean;
  exportLabel?: string;
}) {
  // Hide the share/export buttons while the tour spotlights the steer panel so
  // the step stays visually focused (mirrors the panel's own tour behavior).
  const tourStep = useJlensTourStep();
  const tourStepElement = typeof tourStep?.element === 'string' ? tourStep.element : null;
  const isPanelTourStep =
    tourStepElement === `#${JLENS_STEER_PANEL_ID}` ||
    tourStepElement === `#${JLENS_STEER_COLUMNS_ID}` ||
    tourStepElement === `#${JLENS_DEMOS_BAR_ID}` ||
    tourStepElement === `#${JLENS_RIGHT_BUTTONS_ELEMENT_ID}`;

  return (
    <div className="flex shrink-0 flex-row items-center justify-between gap-x-3 text-xs text-slate-700">
      <button
        type="button"
        onClick={onExit}
        className="flex h-8 w-full max-w-36 flex-row items-center justify-center gap-x-1 rounded-lg border-slate-400 bg-white px-1 py-2 text-[10px] font-semibold uppercase leading-none text-slate-500 shadow-lg transition-colors hover:border-slate-600 hover:bg-slate-200 hover:text-slate-800 sm:h-9 sm:max-w-48 sm:rounded-xl sm:py-3 sm:text-[11.5px]"
      >
        <ArrowLeft className="h-3.5 w-3.5" />
        Back to J-Space
      </button>
      <div className={`flex flex-row items-center gap-x-1.5 ${isPanelTourStep ? 'hidden' : ''}`}>
        <button
          type="button"
          onClick={onExport}
          disabled={exportDisabled}
          title={exportLabel}
          aria-label={exportLabel}
          className="flex h-8 w-8 items-center justify-center rounded-md border-slate-300 bg-white text-slate-500 shadow-lg transition-colors hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-40 sm:h-9 sm:w-9"
        >
          <Download className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
        </button>
        <button
          type="button"
          onClick={onShare}
          disabled={!canShare || shareDisabled}
          title={shareLabel}
          aria-label={shareLabel}
          className="flex h-8 w-8 items-center justify-center rounded-md border-slate-300 bg-white text-slate-500 shadow-lg transition-colors hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-40 sm:h-9 sm:w-9"
        >
          <Share2 className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
        </button>
      </div>
    </div>
  );
}

// The label shown above the default (unsteered) output once steering is active,
// so the user can tell the two transcripts apart.
export function DefaultOutputHeader() {
  return (
    <div className="sticky top-0 z-10 mt-0 flex h-6 max-h-6 min-h-6 w-full items-center justify-center rounded-md bg-slate-200 px-1 text-center text-[10px] font-bold tracking-wide text-slate-500 sm:h-8 sm:max-h-8 sm:min-h-8 sm:w-auto sm:rounded-full sm:text-[11px]">
      Default Output
    </div>
  );
}

// The full-width divider + label + description shown between the default output
// and the steered/swapped transcript.
export function SteerOutputHeader({ steer }: { steer: SteerConfig }) {
  const { label, description } = steerOutputSummary(steer);
  return (
    <div className="sticky top-0 z-10 flex h-6 max-h-6 min-h-6 flex-col items-center justify-center gap-y-0.5 rounded-md bg-slate-200 px-1 sm:h-8 sm:max-h-8 sm:min-h-8 sm:rounded-full">
      <div className="text-[10px] font-normal text-slate-500 sm:text-[11px]">{description}</div>
    </div>
  );
}

// Interactive per-layer selector: a row of cells (one per layer of the lens
// type) tinted by the token's occurrence count, where clicking toggles a layer
// and dragging paints a run (select or deselect). Non-contiguous selections are
// allowed. Commits the new selection on pointer-up (so a drag re-runs once).
// Unselected cells are slate-tinted; selected cells use the token color and
// show their layer number above them.
function SteerLayerSelector({
  layerCounts,
  selected,
  onCommit,
  colorRgb,
  disabled = false,
}: {
  layerCounts: { layer: number; count: number }[];
  selected: number[];
  onCommit: (layers: number[]) => void;
  colorRgb: string;
  disabled?: boolean;
}) {
  const [working, setWorking] = useState<Set<number>>(() => new Set(selected));
  const workingRef = useRef(working);
  useEffect(() => {
    workingRef.current = working;
  }, [working]);
  // Sync from props when not mid-gesture (e.g. reset / new token).
  const gestureRef = useRef<{ mode: 'select' | 'deselect' } | null>(null);
  useEffect(() => {
    if (!gestureRef.current) {
      setWorking(new Set(selected));
    }
  }, [selected]);

  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const maxCount = Math.max(0, ...layerCounts.map((l) => l.count));
  const n = layerCounts.length;
  const step = n > 0 ? 100 / n : 0;

  const apply = (layer: number, mode: 'select' | 'deselect') => {
    setWorking((cur) => {
      const next = new Set(cur);
      if (mode === 'select') {
        next.add(layer);
      } else {
        next.delete(layer);
      }
      return next;
    });
  };

  useEffect(() => {
    const onUp = () => {
      if (gestureRef.current) {
        gestureRef.current = null;
        onCommit([...workingRef.current].sort((a, b) => a - b));
      }
    };
    window.addEventListener('pointerup', onUp);
    return () => window.removeEventListener('pointerup', onUp);
  }, [onCommit]);

  return (
    <div className={`flex flex-col gap-y-1 ${disabled ? 'pointer-events-none opacity-50' : ''}`}>
      <div
        className="relative flex h-6 w-full select-none flex-row overflow-hidden rounded-md border border-slate-200 bg-slate-50"
        onMouseLeave={() => setHoverIdx(null)}
      >
        {layerCounts.map((lc, i) => {
          const isSelected = working.has(lc.layer);
          const alpha = maxCount > 0 ? lc.count / maxCount : 0;
          const tintRgb = isSelected ? colorRgb : SLATE_RGB;
          return (
            <span
              key={lc.layer}
              onPointerDown={(e) => {
                if (disabled) {
                  return;
                }
                e.preventDefault();
                const mode = workingRef.current.has(lc.layer) ? 'deselect' : 'select';
                gestureRef.current = { mode };
                apply(lc.layer, mode);
              }}
              onPointerEnter={() => {
                setHoverIdx(i);
                if (!disabled && gestureRef.current) {
                  apply(lc.layer, gestureRef.current.mode);
                }
              }}
              className={`relative h-full flex-1 ${disabled ? '' : 'cursor-pointer'} ${i > 0 ? 'border-l border-white/40' : ''}`}
            >
              <span className="absolute inset-0" style={{ backgroundColor: `rgba(${tintRgb}, ${alpha})` }} />
              {isSelected && (
                <span className="absolute inset-0" style={{ backgroundColor: `rgba(${colorRgb}, 0.25)` }} />
              )}
            </span>
          );
        })}
        {hoverIdx != null && (
          <div
            className="pointer-events-none absolute -top-0.5 z-10 -translate-x-1/2 -translate-y-full rounded border border-slate-500 bg-white px-1.5 py-0.5 text-[8px] font-bold tabular-nums leading-none text-slate-700 shadow-sm"
            style={{ left: `${(hoverIdx + 0.5) * step}%` }}
          >
            Layer {layerCounts[hoverIdx].layer} · {layerCounts[hoverIdx].count}×
          </div>
        )}
      </div>
      {/* <div className="flex flex-row justify-between text-[9px] uppercase tracking-wide text-slate-400">
        <span>Layer {layerCounts[0]?.layer ?? 0}</span>
        <span>Layer {layerCounts[n - 1]?.layer ?? 0}</span>
      </div> */}
    </div>
  );
}

// The steer interface that temporarily replaces the whole left panel while a
// readout is being steered. Shows the steered token, an interactive layer
// selector (defaulting to the peak layer), and a signed strength slider
// (negative suppresses the readout).
export function JlensSteerPanel({
  analysis,
  numCompletionTokens,
  setNumCompletionTokens,
  maxCompletionTokens,
}: {
  analysis: JlensAnalysis;
  // Number of tokens the steered run generates — the SAME value the advanced
  // panel controls (shared state, owned by the chat/completion interface).
  numCompletionTokens?: number;
  setNumCompletionTokens?: (n: number) => void;
  maxCompletionTokens?: number;
}) {
  const {
    steer,
    steerInfo,
    steerStreaming,
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
  } = analysis;
  const [strength, setStrength] = useState(steer?.strength ?? 0);
  useEffect(() => {
    setStrength(steer?.strength ?? 0);
  }, [steer?.strength]);

  // Advanced settings (gear toggle): reveals the "steer/swap generated tokens"
  // toggle and the "generated tokens" count slider. Hidden by default.
  const [showAdvanced, setShowAdvanced] = useState(false);

  // While the tour is on the steer panel — and the subsequent steps that keep it
  // open (the columns comparison and the final wrap-up) — strip it down to the
  // essentials: hide the mode (steer/swap) toggle, the layer selector, the
  // cancel button, and the share/export buttons so it stays visually focused.
  const tourStep = useJlensTourStep();
  const tourStepElement = typeof tourStep?.element === 'string' ? tourStep.element : null;
  const isPanelTourStep =
    tourStepElement === `#${JLENS_STEER_PANEL_ID}` ||
    tourStepElement === `#${JLENS_STEER_COLUMNS_ID}` ||
    tourStepElement === `#${JLENS_DEMOS_BAR_ID}` ||
    tourStepElement === `#${JLENS_RIGHT_BUTTONS_ELEMENT_ID}`;

  // Prefill the swap-in token with " ants" (leading space included) when the
  // tour reaches this step, so the scripted spiders → ants swap is ready to run.
  useEffect(() => {
    if (isPanelTourStep) {
      setSwapToken(' ants');
    }
  }, [isPanelTourStep, setSwapToken]);

  if (!steer) {
    return null;
  }
  const locked = steerStreaming;
  const layerCounts = steerInfo?.layerCounts ?? [];
  const selectedLayers = [...steer.layers].sort((a, b) => a - b);
  const isDefault =
    steerInfo != null &&
    selectedLayers.length === steerInfo.defaultLayers.length &&
    selectedLayers.every((l, i) => l === [...steerInfo.defaultLayers].sort((a, b) => a - b)[i]);
  const isSwap = steer.mode === 'swap';
  const swapMissing = isSwap && !steer.swapToken.trim();
  // Caution the user when the token being swapped out starts with a leading
  // space but their swap-in token doesn't (only once they've typed something).
  const leadingSpaceMismatch =
    isSwap && steer.swapToken !== '' && steer.token.startsWith(' ') && !steer.swapToken.startsWith(' ');
  const typeLabel = steer.type === LensType.JACOBIAN_LENS ? 'Jacobian Lens' : 'Logit Lens';
  const steerByLabel = `${isSwap ? 'Swap' : 'Steer'} ${typeLabel} Readout`;

  // The color this readout has (or would get) as a sidebar selection, so the
  // steer panel's readout and layer selector match it. The steer token is the
  // exact sidebar key (no trimming), so it matches the selection directly.
  const steerSelIdx = analysis.selected.findIndex((s) => s.key === steer.token && s.type === steer.type);
  const steerColor: SelectColor =
    steerSelIdx >= 0
      ? SELECT_COLORS[steerSelIdx]
      : analysis.selected.length < MAX_SELECT
        ? SELECT_COLORS[analysis.selected.length]
        : 'red';
  const steerColorRgb = COLOR_RGB[steerColor];
  const steerColorRgbDark = COLOR_RGB_DARK[steerColor];
  const steerPill = COLOR_PILL[steerColor];

  // Shared Steer/Stop trigger — rendered as a mobile-only third column (right of
  // steer strength) and, on desktop, at the bottom of the settings panel.
  const steerButton = (
    <button
      type="button"
      onClick={() => {
        if (steerStreaming) {
          // During the guided tour's swap step the run can't be cancelled.
          if (!isPanelTourStep) {
            stopSteer();
          }
          return;
        }
        void runSteer();
      }}
      disabled={
        (!steerStreaming && (selectedLayers.length === 0 || swapMissing)) || (steerStreaming && isPanelTourStep)
      }
      style={steerStreaming ? undefined : { backgroundColor: `rgb(${steerColorRgbDark})` }}
      className={`flex w-20 flex-row items-center justify-center gap-x-1.5 rounded-lg px-2 py-2 text-[12px] font-semibold uppercase text-white transition-all disabled:cursor-not-allowed disabled:opacity-40 sm:w-auto sm:flex-1 sm:basis-3/4 sm:rounded-lg sm:px-4 sm:py-2.5 sm:text-[12px] ${
        steerStreaming ? 'bg-red-500 hover:bg-red-600' : 'hover:brightness-95'
      }`}
    >
      {steerStreaming ? (
        <>
          <X className="h-3.5 w-3.5 sm:h-3.5 sm:w-3.5" />
          Stop
        </>
      ) : (
        <>
          <PlayIcon className="h-3.5 w-3.5 sm:h-3.5 sm:w-3.5" /> {isSwap ? 'Swap' : 'Steer'}
        </>
      )}
    </button>
  );

  // Cancel (exit steering) — sits to the left of the Steer/Swap trigger. Hidden
  // while the tour spotlights this panel so the step stays focused on Swap.
  const cancelButton = isPanelTourStep ? null : (
    <button
      type="button"
      onClick={exitSteer}
      disabled={locked}
      className="flex shrink-0 flex-row items-center justify-center rounded-lg border border-slate-300 bg-white px-4 py-2 text-[12px] font-semibold uppercase text-slate-500 transition-colors hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-40 sm:rounded-lg sm:py-2.5 sm:text-[11px]"
    >
      Cancel
    </button>
  );

  return (
    <div
      id={JLENS_STEER_PANEL_ID}
      className="flex shrink-0 flex-col gap-y-0 overflow-hidden rounded-xl border-slate-200 bg-white shadow-lg"
    >
      <div
        className={`flex w-full flex-row items-center border-b bg-white px-4 py-2 sm:py-3 ${
          isPanelTourStep ? 'justify-center' : 'justify-center sm:justify-between'
        }`}
      >
        <div
          className={`flex flex-row items-center gap-x-2 text-xs font-semibold text-slate-600 sm:text-[14px] ${
            isPanelTourStep ? 'text-center' : 'text-left'
          }`}
        >
          {steerByLabel}
        </div>
        {!isPanelTourStep && (
          <button
            type="button"
            onClick={() => setShowAdvanced((v) => !v)}
            title="Advanced settings"
            aria-pressed={showAdvanced}
            className={`hidden h-6 w-6 items-center justify-center rounded-md border transition-colors sm:flex sm:h-7 sm:w-7 ${
              showAdvanced
                ? 'border-slate-400 bg-slate-200 text-slate-700'
                : 'border-slate-300 bg-slate-50 text-slate-500 hover:bg-slate-200'
            }`}
          >
            <Settings className="h-3 w-3 sm:h-3.5 sm:w-3.5" />
          </button>
        )}
      </div>
      <div
        className={`relative flex-row items-center justify-start border-slate-200 px-2.5 py-3 pb-1 pt-3 sm:px-4 ${
          isPanelTourStep ? 'hidden' : 'flex'
        }`}
      >
        <div className="mb-2 mt-0 flex w-full flex-col items-center justify-center">
          <div className="mb-1 hidden flex-row items-center justify-center text-[10px] font-medium normal-case text-slate-400 sm:flex">
            <div className="uppercase">Mode</div>
          </div>
          <div className="inline-flex shrink-0 rounded-md border-slate-100 bg-slate-200 p-0.5 sm:rounded-lg">
            {(['steer', 'swap'] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setSteerMode(m)}
                disabled={locked}
                className={`rounded-md px-6 py-0.5 text-[10px] font-semibold capitalize transition-colors disabled:cursor-not-allowed disabled:opacity-40 sm:px-8 sm:py-1.5 sm:text-[12px] ${
                  steer.mode === m ? 'bg-white text-slate-800 shadow-sm' : 'text-slate-600 hover:text-slate-700'
                }`}
              >
                {m}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Settings (locked while a steered run is streaming). */}
      <div className="relative flex shrink-0 flex-col gap-y-1 overflow-y-auto px-4 pb-2 pt-1 sm:py-3 sm:pb-3 sm:pt-1">
        {/* On mobile, lay out the token/layers stack and the steer-strength
            column side by side; on desktop they stack vertically (strength
            below). */}
        <div className="flex flex-col gap-x-0 gap-x-4 gap-y-1">
          {/* Left column: token to steer + selected layers. */}
          <div className="flex min-w-0 flex-1 flex-col gap-y-1 sm:flex-none">
            {/* The action label + source readout pill, and (in swap mode) the
            target-token input the source is swapped into. The label is hidden
            while the tour spotlights this panel (the pill/input stay visible). */}
            {!isPanelTourStep &&
              (isSwap ? (
                <div className="flex flex-row items-center justify-center text-[10px] font-medium normal-case text-slate-400">
                  <div className="uppercase">
                    {steer.type === LensType.JACOBIAN_LENS
                      ? 'Readout to Swap - Include leading space'
                      : 'Readout to Swap - Include leading space'}
                  </div>
                </div>
              ) : (
                <div className="flex flex-row items-center justify-center text-[10px] font-medium normal-case text-slate-400">
                  <div>TOKEN TO STEER</div>
                </div>
              ))}
            <div className="flex w-full flex-col items-center justify-center gap-y-1 sm:mb-2">
              <div
                className={`flex min-h-6 flex-row items-center gap-x-1 text-[11px] font-semibold text-slate-700 sm:min-h-8 sm:gap-x-1.5 sm:text-[13px] ${
                  isPanelTourStep ? 'justify-center' : ''
                }`}
              >
                <div
                  className="rounded border px-2 py-0.5 font-mono text-[9px] font-semibold text-slate-800 sm:px-2 sm:py-1 sm:text-xs"
                  style={{ backgroundColor: `rgb(${steerPill.bg})`, borderColor: `rgb(${steerPill.ring})` }}
                >
                  {displayToken(steer.token)}
                </div>
                {isSwap && (
                  <>
                    <ArrowRight className="h-3 w-3 shrink-0 text-slate-400 sm:h-4 sm:w-4" />
                    <input
                      type="text"
                      // Show the space-replacement glyph (matching `displayToken`) while
                      // typing, but keep real spaces under the hood.
                      value={steer.swapToken.replace(/ /g, '␣')}
                      onChange={(e) => setSwapToken(e.target.value.replace(/␣/g, ' '))}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && !locked && selectedLayers.length > 0 && !swapMissing) {
                          e.preventDefault();
                          void runSteer();
                        }
                      }}
                      disabled={locked || isPanelTourStep}
                      placeholder="e.g. ' ants'"
                      style={{ color: `rgb(${COLOR_RGB_DARK[steerColor]})` }}
                      className={`rounded border px-2 py-0.5 font-mono text-[11px] font-semibold leading-normal placeholder:text-slate-300 focus:outline-none disabled:cursor-not-allowed sm:rounded-md sm:py-1.5 sm:text-xs sm:leading-normal ${
                        isPanelTourStep ? 'w-20 sm:w-24' : 'w-28 sm:w-32'
                      } ${
                        isPanelTourStep ? 'disabled:opacity-100' : 'disabled:opacity-100'
                      } ${isPanelTourStep ? 'text-center' : 'text-center'} ${
                        swapMissing
                          ? 'border-rose-400 ring-1 ring-rose-400 focus:border-rose-500 focus:ring-rose-500'
                          : 'border-slate-300 focus:border-slate-500'
                      }`}
                    />
                    {leadingSpaceMismatch && (
                      <>
                        <div className="-mt-9 ml-1 hidden text-center text-[8px] font-medium leading-snug text-rose-600 sm:-mt-0 sm:ml-3 sm:block">
                          Prepend leading space to match the original token.
                        </div>
                        <button
                          type="button"
                          onClick={() => setSwapToken(` ${steer.swapToken}`)}
                          disabled={locked}
                          className="hidden shrink-0 whitespace-nowrap rounded-md border border-rose-300 bg-rose-50 px-2 py-1 font-mono text-[9px] font-semibold text-rose-700 transition-colors hover:bg-rose-100 disabled:cursor-not-allowed disabled:opacity-40 sm:block"
                        >
                          Apply
                        </button>
                      </>
                    )}
                  </>
                )}
              </div>
            </div>

            {/* Interactive layer selector. */}
            <div className={`flex-col gap-y-0 sm:mt-1 ${isPanelTourStep ? 'hidden' : 'flex'}`}>
              <div className="hidden flex-row items-center justify-center pb-1 sm:flex">
                {/* <div className="text-[10px] font-medium uppercase text-slate-400">
                  {isSwap ? 'Swap' : 'Steer'} Layers{' '}
                </div> */}
                <div className="py-[2.5px] text-[10px] font-medium leading-none text-slate-400">
                  {isSwap ? 'SWAP LAYER' : 'STEER LAYER'}
                  {selectedLayers.length > 1 ? 'S' : ''}{' '}
                  <span className="text-slate-500">{formatLayerSelection(selectedLayers)}</span>
                </div>
              </div>

              <div className="hidden sm:block">
                <SteerLayerSelector
                  layerCounts={layerCounts}
                  selected={steer.layers}
                  onCommit={setSteerLayers}
                  colorRgb={steerColorRgb}
                  disabled={locked}
                />
              </div>

              <div className="flex flex-row items-center justify-between pt-1">
                {!isDefault && (
                  <button
                    type="button"
                    onClick={resetSteerLayers}
                    disabled={isDefault || locked}
                    className="flex items-center justify-center rounded-[3px] bg-slate-300 px-1.5 py-[2.5px] text-[8px] font-bold uppercase leading-none tracking-wide text-slate-500 transition-colors hover:bg-slate-300 hover:text-slate-700 disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    Reset
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Right column on mobile / below on desktop: steer strength (+ generated tokens). */}
          <div
            className={`flex flex-row gap-x-6 pb-0 pt-0 ${isPanelTourStep ? 'sm:pt-0' : 'sm:pt-3'} ${!isSwap ? 'flex-1 sm:flex-none' : ''} ${locked ? 'pointer-events-none opacity-50' : ''}`}
          >
            {!isSwap && (
              <div className="mb-3 flex w-full flex-1 flex-col items-center justify-center gap-y-0.5">
                <div className="text-[10px] font-medium uppercase text-slate-400">Steer Strength</div>
                <div className="flex w-full flex-col items-start gap-x-2 gap-y-2 sm:flex-row sm:items-center">
                  {/* Ablate toggle: mutually exclusive with strength. Turning it on
                  projects the readout direction out of the residual (strength
                  ignored); changing the slider turns it back off. */}
                  <button
                    type="button"
                    onClick={() => setSteerAblate(!steer.ablate)}
                    disabled={locked}
                    title="Ablate: remove this readout's direction from the residual (ignores strength)"
                    style={
                      steer.ablate
                        ? { backgroundColor: `rgb(${steerColorRgb})`, borderColor: `rgb(${steerColorRgb})` }
                        : undefined
                    }
                    className={`mt-0.5 flex h-5 shrink-0 items-center rounded-full border px-4 text-[9px] font-semibold uppercase transition-colors disabled:cursor-not-allowed disabled:opacity-40 sm:mt-0 sm:h-7 sm:text-[10px] ${
                      steer.ablate ? 'text-white' : 'border-slate-300 bg-white text-slate-500 hover:bg-slate-100'
                    }`}
                  >
                    Ablate
                  </button>
                  <Slider.Root
                    value={[strength]}
                    min={-MAX_LENS_STEER_STRENGTH}
                    max={MAX_LENS_STEER_STRENGTH}
                    step={LENS_STEER_STRENGTH_STEP}
                    disabled={locked}
                    onValueChange={(v) => setStrength(v[0])}
                    onValueCommit={(v) => setSteerStrength(v[0])}
                    className={`relative flex h-8 w-full flex-1 cursor-pointer items-center data-[disabled]:cursor-not-allowed ${
                      steer.ablate ? 'opacity-40' : ''
                    }`}
                  >
                    <Slider.Track className="relative h-[10px] grow rounded-full border border-slate-300 bg-white">
                      <Slider.Range
                        className="absolute h-full rounded-full"
                        style={{ backgroundColor: `rgb(${steerColorRgb})` }}
                      />
                    </Slider.Track>
                    <Slider.Thumb className="flex h-5 w-10 items-center justify-center rounded-full border border-slate-300 bg-white text-[9px] font-semibold tabular-nums text-slate-500 shadow hover:bg-slate-50 focus:outline-none sm:h-7 sm:w-12 sm:text-[11px]">
                      {strength >= 0 ? `+${strength.toFixed(1)}` : strength.toFixed(1)}x
                    </Slider.Thumb>
                  </Slider.Root>
                </div>
              </div>
            )}

            {showAdvanced && setNumCompletionTokens && numCompletionTokens !== undefined && (
              <div className="flex flex-1 flex-col gap-y-0.5">
                <div className="text-[10px] font-medium uppercase text-slate-400">Generated Tokens</div>
                <Slider.Root
                  value={[numCompletionTokens]}
                  min={0}
                  max={maxCompletionTokens ?? 256}
                  step={1}
                  disabled={locked}
                  onValueChange={(v) => setNumCompletionTokens(v[0])}
                  className="relative flex h-7 w-full cursor-pointer items-center data-[disabled]:cursor-not-allowed"
                >
                  <Slider.Track className="relative h-[10px] grow rounded-full border border-slate-300 bg-white">
                    <Slider.Range
                      className="absolute h-full rounded-full"
                      style={{ backgroundColor: `rgb(${steerColorRgb})` }}
                    />
                  </Slider.Track>
                  <Slider.Thumb className="flex h-7 w-12 items-center justify-center rounded-full border border-slate-300 bg-white text-[11px] font-semibold tabular-nums text-slate-500 shadow hover:bg-slate-50 focus:outline-none">
                    {numCompletionTokens}
                  </Slider.Thumb>
                </Slider.Root>
              </div>
            )}
          </div>

          {/* Mobile-only: Steer/Swap trigger as a third column to the right of
              steer strength (on desktop it lives at the bottom instead). */}
          <div className="flex shrink-0 items-center justify-center gap-x-2 sm:hidden">
            {cancelButton}
            {steerButton}
          </div>
        </div>

        {/* Apply the intervention to generated tokens too (default on). */}
        <div
          className={`${showAdvanced ? 'flex' : 'hidden'} mb-2 flex-row items-center justify-start gap-x-2.5 ${locked ? 'pointer-events-none opacity-50' : ''}`}
        >
          <button
            type="button"
            role="switch"
            aria-checked={steer.steerGenerated}
            onClick={() => setSteerGenerated(!steer.steerGenerated)}
            disabled={locked}
            style={steer.steerGenerated ? { backgroundColor: `rgb(${steerColorRgb})` } : undefined}
            className={`relative h-5 w-9 shrink-0 rounded-full transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
              steer.steerGenerated ? '' : 'bg-slate-300'
            }`}
          >
            <span
              className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-all ${
                steer.steerGenerated ? 'left-[18px]' : 'left-0.5'
              }`}
            />
          </button>
          <div className="text-[10px] font-medium uppercase text-slate-400">
            {isSwap ? 'Swap' : 'Steer'} Generated Tokens
          </div>
        </div>

        {/* Manual trigger (Steer -> Stop while streaming). Desktop-only here; on
            mobile this button is rendered as a third column above instead. */}
        <div className="relative mb-2 hidden flex-row items-center justify-center gap-x-2 sm:flex">
          {cancelButton}
          {steerButton}
        </div>
      </div>
    </div>
  );
}
