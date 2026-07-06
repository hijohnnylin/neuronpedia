'use client';

import { useGlobalContext } from '@/components/provider/global-provider';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/shadcn/dialog';
import { LoadingSquare } from '@/components/svg/loading-square';
import {
  DEFAULT_LENS_COMPLETION_TOKENS_COMPLETION,
  DEFAULT_LENS_TEMPERATURE,
  DEFAULT_LENS_TOP_N,
  LENS_TYPE_ORDER,
  LensMetaMessage,
  LensMode,
  LensTokenMessage,
  LensType,
  MAX_LENS_COMPLETION_PROMPT_CHARS,
  MAX_LENS_COMPLETION_TOKENS_COMPLETION,
} from '@/lib/utils/lens';
import { ArrowUp, Check, Copy, Download, Settings, Share2, Trash2, X } from 'lucide-react';
import { useCallback, useContext, useEffect, useRef, useState } from 'react';
import ReactTextareaAutosize from 'react-textarea-autosize';
import JlensAdvanced from './jlens-advanced';
import { MAX_SELECT } from './jlens-analysis';
import { JlensAnalysisPanel, JlensProviders } from './jlens-analysis-panel';
import { JlensCommentary, useSharedCommentary } from './jlens-commentary';
import {
  buildSteerShareBody,
  defaultExportFilename,
  downloadJson,
  JlensExportCompletion,
  JlensExportSteer,
} from './jlens-export';
import { LensModeSetContext } from './jlens-lens-mode';
import { JlensShareDialog } from './jlens-share-dialog';
import { DefaultOutputHeader, SteerOutputHeader } from './jlens-steer-panel';
import { runLensStream as baseRunLensStream, RunLensStreamParams } from './jlens-stream';
import JlensTokenChip, { scrollContainerToTokenPositions, TokenBand } from './jlens-token';
import { LayerRange } from './jlens-token-popup';
import { SteerConfig, useJlensAnalysis } from './use-jlens-analysis';

export default function JlensCompletion({
  modelId,
  loadedData = null,
  sharedDescription = null,
  sharedAttribution = null,
  inferenceAvailable = true,
  onClear,
  onSidebarSelectionChange,
  onNextDemo,
  nextDemoLabel,
  onFreeChatDemo,
}: {
  modelId: string;
  loadedData?: JlensExportCompletion | null;
  // When viewing a shared link: the sharer's description + attribution line.
  sharedDescription?: string | null;
  sharedAttribution?: string | null;
  // Whether an inference host currently serves this model. When false, cached
  // results still render but live actions (send / steer / swap / filter toggle)
  // are gated behind a "model unavailable" notice.
  inferenceAvailable?: boolean;
  // Called when the run is cleared (trash icon) — used by the shared view to
  // drop the shareId from the URL.
  onClear?: () => void;
  // Called when the user locks/unlocks a sidebar token. The shared view uses
  // this to drop the shareId (URL + state) once the selection diverges from
  // the shared snapshot.
  onSidebarSelectionChange?: () => void;
  // When viewing a predefined demo, advances to the next demo (or free chat on
  // the last one). Rendered as a button in the commentary banner.
  onNextDemo?: () => void;
  // Label for the next-demo button ("Next demo", or "Free Chat" on the last one).
  nextDemoLabel?: string;
  // On non-last demos, jumps straight to free chat via a secondary slate button.
  onFreeChatDemo?: () => void;
}) {
  const [prompt, setPrompt] = useState('');
  const [streaming, setStreaming] = useState(false);
  // True for the duration of a generation stream (runLens). Drives the loading
  // square shown below the in-progress completion. Distinct from `streaming`,
  // which is also set during non-generating read-out replays.
  const [generating, setGenerating] = useState(false);
  const [awaitingFirstResponse, setAwaitingFirstResponse] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  // Remaining requests in the current hourly window for `/api/lens/prompt`,
  // surfaced next to the send button (see `x-limit-remaining` in middleware).
  // `null` until the first response. Every jlens action (run, re-analyze,
  // steer, filter toggle) hits the same endpoint/bucket, so all go through
  // `runLensStream` below, which updates this via `onRateLimit`.
  const [limitRemaining, setLimitRemaining] = useState<number | null>(null);
  // Wrap the shared stream helper so every call reports its remaining rate
  // limit into `limitRemaining` without threading the callback through each
  // call site.
  const runLensStream = useCallback(
    (params: Omit<RunLensStreamParams, 'onRateLimit'>) =>
      baseRunLensStream({ ...params, onRateLimit: setLimitRemaining }),
    [],
  );
  // Shown when a live action is attempted but no inference host serves the model.
  const [unavailableOpen, setUnavailableOpen] = useState(false);
  const notifyUnavailable = useCallback(() => setUnavailableOpen(true), []);

  const [shareOpen, setShareOpen] = useState(false);

  const [temperature, setTemperature] = useState(DEFAULT_LENS_TEMPERATURE);
  const [numCompletionTokens, setNumCompletionTokens] = useState(DEFAULT_LENS_COMPLETION_TOKENS_COMPLETION);
  const [topN, setTopN] = useState(DEFAULT_LENS_TOP_N);

  const [meta, setMeta] = useState<LensMetaMessage | null>(null);
  const [tokens, setTokens] = useState<LensTokenMessage[]>([]);

  // Steered run: a SEPARATE token stream + meta (the original run above is left
  // untouched) rendered in the steer panel.
  const [steerTokens, setSteerTokens] = useState<LensTokenMessage[]>([]);
  const [steerMeta, setSteerMeta] = useState<LensMetaMessage | null>(null);

  const { user } = useGlobalContext();
  const setLensMode = useContext(LensModeSetContext);

  // The shared-commentary banner state, lifted here so the mobile banner (above
  // the run) and the desktop banner (in the analysis panel) share one dismiss.
  const { showCommentary, dismiss: dismissCommentary } = useSharedCommentary(sharedDescription);

  const analysis = useJlensAnalysis({
    tokens,
    meta,
    modelId,
    busy: streaming || awaitingFirstResponse,
    inferenceAvailable,
    onInferenceUnavailable: notifyUnavailable,
    onSidebarSelectionChange,
  });
  // A second analysis instance drives the steered-output transcript (its own
  // hover popups + readout highlights), independent of the primary one.
  const steerAnalysis = useJlensAnalysis({
    tokens: steerTokens,
    meta: steerMeta,
    modelId,
    busy: analysis.steerStreaming,
  });
  const {
    lensMode,
    layersByType,
    effectiveRange,
    hideNonWordTokens,
    setHideNonWordTokens,
    selected,
    setSelected,
    setLockedLayer,
    selectedPositions,
    setSelectedPositions,
    highlightedPosition,
    onChatPointerDown,
    dragging,
    handleTokenHover,
    bandsByPosition,
  } = analysis;

  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  // Positions to bring into view on share load. Set while restoring a shared run
  // that carried a selection; kept set (suppressing the auto-scroll-to-bottom
  // below) until the user starts a fresh run, then cleared.
  const pendingScrollSelectionRef = useRef<number[] | null>(null);
  // The in-flight steered run's abort controller.
  const steerAbortRef = useRef<AbortController | null>(null);
  const steering = analysis.steer != null;
  // A steered run restored from a shared link, applied once the main tokens have
  // been hydrated (so the steer panel's per-layer counts compute correctly).
  const pendingSteerRef = useRef<JlensExportSteer | null>(null);
  // The readout to highlight in the steered transcript, snapshotted when a
  // steered run is started (or restored). Editing the swap token afterwards does
  // NOT change it — the highlight only updates on a new steered run.
  const steerHighlightRef = useRef<{ key: string; type: LensType } | null>(null);

  // Auto-scroll to bottom as tokens stream in (including the steered output,
  // which now renders inline below the default output). Skipped while a shared
  // run's restored selection is still waiting to be scrolled into view.
  // `analysis.steer` is a dep so that entering steer mode — which, on a shared
  // run, happens a commit AFTER the tokens render and adds the steered column —
  // re-scrolls to the true bottom rather than leaving the view mid-transcript.
  useEffect(() => {
    if (pendingScrollSelectionRef.current) {
      return;
    }
    const el = scrollRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [tokens, awaitingFirstResponse, steerTokens, analysis.steer, analysis.steerStreaming]);

  const runLens = useCallback(
    async (promptValue: string) => {
      if (!promptValue.trim() || streaming) {
        return;
      }
      setError(null);
      // A fresh run supersedes any share-load selection scroll; re-enable the
      // auto-scroll-to-bottom.
      pendingScrollSelectionRef.current = null;

      const priorTokens = tokens;
      const priorMeta = meta;

      // Prefix-reuse: hand the server the token ids we already have read-outs
      // for so it only recomputes the changed suffix (matches the chat flow).
      const priorIds = priorTokens.map((t) => t.id);
      const cachedTokenIds = priorIds.every((id) => typeof id === 'number') ? (priorIds as number[]) : undefined;
      let reuseLen = 0;

      setStreaming(true);
      setGenerating(true);
      setAwaitingFirstResponse(true);
      // Clear the previous run's results immediately so a re-submit shows the
      // loading state instead of leaving stale tokens up until the first token
      // streams back. `priorTokens`/`priorMeta` are captured above, so
      // prefix-reuse and error-restore still work.
      setTokens([]);
      setMeta(null);

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        await runLensStream({
          modelId,
          prompt: promptValue,
          type: LENS_TYPE_ORDER,
          topN,
          temperature,
          numCompletionTokens,
          cachedTokenIds,
          filterNonWordTokens: hideNonWordTokens,
          signal: controller.signal,
          onMeta: (m) => {
            setMeta(m);
            reuseLen = m.reuse_len ?? 0;
          },
          onPromptTokens: (p) => {
            setAwaitingFirstResponse(false);
            setTokens(
              p.tokens.map((t) => ({
                kind: 'token' as const,
                position: t.position,
                token: t.token,
                id: t.id,
                is_generated: t.is_generated,
                results: t.position < reuseLen ? (priorTokens[t.position]?.results ?? []) : [],
              })),
            );
          },
          onToken: (t) => {
            setAwaitingFirstResponse(false);
            setTokens((prev) => {
              const next = prev.slice();
              next[t.position] = t;
              return next;
            });
          },
        });
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') {
          // user cancelled — keep whatever streamed so far
        } else {
          setTokens(priorTokens);
          setMeta(priorMeta);
          setError(err instanceof Error ? err.message : String(err));
        }
      } finally {
        setStreaming(false);
        setGenerating(false);
        setAwaitingFirstResponse(false);
        abortRef.current = null;
      }
    },
    [streaming, tokens, meta, modelId, topN, temperature, numCompletionTokens, hideNonWordTokens, analysis],
  );

  // Run the same prompt with steering applied into the SEPARATE steered-output
  // state (or, with `null`, clear it). The original run is never touched.
  const runLensSteer = useCallback(
    async (config: SteerConfig | null) => {
      steerAbortRef.current?.abort();
      steerAbortRef.current = null;

      if (!config) {
        steerHighlightRef.current = null;
        setSteerTokens([]);
        setSteerMeta(null);
        return;
      }

      if (!prompt.trim()) {
        return;
      }
      const promptValue = prompt;
      // A fresh steered run supersedes any share-load selection scroll.
      pendingScrollSelectionRef.current = null;

      // Snapshot which readout to highlight (the swap target, or the steered
      // token) so later edits to the swap token don't change the highlight.
      steerHighlightRef.current = {
        key: config.mode === 'swap' && config.swapToken.trim() ? config.swapToken : config.token,
        type: config.type,
      };

      setError(null);
      setSteerTokens([]);
      setSteerMeta(null);
      const controller = new AbortController();
      steerAbortRef.current = controller;

      try {
        await runLensStream({
          modelId,
          prompt: promptValue,
          type: LENS_TYPE_ORDER,
          topN,
          temperature,
          numCompletionTokens,
          steerTokens: [{ token: config.token, type: config.type }],
          steerLayers: config.layers,
          steerStrength: config.strength,
          steerAblate: config.ablate,
          swapToken:
            config.mode === 'swap' && config.swapToken.trim()
              ? { token: config.swapToken, type: config.type }
              : undefined,
          steerGeneratedTokens: config.steerGenerated,
          filterNonWordTokens: hideNonWordTokens,
          signal: controller.signal,
          onMeta: (m) => setSteerMeta(m),
          onPromptTokens: (p) => {
            setSteerTokens(
              p.tokens.map((t) => ({
                kind: 'token' as const,
                position: t.position,
                token: t.token,
                id: t.id,
                is_generated: t.is_generated,
                results: [],
              })),
            );
          },
          onToken: (t) => {
            setSteerTokens((prev) => {
              const next = prev.slice();
              next[t.position] = t;
              return next;
            });
          },
        });
      } catch (err) {
        if (!(err instanceof DOMException && err.name === 'AbortError')) {
          setError(err instanceof Error ? err.message : String(err));
        }
      } finally {
        if (steerAbortRef.current === controller) {
          steerAbortRef.current = null;
        }
      }
    },
    [prompt, modelId, topN, temperature, numCompletionTokens, hideNonWordTokens],
  );

  // Re-run the existing run's read-outs with a different non-word filter, WITHOUT
  // re-tokenizing or re-generating: we replay the exact token ids
  // (`inputTokenIds`) with generation disabled, so only the per-layer top-n
  // changes. Generated-token styling is preserved from the prior run (the replay
  // reports every position as a prompt token). The steered run, if any, is
  // replayed the same way.
  const rerunReadouts = useCallback(
    async (filter: boolean) => {
      const priorTokens = tokens;
      const ids = priorTokens.map((t) => t.id);
      if (ids.length === 0 || ids.some((id) => typeof id !== 'number')) {
        return;
      }
      abortRef.current?.abort();
      setError(null);
      const priorMeta = meta;
      // Re-analysis of the EXISTING run: keep the current tokens on screen and
      // update chips in place as the replay streams (no clear, no loading
      // placeholder), preserving prior results until the re-filtered ones arrive.
      setStreaming(true);
      const controller = new AbortController();
      abortRef.current = controller;
      try {
        await runLensStream({
          modelId,
          inputTokenIds: ids as number[],
          type: LENS_TYPE_ORDER,
          topN,
          temperature,
          numCompletionTokens: 0,
          filterNonWordTokens: filter,
          signal: controller.signal,
          onMeta: (m) => setMeta(m),
          onPromptTokens: (p) => {
            setTokens(
              p.tokens.map((t) => ({
                kind: 'token' as const,
                position: t.position,
                token: t.token,
                id: t.id,
                is_generated: priorTokens[t.position]?.is_generated ?? t.is_generated,
                results: priorTokens[t.position]?.results ?? [],
              })),
            );
          },
          onToken: (t) => {
            setTokens((prev) => {
              const next = prev.slice();
              next[t.position] = { ...t, is_generated: priorTokens[t.position]?.is_generated ?? t.is_generated };
              return next;
            });
          },
        });
      } catch (err) {
        if (!(err instanceof DOMException && err.name === 'AbortError')) {
          setTokens(priorTokens);
          setMeta(priorMeta);
          setError(err instanceof Error ? err.message : String(err));
        }
      } finally {
        setStreaming(false);
        abortRef.current = null;
      }

      // Replay the steered run too (if present) so its read-outs match the filter.
      const priorSteer = steerTokens;
      const steerIds = priorSteer.map((t) => t.id);
      const steerConfig = analysis.steer;
      if (steerConfig && steerIds.length > 0 && steerIds.every((id) => typeof id === 'number')) {
        steerAbortRef.current?.abort();
        const steerController = new AbortController();
        steerAbortRef.current = steerController;
        try {
          await runLensStream({
            modelId,
            inputTokenIds: steerIds as number[],
            type: LENS_TYPE_ORDER,
            topN,
            temperature,
            numCompletionTokens: 0,
            steerTokens: [{ token: steerConfig.token, type: steerConfig.type }],
            steerLayers: steerConfig.layers,
            steerStrength: steerConfig.strength,
            steerAblate: steerConfig.ablate,
            swapToken:
              steerConfig.mode === 'swap' && steerConfig.swapToken.trim()
                ? { token: steerConfig.swapToken, type: steerConfig.type }
                : undefined,
            steerGeneratedTokens: steerConfig.steerGenerated,
            filterNonWordTokens: filter,
            signal: steerController.signal,
            onMeta: (m) => setSteerMeta(m),
            onPromptTokens: (p) => {
              setSteerTokens(
                p.tokens.map((t) => ({
                  kind: 'token' as const,
                  position: t.position,
                  token: t.token,
                  id: t.id,
                  is_generated: priorSteer[t.position]?.is_generated ?? t.is_generated,
                  results: [],
                })),
              );
            },
            onToken: (t) => {
              setSteerTokens((prev) => {
                const next = prev.slice();
                next[t.position] = { ...t, is_generated: priorSteer[t.position]?.is_generated ?? t.is_generated };
                return next;
              });
            },
          });
        } catch (err) {
          if (!(err instanceof DOMException && err.name === 'AbortError')) {
            setError(err instanceof Error ? err.message : String(err));
          }
        } finally {
          if (steerAbortRef.current === steerController) {
            steerAbortRef.current = null;
          }
        }
      }
    },
    [tokens, meta, modelId, topN, temperature, steerTokens, analysis.steer],
  );

  // Toggle the non-word filter: update the flag AND re-run the read-outs (the
  // filter is applied server-side now, so we must recompute). With no run yet,
  // just store the flag so the next run uses it.
  const handleSetHideNonWord = useCallback(
    (next: boolean) => {
      // Toggling re-runs the read-outs server-side; without a live host, keep
      // the current results and surface the notice instead.
      if (!inferenceAvailable) {
        notifyUnavailable();
        return;
      }
      setHideNonWordTokens(next);
      if (tokens.length > 0) {
        void rerunReadouts(next);
      }
    },
    [inferenceAvailable, notifyUnavailable, setHideNonWordTokens, tokens.length, rerunReadouts],
  );

  const { setSelected: setSteerSelected } = steerAnalysis;
  const { registerSteerRunner: registerPrimarySteerRunner, registerSteerStopper: registerPrimarySteerStopper } =
    analysis;
  useEffect(() => {
    registerPrimarySteerRunner(runLensSteer);
    return () => registerPrimarySteerRunner(null);
  }, [registerPrimarySteerRunner, runLensSteer]);
  useEffect(() => {
    const stop = () => steerAbortRef.current?.abort();
    registerPrimarySteerStopper(stop);
    return () => registerPrimarySteerStopper(null);
  }, [registerPrimarySteerStopper]);
  // Highlight the steered readout in the steered transcript once results arrive.
  // The highlighted readout is the snapshot taken when the run started (see
  // `steerHighlightRef`), so editing the swap token later doesn't move it; it
  // only changes on a new steered run (when `steerTokens` changes).
  useEffect(() => {
    const hl = steerHighlightRef.current;
    if (hl && steerTokens.length > 0) {
      setSteerSelected([hl]);
    }
  }, [steerTokens, setSteerSelected]);

  // Keep the steered transcript's highlight layer range in sync with the main
  // sidebar's selected layer range (the chat/completion on the right), rather
  // than the steered analysis's own default range.
  useEffect(() => {
    steerAnalysis.setLayerRange(effectiveRange);
  }, [effectiveRange, steerMeta, steerAnalysis.setLayerRange]);

  // The steered-output transcript, rendered inline beneath the default output.
  const renderSteerResults = useCallback(
    () => (
      <JlensProviders analysis={steerAnalysis} steering>
        {steerTokens.length > 0 ? (
          <div className="group relative w-full rounded-lg bg-white px-3 py-2 text-slate-800 sm:rounded-xl sm:px-4 sm:py-5">
            <CompletionCopyButton text={steerTokens.map((t) => t.token).join('')} />
            <CompletionTokens
              tokens={steerTokens}
              layersByType={steerAnalysis.layersByType}
              bandsByPosition={steerAnalysis.bandsByPosition}
              layerRange={steerAnalysis.effectiveRange}
              onTokenHover={steerAnalysis.handleTokenHover}
              selectedPositions={steerAnalysis.selectedPositions}
              highlightedPosition={steerAnalysis.highlightedPosition}
            />
          </div>
        ) : (
          <div className="flex items-center justify-center gap-x-2 px-1 text-center text-slate-400">
            {analysis.steerStreaming ? (
              <>
                <LoadingSquare size={16} />
                <span className="text-[13px]">{analysis.steer?.mode === 'swap' ? 'Swapping…' : 'Steering…'}</span>
              </>
            ) : (
              <span className="my-10 mb-12 text-[10px] sm:text-[13px]">
                Press {analysis.steer?.mode === 'swap' ? 'Swap' : 'Steer'} to generate the steered completion.
              </span>
            )}
          </div>
        )}
      </JlensProviders>
    ),
    [steerAnalysis, steerTokens, analysis.steerStreaming, analysis.steer?.mode],
  );

  function handleStop() {
    abortRef.current?.abort();
  }

  function handleClear() {
    if (streaming) {
      return;
    }
    // eslint-disable-next-line no-alert
    if (tokens.length > 0 && !window.confirm('Clear this run?')) {
      return;
    }
    setTokens([]);
    setMeta(null);
    setError(null);
    setSelected([]);
    setSelectedPositions(new Set());
    setLockedLayer(null);
    analysis.clearSidebarSearch();
    onClear?.();
  }

  // When a fixture/share is loaded, hydrate state from it so the run renders
  // without hitting the inference server.
  useEffect(() => {
    if (!loadedData) {
      return;
    }
    abortRef.current?.abort();
    steerAbortRef.current?.abort();
    setStreaming(false);
    setGenerating(false);
    setAwaitingFirstResponse(false);
    setError(null);
    setPrompt(loadedData.prompt ?? '');
    setMeta(loadedData.meta ?? null);
    // Shares are re-run server-side with generation disabled, so their tokens
    // come back without `is_generated` set. Rebuild the prompt→generated
    // boundary from the persisted `numPromptTokens` (absent/null on older shares
    // and plain fixtures, in which case we keep whatever the blob carried).
    const numPromptTokens = loadedData.uiState?.numPromptTokens;
    const restoreGenerated = (t: LensTokenMessage, i: number): LensTokenMessage =>
      typeof numPromptTokens === 'number' ? { ...t, is_generated: i >= numPromptTokens } : t;
    setTokens((loadedData.tokens ?? []).map(restoreGenerated));
    // Restore a saved steered run (if any). The steered results are applied
    // immediately; entering steer mode is deferred to the effect below so the
    // per-layer counts compute against the just-set main tokens.
    if (loadedData.steer) {
      const c = loadedData.steer.config;
      steerHighlightRef.current = {
        key: c.mode === 'swap' && c.swapToken?.trim() ? c.swapToken : c.token,
        type: c.type,
      };
      setSteerMeta(loadedData.steer.meta ?? null);
      setSteerTokens((loadedData.steer.tokens ?? []).map(restoreGenerated));
      pendingSteerRef.current = loadedData.steer;
    } else {
      steerHighlightRef.current = null;
      setSteerMeta(null);
      setSteerTokens([]);
      pendingSteerRef.current = null;
    }
    const ui = loadedData.uiState;
    if (ui) {
      setSelected(ui.lockedTokens.map((t) => ({ key: t.key, type: t.type as LensType })));
      setSelectedPositions(new Set(ui.selectedPositions));
      setTopN(ui.topN);
      setHideNonWordTokens(ui.hideNonWordTokens);
      setTemperature(ui.temperature);
      setNumCompletionTokens(ui.numCompletionTokens);
      pendingScrollSelectionRef.current = ui.selectedPositions.length > 0 ? ui.selectedPositions : null;
      if (
        ui.activeLensModeTab === LensMode.JACOBIAN_LENS ||
        ui.activeLensModeTab === LensMode.LOGIT_LENS ||
        ui.activeLensModeTab === LensMode.DIFF
      ) {
        setLensMode(ui.activeLensModeTab);
      }
    } else {
      setSelectedPositions(new Set());
      pendingScrollSelectionRef.current = null;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadedData, setLensMode]);

  // Once a shared run's tokens have rendered, bring its restored selection into
  // view by scrolling ONLY the transcript container (never the page). Re-runs
  // while the scroll is pending (e.g. after a late-restored steered column) so
  // the selection stays visible.
  useEffect(() => {
    if (!pendingScrollSelectionRef.current || tokens.length === 0) {
      return undefined;
    }
    const el = scrollRef.current;
    if (!el) {
      return undefined;
    }
    const positions = pendingScrollSelectionRef.current;
    const raf = requestAnimationFrame(() => scrollContainerToTokenPositions(el, positions));
    return () => cancelAnimationFrame(raf);
  }, [tokens, steerTokens]);

  // Enter steer mode for a restored share once the main tokens are in place.
  useEffect(() => {
    const pending = pendingSteerRef.current;
    if (pending && tokens.length > 0) {
      pendingSteerRef.current = null;
      analysis.restoreSteer(pending.config);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tokens]);

  function handleExport() {
    const data: JlensExportCompletion = {
      version: 1,
      kind: 'completion',
      modelId,
      exportedAt: new Date().toISOString(),
      prompt,
      meta,
      tokens,
      steer:
        analysis.steer && steerTokens.length > 0
          ? { config: analysis.steer, meta: steerMeta, tokens: steerTokens }
          : undefined,
    };
    downloadJson(data, defaultExportFilename('completion', modelId));
  }

  const canShare = tokens.length > 0 && tokens.every((t) => typeof t.id === 'number');
  const hasRun = tokens.length > 0;

  const defaultOutput = (
    <div className="group relative w-full rounded-lg bg-white px-3 py-2 text-slate-800 sm:rounded-xl sm:px-4 sm:py-5">
      {tokens.length > 0 && <CompletionCopyButton text={tokens.map((t) => t.token).join('')} />}
      <CompletionTokens
        tokens={tokens}
        layersByType={layersByType}
        bandsByPosition={bandsByPosition}
        layerRange={effectiveRange}
        onTokenHover={handleTokenHover}
        selectedPositions={selectedPositions}
        highlightedPosition={highlightedPosition}
      />
      {generating && (
        <div className="flex items-center pt-1 text-slate-400">
          <LoadingSquare size={14} />
        </div>
      )}
    </div>
  );

  const inputArea = (
    <>
      <div className="relative mb-0 flex flex-row items-end gap-x-2 rounded-xl border border-sky-100 bg-white px-0 py-2 shadow-md sm:mb-5">
        <ReactTextareaAutosize
          value={prompt}
          onChange={(e) => setPrompt(e.target.value.slice(0, MAX_LENS_COMPLETION_PROMPT_CHARS))}
          maxLength={MAX_LENS_COMPLETION_PROMPT_CHARS}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              if (!inferenceAvailable) {
                notifyUnavailable();
                return;
              }
              void runLens(prompt);
            }
          }}
          minRows={4}
          maxRows={8}
          disabled={streaming || steering}
          placeholder={steering ? 'Exit steering to edit the prompt…' : 'Enter a prompt to run the lens on…'}
          className="flex-1 resize-none border-0 px-4 py-1 pb-3 font-mono text-[11px] leading-normal text-slate-800 outline-none placeholder:text-slate-400 focus:ring-0 disabled:cursor-not-allowed disabled:opacity-40 sm:pb-10 sm:text-[13px]"
        />
        <div className="absolute bottom-2 right-2 flex flex-row items-center gap-x-1.5">
          <button
            type="button"
            onClick={() => setShareOpen(true)}
            disabled={!canShare || streaming}
            title="Share"
            aria-label="Share"
            className="flex h-8 w-8 items-center justify-center rounded-md border border-slate-200 bg-white text-slate-400 transition-colors hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Share2 className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={handleExport}
            disabled={tokens.length === 0}
            title="Export this run to JSON"
            aria-label="Export this run to JSON"
            className="hidden h-8 w-8 items-center justify-center rounded-md border border-slate-200 bg-white text-slate-400 transition-colors hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-40 sm:flex"
          >
            <Download className="h-4 w-4" />
          </button>
          {hasRun && (
            <button
              type="button"
              onClick={handleClear}
              disabled={streaming || steering}
              title="Clear run"
              aria-label="Clear run"
              className="flex h-8 w-8 items-center justify-center rounded-md border border-slate-200 bg-white text-slate-400 transition-colors hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-40"
            >
              <Trash2 className="h-4 w-4" />
            </button>
          )}
          <button
            type="button"
            onClick={() => setShowAdvanced((s) => !s)}
            title="Advanced settings"
            aria-label="Advanced settings"
            className={`flex h-8 w-8 items-center justify-center rounded-md border transition-colors ${
              showAdvanced
                ? 'border-sky-400 bg-sky-50 text-sky-600'
                : 'border-slate-200 bg-white text-slate-400 hover:bg-slate-100'
            }`}
          >
            <Settings className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={() => {
              if (!inferenceAvailable) {
                notifyUnavailable();
                return;
              }
              if (streaming) {
                handleStop();
              } else {
                void runLens(prompt);
              }
            }}
            disabled={steering || (inferenceAvailable && !streaming && !prompt.trim())}
            className="flex h-8 w-8 items-center justify-center rounded-full transition-colors disabled:cursor-not-allowed disabled:opacity-40"
          >
            {streaming ? (
              <X className="h-8 w-8 rounded-lg bg-red-400 p-1.5 text-white hover:bg-red-500" />
            ) : (
              <ArrowUp className="h-8 w-8 rounded-lg bg-sky-700 p-1.5 text-white hover:bg-sky-800" />
            )}
          </button>
        </div>
        {limitRemaining !== null && (
          <div className="absolute -bottom-4 right-1 hidden text-[9px] leading-none text-slate-400 sm:block">
            {limitRemaining > 0 ? `Hourly Limit Left: ${limitRemaining}` : 'Limit reached. Try again later.'}
          </div>
        )}
      </div>

      {showAdvanced && (
        <JlensAdvanced
          isBusy={streaming}
          temperature={temperature}
          setTemperature={setTemperature}
          numCompletionTokens={numCompletionTokens}
          setNumCompletionTokens={setNumCompletionTokens}
          maxCompletionTokens={MAX_LENS_COMPLETION_TOKENS_COMPLETION}
          topN={topN}
          setTopN={setTopN}
          hideNonWordTokens={hideNonWordTokens}
          setHideNonWordTokens={handleSetHideNonWord}
        />
      )}
    </>
  );

  return (
    <JlensProviders analysis={analysis}>
      <div className="relative flex h-full min-h-full flex-1 flex-col rounded-xl bg-slate-50 sm:flex-row sm:gap-x-3 sm:gap-y-0 sm:gap-y-3">
        <Dialog open={analysis.maxSelectError} onOpenChange={analysis.setMaxSelectError}>
          <DialogContent className="max-w-sm bg-white">
            <DialogHeader>
              <DialogTitle>Selection limit reached</DialogTitle>
              <DialogDescription>
                You can only select {MAX_SELECT} tokens to analyze at a time. Please deselect one first.
              </DialogDescription>
            </DialogHeader>
            <button
              type="button"
              onClick={() => analysis.setMaxSelectError(false)}
              className="w-full rounded-lg bg-sky-600 px-4 py-2 text-[13px] font-semibold text-white transition-colors hover:bg-sky-700"
            >
              Okay
            </button>
          </DialogContent>
        </Dialog>

        <Dialog open={unavailableOpen} onOpenChange={setUnavailableOpen}>
          <DialogContent className="max-w-sm bg-white">
            <DialogHeader>
              <DialogTitle>Inference unavailable</DialogTitle>
              <DialogDescription>This model is no longer available for inference.</DialogDescription>
            </DialogHeader>
            <button
              type="button"
              onClick={() => setUnavailableOpen(false)}
              className="w-full rounded-lg bg-sky-600 px-4 py-2 text-[13px] font-semibold text-white transition-colors hover:bg-sky-700"
            >
              Okay
            </button>
          </DialogContent>
        </Dialog>

        <JlensShareDialog
          open={shareOpen}
          onOpenChange={setShareOpen}
          userName={user?.name ?? null}
          buildBody={(description) => ({
            modelId,
            kind: 'completion',
            inputTokenIds: tokens.map((t) => t.id),
            prompt,
            topN,
            temperature,
            numCompletionTokens,
            numPromptTokens: (() => {
              const firstGen = tokens.findIndex((t) => t.is_generated);
              return firstGen === -1 ? tokens.length : firstGen;
            })(),
            activeLensModeTab: lensMode,
            hideNonWordTokens,
            lockedTokens: selected.map((s) => ({ key: s.key, type: s.type })),
            selectedPositions: Array.from(selectedPositions),
            description: description || undefined,
            steer: buildSteerShareBody(analysis.steer, steerTokens),
          })}
        />

        {/* Mobile only: commentary above the run, just below the top bar. On
            desktop it renders in the analysis panel instead. */}
        {showCommentary && sharedDescription && (
          <div className="w-full px-0 sm:hidden">
            <JlensCommentary
              description={sharedDescription}
              attribution={sharedAttribution}
              onDismiss={dismissCommentary}
              onNext={onNextDemo}
              nextLabel={nextDemoLabel}
              onFreeChat={onFreeChatDemo}
            />
          </div>
        )}

        <div className="flex min-h-0 w-full max-w-screen-lg grow-[2] basis-0 flex-col gap-y-0 bg-slate-200/40 px-1 py-0 sm:flex-1 sm:rounded-2xl sm:px-3 sm:py-3">
          <div className="relative flex min-h-0 flex-1 flex-col overflow-hidden px-0 pb-4 pt-0">
            {hasRun || awaitingFirstResponse ? (
              <>
                {/* Desktop only: commentary above the action bar. On mobile it
                    renders above the run instead (see above). */}
                {showCommentary && sharedDescription && (
                  <div className="hidden w-full px-0 sm:flex sm:pb-3">
                    <JlensCommentary
                      description={sharedDescription}
                      attribution={sharedAttribution}
                      onDismiss={dismissCommentary}
                      onNext={onNextDemo}
                      nextLabel={nextDemoLabel}
                      onFreeChat={onFreeChatDemo}
                      className="sm:rounded-xl"
                    />
                  </div>
                )}
                {!steering && <div className="relative flex flex-col gap-y-2 px-0 pt-1">{inputArea}</div>}

                {error && (
                  <div className="mt-2 px-4">
                    <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-[11px] text-red-700">
                      {error}
                    </div>
                  </div>
                )}

                <div
                  ref={scrollRef}
                  onPointerDown={steering ? undefined : onChatPointerDown}
                  className={`mt-2 min-h-0 flex-1 overflow-y-auto ${dragging ? 'select-none' : ''}`}
                >
                  {/* Inner row is `min-h-full` so it grows to the taller of the
                      viewport and the scrolled content — otherwise the steered
                      column's divider (align-stretch) would only span the first
                      visible page and stop mid-transcript when scrolled. */}
                  <div className="flex min-h-full flex-row gap-x-0 gap-y-2.5">
                    {hasRun ? (
                      <>
                        {steering ? (
                          <div className="sm:min-w-auto mt-1 flex flex-1 flex-col gap-y-2.5">
                            <DefaultOutputHeader />
                            {defaultOutput}
                          </div>
                        ) : (
                          <div className="flex min-h-0 flex-1 flex-col gap-y-2.5">{defaultOutput}</div>
                        )}
                        {analysis.steer && (
                          <div className="sm:min-w-auto ml-4 mt-1 flex flex-1 flex-col gap-y-2.5 border-l border-slate-300 pl-4">
                            <SteerOutputHeader steer={analysis.steer} />
                            {renderSteerResults()}
                          </div>
                        )}
                      </>
                    ) : (
                      <div className="flex max-h-[64px] w-full items-center gap-x-2 self-start rounded-xl bg-white px-4 py-3 text-slate-400 shadow">
                        <LoadingSquare size={16} />
                        <span className="text-[13px]">Generating…</span>
                      </div>
                    )}
                  </div>
                </div>
              </>
            ) : (
              <div className="flex flex-1 flex-col items-center justify-center gap-y-3 px-1">
                <div className="mb-1 flex flex-col items-center gap-y-1 text-center text-slate-500">
                  <span className="text-xl font-normal text-slate-700">
                    Jacobian Lens for{' '}
                    <span className="font-bold">
                      {modelId
                        .split('-')
                        .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
                        .join(' ')}
                    </span>
                    <div className="pt-1 text-center text-[13px] leading-snug text-slate-500 sm:text-base sm:leading-normal">
                      Enter a prompt to see what's in the model's <strong>J-Space</strong>.
                    </div>
                  </span>
                </div>
                {error && (
                  <div className="w-full rounded-md border border-red-200 bg-red-50 px-3 py-2 text-[11px] text-red-700">
                    {error}
                  </div>
                )}
                <div className="flex w-full flex-col gap-y-2 px-4">{inputArea}</div>
              </div>
            )}
          </div>
        </div>

        <JlensAnalysisPanel
          analysis={analysis}
          tokens={tokens}
          numCompletionTokens={numCompletionTokens}
          setNumCompletionTokens={setNumCompletionTokens}
          maxCompletionTokens={MAX_LENS_COMPLETION_TOKENS_COMPLETION}
          onShare={() => setShareOpen(true)}
          canShare={canShare}
          shareDisabled={streaming}
          shareLabel="Share"
          onExport={handleExport}
          exportDisabled={tokens.length === 0}
          exportLabel="Export this run to JSON"
        />
      </div>
    </JlensProviders>
  );
}

// The flat token transcript for a completion run: the prompt tokens followed by
// the generated continuation, every token a hoverable + selectable lens chip.
// Always monospaced.
function CompletionTokens({
  tokens,
  layersByType,
  bandsByPosition,
  layerRange,
  onTokenHover,
  selectedPositions,
  highlightedPosition,
}: {
  tokens: LensTokenMessage[];
  layersByType: Record<string, number[]>;
  bandsByPosition: Map<number, TokenBand[]>;
  layerRange: LayerRange | null;
  onTokenHover: (token: LensTokenMessage, open: boolean) => void;
  selectedPositions: Set<number>;
  highlightedPosition: number | null;
}) {
  // Index of the first generated token, so we can mark the prompt→generated
  // boundary with a vertical bar. Only meaningful when there's a prompt before
  // it (> 0); -1 means nothing was generated.
  const firstGeneratedIdx = tokens.findIndex((t) => t.is_generated);
  return (
    <div className="select-text whitespace-pre-wrap break-words font-mono text-[13px] leading-relaxed">
      {tokens.map((token, tokenIdx) => {
        const newlineCount = (token.token.match(/\n/g) || []).length;
        const prevToken = tokenIdx > 0 ? tokens[tokenIdx - 1].token : undefined;
        const prevEndsWithLineBreak = tokenIdx === 0 || (prevToken !== undefined && /\n/.test(prevToken));
        const nextStartsNewLine = tokenIdx === tokens.length - 1;
        return (
          <span key={token.position}>
            {tokenIdx === firstGeneratedIdx && firstGeneratedIdx > 0 && (
              // Zero-width inline anchor so the (absolutely positioned) bar can
              // extend beyond the line height without growing the line box.
              <span
                className="relative mt-0 inline-block w-0 bg-slate-600 align-middle sm:-mt-[7px]"
                aria-hidden="true"
              >
                <span
                  className="absolute left-0 top-1/2 h-[1.6em] w-[2px] -translate-y-1/2 border-l-2 border-dotted border-slate-400 sm:h-[2.2em]"
                  aria-hidden="true"
                />
              </span>
            )}
            <JlensTokenChip
              token={token}
              layersByType={layersByType}
              variant={token.is_generated ? 'generated' : 'content'}
              bands={bandsByPosition.get(token.position)}
              layerRange={layerRange}
              onHoverChange={onTokenHover}
              positionSelected={highlightedPosition == null && selectedPositions.has(token.position)}
              prevSelected={highlightedPosition == null && selectedPositions.has(token.position - 1)}
              nextSelected={highlightedPosition == null && selectedPositions.has(token.position + 1)}
              highlighted={highlightedPosition === token.position}
              prevEndsWithLineBreak={prevEndsWithLineBreak}
              nextStartsNewLine={nextStartsNewLine}
            />
            {Array.from({ length: newlineCount }).map((_, i) => (
              <div key={i} className="h-1 max-h-1 leading-[0em]" />
            ))}
          </span>
        );
      })}
    </div>
  );
}

// Copy-to-clipboard button pinned to the top-right of a completion output box.
// Manages its own transient "copied" checkmark state.
function CompletionCopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    if (!copied) {
      return undefined;
    }
    const t = setTimeout(() => setCopied(false), 1500);
    return () => clearTimeout(t);
  }, [copied]);
  const handleCopy = async () => {
    try {
      if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
      }
      setCopied(true);
    } catch {
      // ignore — clipboard access may be denied in some contexts
    }
  };
  return (
    <div className="absolute right-0 top-full mt-1 hidden h-5 items-center gap-x-1 px-1 opacity-0 transition-opacity duration-150 group-hover:opacity-100 sm:flex">
      <button
        type="button"
        onClick={handleCopy}
        disabled={!text}
        title="Copy output"
        aria-label="Copy output"
        className="flex h-5 w-5 items-center justify-center rounded text-slate-400 transition-colors hover:bg-slate-200 hover:text-slate-600 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-transparent disabled:hover:text-slate-400"
      >
        {copied ? <Check className="h-3 w-3 text-sky-600" /> : <Copy className="h-3 w-3" />}
      </button>
    </div>
  );
}
