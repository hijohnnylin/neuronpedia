'use client';

import {
  JLENS_CHAT_ID,
  JLENS_STEER_COLUMNS_ID,
  JLENS_STEER_OUTPUT_ID,
  JLENS_STEER_PANEL_ID,
} from '@/app/[modelId]/jlens/jlens-tour-constants';
import { useJlensTourStep } from '@/app/[modelId]/jlens/jlens-tour-context';
import { useGlobalContext } from '@/components/provider/global-provider';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/shadcn/dialog';
import { LoadingSquare } from '@/components/svg/loading-square';
import {
  DEFAULT_LENS_COMPLETION_TOKENS,
  DEFAULT_LENS_TEMPERATURE,
  DEFAULT_LENS_TOP_N,
  LENS_TYPE_ORDER,
  LensMetaMessage,
  LensMode,
  LensTokenMessage,
  LensType,
  MAX_LENS_CHAT_PREFILL_CHARS,
  MAX_LENS_CHAT_USER_CHARS,
} from '@/lib/utils/lens';
import { ArrowUp, Check, Copy, Download, Pencil, Settings, Share2, Trash2, X } from 'lucide-react';
import { useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import ReactTextareaAutosize from 'react-textarea-autosize';
import JlensAdvanced from './jlens-advanced';
import { MAX_SELECT } from './jlens-analysis';
import { JlensAnalysisPanel, JlensProviders } from './jlens-analysis-panel';
import {
  detectChatFormat,
  groupTokensIntoMessages,
  JlensChatFormat,
  JlensTokenGroup,
  toChatPayload,
} from './jlens-chat-format';
import { JlensCommentary, useSharedCommentary } from './jlens-commentary';
import {
  buildSteerShareBody,
  defaultExportFilename,
  downloadJson,
  JlensExportChat,
  JlensExportSteer,
  parseFixture,
} from './jlens-export';
import { LensModeSetContext } from './jlens-lens-mode';
import { JlensShareDialog } from './jlens-share-dialog';
import { DefaultOutputHeader, SteerOutputHeader } from './jlens-steer-panel';
import { runLensStream as baseRunLensStream, RunLensStreamParams } from './jlens-stream';
import JlensTokenChip, { scrollContainerToTokenPositions, TokenBand } from './jlens-token';
import { LayerRange } from './jlens-token-popup';
import { SteerConfig, useJlensAnalysis } from './use-jlens-analysis';

type ChatMessage = { role: 'user' | 'assistant'; content: string };

export default function JlensChat({
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
  loadedData?: JlensExportChat | null;
  // When viewing a shared link: the sharer's description (shown above the chat)
  // and an optional attribution line below it (null for the default creator).
  sharedDescription?: string | null;
  sharedAttribution?: string | null;
  // Whether an inference host currently serves this model. When false, cached
  // results still render but live actions (send / steer / swap / filter toggle)
  // are gated behind a "model unavailable" notice.
  inferenceAvailable?: boolean;
  // Called when the conversation is cleared (trash icon). Used by the shared
  // view to drop the shareId from the URL and reset to the live model page.
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
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [typedText, setTypedText] = useState('');
  // Optional assistant prefill: when non-empty, it's sent as a trailing
  // assistant turn so the model continues from it instead of starting fresh.
  const [prefillText, setPrefillText] = useState('');
  const [streaming, setStreaming] = useState(false);
  // True for the duration of a generation stream (sendMessage). Drives the
  // loading square shown below the in-progress assistant response. Distinct from
  // `streaming`, which is also set during non-generating re-analysis/replays.
  const [generating, setGenerating] = useState(false);
  // True between hitting submit and the server's first response (the
  // chat-formatted prompt tokens). While true we show a loading placeholder.
  const [awaitingFirstResponse, setAwaitingFirstResponse] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  // Remaining requests in the current hourly window for `/api/lens/prompt`,
  // surfaced next to the send button (see `x-limit-remaining` in middleware).
  // `null` until the first response. Every jlens action (send, re-analyze,
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

  const [shareOpen, setShareOpen] = useState(false);
  // Shown when a live action is attempted but no inference host serves the model.
  const [unavailableOpen, setUnavailableOpen] = useState(false);
  const notifyUnavailable = useCallback(() => setUnavailableOpen(true), []);

  // Inline-edit state for assistant messages. `editingIdx` is the `messages`
  // index of the bubble being edited (`null` when not editing); `editingText`
  // is the working copy of just the assistant `content` (chat-template special
  // tokens are re-applied by the server on re-analysis, so the user can't
  // corrupt the templating).
  const [editingIdx, setEditingIdx] = useState<number | null>(null);
  const [editingText, setEditingText] = useState('');
  const isEditing = editingIdx !== null;
  // True between committing an assistant edit and the re-analysis stream's first
  // response. While true the edited assistant bubble is replaced by a loading
  // placeholder (the surviving earlier turns stay visible).
  const [awaitingReanalyze, setAwaitingReanalyze] = useState(false);
  // Which message is currently flashing the "copied" checkmark; cleared on a
  // short timer so the icon flips back to the copy glyph.
  const [copiedMessageIdx, setCopiedMessageIdx] = useState<number | null>(null);
  // Same as above, but for the separate steered-output transcript (its bubble
  // indices are independent of the main conversation's).
  const [copiedSteerIdx, setCopiedSteerIdx] = useState<number | null>(null);

  const [temperature, setTemperature] = useState(DEFAULT_LENS_TEMPERATURE);
  const [numCompletionTokens, setNumCompletionTokens] = useState(DEFAULT_LENS_COMPLETION_TOKENS);
  const [topN, setTopN] = useState(DEFAULT_LENS_TOP_N);

  const [meta, setMeta] = useState<LensMetaMessage | null>(null);
  const [tokens, setTokens] = useState<LensTokenMessage[]>([]);
  // Live assistant text accumulated from generated tokens during a stream.
  const [liveAssistantText, setLiveAssistantText] = useState('');

  // Steered run: a SEPARATE token stream + meta (the original conversation above
  // is left untouched) rendered in the steer panel.
  const [steerTokens, setSteerTokens] = useState<LensTokenMessage[]>([]);
  const [steerMeta, setSteerMeta] = useState<LensMetaMessage | null>(null);

  const { user } = useGlobalContext();
  const setLensMode = useContext(LensModeSetContext);

  // The shared-commentary banner state, lifted here so the mobile banner (above
  // the chat) and the desktop banner (in the analysis panel) share one dismiss.
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

  // While the guided tour spotlights the steer panel, the swap runs against a
  // precached exported result (see `runLensSteer`) instead of live inference.
  // Kept in a ref so the steer runner's identity stays stable (it's registered
  // via effect) while still reading the latest value.
  const tourStep = useJlensTourStep();
  const isSteerPanelTourStep = typeof tourStep?.element === 'string' && tourStep.element === `#${JLENS_STEER_PANEL_ID}`;
  const isSteerPanelTourStepRef = useRef(isSteerPanelTourStep);
  useEffect(() => {
    isSteerPanelTourStepRef.current = isSteerPanelTourStep;
  }, [isSteerPanelTourStep]);

  const fmt = useMemo(() => detectChatFormat(modelId), [modelId]);

  // Auto-scroll to bottom as content arrives (including the steered output,
  // which now renders inline below the default conversation). Skipped while a
  // shared run's restored selection is still waiting to be scrolled into view.
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
  }, [
    tokens,
    liveAssistantText,
    awaitingFirstResponse,
    awaitingReanalyze,
    steerTokens,
    analysis.steer,
    analysis.steerStreaming,
  ]);

  // Clear the "copied" confirmation after a short delay.
  useEffect(() => {
    if (copiedMessageIdx === null) {
      return undefined;
    }
    const t = setTimeout(() => setCopiedMessageIdx(null), 1500);
    return () => clearTimeout(t);
  }, [copiedMessageIdx]);

  useEffect(() => {
    if (copiedSteerIdx === null) {
      return undefined;
    }
    const t = setTimeout(() => setCopiedSteerIdx(null), 1500);
    return () => clearTimeout(t);
  }, [copiedSteerIdx]);

  // Group the token stream into bubbles so every token is hoverable.
  const groupedMessages = useMemo<JlensTokenGroup[] | null>(() => {
    if (tokens.length === 0) {
      return null;
    }
    const { messages: groups, hasChatFormat } = groupTokensIntoMessages(tokens, fmt);
    return hasChatFormat ? groups : null;
  }, [tokens, fmt]);

  const sendMessage = useCallback(async () => {
    if (!typedText.trim() || streaming || isEditing) {
      return;
    }
    setError(null);
    // A fresh run supersedes any share-load selection scroll; re-enable the
    // auto-scroll-to-bottom.
    pendingScrollSelectionRef.current = null;

    const userTurn: ChatMessage = { role: 'user', content: typedText };
    const nextMessages = [...messages, userTurn];
    const prefill = prefillText;
    const hasPrefill = prefill.trim().length > 0;
    const requestMessages = hasPrefill
      ? [...nextMessages, { role: 'assistant' as const, content: prefill }]
      : nextMessages;
    const priorTokens = tokens;
    const priorMeta = meta;

    const priorIds = priorTokens.map((t) => t.id);
    const cachedTokenIds = priorIds.every((id) => typeof id === 'number') ? (priorIds as number[]) : undefined;
    let reuseLen = 0;

    setMessages(nextMessages);
    setTypedText('');
    setPrefillText('');
    setStreaming(true);
    setGenerating(true);
    setAwaitingFirstResponse(true);
    setLiveAssistantText('');

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      let completion = '';
      await runLensStream({
        modelId,
        chat: toChatPayload(requestMessages),
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
          if (t.is_generated) {
            setLiveAssistantText((prev) => prev + t.token);
          }
        },
        onDone: (d) => {
          completion = d.completion;
        },
      });
      const raw = completion || liveAssistantText;
      // Harmony (and other custom formats) may need bespoke extraction of the
      // clean assistant text (e.g. gpt-oss returns analysis + final channels
      // wrapped in structural tokens); fall back to stripping the turn-end.
      const generated = fmt.parseCompletion ? fmt.parseCompletion(raw) : stripTurnEnd(raw, fmt.turnEndToken);
      const assistantText = hasPrefill ? prefill + generated : generated;
      setMessages([...nextMessages, { role: 'assistant', content: assistantText }]);
    } catch (err) {
      setTokens(priorTokens);
      setMeta(priorMeta);
      setMessages(messages);
      setTypedText(typedText);
      setPrefillText(prefill);
      if (!(err instanceof DOMException && err.name === 'AbortError')) {
        setError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      setStreaming(false);
      setGenerating(false);
      setAwaitingFirstResponse(false);
      setLiveAssistantText('');
      abortRef.current = null;
    }
  }, [
    typedText,
    prefillText,
    streaming,
    isEditing,
    messages,
    tokens,
    meta,
    modelId,
    topN,
    temperature,
    numCompletionTokens,
    hideNonWordTokens,
    liveAssistantText,
    fmt,
    analysis,
  ]);

  // Re-run the lens stream over a fixed conversation WITHOUT generating any new
  // tokens (`numCompletionTokens: 0`) — used after an assistant-message edit so
  // the chips/readouts reflect the edited conversation. The longest common
  // token-id prefix is reused so unchanged earlier turns keep their results
  // and only the edited (and any following) positions are recomputed.
  const reanalyzeConversation = useCallback(
    async (conversation: ChatMessage[], prefixTokens: LensTokenMessage[]) => {
      if (conversation.length === 0) {
        setMessages([]);
        setTokens([]);
        setMeta(null);
        return;
      }
      setError(null);
      const priorTokens = tokens;
      const priorMeta = meta;
      const priorMessages = messages;
      const priorIds = priorTokens.map((t) => t.id);
      const cachedTokenIds = priorIds.every((id) => typeof id === 'number') ? (priorIds as number[]) : undefined;
      let reuseLen = 0;

      setMessages(conversation);
      // Drop the edited (last) assistant turn from the displayed tokens so its
      // stale content disappears and a loading placeholder shows in its place;
      // the surviving earlier turns keep rendering from the prefix tokens.
      setTokens(prefixTokens);
      // Clear the chat-position selection (the red-border picks) since those
      // positions may now point at different tokens; the sidebar-locked tokens
      // (key-based) are intentionally kept.
      setSelectedPositions(new Set());
      // The selection is gone, so drop any pending share-load selection scroll.
      pendingScrollSelectionRef.current = null;
      setAwaitingReanalyze(true);
      setStreaming(true);

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        await runLensStream({
          modelId,
          chat: toChatPayload(conversation),
          type: LENS_TYPE_ORDER,
          topN,
          temperature,
          numCompletionTokens: 0,
          cachedTokenIds,
          filterNonWordTokens: hideNonWordTokens,
          signal: controller.signal,
          onMeta: (m) => {
            setMeta(m);
            reuseLen = m.reuse_len ?? 0;
          },
          onPromptTokens: (p) => {
            setAwaitingReanalyze(false);
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
            setAwaitingReanalyze(false);
            setTokens((prev) => {
              const next = prev.slice();
              next[t.position] = t;
              return next;
            });
          },
        });
      } catch (err) {
        setTokens(priorTokens);
        setMeta(priorMeta);
        setMessages(priorMessages);
        if (!(err instanceof DOMException && err.name === 'AbortError')) {
          setError(err instanceof Error ? err.message : String(err));
        }
      } finally {
        setStreaming(false);
        setAwaitingReanalyze(false);
        abortRef.current = null;
      }
    },
    [tokens, meta, messages, modelId, topN, temperature, hideNonWordTokens, setSelectedPositions],
  );

  // The position of the first token belonging to message `idx` (its turn-start),
  // used to slice the token stream so surviving turns keep their results when a
  // message is edited. Returns null if the grouping isn't available.
  const tokenCutoffForMessage = useCallback(
    (idx: number): number | null => {
      if (!groupedMessages || idx < 0 || idx >= groupedMessages.length) {
        return null;
      }
      const g = groupedMessages[idx];
      const first = g.headerTokens[0] ?? g.contentTokens[0] ?? g.footerTokens[0];
      return first ? first.position : null;
    },
    [groupedMessages],
  );

  const handleCopyMessage = useCallback(async (idx: number, content: string) => {
    try {
      if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(content);
      }
      setCopiedMessageIdx(idx);
    } catch {
      // ignore — clipboard access may be denied in some contexts
    }
  }, []);

  const handleCopySteerMessage = useCallback(async (idx: number, content: string) => {
    try {
      if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(content);
      }
      setCopiedSteerIdx(idx);
    } catch {
      // ignore — clipboard access may be denied in some contexts
    }
  }, []);

  // Edit a user message: drop it (and any later turns) and drop it back into the
  // composer for the user to revise and re-send. Surviving earlier turns keep
  // their tokens/results.
  const handleEditUserMessage = useCallback(
    (idx: number, content: string) => {
      if (streaming || steering || isEditing) {
        return;
      }
      const willClear = idx < messages.length - 1;
      // eslint-disable-next-line no-alert
      if (willClear && !window.confirm('Editing this message will clear all messages after it. Continue?')) {
        return;
      }
      const cutoff = tokenCutoffForMessage(idx);
      setMessages(messages.slice(0, idx));
      setTokens(cutoff == null ? [] : tokens.filter((t) => t.position < cutoff));
      setSelected([]);
      setSelectedPositions(new Set());
      setLockedLayer(null);
      setError(null);
      setPrefillText('');
      setTypedText(content);
    },
    [
      streaming,
      steering,
      isEditing,
      messages,
      tokens,
      tokenCutoffForMessage,
      setSelected,
      setSelectedPositions,
      setLockedLayer,
    ],
  );

  // Enter the inline editor for an assistant message.
  const handleStartAssistantEdit = useCallback(
    (idx: number, content: string) => {
      if (streaming || steering || isEditing) {
        return;
      }
      // Committing an assistant edit re-analyzes via inference; gate it when no
      // host is available so shared/cached runs stay viewable.
      if (!inferenceAvailable) {
        notifyUnavailable();
        return;
      }
      const willClear = idx < messages.length - 1;
      // eslint-disable-next-line no-alert
      if (willClear && !window.confirm('Editing this message will clear all messages after it. Continue?')) {
        return;
      }
      setEditingIdx(idx);
      setEditingText(content);
      setError(null);
    },
    [streaming, steering, isEditing, messages, inferenceAvailable, notifyUnavailable],
  );

  const handleCancelAssistantEdit = useCallback(() => {
    setEditingIdx(null);
    setEditingText('');
  }, []);

  // Commit an assistant edit: truncate from the edited message onward, re-append
  // it with the new content, then re-analyze the resulting conversation.
  const handleSaveAssistantEdit = useCallback(async () => {
    if (editingIdx === null) {
      return;
    }
    const idx = editingIdx;
    const conversation: ChatMessage[] = [...messages.slice(0, idx), { role: 'assistant', content: editingText }];
    // Tokens for the turns before the edited assistant, shown while the
    // re-analysis streams back so they don't flicker into placeholders.
    const cutoff = tokenCutoffForMessage(idx);
    const prefixTokens = cutoff == null ? [] : tokens.filter((t) => t.position < cutoff);
    setEditingIdx(null);
    setEditingText('');
    await reanalyzeConversation(conversation, prefixTokens);
  }, [editingIdx, editingText, messages, tokens, tokenCutoffForMessage, reanalyzeConversation]);

  // Re-run the last assistant turn with steering applied into the SEPARATE
  // steered-output state (or, with `null`, clear it). The conversation is kept
  // fixed: we re-send the messages up to and excluding the last assistant turn
  // so it regenerates under the steering. The original run is never touched.
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
      // A fresh steered run supersedes any share-load selection scroll.
      pendingScrollSelectionRef.current = null;

      const convo = messages.slice();
      // If the last assistant turn was generated from a user-supplied prefill,
      // preserve it: drop the assistant turn but re-send its prefill as a
      // trailing assistant message so the steered regeneration continues from
      // the same prefix (otherwise the steered output loses the prefill). The
      // prefill is the leading non-generated content of that assistant turn.
      let prefill = '';
      if (convo.length > 0 && convo[convo.length - 1].role === 'assistant') {
        convo.pop();
        const lastGroup = groupedMessages?.[groupedMessages.length - 1];
        if (lastGroup && lastGroup.role === 'assistant') {
          const firstGen = lastGroup.contentTokens.findIndex((t) => t.is_generated);
          if (firstGen > 0) {
            const prefixTokens = lastGroup.contentTokens.slice(0, firstGen);
            // Ignore the template's auto-injected <think></think> scaffold — only
            // a real user prefix should be re-sent as the steered prefill.
            const isRealPrefill = prefixTokens.some((t, i) => {
              const prev = i > 0 ? prefixTokens[i - 1].token : undefined;
              return t.token.trim() !== '' && !isThinkToken(t.token, prev);
            });
            if (isRealPrefill) {
              prefill = prefixTokens.map((t) => t.token).join('');
            }
          }
        }
      }
      if (convo.length === 0) {
        return;
      }
      const requestConvo: ChatMessage[] =
        prefill.trim().length > 0 ? [...convo, { role: 'assistant', content: prefill }] : convo;

      // Snapshot which readout to highlight (the swap target, or the steered
      // token) so later edits to the swap token don't change the highlight.
      steerHighlightRef.current = {
        key: config.mode === 'swap' && config.swapToken.trim() ? config.swapToken : config.token,
        type: config.type,
      };

      setError(null);
      setSteerTokens([]);
      setSteerMeta(null);

      // Guided-tour shortcut: on the steer-panel step the swap doesn't hit the
      // inference server. Instead we load a precached exported result from
      // `/public` (the same shape the "Export" feature produces) and render its
      // saved steered run, so the scripted spiders → ants swap is instant and
      // deterministic.
      if (isSteerPanelTourStepRef.current) {
        try {
          const res = await fetch('/qwen-output.json', { cache: 'force-cache' });
          if (!res.ok) {
            throw new Error(`Failed to load the precached swap result (${res.status}).`);
          }
          const parsed = parseFixture(await res.json());
          // Artificial delay so the swap feels like a real (streaming) run —
          // the steered column shows the LoadingSquare "Steering…" state (the
          // `runSteer` wrapper keeps `steerStreaming` true while we await here)
          // before the precached result pops in.
          await new Promise((resolve) => setTimeout(resolve, 2000));
          // Prefer the export's saved steered run; fall back to its main run if
          // the file itself already is the swapped output.
          const steerResult = parsed.steer;
          setSteerMeta(steerResult?.meta ?? parsed.meta ?? null);
          setSteerTokens(steerResult?.tokens ?? parsed.tokens ?? []);
        } catch (err) {
          setError(err instanceof Error ? err.message : String(err));
        }
        return;
      }

      const controller = new AbortController();
      steerAbortRef.current = controller;

      try {
        await runLensStream({
          modelId,
          chat: toChatPayload(requestConvo),
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
    [messages, groupedMessages, modelId, topN, temperature, numCompletionTokens, hideNonWordTokens],
  );

  // Re-run the existing run's read-outs with a different non-word filter, WITHOUT
  // re-tokenizing or re-generating: replay the exact token ids with generation
  // disabled, so only the per-layer top-n changes. Generated-token styling is
  // preserved from the prior run. The steered run, if any, is replayed too.
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
      // This is a re-analysis of the EXISTING conversation (not a new turn), so
      // we keep the current tokens/messages on screen and update chips in place
      // as the replay streams. We deliberately do NOT set `awaitingFirstResponse`
      // (that drives the pending USER-turn bubble, which here would wrongly show
      // the last assistant turn as a new user message).
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
            // Preserve prior per-token results so the chips keep their data
            // until the fresh (re-filtered) read-outs stream in (no flicker).
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
  // filter is applied server-side now). With no run yet, just store the flag.
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

  // Send a message, gating on inference availability (shared runs whose model is
  // no longer served show the notice instead of attempting a request).
  const attemptSend = useCallback(() => {
    if (!inferenceAvailable) {
      notifyUnavailable();
      return;
    }
    void sendMessage();
  }, [inferenceAvailable, notifyUnavailable, sendMessage]);

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
  // sidebar's selected layer range, rather than the steered analysis's own default.
  useEffect(() => {
    steerAnalysis.setLayerRange(effectiveRange);
  }, [effectiveRange, steerMeta, steerAnalysis.setLayerRange]);

  // The steered-output transcript (grouped chat bubbles), rendered inline below
  // the default conversation.
  const steerGroupedMessages = useMemo<JlensTokenGroup[] | null>(() => {
    if (steerTokens.length === 0) {
      return null;
    }
    const { messages: groups, hasChatFormat } = groupTokensIntoMessages(steerTokens, fmt);
    return hasChatFormat ? groups : null;
  }, [steerTokens, fmt]);

  const renderSteerResults = useCallback(
    () => (
      <JlensProviders analysis={steerAnalysis} steering>
        {steerTokens.length > 0 ? (
          <div className="flex flex-col gap-y-2.5 pb-5">
            {steerGroupedMessages
              ? steerGroupedMessages.map((group, idx) => {
                  const content = group.contentTokens.map((t) => t.token).join('');
                  return (
                    <GroupBubble
                      key={idx}
                      group={group}
                      streaming={analysis.steerStreaming}
                      isLast={idx === steerGroupedMessages.length - 1}
                      layersByType={steerAnalysis.layersByType}
                      bandsByPosition={steerAnalysis.bandsByPosition}
                      layerRange={steerAnalysis.effectiveRange}
                      onTokenHover={steerAnalysis.handleTokenHover}
                      selectedPositions={steerAnalysis.selectedPositions}
                      highlightedPosition={steerAnalysis.highlightedPosition}
                      editControls={
                        content.length > 0
                          ? {
                              idx,
                              content,
                              copied: copiedSteerIdx === idx,
                              canEdit: false,
                              canCopy: true,
                              showEdit: false,
                              onCopy: handleCopySteerMessage,
                              onEdit: () => {},
                            }
                          : undefined
                      }
                    />
                  );
                })
              : renderPlainBubbles(messages, false, '')}
          </div>
        ) : (
          <div className="flex min-h-40 items-center justify-center gap-x-2 px-1 text-slate-400">
            {analysis.steerStreaming ? (
              <>
                <LoadingSquare size={16} />
                <span className="text-[13px]">{analysis.steer?.mode === 'swap' ? 'Swapping…' : 'Steering…'}</span>
              </>
            ) : (
              <span className="text-center text-[10px] sm:text-[13px]">
                Press {analysis.steer?.mode === 'swap' ? 'Swap' : 'Steer'} to generate a new assistant response.
              </span>
            )}
          </div>
        )}
      </JlensProviders>
    ),
    [
      steerAnalysis,
      steerTokens,
      steerGroupedMessages,
      analysis.steerStreaming,
      analysis.steer?.mode,
      messages,
      copiedSteerIdx,
      handleCopySteerMessage,
    ],
  );

  function handleStop() {
    abortRef.current?.abort();
  }

  function handleClear() {
    if (streaming) {
      return;
    }
    // eslint-disable-next-line no-alert
    if (messages.length > 0 && !window.confirm('Clear this conversation?')) {
      return;
    }
    setMessages([]);
    setTokens([]);
    setMeta(null);
    setError(null);
    setPrefillText('');
    setEditingIdx(null);
    setEditingText('');
    setAwaitingReanalyze(false);
    setSelected([]);
    setSelectedPositions(new Set());
    setLockedLayer(null);
    analysis.clearSidebarSearch();
    onClear?.();
  }

  // When a fixture is loaded from /public, hydrate state from it so the chat
  // renders without hitting the inference server.
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
    setLiveAssistantText('');
    setEditingIdx(null);
    setEditingText('');
    setAwaitingReanalyze(false);
    setMessages(loadedData.messages ?? []);
    setMeta(loadedData.meta ?? null);
    // Shares are re-run server-side with generation disabled, so their tokens
    // come back without `is_generated` set. Rebuild the prompt→generated
    // boundary from the persisted `numPromptTokens` (absent/null on older shares
    // and plain fixtures, in which case we keep whatever the blob carried) so
    // an assistant-prefill boundary marker renders on reload.
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
    const data: JlensExportChat = {
      version: 1,
      kind: 'chat',
      modelId,
      exportedAt: new Date().toISOString(),
      messages,
      meta,
      tokens,
      steer:
        analysis.steer && steerTokens.length > 0
          ? { config: analysis.steer, meta: steerMeta, tokens: steerTokens }
          : undefined,
    };
    downloadJson(data, defaultExportFilename('chat', modelId));
  }

  // Sharing requires every token to carry an id so the server can faithfully
  // re-run inference over the exact sequence (older fixtures may lack ids).
  const canShare = tokens.length > 0 && tokens.every((t) => typeof t.id === 'number');

  const hasConversation = messages.length > 0 || tokens.length > 0;

  // Formats with a custom grouper (gpt-oss harmony) may emit more bubbles than
  // there are chat messages (an injected system turn, and separate analysis /
  // final assistant channels), so the group index no longer maps 1:1 onto the
  // `messages` array. Inline editing (which slices `messages` by that index) is
  // therefore disabled for such formats; copy still works off the bubble's own
  // rendered text.
  const supportsInlineEdit = !fmt.groupTokens;

  const defaultMessages = (
    <>
      {groupedMessages &&
        groupedMessages.map((group, idx) => {
          const content = supportsInlineEdit
            ? (messages[idx]?.content ?? '')
            : group.contentTokens.map((t) => t.token).join('');
          if (supportsInlineEdit && editingIdx === idx) {
            return (
              <AssistantEditBubble
                key={idx}
                value={editingText}
                onChange={setEditingText}
                onCancel={handleCancelAssistantEdit}
                onSave={handleSaveAssistantEdit}
              />
            );
          }
          return (
            <GroupBubble
              key={idx}
              group={group}
              streaming={streaming}
              showLoading={generating && !awaitingFirstResponse && idx === groupedMessages.length - 1}
              isLast={!awaitingFirstResponse && idx === groupedMessages.length - 1}
              layersByType={layersByType}
              bandsByPosition={bandsByPosition}
              layerRange={effectiveRange}
              onTokenHover={handleTokenHover}
              selectedPositions={selectedPositions}
              highlightedPosition={highlightedPosition}
              editControls={
                content.length > 0
                  ? {
                      idx,
                      content,
                      copied: copiedMessageIdx === idx,
                      canEdit: supportsInlineEdit && !streaming && !steering && !isEditing,
                      canCopy: !isEditing,
                      showEdit: supportsInlineEdit,
                      onCopy: handleCopyMessage,
                      onEdit: group.role === 'user' ? handleEditUserMessage : handleStartAssistantEdit,
                    }
                  : undefined
              }
            />
          );
        })}
      {!groupedMessages && !awaitingFirstResponse && renderPlainBubbles(messages, streaming, liveAssistantText)}
      {awaitingReanalyze && <PendingAssistantBubble fmt={fmt} />}
      {awaitingFirstResponse && (
        <PendingTurnBubbles userText={messages[messages.length - 1]?.content ?? ''} fmt={fmt} />
      )}
    </>
  );

  const inputArea = (
    <>
      <div className="relative flex flex-row items-end gap-x-2 rounded-lg border border-sky-100 bg-white px-0 py-2 shadow-md sm:rounded-xl">
        <div className="flex flex-1 flex-col">
          <ReactTextareaAutosize
            value={typedText}
            onChange={(e) => setTypedText(e.target.value.slice(0, MAX_LENS_CHAT_USER_CHARS))}
            maxLength={MAX_LENS_CHAT_USER_CHARS}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                attemptSend();
              }
            }}
            minRows={4}
            maxRows={6}
            disabled={streaming || steering || isEditing}
            placeholder={
              steering
                ? 'Exit steering to send messages…'
                : isEditing
                  ? 'Finish editing to send messages…'
                  : 'Send a message…'
            }
            className="max-h-[48px] resize-none border-0 px-2.5 py-0 text-[11px] leading-normal text-slate-800 outline-none placeholder:text-slate-400 focus:ring-0 disabled:cursor-not-allowed disabled:opacity-40 sm:max-h-full sm:px-4 sm:py-1 sm:text-[13px]"
          />
          <ReactTextareaAutosize
            value={prefillText}
            onChange={(e) => setPrefillText(e.target.value.slice(0, MAX_LENS_CHAT_PREFILL_CHARS))}
            maxLength={MAX_LENS_CHAT_PREFILL_CHARS}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                attemptSend();
              }
            }}
            minRows={3}
            maxRows={6}
            disabled={streaming || steering || isEditing}
            placeholder={'Optional: Prefill assistant (eg "You\'re absolutely right!")'}
            className="hidden max-h-[48px] resize-none border-0 border-t border-slate-200 px-2.5 py-0 pt-2 text-[10px] leading-normal text-slate-800 outline-none placeholder:text-slate-400 focus:ring-0 sm:block sm:max-h-[36px] sm:max-h-full sm:px-4 sm:py-2 sm:text-[13px]"
          />
        </div>
        <div className="absolute bottom-2 right-2 flex flex-row items-center gap-x-1.5">
          <button
            type="button"
            onClick={() => setShareOpen(true)}
            disabled={!canShare || streaming}
            title="Share this chat"
            aria-label="Share this chat"
            className="flex h-6 w-6 items-center justify-center rounded-md border border-slate-200 bg-white text-slate-400 transition-colors hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-40 sm:h-8 sm:w-8"
          >
            <Share2 className="h-3 w-3 sm:h-4 sm:w-4" />
          </button>
          <button
            type="button"
            onClick={handleExport}
            disabled={tokens.length === 0}
            title="Export this chat to JSON"
            aria-label="Export this chat to JSON"
            className="flex h-6 w-6 items-center justify-center rounded-md border border-slate-200 bg-white text-slate-400 transition-colors hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-40 sm:h-8 sm:w-8"
          >
            <Download className="h-3 w-3 sm:h-4 sm:w-4" />
          </button>
          {hasConversation && (
            <button
              type="button"
              onClick={handleClear}
              disabled={streaming || steering || isEditing}
              title="Clear chat"
              aria-label="Clear chat"
              className="flex h-6 w-6 items-center justify-center rounded-md border border-slate-200 bg-white text-slate-400 transition-colors hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-40 sm:h-8 sm:w-8"
            >
              <Trash2 className="h-3 w-3 sm:h-4 sm:w-4" />
            </button>
          )}
          <button
            type="button"
            onClick={() => setShowAdvanced((s) => !s)}
            title="Advanced settings"
            aria-label="Advanced settings"
            className={`flex h-6 w-6 items-center justify-center rounded-md border transition-colors sm:h-8 sm:w-8 ${
              showAdvanced
                ? 'border-sky-400 bg-sky-50 text-sky-600'
                : 'border-slate-200 bg-white text-slate-400 hover:bg-slate-100'
            }`}
          >
            <Settings className="h-3 w-3 sm:h-4 sm:w-4" />
          </button>
          <button
            type="button"
            onClick={() => (streaming ? handleStop() : attemptSend())}
            disabled={steering || isEditing || (inferenceAvailable && !streaming && !typedText.trim())}
            className="flex h-6 w-6 items-center justify-center rounded-full transition-colors disabled:cursor-not-allowed disabled:opacity-40 sm:h-8 sm:w-8"
          >
            {streaming ? (
              <X className="h-6 w-6 rounded-md bg-red-400 p-1.5 text-white hover:bg-red-500 sm:h-8 sm:w-8 sm:rounded-lg" />
            ) : (
              <ArrowUp className="h-6 w-6 rounded-md bg-sky-700 p-1.5 text-white hover:bg-sky-800 sm:h-8 sm:w-8 sm:rounded-lg" />
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
      <div className="relative flex h-full min-h-full flex-1 flex-col rounded-xl bg-slate-50 sm:flex-row sm:gap-x-3 sm:gap-y-0 sm:gap-y-1.5 sm:gap-y-3">
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
            kind: 'chat',
            inputTokenIds: tokens.map((t) => t.id),
            messages,
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

        {/* Mobile only: commentary above the chat, just below the top bar. On
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

        <div
          className={`flex min-h-0 w-full max-w-screen-lg basis-0 flex-col gap-y-0 border-slate-300 bg-slate-200/40 transition-all sm:flex-1 sm:rounded-2xl sm:px-3 ${steering ? 'grow-[2]' : 'grow-[3]'}`}
        >
          <div className="relative flex min-h-0 flex-1 flex-col sm:pb-4">
            {hasConversation ? (
              <>
                {/* Desktop only: commentary above the action bar. On mobile it
                    renders above the chat instead (see above). */}
                {showCommentary && sharedDescription && (
                  <div className="hidden w-full px-0 pt-3 sm:flex sm:pb-3">
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
                <div
                  ref={scrollRef}
                  id={JLENS_STEER_COLUMNS_ID}
                  onPointerDown={isEditing || steering ? undefined : onChatPointerDown}
                  className={`min-h-0 flex-1 overflow-y-auto ${dragging ? 'select-none' : ''}`}
                >
                  {/* Inner row is `min-h-full` so it grows to the taller of the
                      viewport and the scrolled content — otherwise the steered
                      column's divider (align-stretch) would only span the first
                      visible page and stop mid-transcript when scrolled. */}
                  <div className="flex min-h-full flex-row gap-x-0 gap-y-2.5">
                    {steering ? (
                      <div className="sm:min-w-auto mt-1 flex flex-1 flex-col gap-y-2.5 px-2 sm:px-0 sm:pt-2">
                        <DefaultOutputHeader />
                        {defaultMessages}
                      </div>
                    ) : (
                      <div
                        id={JLENS_CHAT_ID}
                        className="flex min-h-0 flex-1 flex-col gap-y-2.5 px-2 pt-2 sm:px-4 sm:pt-4"
                      >
                        {defaultMessages}
                      </div>
                    )}
                    {analysis.steer && (
                      <div
                        id={JLENS_STEER_OUTPUT_ID}
                        className="sm:min-w-auto mt-1 flex flex-1 flex-col gap-y-2.5 border-l border-slate-300 px-2 sm:ml-4 sm:pl-4 sm:pt-2"
                      >
                        <SteerOutputHeader steer={analysis.steer} />
                        {renderSteerResults()}
                      </div>
                    )}
                  </div>
                </div>

                {error && (
                  <div className="mt-2 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-[11px] text-red-700">
                    {error}
                  </div>
                )}

                {!steering && <div className="relative mt-0 flex flex-col gap-y-2 pb-3 sm:px-2">{inputArea}</div>}
              </>
            ) : (
              <div className="flex flex-1 flex-col items-center justify-center gap-y-3 px-1">
                <div className="mb-1 flex flex-col items-center gap-y-1 text-center text-slate-500">
                  <span className="text-lg font-normal text-slate-700 sm:text-3xl">
                    Hey! I'm{' '}
                    <span className="font-bold">
                      {modelId
                        .split('-')
                        .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
                        .join(' ')}
                    </span>
                    .<br />
                    <div className="mt-1.5 text-sm leading-normal text-slate-500 sm:text-base sm:leading-normal">
                      Send a message to see what pops up in my mind's workspace, or{' '}
                      <strong className="leading-none">J-Space</strong>.
                    </div>
                  </span>
                </div>
                {error && (
                  <div className="w-full rounded-md border border-red-200 bg-red-50 px-3 py-2 text-[11px] text-red-700">
                    {error}
                  </div>
                )}
                <div className="flex w-full flex-col gap-y-2 px-1 sm:px-4">{inputArea}</div>
              </div>
            )}
          </div>
        </div>

        <JlensAnalysisPanel
          analysis={analysis}
          tokens={tokens}
          numCompletionTokens={numCompletionTokens}
          setNumCompletionTokens={setNumCompletionTokens}
          onShare={() => setShareOpen(true)}
          canShare={canShare}
          shareDisabled={streaming}
          shareLabel="Share this chat"
          onExport={handleExport}
          exportDisabled={tokens.length === 0}
          exportLabel="Export this chat to JSON"
        />
      </div>
    </JlensProviders>
  );
}

function stripTurnEnd(content: string, turnEndToken: string): string {
  let out = content;
  while (out.endsWith(turnEndToken)) {
    out = out.slice(0, -turnEndToken.length);
  }
  return out.replace(/\s+$/, '');
}

// `<think>` / `</think>` reasoning markers. Rendered inline with the content
// but in the dim, monospaced special-token style.
function isThinkToken(token: string, prevToken?: string): boolean {
  const t = token.trim();
  if (t === '<think>' || t === '</think>') {
    return true;
  }
  if (token === '\n\n' && prevToken !== undefined) {
    const p = prevToken.trim();
    return p === '<think>' || p === '</think>';
  }
  return false;
}

// Per-message copy/edit affordances shown beneath a bubble. Omitted (undefined)
// when the bubble isn't user-editable (e.g. the steered-output transcript).
type MessageEditControls = {
  idx: number;
  content: string;
  copied: boolean;
  canEdit: boolean;
  canCopy: boolean;
  // When false, the edit affordance is omitted entirely (only copy remains) —
  // used by the steered-output transcript, which can be copied but not edited.
  showEdit?: boolean;
  onCopy: (idx: number, content: string) => void;
  onEdit: (idx: number, content: string) => void;
};

// A grouped (tokenized) message bubble: header/footer special tokens render
// dim and small; content tokens are hoverable lens chips.
function GroupBubble({
  group,
  streaming,
  showLoading = false,
  isLast = false,
  layersByType,
  bandsByPosition,
  layerRange,
  onTokenHover,
  selectedPositions,
  highlightedPosition,
  editControls,
}: {
  group: JlensTokenGroup;
  streaming: boolean;
  // When true, render a loading square below the bubble's content to indicate
  // the assistant response is still streaming (not yet complete).
  showLoading?: boolean;
  // Whether this is the final group in the transcript. The streaming loader is
  // only meaningful for the in-flight (last) assistant turn; gating on this
  // prevents earlier assistant bubbles from flashing a loader during a
  // streamed re-analysis when their content tokens haven't been rebuilt yet.
  isLast?: boolean;
  layersByType: Record<string, number[]>;
  bandsByPosition: Map<number, TokenBand[]>;
  layerRange: LayerRange | null;
  onTokenHover: (token: LensTokenMessage, open: boolean) => void;
  selectedPositions: Set<number>;
  highlightedPosition: number | null;
  editControls?: MessageEditControls;
}) {
  const isUser = group.role === 'user';
  const showContentLoading = !isUser && isLast && (showLoading || (streaming && group.contentTokens.length === 0));
  // Where generation begins within this bubble's content. Only > 0 when the
  // non-generated prefix is a real user assistant prefill, so the boundary
  // marker is shown only in that case — never at the very start of an assistant
  // bubble (the bubble already makes that clear). The chat template can
  // auto-inject an empty `<think></think>` scaffold (non-generated) before
  // generation; that isn't a prefill, so we ignore think/whitespace-only
  // prefixes.
  const firstGeneratedContentIdx = group.contentTokens.findIndex((t) => t.is_generated);
  const hasRealPrefill =
    firstGeneratedContentIdx > 0 &&
    group.contentTokens.slice(0, firstGeneratedContentIdx).some((t, i, arr) => {
      const prev = i > 0 ? arr[i - 1].token : undefined;
      return t.token.trim() !== '' && !isThinkToken(t.token, prev);
    });
  return (
    <div className={`group flex w-full ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`relative flex max-w-[80%] flex-col ${isUser ? 'items-end' : 'items-start'}`}>
        <div className={`flex flex-col rounded-xl bg-white px-3 py-2 sm:gap-y-0.5`}>
          {group.headerTokens.length > 0 && (
            <div className="whitespace-pre-wrap break-words font-mono text-[9px] leading-tight text-slate-400">
              {group.headerTokens.map((t, i) => (
                <JlensTokenChip
                  key={t.position}
                  token={t}
                  layersByType={layersByType}
                  variant="special"
                  bands={bandsByPosition.get(t.position)}
                  layerRange={layerRange}
                  onHoverChange={onTokenHover}
                  positionSelected={highlightedPosition == null && selectedPositions.has(t.position)}
                  prevSelected={highlightedPosition == null && selectedPositions.has(t.position - 1)}
                  nextSelected={highlightedPosition == null && selectedPositions.has(t.position + 1)}
                  highlighted={highlightedPosition === t.position}
                  prevEndsWithLineBreak={i > 0 && /\n/.test(group.headerTokens[i - 1].token)}
                  nextStartsNewLine={i === group.headerTokens.length - 1}
                />
              ))}
            </div>
          )}
          <div className="whitespace-pre-wrap break-words font-serif text-[13px] leading-none sm:leading-relaxed">
            {group.contentTokens.map((token, tokenIdx) => {
              const newlineCount = (token.token.match(/\n/g) || []).length;
              const prevToken = tokenIdx > 0 ? group.contentTokens[tokenIdx - 1].token : undefined;
              const prevEndsWithLineBreak = tokenIdx === 0 || (prevToken !== undefined && /\n/.test(prevToken));
              const nextStartsNewLine = tokenIdx === group.contentTokens.length - 1;
              const variant = isThinkToken(token.token, prevToken)
                ? 'think'
                : token.is_generated
                  ? 'generated'
                  : 'content';
              return (
                <span key={token.position}>
                  {tokenIdx === firstGeneratedContentIdx && hasRealPrefill && (
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
                    variant={variant}
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
                    <div key={i} className="h-[0px] max-h-[0px] leading-[0em] sm:h-1 sm:max-h-1" />
                  ))}
                </span>
              );
            })}
          </div>
          {showContentLoading && (
            <div className="flex items-center pt-0.5 text-slate-400">
              <LoadingSquare size={14} />
            </div>
          )}
          {group.footerTokens.length > 0 && (
            <div className="whitespace-pre-wrap break-words font-mono text-[9px] leading-tight text-slate-400">
              {group.footerTokens.map((t, i) => (
                <JlensTokenChip
                  key={t.position}
                  token={t}
                  layersByType={layersByType}
                  variant="special"
                  bands={bandsByPosition.get(t.position)}
                  layerRange={layerRange}
                  onHoverChange={onTokenHover}
                  positionSelected={highlightedPosition == null && selectedPositions.has(t.position)}
                  prevSelected={highlightedPosition == null && selectedPositions.has(t.position - 1)}
                  nextSelected={highlightedPosition == null && selectedPositions.has(t.position + 1)}
                  highlighted={highlightedPosition === t.position}
                  prevEndsWithLineBreak={i === 0 || /\n/.test(group.footerTokens[i - 1].token)}
                  nextStartsNewLine={i === group.footerTokens.length - 1}
                />
              ))}
            </div>
          )}
        </div>
        {editControls && (
          <div
            className={`absolute top-full mt-1 hidden h-5 items-center gap-x-1 px-1 opacity-0 transition-opacity duration-150 group-hover:opacity-100 sm:flex ${
              isUser ? 'right-0 justify-end' : 'left-0 justify-start'
            }`}
          >
            <button
              type="button"
              onClick={() => editControls.onCopy(editControls.idx, editControls.content)}
              disabled={!editControls.canCopy}
              title="Copy message"
              aria-label="Copy message"
              className="flex h-5 w-5 items-center justify-center rounded text-slate-400 transition-colors hover:bg-slate-200 hover:text-slate-600 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-transparent disabled:hover:text-slate-400"
            >
              {editControls.copied ? <Check className="h-3 w-3 text-sky-600" /> : <Copy className="h-3 w-3" />}
            </button>
            {editControls.showEdit !== false && (
              <button
                type="button"
                onClick={() => editControls.onEdit(editControls.idx, editControls.content)}
                disabled={!editControls.canEdit}
                title={isUser ? 'Edit message' : 'Edit assistant message'}
                aria-label={isUser ? 'Edit message' : 'Edit assistant message'}
                className="flex h-5 w-5 items-center justify-center rounded text-slate-400 transition-colors hover:bg-slate-200 hover:text-slate-600 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-transparent disabled:hover:text-slate-400"
              >
                <Pencil className="h-3 w-3" />
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// Inline editor that replaces an assistant bubble while the user revises its
// content. The chat-template special tokens are not exposed (they're re-applied
// by the server on re-analysis), so the user can't corrupt the templating.
function AssistantEditBubble({
  value,
  onChange,
  onCancel,
  onSave,
}: {
  value: string;
  onChange: (value: string) => void;
  onCancel: () => void;
  onSave: () => void;
}) {
  return (
    <div className="flex w-full justify-start">
      <div className="flex w-full max-w-[90%] flex-col gap-y-1.5 rounded-xl bg-white px-3 py-2 shadow">
        <ReactTextareaAutosize
          value={value}
          onChange={(e) => onChange(e.target.value)}
          minRows={3}
          maxRows={12}
          autoFocus
          aria-label="Edit assistant message content"
          className="w-full resize-none rounded-md border border-sky-200 px-2 py-1.5 font-serif text-[13px] leading-relaxed text-slate-700 outline-none focus:border-sky-400 focus:ring-1 focus:ring-sky-400"
        />
        <div className="flex flex-row items-center justify-end gap-x-1.5">
          <button
            type="button"
            onClick={onCancel}
            className="rounded-md border border-slate-200 bg-slate-100 px-3 py-1 text-[11px] font-medium text-slate-600 transition-colors hover:bg-slate-200"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onSave}
            disabled={value.length === 0}
            title="Save and re-analyze"
            className="rounded-md border border-sky-600 bg-sky-600 px-3 py-1 text-[11px] font-semibold text-white transition-colors hover:bg-sky-500/90 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Save
          </button>
        </div>
      </div>
    </div>
  );
}

// Loading placeholder for a single assistant turn, shown in place of an edited
// assistant bubble while the re-analysis streams back (the earlier turns stay
// rendered from their tokens).
function PendingAssistantBubble({ fmt }: { fmt: JlensChatFormat }) {
  return (
    <div className="flex w-full justify-start">
      <div className="flex max-w-[90%] flex-col gap-y-0.5 rounded-xl bg-white px-3 py-2 shadow">
        <div className="whitespace-pre-wrap break-words font-mono text-[9px] leading-tight text-slate-400">
          {fmt.turnStartToken}
          {fmt.assistantRoleName}
        </div>
        <div className="whitespace-pre-wrap break-words pl-2 pt-2 font-serif text-[13px] leading-relaxed">
          <span className="inline-flex items-center gap-1 text-slate-400">
            <LoadingSquare size={24} />
          </span>
        </div>
      </div>
    </div>
  );
}

// Optimistic placeholder shown for the just-submitted turn while we wait for
// the server's first response.
function PendingTurnBubbles({ userText, fmt }: { userText: string; fmt: JlensChatFormat }) {
  return (
    <>
      <div className="flex w-full justify-end">
        <div className="flex max-w-[80%] flex-col rounded-xl bg-white px-3 py-2 sm:gap-y-0.5">
          <div className="whitespace-pre-wrap break-words font-mono text-[9px] leading-tight text-slate-400">
            {fmt.turnStartToken}user
          </div>
          <div className="whitespace-pre-wrap break-words font-serif text-[10px] leading-relaxed sm:text-[13px] sm:leading-[27.5px]">
            {userText}
          </div>
          <div className="whitespace-pre-wrap break-words font-mono text-[9px] leading-tight text-slate-400">
            {fmt.turnEndToken}
          </div>
        </div>
      </div>
      <div className="flex w-full justify-start">
        <div className="flex max-w-[80%] flex-col rounded-xl bg-white px-3 py-2 sm:gap-y-0.5">
          <div className="whitespace-pre-wrap break-words font-mono text-[9px] leading-tight text-slate-400">
            {fmt.turnStartToken}
            {fmt.assistantRoleName}
          </div>
          <div className="whitespace-pre-wrap break-words pl-2 pt-2 font-serif text-[13px] leading-relaxed">
            <span className="inline-flex items-center gap-1 text-slate-400">
              <LoadingSquare size={24} />
            </span>
          </div>
        </div>
      </div>
    </>
  );
}

// Plain-text bubbles used while streaming (before the full token stream is
// grouped) and for the optimistic user turn.
function renderPlainBubbles(messages: ChatMessage[], streaming: boolean, liveAssistantText: string) {
  const bubbles = [...messages];
  if (streaming) {
    bubbles.push({ role: 'assistant', content: liveAssistantText });
  }
  return bubbles.map((msg, idx) => {
    const isUser = msg.role === 'user';
    return (
      <div key={idx} className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'}`}>
        <div
          className={`max-w-[90%] whitespace-pre-wrap break-words rounded-xl px-3 py-2 font-serif text-[13px] leading-relaxed ${
            isUser ? 'bg-white text-slate-800 shadow' : 'bg-transparent text-slate-600'
          }`}
        >
          {msg.content.length > 0 ? (
            msg.content
          ) : (
            <span className="inline-flex items-center gap-1 text-slate-400">
              <LoadingSquare size={14} />
            </span>
          )}
        </div>
      </div>
    );
  });
}
