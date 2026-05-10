'use client';

import { useNlaContext } from '@/components/provider/nla-provider';
import { LoadingSquare } from '@/components/svg/loading-square';
import { CHARS_PER_TOKEN_ESTIMATE, CONFIDENCE_THRESHOLD, EXPLANATION_TOKEN_ESTIMATE } from '@/lib/nla-constants';
import * as Slider from '@radix-ui/react-slider';
import { EventSourceParserStream } from 'eventsource-parser/stream';
import {
  ArrowDown,
  ArrowUp,
  Check,
  ChevronLeft,
  ChevronRight,
  Copy,
  Pencil,
  Play,
  Search,
  Trash2,
  X,
} from 'lucide-react';
import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ReactTextareaAutosize from 'react-textarea-autosize';
import NLAInputChatAdvanced from './nla-input-chat-advanced';
import NLAInputChatControls from './nla-input-chat-controls';
import {
  NLA_TOUR_LLAMA_LIE_CACHE_ID,
  NLA_TOUR_LLAMA_LIE_EXPLAIN_ELEMENT_ID,
  NLA_TOUR_LLAMA_LIE_QUESTION_ELEMENT_ID,
  NLA_TOUR_LLAMA_LIE_QUESTION_POSITION,
} from './nla-tour-constants';
import { ChatMessage, TokenInfo } from './nla-types';
import {
  buildSyntheticUserTurnPreviewGroup,
  cleanPartialText,
  computeRelativeMse,
  groupTokensIntoMessages,
  MAX_TOKENS_TO_EXPLAIN,
} from './nla-utils';

export default function NLAInputChat() {
  const {
    chatMessages,
    setChatMessages,
    tokenizerFormat,
    selectedModelId: modelId,
    selectedNlaSource,
    temperature,
    setTemperature,
    maxNewTokens,
    setMaxNewTokens,
    showAdvanced,
    setShowAdvanced,
    isChatStreaming,
    setIsChatStreaming,
    isLoading,
    tokenList,
    setTokenList,
    setLastTokenizedText,
    selectedTokenPositions,
    handleApplySelection: onApplySelection,
    topLevelMode,
    handleTopLevelModeChange: onTopLevelModeChange,
    selectedPosition,
    setSelectedPosition,
    lockedPosition,
    setLockedPosition,
    setHighlightedParagraph,
    setHighlightedRange,
    setHighlightComment,
    resultMap,
    partialMap,
    handleSubmit: onExplain,
    explainDisabled,
    handleClear: onClear,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    handleShare: onShare,
    isEmbed,
    onUserEdit,
    chatScrollNonce,
    cancelPendingAutoExplain,
    truncateChatFrom,
    setExplanationSearchNeedle,
    explanationSearchResetNonce,
    activeDemoCacheId,
    isHydratingDemo,
    error: providerError,
    pendingChatInputRestore,
    setPendingChatInputRestore,
  } = useNlaContext();
  const nlaSourceId = selectedNlaSource?.id;
  const [typedText, setTypedText] = useState('');
  const [error, setError] = useState<string | null>(null);
  // Remaining requests in the current hourly window for `/api/nla/completion`,
  // surfaced via the `x-limit-remaining` response header (set by the
  // top-level rate-limit middleware). `null` until the first response.
  const [limitRemaining, setLimitRemaining] = useState<number | null>(null);

  // The provider sets `pendingChatInputRestore` when an upstream failure
  // (e.g. explain 429) rolls the chat back; we drain it here into the
  // textarea so the user's last message is ready to retry. Drained in a
  // single tick to avoid clobbering subsequent user typing.
  useEffect(() => {
    if (pendingChatInputRestore !== null) {
      setTypedText(pendingChatInputRestore);
      setPendingChatInputRestore(null);
    }
  }, [pendingChatInputRestore, setPendingChatInputRestore]);
  // Whether the chat-template special tokens (`<|im_start|>...`, `<|im_end|>`,
  // role tokens, the trailing newlines after them) are rendered as chips.
  const [showChatTokens, setShowChatTokens] = useState(true);

  // Restore default (show) on cache hydrate (demo button or `?id=...&position=N` deep-link).
  // Keyed only on `chatScrollNonce` so chip clicks don't steer the toggle.
  useEffect(() => {
    if (chatScrollNonce === 0) return;
    setShowChatTokens(true);
  }, [chatScrollNonce]);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  // Auto-scroll to bottom when new content arrives.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [chatMessages, tokenList]);

  // After hydrating from a cache the conversation is swapped in
  // synchronously across multiple state updates; the existing effect above
  // races with subsequent layout passes (chips render after tokenList
  // updates). Force a scroll-to-bottom after the next paint so the user
  // always lands at the most recent turn.
  useEffect(() => {
    if (chatScrollNonce === 0) return undefined;
    let raf2 = 0;
    const raf1 = requestAnimationFrame(() => {
      raf2 = requestAnimationFrame(() => {
        const el = scrollRef.current;
        if (el) el.scrollTop = el.scrollHeight;
      });
    });
    return () => {
      cancelAnimationFrame(raf1);
      if (raf2) cancelAnimationFrame(raf2);
    };
  }, [chatScrollNonce]);

  // ── Chip selection / drag state
  // After the very first explain run (when `resultMap`/`partialMap` have
  // anything in them) we drop the auto/manual toggle and let the user keep
  // adding more tokens to explain by clicking/dragging chips directly. The
  // already-explained chips display their result style and are locked
  // (cannot be deselected) — they also don't count toward the 32-token
  // pending-selection limit.
  // const hasResults = resultMap.size > 0 || partialMap.size > 0;
  const explainedPositions = useMemo(() => {
    const s = new Set<number>();
    resultMap.forEach((_, k) => s.add(k));
    partialMap.forEach((_, k) => s.add(k));
    return s;
  }, [resultMap, partialMap]);
  const pendingPositions = useMemo(() => {
    const s = new Set<number>();
    selectedTokenPositions.forEach((p) => {
      if (!explainedPositions.has(p)) s.add(p);
    });
    return s;
  }, [selectedTokenPositions, explainedPositions]);
  const isSelectionPhase = !isLoading && !isChatStreaming;
  const selectionLimitReached = pendingPositions.size >= MAX_TOKENS_TO_EXPLAIN;
  const sourceNorm = selectedNlaSource?.norm ?? 0;

  const [dragMode, setDragMode] = useState<'select' | 'deselect' | null>(null);
  const dragStartPositionRef = useRef<number | null>(null);
  const preDragSelectionRef = useRef<Set<number> | null>(null);
  // Anchor for shift+click range selection. Updated on every non-shift
  // mousedown that starts a drag/select, so subsequent shift+clicks extend
  // from the last "starting" click (Finder/GitHub-style multi-select).
  // Survives across drag-end (unlike `dragStartPositionRef`, which is
  // cleared when `dragMode` falls back to null).
  const selectionAnchorRef = useRef<number | null>(null);
  const [limitPulse, setLimitPulse] = useState(0);

  useEffect(() => {
    if (dragMode === null) return undefined;
    const handleUp = () => setDragMode(null);
    window.addEventListener('mouseup', handleUp);
    return () => window.removeEventListener('mouseup', handleUp);
  }, [dragMode]);

  useEffect(() => {
    if (!isSelectionPhase && dragMode !== null) setDragMode(null);
  }, [isSelectionPhase, dragMode]);

  useEffect(() => {
    if (dragMode === null) {
      dragStartPositionRef.current = null;
      preDragSelectionRef.current = null;
    }
  }, [dragMode]);

  useEffect(() => {
    if (limitPulse === 0) return undefined;
    const t = setTimeout(() => setLimitPulse(0), 700);
    return () => clearTimeout(t);
  }, [limitPulse]);

  // Shared range-select core used by both click-and-drag and shift+click.
  // `anchorPosition` and `currentPosition` define the inclusive token range;
  // `basis` is the selection set the result is built off of (the pre-drag
  // snapshot for drags, the live selection for shift+click).
  const applySelectRangeFrom = useCallback(
    (anchorPosition: number, currentPosition: number, basis: Set<number>, mode: 'select' | 'deselect') => {
      const lo = Math.min(anchorPosition, currentPosition);
      const hi = Math.max(anchorPosition, currentPosition);
      // Skip already-explained positions — they're locked (can't be
      // deselected, and shouldn't count toward the pending-selection limit).
      const inRange = tokenList
        .filter((t) => t.position >= lo && t.position <= hi && !explainedPositions.has(t.position))
        .map((t) => t.position)
        .sort((a, b) => Math.abs(a - anchorPosition) - Math.abs(b - anchorPosition));

      const next = new Set(basis);
      let pendingCount = 0;
      next.forEach((p) => {
        if (!explainedPositions.has(p)) pendingCount += 1;
      });
      let limitHit = false;
      if (mode === 'select') {
        for (const p of inRange) {
          if (next.has(p)) continue;
          if (pendingCount >= MAX_TOKENS_TO_EXPLAIN) {
            limitHit = true;
            continue;
          }
          next.add(p);
          pendingCount += 1;
        }
      } else {
        for (const p of inRange) next.delete(p);
      }
      if (limitHit) setLimitPulse((c) => c + 1);
      onApplySelection(next);
    },
    [onApplySelection, tokenList, explainedPositions],
  );

  const applyDragRange = useCallback(
    (currentPosition: number, mode: 'select' | 'deselect') => {
      const start = dragStartPositionRef.current;
      const preDrag = preDragSelectionRef.current;
      if (start === null || preDrag === null) return;
      applySelectRangeFrom(start, currentPosition, preDrag, mode);
    },
    [applySelectRangeFrom],
  );

  const toggleSinglePosition = useCallback(
    (position: number) => {
      if (explainedPositions.has(position)) return;
      const next = new Set(selectedTokenPositions);
      if (next.has(position)) {
        next.delete(position);
      } else {
        let pendingCount = 0;
        next.forEach((p) => {
          if (!explainedPositions.has(p)) pendingCount += 1;
        });
        if (pendingCount >= MAX_TOKENS_TO_EXPLAIN) {
          setLimitPulse((c) => c + 1);
          return;
        }
        next.add(position);
      }
      onApplySelection(next);
    },
    [onApplySelection, selectedTokenPositions, explainedPositions],
  );

  const tokenGroups = useMemo(() => {
    if (tokenList.length === 0) return null;
    return groupTokensIntoMessages(tokenList, tokenizerFormat);
  }, [tokenList, tokenizerFormat]);

  // Bridges chat end → explain start (auto mode). Hoisted so explanation search can use `isBusy`.
  const [autoExplainPending, setAutoExplainPending] = useState(false);
  const prevChatStreamingRef = useRef(false);
  useEffect(() => {
    const wasStreaming = prevChatStreamingRef.current;
    prevChatStreamingRef.current = isChatStreaming;
    if (isChatStreaming) {
      setAutoExplainPending(false);
      return;
    }
    if (wasStreaming && topLevelMode === 'auto') {
      setAutoExplainPending(true);
    }
  }, [isChatStreaming, topLevelMode]);
  useEffect(() => {
    if (isLoading) setAutoExplainPending(false);
  }, [isLoading]);
  useEffect(() => {
    if (!autoExplainPending) return undefined;
    const t = setTimeout(() => setAutoExplainPending(false), 4000);
    return () => clearTimeout(t);
  }, [autoExplainPending]);
  const isExplainInFlight = isLoading || autoExplainPending;
  // Demo hydrate counts as "busy" so the textarea, send button, and
  // selection chips are disabled while we're swapping in a new cache.
  const isBusy = isChatStreaming || isExplainInFlight || isHydratingDemo;

  const [explanationSearchInput, setExplanationSearchInput] = useState('');
  const [explanationSearchDebouncedTrimmed, setExplanationSearchDebouncedTrimmed] = useState('');
  const [explanationSearchResultIndex, setExplanationSearchResultIndex] = useState(0);
  const prevExplanationSearchTrimmedRef = useRef('');
  useEffect(() => {
    const h = window.setTimeout(() => {
      setExplanationSearchDebouncedTrimmed(explanationSearchInput.trim());
    }, 200);
    return () => clearTimeout(h);
  }, [explanationSearchInput]);

  const scrollToExplanationSearchMatch = useCallback((pos: number) => {
    requestAnimationFrame(() => {
      const root = scrollRef.current;
      if (!root) return;
      const el = root.querySelector(`[data-token-position="${pos}"]`);
      el?.scrollIntoView({ block: 'center', behavior: 'smooth' });
    });
  }, []);

  const clearExplanationCommentaryAndSelection = useCallback(() => {
    setHighlightedParagraph(null);
    setHighlightedRange(null);
    setHighlightComment(null);
  }, [setHighlightedParagraph, setHighlightedRange, setHighlightComment]);

  useEffect(() => {
    if (explanationSearchResetNonce === 0) return;
    setExplanationSearchInput('');
    setExplanationSearchDebouncedTrimmed('');
    setExplanationSearchResultIndex(0);
    prevExplanationSearchTrimmedRef.current = '';
    setExplanationSearchNeedle('');
    clearExplanationCommentaryAndSelection();
  }, [explanationSearchResetNonce, setExplanationSearchNeedle, clearExplanationCommentaryAndSelection]);

  const explanationSearchMatches = useMemo(() => {
    if (explanationSearchDebouncedTrimmed.length < 2) return [];
    const q = explanationSearchDebouncedTrimmed.toLowerCase();
    const out: number[] = [];
    for (const tok of tokenList) {
      const r = resultMap.get(tok.position);
      const partial = partialMap.get(tok.position);
      let text = '';
      if (r?.description) text = r.description;
      else if (partial !== undefined) text = cleanPartialText(partial);
      if (text.length > 0 && text.toLowerCase().includes(q)) out.push(tok.position);
    }
    return out.sort((a, b) => a - b);
  }, [explanationSearchDebouncedTrimmed, tokenList, resultMap, partialMap]);
  const explanationSearchMatchSet = useMemo(() => new Set(explanationSearchMatches), [explanationSearchMatches]);
  const explanationSearchMatchesKey = explanationSearchMatches.join(',');
  const explanationSearchActive = explanationSearchDebouncedTrimmed.length >= 2;

  useEffect(() => {
    if (explanationSearchDebouncedTrimmed.length < 2) {
      setExplanationSearchResultIndex(0);
    }
  }, [explanationSearchDebouncedTrimmed]);

  useEffect(() => {
    if (explanationSearchDebouncedTrimmed.length < 2) return;
    setShowChatTokens(true);
  }, [explanationSearchDebouncedTrimmed]);

  useEffect(() => {
    if (explanationSearchDebouncedTrimmed.length < 2) {
      setExplanationSearchNeedle('');
      prevExplanationSearchTrimmedRef.current = '';
      return;
    }
    setExplanationSearchNeedle(explanationSearchDebouncedTrimmed);
    const queryChanged = prevExplanationSearchTrimmedRef.current !== explanationSearchDebouncedTrimmed;
    prevExplanationSearchTrimmedRef.current = explanationSearchDebouncedTrimmed;
    if (!queryChanged) return;
    clearExplanationCommentaryAndSelection();
    if (explanationSearchMatches.length === 0) {
      setExplanationSearchResultIndex(0);
      setLockedPosition(null);
      setSelectedPosition(null);
      return;
    }
    setExplanationSearchResultIndex(0);
    const p = explanationSearchMatches[0];
    setLockedPosition(p);
    setSelectedPosition(p);
    scrollToExplanationSearchMatch(p);
  }, [
    explanationSearchDebouncedTrimmed,
    explanationSearchMatches,
    setExplanationSearchNeedle,
    setLockedPosition,
    setSelectedPosition,
    scrollToExplanationSearchMatch,
    clearExplanationCommentaryAndSelection,
  ]);

  useEffect(() => {
    if (explanationSearchDebouncedTrimmed.length < 2 || explanationSearchMatches.length === 0) return;
    if (explanationSearchResultIndex < explanationSearchMatches.length) return;
    clearExplanationCommentaryAndSelection();
    const ni = explanationSearchMatches.length - 1;
    const p = explanationSearchMatches[ni];
    setExplanationSearchResultIndex(ni);
    setLockedPosition(p);
    setSelectedPosition(p);
    scrollToExplanationSearchMatch(p);
  }, [
    explanationSearchResultIndex,
    explanationSearchMatches,
    explanationSearchMatchesKey,
    explanationSearchDebouncedTrimmed.length,
    setLockedPosition,
    setSelectedPosition,
    scrollToExplanationSearchMatch,
    clearExplanationCommentaryAndSelection,
  ]);

  const goExplanationSearchPrev = useCallback(() => {
    if (explanationSearchMatches.length === 0) return;
    clearExplanationCommentaryAndSelection();
    let idx = explanationSearchResultIndex;
    if (idx < 0 || idx >= explanationSearchMatches.length) idx = 0;
    const n = explanationSearchMatches.length;
    const nextIdx = (idx - 1 + n) % n;
    setExplanationSearchResultIndex(nextIdx);
    const p = explanationSearchMatches[nextIdx];
    setLockedPosition(p);
    setSelectedPosition(p);
    scrollToExplanationSearchMatch(p);
  }, [
    explanationSearchMatches,
    explanationSearchResultIndex,
    scrollToExplanationSearchMatch,
    setLockedPosition,
    setSelectedPosition,
    clearExplanationCommentaryAndSelection,
  ]);

  const goExplanationSearchNext = useCallback(() => {
    if (explanationSearchMatches.length === 0) return;
    clearExplanationCommentaryAndSelection();
    let idx = explanationSearchResultIndex;
    if (idx < 0 || idx >= explanationSearchMatches.length) idx = 0;
    const n = explanationSearchMatches.length;
    const nextIdx = (idx + 1) % n;
    setExplanationSearchResultIndex(nextIdx);
    const p = explanationSearchMatches[nextIdx];
    setLockedPosition(p);
    setSelectedPosition(p);
    scrollToExplanationSearchMatch(p);
  }, [
    explanationSearchMatches,
    explanationSearchResultIndex,
    scrollToExplanationSearchMatch,
    setLockedPosition,
    setSelectedPosition,
    clearExplanationCommentaryAndSelection,
  ]);

  function renderChip(tok: TokenInfo, small?: boolean, preview?: boolean, tooltipTriggerClassName?: string) {
    const newlineCount = (tok.token.match(/\n/g) || []).length;
    const hasNewline = newlineCount > 0;
    // Bump non-small chips one size up while the Llama's Lie tour demo
    // is loaded — the tour spotlights individual tokens and the default
    // 14px reads too thin under the driver.js overlay. Looser leading
    // gives the larger glyphs room to breathe across wrapped lines.
    const isTourDemo = activeDemoCacheId === NLA_TOUR_LLAMA_LIE_CACHE_ID;
    const nonSmallSizeClass = isTourDemo
      ? 'px-[0.5px] py-0 text-[14px] sm:text-[18px] leading-relaxed'
      : 'px-[0.5px] py-0 text-[12px] sm:text-[14px] leading-tight';

    if (preview) {
      const colorClass = small ? 'text-slate-400' : 'text-slate-500';
      const inner =
        tok.token.trim() === '' ? (
          hasNewline ? (
            <span className="opacity-35">{'\u21B5'.repeat(newlineCount)}</span>
          ) : (
            <span className="text-slate-300">{'\u00B7'}</span>
          )
        ) : tok.token.includes('\n') ? (
          <span className="whitespace-pre-wrap break-words text-left">{tok.token}</span>
        ) : (
          <>
            {tok.token.startsWith(' ') && <span className="text-white">{'\u00B7'}</span>}
            {tok.token.trim()}
            {tok.token.endsWith(' ') && <span className="text-white">{'\u00B7'}</span>}
          </>
        );
      // Same outer wrapper as live chips: synthetic preview (`buildSyntheticUserTurnPreviewGroup`) uses the
      // same `tooltipTriggerClassName` header/footer strips from `buildLines`.
      const previewOuterClass = tooltipTriggerClassName
        ? `inline-block ${tooltipTriggerClassName}`
        : small
          ? 'inline-block align-top leading-none'
          : 'inline-block';
      return (
        <Fragment key={tok.position}>
          <span className={previewOuterClass}>
            <span
              className={`relative cursor-default border-b-[1.5px] border-transparent font-serif ${
                small ? 'px-[1px] py-0 text-[9px] leading-none' : nonSmallSizeClass
              } ${colorClass}`}
            >
              {inner}
            </span>
          </span>
        </Fragment>
      );
    }

    // // for now, always render the chips in normal size
    // small = false;
    const result = resultMap.get(tok.position);
    const partialText = partialMap.get(tok.position);
    const isInSelection = selectedTokenPositions.has(tok.position);
    // A chip is "explained" once any of these is true:
    //   1. the streaming partial text contains a closing </explanation>
    //      tag (description locked in, score still pending),
    //   2. the result event has fired (covers the truncated case where
    //      </explanation> never streamed in — once the server gives up
    //      and emits the final result, we still want to flip the bg from
    //      amber → white, even if that happens at the same instant the
    //      score lands), or
    //   3. the result event arrived without any prior partial at all
    //      (e.g. when hydrating from cache).
    // It is "scored" only once an MSE value lands. Decoupling these two
    // lets the chip flip from the amber "generating" bg to white the
    // instant the explanation finishes, then keep the slate sweep
    // underline going until the score comes in.
    const partialExplanationDone = partialText !== undefined && partialText.includes('</explanation>');
    const isExplained = !!result || partialExplanationDone;
    const isScored = !!result && result.mse !== null;
    // A selected chip that we're explaining counts as "generating" the
    // moment Explain is clicked, so the orbit animation starts immediately
    // — not when the server first emits a partial update for that token.
    const chipIsGenerating = !isExplained && (partialText !== undefined || (isLoading && isInSelection));
    // Three independent visual layers for explained chips:
    //   - the LOCKED chip always retains the full sky highlight,
    //   - the HOVERED chip (when nothing is locked) gets the same full
    //     highlight,
    //   - the HOVERED chip while another chip is locked gets a softer
    //     sky tint so the user can tell it's not the pinned one.
    const isLockedChip = lockedPosition !== null && lockedPosition === tok.position;
    const isHoveredChip = selectedPosition === tok.position;
    const hasLockElsewhere = lockedPosition !== null && lockedPosition !== tok.position;
    const showFullFocus = isExplained && (isLockedChip || (isHoveredChip && !hasLockElsewhere));
    const showSoftFocus = isExplained && !showFullFocus && isHoveredChip && hasLockElsewhere;
    const chipScore = isScored ? computeRelativeMse(result.mse, sourceNorm) : null;
    // Low-confidence chips get an orange underline to flag explanations the
    // model is least sure about; everything below the threshold (or where
    // we have no score yet) keeps the default sky underline below. Higher
    // RMSE = worse reconstruction, so "low confidence" is `>= threshold`.
    const isLowConfidence = chipScore !== null && chipScore >= CONFIDENCE_THRESHOLD;
    const cannotAdd = !isInSelection && selectionLimitReached;
    // The slate sweep keeps animating until we have a score: it covers
    // both "not yet started / mid-explain" (paired with the amber bg) and
    // the brief "explained but not scored" window (paired with the white
    // bg, so the chip clearly reads as "explanation done, awaiting score").
    const showLoadingSweep = !isScored && (chipIsGenerating || isExplained);
    const isExplanationSearchHit = explanationSearchActive && explanationSearchMatchSet.has(tok.position);

    let colorClass: string;
    if (isExplained) {
      // Use a transparent CSS border to preserve spacing, then render a
      // 1px-inset bar below as the visible "underline". This keeps adjacent
      // result chips visually separated instead of merging into one bar.
      // Explicit `bg-white` so when a chip transitions from "generating"
      // (amber) to "explained" the background animates from amber-200 →
      // white via the chip's existing `transition-all`.
      colorClass = small ? 'text-slate-400' : 'text-slate-700' + ' rounded-t border-b-transparent bg-transparent';
    } else if (chipIsGenerating) {
      // Mid-explain: use the same amber palette as the "selected for
      // explanation" state so the visual flow is selected → generating →
      // explained → scored, with the animated underline sweep on top until
      // a score lands.
      colorClass = 'border-b-transparent bg-amber-200 text-amber-700';
    } else if (isInSelection && isSelectionPhase) {
      colorClass = 'border-b-slate-400 bg-amber-200 text-amber-700';
    } else if (isInSelection && isLoading) {
      colorClass = 'border-slate-300 bg-slate-100 text-slate-600 opacity-50';
    } else {
      colorClass = small ? 'border-transparent bg-transparent text-slate-400' : 'border-transparent text-slate-500';
    }

    const handleActivateKeyboard = () => {
      // Any chip click clears the per-paragraph and per-range highlights
      // — they're anchored to whatever the previous focus token was, and
      // a deep link's preset highlight shouldn't bleed across navigations.
      setHighlightedParagraph(null);
      setHighlightedRange(null);
      setHighlightComment(null);
      if (isExplained) {
        // Toggle: clicking the already-locked chip unlocks it.
        setLockedPosition(lockedPosition === tok.position ? null : tok.position);
        setSelectedPosition(tok.position);
        return;
      }
      if (!isSelectionPhase) {
        setLockedPosition(null);
        setSelectedPosition(tok.position);
        return;
      }
      const willSelect = !isInSelection && !cannotAdd;
      setLockedPosition(null);
      toggleSinglePosition(tok.position);
      setSelectedPosition(willSelect ? tok.position : null);
    };

    const handleMouseDown = (e: React.MouseEvent) => {
      if (e.button !== 0) return;
      // Stop the chat-area unlock handler from clearing the lock the
      // moment we set it (mousedown bubbles up from the chip).
      e.stopPropagation();
      // Same reset as the keyboard path — see handleActivateKeyboard.
      setHighlightedParagraph(null);
      setHighlightedRange(null);
      setHighlightComment(null);
      if (isExplained) {
        // Toggle: clicking the already-locked chip unlocks it.
        setLockedPosition(lockedPosition === tok.position ? null : tok.position);
        setSelectedPosition(tok.position);
        return;
      }
      if (!isSelectionPhase) {
        setLockedPosition(null);
        setSelectedPosition(tok.position);
        return;
      }
      e.preventDefault();
      // Shift+click extends selection from the last anchor (the most
      // recent click that initiated a drag/select) up to this chip — a
      // one-shot range select that doesn't enter drag mode and doesn't
      // move the anchor, so subsequent shift+clicks keep extending from
      // the same origin (Finder/GitHub-style multi-select). Range filter
      // skips already-explained positions and caps at the per-run limit
      // (pulses the counter if the cap is hit, same as drag).
      if (e.shiftKey && selectionAnchorRef.current !== null) {
        setLockedPosition(null);
        applySelectRangeFrom(selectionAnchorRef.current, tok.position, selectedTokenPositions, 'select');
        setSelectedPosition(tok.position);
        return;
      }
      const shouldSelect = !isInSelection;
      if (shouldSelect && cannotAdd) {
        setLimitPulse((c) => c + 1);
        return;
      }
      // Clicking an unexplained chip during the selection phase counts
      // as a "click that's not the locked token" — unlock so the user
      // can see the details for whatever they hover next.
      setLockedPosition(null);
      const mode: 'select' | 'deselect' = shouldSelect ? 'select' : 'deselect';
      dragStartPositionRef.current = tok.position;
      preDragSelectionRef.current = selectedTokenPositions;
      setDragMode(mode);
      applyDragRange(tok.position, mode);
      setSelectedPosition(mode === 'select' ? tok.position : null);
      selectionAnchorRef.current = tok.position;
    };

    const handleMouseEnter = () => {
      if (dragMode !== null && isSelectionPhase && !isExplained) {
        applyDragRange(tok.position, dragMode);
        setSelectedPosition(dragMode === 'select' ? tok.position : null);
        return;
      }
      // Always update focus on hover — locked (already-explained) chips
      // still get hover focus styling so the visual feedback works even
      // when the details panel is pinned to a different token.
      setSelectedPosition(tok.position);
    };

    // Tour hook: tag the "?" token (position 48) of the Llama's Lie demo
    // with a stable id so `nla-tour.tsx` can spotlight it in step 3.
    const chipDomId =
      activeDemoCacheId === NLA_TOUR_LLAMA_LIE_CACHE_ID && tok.position === NLA_TOUR_LLAMA_LIE_QUESTION_POSITION
        ? NLA_TOUR_LLAMA_LIE_QUESTION_ELEMENT_ID
        : undefined;
    const chipEl = (
      <span
        role="button"
        tabIndex={0}
        id={chipDomId}
        data-token-position={tok.position}
        aria-pressed={isSelectionPhase ? isInSelection : undefined}
        onMouseDown={handleMouseDown}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={() => {
          // Clear the focused token (and the details column) when the
          // cursor leaves a chip — but not mid-drag, since the next chip's
          // mouseenter will set the new focus.
          if (dragMode === null) setSelectedPosition(null);
        }}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            handleActivateKeyboard();
          }
        }}
        className={`relative font-serif transition-colors duration-300 ease-out ${
          small ? 'px-[1px] py-0 text-[9px] leading-none' : nonSmallSizeClass
        } ${colorClass} ${isInSelection && isSelectionPhase && !isExplained && !chipIsGenerating ? '[animation:chip-amber-fade-in_700ms_ease-out_both]' : ''} border-b-[1.5px] ${
          isExplanationSearchHit
            ? lockedPosition === tok.position
              ? 'z-20 rounded border-b-transparent !bg-emerald-200 !ring-2 !ring-emerald-600'
              : 'z-[1] rounded border-b-transparent !bg-emerald-200'
            : showFullFocus
              ? 'z-10 rounded !bg-sky-200 ring-[1.5px] ring-sky-600'
              : showSoftFocus
                ? 'z-10 rounded !bg-sky-100 ring-[1.5px] ring-sky-300'
                : isBusy
                  ? ''
                  : !isInSelection
                    ? 'hover:bg-amber-100 hover:text-amber-600'
                    : 'hover:bg-amber-300 hover:text-amber-800'
        } ${isBusy && !isExplained ? 'cursor-default' : cannotAdd ? 'cursor-not-allowed' : 'cursor-pointer'}`}
      >
        {tok.token.trim() === '' ? (
          hasNewline ? (
            <span className="opacity-35">{'\u21B5'.repeat(newlineCount)}</span>
          ) : (
            <span className="text-slate-300">{'\u00B7'}</span>
          )
        ) : (
          <>
            {tok.token.startsWith(' ') && <span className="text-white">{'\u00B7'}</span>}
            {/* Replace newlines with ↵ BEFORE trimming. The `↵` glyph is
                not whitespace, so trim() only strips edge spaces and
                preserves the newline marker — important for tokenizers
                that emit content+trailing-newline as a single token
                (e.g. Llama 3's "Hi!\n" rather than "Hi!" + "\n"). */}
            {tok.token.replaceAll('\n', '\u21B5').trim()}
            {tok.token.endsWith(' ') && <span className="text-white">{'\u00B7'}</span>}
          </>
        )}
        {showLoadingSweep && (
          <span
            aria-hidden
            className={`pointer-events-none absolute z-10 h-[1.5px] rounded-full bg-slate-500 [animation:chip-loading-sweep_1.6s_linear_infinite] ${small ? '-bottom-1' : 'bottom-0'}`}
          />
        )}
        {isScored && !showFullFocus && !showSoftFocus && (
          <span
            aria-hidden
            className={`pointer-events-none absolute left-[1px] right-[1px] h-[1.5px] ${small ? '-bottom-0.5' : 'bottom-0'} ${
              isLowConfidence ? 'bg-sky-500' : 'bg-sky-500'
            }`}
          />
        )}
      </span>
    );

    const isPendingSelection = pendingPositions.has(tok.position);
    // Match `components/shadcn/tooltip.tsx`: Tooltip wraps the trigger in
    // `span.inline-block ${className}`. Busy/pending chips skip CustomTooltip but must use the same
    // outer classes (e.g. header/footer flex alignment) so selection does not jump layout.
    const tooltipTriggerWrapperClass = tooltipTriggerClassName
      ? `inline-block ${tooltipTriggerClassName}`
      : 'inline-block';
    // Skip tooltip only for tokens still queued for explain (amber selection);
    // explained tokens stay in selectedTokenPositions but are not "pending".
    if (isBusy || isPendingSelection) {
      return (
        <Fragment key={tok.position}>
          <span className={tooltipTriggerWrapperClass}>{chipEl}</span>
        </Fragment>
      );
    }

    return <span className={tooltipTriggerWrapperClass}>{chipEl}</span>;

    // return (
    //   <Fragment key={tok.position}>
    //     <CustomTooltip delayDuration={400} side="top" trigger={chipEl} minMargin className={tooltipTriggerClassName}>
    //       <div className="flex flex-col items-center justify-center text-[10px] font-medium leading-normal text-slate-500">
    //         {isExplained ? (
    //           'Click to lock explanation details or share.'
    //         ) : (
    //           <>
    //             Click to select for explanation.
    //             <br />
    //             Click and drag to select multiple tokens.
    //           </>
    //         )}
    //       </div>
    //     </CustomTooltip>
    //   </Fragment>
    // );
  }

  function renderTokenGroup(
    group: {
      role: 'user' | 'assistant';
      headerTokens: TokenInfo[];
      contentTokens: TokenInfo[];
      footerTokens: TokenInfo[];
    },
    opts?: { preview?: boolean },
  ) {
    const preview = opts?.preview ?? false;
    // Render tokens as a sequence of per-line rows (one `flex flex-row
    // flex-wrap` per visual line). Splitting newline-delimited runs into
    // separate rows means each row's natural width is just its widest line,
    // so a short two-line message no longer expands the bubble to its
    // 85% max — it stays as wide as the longest line. The bubble's overall
    // width is determined by the widest row.
    //
    // `allowBlankLines` controls how consecutive newlines are handled:
    //  - true  (user content): every newline emits a row break, and any
    //    extras within a single token render as `h-[14px]` blank rows.
    //  - false (assistant content + header/footer special tokens): runs
    //    of consecutive newlines (within or across tokens) collapse to a
    //    single break — the ↵ glyphs inside each chip still show, but we
    //    never render extra blank rows. Keeps models that emit "\n\n\n"
    //    from blowing up the visual layout, and keeps gemma-style
    //    "<start_of_turn>model\n" headers compact.
    const isAssistant = group.role === 'assistant';
    function buildLines(
      tokens: TokenInfo[],
      small: boolean,
      allowBlankLines: boolean,
      /** Pre-tokenization preview: one chip may contain real line breaks via CSS; do not split flex rows on `\n`. */
      collapseNewlinesForRowLayout?: boolean,
      /** Radix Tooltip trigger wrapper classes (header/footer strips align differently in flex rows). */
      tooltipTriggerClassName?: string,
    ): { rows: React.ReactNode[][]; blankRowsAfter: number[] } {
      const rows: React.ReactNode[][] = [];
      const blankRowsAfter: number[] = [];
      let currentRow: React.ReactNode[] = [];
      let prevWrapped = false;
      for (const tok of tokens) {
        currentRow.push(renderChip(tok, small, preview, tooltipTriggerClassName));
        const rawNl = (tok.token.match(/\n/g) || []).length;
        const tokNewlineCount = collapseNewlinesForRowLayout ? 0 : rawNl;
        let breaks: number;
        if (tokNewlineCount === 0) {
          breaks = 0;
          prevWrapped = false;
        } else if (allowBlankLines) {
          breaks = tokNewlineCount;
          prevWrapped = true;
        } else if (prevWrapped) {
          breaks = 0;
        } else {
          breaks = 1;
          prevWrapped = true;
        }
        if (breaks > 0) {
          rows.push(currentRow);
          blankRowsAfter.push(breaks - 1);
          currentRow = [];
        }
      }
      if (currentRow.length > 0) {
        rows.push(currentRow);
        blankRowsAfter.push(0);
      }
      return { rows, blankRowsAfter };
    }

    function renderLines(lines: { rows: React.ReactNode[][]; blankRowsAfter: number[] }, small: boolean) {
      const out: React.ReactNode[] = [];
      const rowLeading = small ? 'leading-none' : 'leading-none sm:leading-tight';
      lines.rows.forEach((row, i) => {
        out.push(
          <div key={`r${i}`} className={`flex flex-row flex-wrap items-start gap-x-[0px] gap-y-0 ${rowLeading}`}>
            {row}
          </div>,
        );
        for (let j = 0; j < lines.blankRowsAfter[i]; j += 1) {
          out.push(<div key={`b${i}-${j}`} className="h-[14px]" />);
        }
      });
      return out;
    }

    const headerLines = buildLines(
      group.headerTokens,
      true,
      false,
      undefined,
      'flex items-start justify-start mb-[1px]',
    );
    const footerLines = buildLines(group.footerTokens, true, false, undefined, 'flex justify-end items-end mt-1.5');
    const contentLines = buildLines(group.contentTokens, false, !isAssistant, preview);
    return (
      <div className="flex flex-col items-start gap-y-0 px-1 py-1.5">
        {showChatTokens && group.headerTokens.length > 0 && <>{renderLines(headerLines, true)}</>}
        {group.contentTokens.length > 0 && <>{renderLines(contentLines, false)}</>}
        {showChatTokens && group.footerTokens.length > 0 && <>{renderLines(footerLines, true)}</>}
      </div>
    );
  }

  function stripTrailingTurnEnd(content: string): string {
    if (content.endsWith(tokenizerFormat.turnEndToken)) {
      return content.slice(0, -tokenizerFormat.turnEndToken.length);
    }
    return content;
  }

  async function sendMessage() {
    const trimmed = typedText.trim();
    if (!trimmed || isChatStreaming || isLoading) return;
    setError(null);
    onUserEdit();

    // If the user had pending tokens selected (the "X Tokens Selected for
    // Explanation" panel was up) and chose to send another message instead
    // of clicking Explain, drop those pending picks so they don't carry
    // over visually into the new turn. Already-explained positions are
    // preserved (their chips are locked anyway).
    if (pendingPositions.size > 0) {
      const preservedSelection = new Set<number>();
      explainedPositions.forEach((p) => preservedSelection.add(p));
      onApplySelection(preservedSelection);
    }

    // Drop any empty turns (e.g. the manual-mode default `{user: ''}` stub)
    // before appending the user turn and an empty assistant turn that the
    // stream will fill in token-by-token.
    const cleaned = chatMessages.filter((m) => m.content.length > 0);
    const userTurn: ChatMessage = { role: 'user', content: trimmed };
    const assistantTurn: ChatMessage = { role: 'assistant', content: '' };
    const nextMessages: ChatMessage[] = [...cleaned, userTurn, assistantTurn];
    // Snapshot the pre-send tokenList so a user-cancelled abort can
    // restore prior turns' chip rendering (the streaming overwrites
    // tokenList in place, so we lose the prior tokenization without
    // this).
    const tokenListBefore = tokenList;
    setChatMessages(nextMessages);
    setTypedText('');
    setIsChatStreaming(true);
    // Don't wipe `tokenList` here — keep the previously-tokenized turns on
    // screen so they don't flicker into "Thinking…" placeholders while we
    // wait for the SSE 'prompt' event. Those tokens' positions are stable:
    // the new prompt event will return a list that starts with the same
    // tokens (the previous conversation) and just appends the new user
    // turn + the start of the assistant turn.
    setSelectedPosition(null);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch('/api/nla/completion', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: tokenizerFormat.formatChat(nextMessages),
          // Sent in addition to `text` so the route can forward to a
          // chat-completion provider (e.g. OpenRouter) that needs
          // structured messages rather than the chat-templated string.
          // The NLA inference server ignores this field.
          messages: [...cleaned, userTurn],
          completion_tokens: maxNewTokens,
          temperature,
          stream: true,
          modelId: modelId || undefined,
          nlaSourceId: nlaSourceId || undefined,
        }),
        signal: controller.signal,
      });

      const remainingHeader = res.headers.get('x-limit-remaining');
      if (remainingHeader !== null) {
        setLimitRemaining(Number(remainingHeader));
      }

      if (res.status === 429) {
        // Two flavors hit this branch (see /api/nla/completion route +
        // top-level rate-limit middleware): the middleware response body
        // carries `limitPerWindow`, while an NLA inference-server 429
        // (busy) just propagates `{ error: "NLA server error: 429 - …" }`.
        // Surface the appropriate message; the catch handler below will
        // roll back the optimistic user/assistant turns and restore the
        // typed text either way.
        let isRateLimit = false;
        try {
          const body = await res.clone().json();
          isRateLimit = typeof body?.limitPerWindow === 'number';
        } catch {
          // body wasn't JSON — treat as NLA server busy
        }
        throw new Error(
          isRateLimit
            ? 'Hourly message limit reached. Please wait a bit and try again later.'
            : 'Servers busy - please try again later.',
        );
      }

      if (!res.ok || !res.body) {
        const errText = await res.text().catch(() => '');
        throw new Error(`Chat error: ${res.status} ${errText}`);
      }

      const reader = res.body
        .pipeThrough(new TextDecoderStream())
        .pipeThrough(new EventSourceParserStream())
        .getReader();

      let accumulatedText = '';
      let accumulatedTokens: TokenInfo[] = [];

      while (true) {
        // eslint-disable-next-line no-await-in-loop
        const { done, value } = await reader.read();
        if (done) break;
        if (!value || !value.data) continue;
        if (value.data === '[DONE]') break;
        let parsed: {
          type?: string;
          tokens?: TokenInfo[];
          token?: TokenInfo;
          prompt_length?: number;
          text?: string;
        } | null = null;
        try {
          parsed = JSON.parse(value.data);
        } catch {
          continue;
        }
        if (!parsed) continue;
        if (parsed.type === 'prompt' && Array.isArray(parsed.tokens)) {
          accumulatedTokens = [...parsed.tokens];
          setTokenList(accumulatedTokens);
        } else if (parsed.type === 'token' && parsed.token && typeof parsed.token.token === 'string') {
          accumulatedText += parsed.token.token;
          accumulatedTokens = [...accumulatedTokens, parsed.token];
          setTokenList(accumulatedTokens);
          const cleanedText = stripTrailingTurnEnd(accumulatedText);
          setChatMessages((prev) => {
            if (prev.length === 0) return prev;
            const arr = [...prev];
            arr[arr.length - 1] = { role: 'assistant', content: cleanedText };
            return arr;
          });
        }
      }

      const finalText = stripTrailingTurnEnd(accumulatedText);
      setChatMessages((prev) => {
        if (prev.length === 0) return prev;
        const arr = [...prev];
        arr[arr.length - 1] = { role: 'assistant', content: finalText };
        return arr;
      });

      // Re-tokenize the canonical formatted chat so the positions match what
      // /explain will see on its end (the streamed tokenList may include
      // trailing turn-end / newline tokens that aren't part of the canonical
      // tokenization of `formatChat(messages)`). Also stash it as
      // lastTokenizedText so a subsequent Explain click skips the redundant
      // tokenize round-trip.
      const finalMessages: ChatMessage[] = [...cleaned, userTurn, { role: 'assistant', content: finalText }];
      const canonicalText = tokenizerFormat.formatChat(finalMessages);
      try {
        const tokRes = await fetch('/api/nla/completion', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: canonicalText,
            completion_tokens: 0,
            modelId: modelId || undefined,
            nlaSourceId: nlaSourceId || undefined,
          }),
          signal: controller.signal,
        });
        const tokRemainingHeader = tokRes.headers.get('x-limit-remaining');
        if (tokRemainingHeader !== null) {
          setLimitRemaining(Number(tokRemainingHeader));
        }
        if (tokRes.ok) {
          const data = (await tokRes.json()) as { tokens?: TokenInfo[] };
          if (Array.isArray(data.tokens)) setTokenList(data.tokens);
        }
      } catch {
        // ignore — fall back to the streamed tokens we already have
      }
      setLastTokenizedText(canonicalText);
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        // User cancelled mid-stream. Roll back to the pre-send state
        // (drop the partially-streamed assistant turn and the user turn
        // we optimistically appended) and put the message back into the
        // input so the user can edit/retry. Also disarm the auto-explain
        // trigger that was armed at stream start — without this, the
        // explain pipeline would fire on the rolled-back conversation as
        // soon as `isChatStreaming` flips back to false.
        cancelPendingAutoExplain();
        setChatMessages((prev) => {
          let next = prev;
          if (next.length > 0 && next[next.length - 1].role === 'assistant') {
            next = next.slice(0, -1);
          }
          if (next.length > 0 && next[next.length - 1].role === 'user' && next[next.length - 1].content === trimmed) {
            next = next.slice(0, -1);
          }
          return next;
        });
        setTypedText(trimmed);
        setTokenList(tokenListBefore);
      } else {
        setError(err instanceof Error ? err.message : 'Chat error');
        // Roll back the optimistic user + empty-assistant turns we appended
        // before the request, and restore the typed text so the user can
        // edit/retry without losing their message.
        setChatMessages((prev) => {
          let next = prev;
          if (next.length > 0 && next[next.length - 1].role === 'assistant' && next[next.length - 1].content === '') {
            next = next.slice(0, -1);
          }
          if (next.length > 0 && next[next.length - 1].role === 'user' && next[next.length - 1].content === trimmed) {
            next = next.slice(0, -1);
          }
          return next;
        });
        setTypedText(trimmed);
      }
    } finally {
      setIsChatStreaming(false);
      abortRef.current = null;
    }
  }

  function handleStop() {
    abortRef.current?.abort();
  }

  // Chat messages we actually want to render. Drop the manual-mode empty
  // default stub; keep the trailing empty assistant during streaming so we
  // can show its bubble filling up. We pair each visible message with its
  // index in `chatMessages` so the per-message edit button can map back
  // to the canonical position when truncating.
  const visibleMessages = chatMessages
    .map((msg, idx) => ({ msg, originalIdx: idx }))
    .filter(({ msg, originalIdx }) => {
      if (msg.content.length > 0) return true;
      return msg.role === 'assistant' && originalIdx === chatMessages.length - 1 && isChatStreaming;
    });
  const hasMessages = visibleMessages.length > 0;
  // Map a visible-message index to its corresponding token group. The
  // tokenizer was fed only the cleaned messages, so the indices line up
  // 1:1 with `visibleMessages` entries that have content; entries that are
  // still empty (the streaming-assistant placeholder when the prompt event
  // hasn't arrived yet) just don't have a group and fall back to the
  // text-bubble rendering.
  const groupForIndex = (idx: number) => tokenGroups?.messages?.[idx];

  // Live count of selected tokens that haven't finished being explained
  // yet. A token counts as explained the moment either (a) the streaming
  // partial has emitted </explanation> or (b) the result event has
  // landed — same rule the chip styling uses, so the caption stays in
  // sync with the chip-level "amber → white" flip.
  const explanationsRemaining = useMemo(() => {
    let n = 0;
    selectedTokenPositions.forEach((p) => {
      if (resultMap.has(p)) return;
      const partial = partialMap.get(p);
      if (partial !== undefined && partial.includes('</explanation>')) return;
      n += 1;
    });
    return n;
  }, [selectedTokenPositions, resultMap, partialMap]);
  // Live count of selected tokens whose score (mse) hasn't landed yet.
  // Always >= explanationsRemaining since scoring follows explanation.
  const scoresRemaining = useMemo(() => {
    let n = 0;
    selectedTokenPositions.forEach((p) => {
      const r = resultMap.get(p);
      if (!r || r.mse === null) n += 1;
    });
    return n;
  }, [selectedTokenPositions, resultMap]);

  // Each token contributes two units of progress: one for its
  // explanation, and one for its score (mse). The explanation unit is
  // fractional — it advances continuously as the partial text streams
  // in, using a `~EXPLANATION_TOKEN_ESTIMATE`-token assumption (with
  // tokens estimated from char count). Without this, the bar would sit
  // motionless during the bulk of each run and only twitch at the
  // result-event moments. The fractional estimate is capped at <1 until
  // the explanation truly finishes (</explanation> in partial OR result
  // event), so a slow stream can never overshoot. The score unit stays
  // boolean (0 or 1) since scoring is atomic from the UI's POV.
  //
  // Performance note: this recomputes on every partialMap update, which
  // ticks per generated token across up to MAX_TOKENS_TO_EXPLAIN parallel
  // streams. The work is O(selectedTokenPositions.size) ≤ 32 with two
  // map lookups per iteration — trivially fast on any modern machine.
  // React 18 auto-batches the async setState calls in the streaming
  // loop, so render counts roughly match the human-perceptible token
  // emission rate rather than every individual update.
  const explanationStepsCompleted = useMemo(() => {
    const explanationCharBudget = EXPLANATION_TOKEN_ESTIMATE * CHARS_PER_TOKEN_ESTIMATE;
    let n = 0;
    selectedTokenPositions.forEach((p) => {
      const r = resultMap.get(p);
      const partial = partialMap.get(p);
      const explanationDone = !!r || (partial !== undefined && partial.includes('</explanation>'));
      if (explanationDone) {
        n += 1;
      } else if (partial !== undefined) {
        n += Math.min(0.999, partial.length / explanationCharBudget);
      }
    });
    return n;
  }, [selectedTokenPositions, resultMap, partialMap]);
  // Score progress is atomic per token: 1 unit when the mse result lands.
  const scoreStepsCompleted = useMemo(() => {
    let n = 0;
    selectedTokenPositions.forEach((p) => {
      const r = resultMap.get(p);
      if (r && r.mse !== null) n += 1;
    });
    return n;
  }, [selectedTokenPositions, resultMap]);

  // Snapshot the number of tokens (and their already-completed steps) at
  // the moment the explain stream begins so the progress bar's denominator
  // stays fixed across the run AND the bar always starts from 0 for THIS
  // run — the live pending count would tick down as results arrive, and
  // already-explained tokens that remain in `selectedTokenPositions` from
  // a prior run would otherwise inflate the completed-step counts and peg the bar
  // at max. Both snapshots MUST be taken off the same `false → true`
  // transition (and reset off the same `true → false` transition), so
  // they live in a single effect with a single guard ref.
  const [tokensSubmittedTotal, setTokensSubmittedTotal] = useState(0);
  const [explanationCompletedAtStart, setExplanationCompletedAtStart] = useState(0);
  const [scoreCompletedAtStart, setScoreCompletedAtStart] = useState(0);
  const prevLoadingForTotalRef = useRef(false);
  useEffect(() => {
    if (isLoading && !prevLoadingForTotalRef.current) {
      let total = 0;
      let explCompleted = 0;
      let scoreCompleted = 0;
      selectedTokenPositions.forEach((p) => {
        const r = resultMap.get(p);
        if (!r) total += 1;
        if (r) explCompleted += 1;
        if (r && r.mse !== null) scoreCompleted += 1;
      });
      setTokensSubmittedTotal(total);
      setExplanationCompletedAtStart(explCompleted);
      setScoreCompletedAtStart(scoreCompleted);
    }
    if (!isLoading && prevLoadingForTotalRef.current) {
      setTokensSubmittedTotal(0);
      setExplanationCompletedAtStart(0);
      setScoreCompletedAtStart(0);
    }
    prevLoadingForTotalRef.current = isLoading;
    // We snapshot only at the loading transition; reading live state inside
    // is intentional.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isLoading]);

  const explanationStepsThisRun = Math.max(0, explanationStepsCompleted - explanationCompletedAtStart);
  const scoreStepsThisRun = Math.max(0, scoreStepsCompleted - scoreCompletedAtStart);
  const explanationProgressFraction =
    tokensSubmittedTotal > 0 ? Math.min(1, Math.max(0, explanationStepsThisRun / tokensSubmittedTotal)) : 0;
  const scoreProgressFraction =
    tokensSubmittedTotal > 0 ? Math.min(1, Math.max(0, scoreStepsThisRun / tokensSubmittedTotal)) : 0;
  const isGeneratingDone = explanationsRemaining === 0;

  const handleClearPending = () => {
    // Drop only the pending positions; keep the already-explained ones in
    // the selection set (their chips are locked, so this just hides the
    // selection styling for the unexplained ones the user picked).
    const next = new Set<number>();
    explainedPositions.forEach((p) => next.add(p));
    onApplySelection(next);
    setError(null);
  };

  const handleExplainPending = () => {
    if (pendingPositions.size === 0) {
      setError('Select at least one token to explain.');
      return;
    }
    setError(null);
    onExplain();
  };

  const isBlankChat = !hasMessages && !isChatStreaming && !isExplainInFlight;

  const handleEditUserMessage = (originalIdx: number, content: string) => {
    if (isChatStreaming || isLoading) return;
    // eslint-disable-next-line no-alert
    const confirmed = window.confirm(
      'Are you sure you want to edit this message? It will clear the rest of the chat messages.',
    );
    if (!confirmed) return;
    onUserEdit();
    truncateChatFrom(originalIdx);
    setTypedText(content);
    setError(null);
  };

  // Tracks which user message is currently showing the "copied" checkmark
  // confirmation. Cleared on a short timer so the icon flicks back to the
  // copy glyph without further interaction.
  const [copiedMessageIdx, setCopiedMessageIdx] = useState<number | null>(null);
  useEffect(() => {
    if (copiedMessageIdx === null) return undefined;
    const t = setTimeout(() => setCopiedMessageIdx(null), 1500);
    return () => clearTimeout(t);
  }, [copiedMessageIdx]);

  const handleCopyMessage = async (originalIdx: number, content: string) => {
    try {
      if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(content);
      }
      setCopiedMessageIdx(originalIdx);
    } catch {
      // ignore — clipboard access may be denied in some contexts
    }
  };

  return (
    <div
      className={`relative flex min-h-0 flex-1 flex-col ${isEmbed ? 'rounded-none bg-slate-50 p-2 pb-2' : 'rounded-xl bg-slate-50 pb-2 pt-0 sm:px-4'}`}
    >
      {/* Animated spacer that pushes the status banner down to roughly the
          vertical center while the chat is blank, then collapses to 0 as
          soon as the user sends a message — letting the banner glide back
          to the top via the same height transition. */}
      <div
        aria-hidden
        className={`shrink-0 overflow-hidden transition-[height] duration-500 ease-out ${isBlankChat ? 'h-[25px] sm:h-[20dvh]' : 'h-0'}`}
      />
      <div
        id="nla-chat-status"
        className={`sticky top-0 mt-0 flex h-10 min-h-10 w-full flex-col items-center justify-center gap-y-1.5 rounded-md border-slate-700/30 px-2 py-0 transition-colors sm:h-[54px] sm:min-h-[54px] sm:rounded-xl sm:px-3.5 ${
          !isChatStreaming &&
          !isExplainInFlight &&
          !isHydratingDemo &&
          hasMessages &&
          (pendingPositions.size > 0 || explainedPositions.size === 0)
            ? 'bg-amber-300/30'
            : 'bg-sky-500/10'
        }`}
      >
        {isHydratingDemo ? (
          <div className="flex flex-row items-center justify-center gap-x-2">
            <LoadingSquare size={20} />
            <span className="text-[12px] font-medium text-slate-600">Loading…</span>
          </div>
        ) : isChatStreaming ? (
          <div className="flex flex-row items-center justify-center gap-x-2">
            <LoadingSquare size={20} />
            <span className="text-[12px] font-medium text-slate-600">Assistant Responding...</span>
          </div>
        ) : isExplainInFlight ? (
          <div className="flex w-full flex-row divide-x divide-sky-800/20">
            <div
              className={`flex flex-1 flex-col items-center gap-y-0 px-2 transition-opacity ${
                isGeneratingDone ? 'opacity-50' : 'opacity-100'
              }`}
            >
              <span className="text-[11.5px] font-medium text-slate-600">
                {isGeneratingDone ? 'Explanations Generated' : 'Generating Explanations...'}
              </span>
              {tokensSubmittedTotal > 0 && (
                <Slider.Root
                  value={[isGeneratingDone ? 100 : Math.max(Math.round(explanationProgressFraction * 100), 10)]}
                  min={0}
                  max={100}
                  step={1}
                  disabled
                  aria-label="Explanation progress"
                  className="relative mb-0.5 flex h-[6px] w-[70%] max-w-[70%] cursor-default items-center"
                >
                  <Slider.Track className="relative h-full grow overflow-hidden rounded-full border border-sky-600 bg-sky-100">
                    <Slider.Range className="absolute h-full rounded-full bg-sky-600 transition-all duration-200 ease-out" />
                  </Slider.Track>
                </Slider.Root>
              )}
              {explanationsRemaining > 0 && (
                <div className="flex flex-col items-center text-[10px] font-medium text-sky-800/50">
                  <span>
                    {explanationsRemaining} token{explanationsRemaining === 1 ? '' : 's'} left to explain
                  </span>
                </div>
              )}
            </div>
            <div
              className={`flex flex-1 flex-col items-center gap-y-0 px-2 transition-opacity ${
                isGeneratingDone ? 'opacity-100' : 'opacity-50'
              }`}
            >
              <span className="text-[11.5px] font-medium text-slate-600">Scoring Explanations...</span>
              {tokensSubmittedTotal > 0 && (
                <Slider.Root
                  value={[isGeneratingDone ? Math.max(Math.round(scoreProgressFraction * 100), 10) : 0]}
                  min={0}
                  max={100}
                  step={1}
                  disabled
                  aria-label="Scoring progress"
                  className="relative mb-0.5 flex h-[6px] w-[70%] max-w-[70%] cursor-default items-center"
                >
                  <Slider.Track className="relative h-full grow overflow-hidden rounded-full border border-sky-600 bg-sky-100">
                    <Slider.Range className="absolute h-full rounded-full bg-sky-600 transition-all duration-200 ease-out" />
                  </Slider.Track>
                </Slider.Root>
              )}
              {scoresRemaining > 0 && (
                <div className="flex flex-col items-center text-[10px] font-medium text-sky-800/50">
                  <span>
                    {scoresRemaining} token{scoresRemaining === 1 ? '' : 's'} left to score
                  </span>
                </div>
              )}
            </div>
          </div>
        ) : !hasMessages ? (
          <div className="flex w-full flex-row items-center justify-center gap-x-2 divide-x divide-sky-800/20">
            <div className="flex flex-1 flex-col items-center justify-center gap-y-0 sm:px-4">
              <span className="text-[11px] font-medium text-sky-700 sm:text-[14px]">Start Chat</span>
              <span className="text-center text-[9px] leading-tight text-sky-600 sm:text-[11px] sm:leading-normal">
                Set up a scenario to analyze the model&apos;s thoughts.
              </span>
            </div>
            <div className="flex flex-1 flex-col items-center justify-center gap-y-0 px-4">
              <span className="text-[11px] font-medium text-sky-700 sm:text-[14px]">Choose Demo</span>
              <span className="text-center text-[9px] leading-tight text-sky-600 sm:text-[11px] sm:leading-normal">
                Click one of the demos at the top to load a scenario.
              </span>
            </div>
          </div>
        ) : pendingPositions.size === 0 && explainedPositions.size > 0 ? (
          <div className="flex w-full flex-col items-center justify-center gap-y-0.5">
            <div className="flex w-full flex-row sm:divide-x sm:divide-sky-800/20">
              <div className="hidden flex-1 flex-col gap-y-0 pr-2 text-center sm:flex">
                <span className="text-[12px] font-medium text-slate-700">See Explanations</span>
                <span className="text-[10.5px] font-medium text-slate-500">
                  Click{' '}
                  <span className="mr-0.5 border-b-[1.5px] border-sky-500 font-serif font-semibold">underlined</span>
                  <span className="border-b-[1.5px] border-sky-500 font-serif font-semibold">tokens</span>.
                </span>
              </div>
              <div className="hidden flex-1 flex-col gap-y-0 px-2 text-center sm:flex">
                <span className="text-[12px] font-medium text-slate-700">Explain Other Tokens</span>
                <span className="text-[10.5px] font-medium text-slate-500">
                  Click <span className="font-serif font-semibold text-slate-400">gray tokens</span> to explain.
                </span>
              </div>
              <div className="flex flex-1 flex-col items-center justify-center gap-y-0 pl-2 text-center">
                <div className="pointer-events-none flex w-[200px] max-w-[200px] flex-col items-center justify-center">
                  <div
                    className={`pointer-events-auto flex flex-row items-center justify-center gap-0.5 ${explanationSearchActive ? 'rounded-2xl' : 'rounded-full'}`}
                    onMouseDown={(e) => e.stopPropagation()}
                  >
                    <div className="group relative w-[170px] max-w-[170px] shrink-0">
                      <Search
                        className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-400 transition-colors group-focus-within:text-emerald-600"
                        strokeWidth={2}
                        aria-hidden
                      />
                      <input
                        type="search"
                        value={explanationSearchInput}
                        onChange={(e) => setExplanationSearchInput(e.target.value)}
                        placeholder="Search Explanations"
                        disabled={isBusy}
                        className={`box-border h-7 w-[170px] rounded-full border border-slate-200 bg-white py-0 pl-7 text-[11px] font-medium leading-none text-slate-700 outline-none placeholder:text-center placeholder:text-slate-400 focus:ring-1 focus:ring-emerald-500 disabled:cursor-not-allowed disabled:opacity-50 sm:h-8 ${
                          explanationSearchActive ? 'pr-9' : 'pr-2'
                        }`}
                        aria-label="Search explanations"
                      />

                      {explanationSearchActive && (
                        <span
                          className="pointer-events-none absolute right-2 top-1/2 z-10 -translate-y-1/2 text-[9px] tabular-nums text-slate-400"
                          aria-hidden
                        >
                          {explanationSearchMatches.length === 0
                            ? '0/0'
                            : `${
                                Math.min(
                                  Math.max(explanationSearchResultIndex, 0),
                                  explanationSearchMatches.length - 1,
                                ) + 1
                              }/${explanationSearchMatches.length}`}
                        </span>
                      )}
                    </div>
                    {explanationSearchActive && (
                      <div className="flex h-7 max-h-7 min-h-7 w-[40px] max-w-[40px] items-center gap-0 px-1 py-0.5">
                        <button
                          type="button"
                          className="ml-auto flex h-6 w-6 shrink-0 items-center justify-center rounded-l-full border-r border-slate-300 bg-slate-300 text-slate-500 transition-colors hover:bg-slate-400 hover:text-slate-700 disabled:cursor-not-allowed disabled:opacity-40"
                          aria-label="Previous explanation match"
                          disabled={explanationSearchMatches.length === 0}
                          onMouseDown={(e) => e.stopPropagation()}
                          onClick={() => goExplanationSearchPrev()}
                        >
                          <ChevronLeft className="h-4 w-4" />
                        </button>
                        <button
                          type="button"
                          className="flex h-6 w-6 shrink-0 items-center justify-center rounded-r-full bg-slate-300 text-slate-500 transition-colors hover:bg-slate-400 hover:text-slate-700 disabled:cursor-not-allowed disabled:opacity-40"
                          aria-label="Next explanation match"
                          disabled={explanationSearchMatches.length === 0}
                          onMouseDown={(e) => e.stopPropagation()}
                          onClick={() => goExplanationSearchNext()}
                        >
                          <ChevronRight className="h-4 w-4" />
                        </button>
                      </div>
                    )}
                  </div>
                </div>
                {/* <span className="text-[12px] font-medium text-slate-700">Continue Chat</span>
                <span className="text-[10.5px] font-medium text-slate-500">Chat to set up scenarios.</span> */}
              </div>
            </div>
          </div>
        ) : (
          <div className="flex w-full flex-row items-center justify-center gap-x-5 sm:divide-x sm:divide-amber-800/30">
            <div className="hidden flex-col sm:flex">
              <div className="flex flex-1 flex-col gap-y-0 px-2 text-center">
                <span className="text-[12px] font-medium text-amber-800">Select Tokens to Explain</span>
                <span className="text-[11px] font-medium text-amber-600">
                  Click <span className="font-serif font-semibold text-slate-500">gray tokens</span>, and use shift/drag
                  for multiple tokens.{' '}
                </span>
              </div>
            </div>
            <div className="flex flex-row">
              <div className="flex w-32 min-w-32 max-w-32 flex-row items-center justify-center gap-x-1 gap-y-0 text-wrap text-[11px] font-medium leading-snug text-amber-800 sm:flex-col">
                <span
                  className={`text-[12px] transition-colors sm:text-sm ${
                    limitPulse > 0 ? 'animate-pulse font-semibold text-red-600' : 'text-amber-800'
                  }`}
                >
                  {pendingPositions.size}
                  {selectionLimitReached ? ' (Max)' : ''}
                </span>
                Selected
              </div>

              <div className="flex flex-row items-center gap-x-1.5 gap-y-1">
                <button
                  type="button"
                  id={
                    activeDemoCacheId === NLA_TOUR_LLAMA_LIE_CACHE_ID
                      ? NLA_TOUR_LLAMA_LIE_EXPLAIN_ELEMENT_ID
                      : undefined
                  }
                  onClick={handleExplainPending}
                  disabled={
                    explainDisabled ||
                    isLoading ||
                    isChatStreaming ||
                    tokenList.length === 0 ||
                    pendingPositions.size === 0
                  }
                  className="flex h-7 items-center justify-center gap-x-1 rounded-md border border-sky-600 bg-sky-600 px-2.5 py-1 text-[12px] font-semibold text-white transition-colors hover:bg-sky-500/90 disabled:cursor-not-allowed disabled:opacity-50 sm:h-9 sm:px-4"
                >
                  <Play className="h-3.5 w-3.5" />
                  Explain
                </button>
                <button
                  type="button"
                  onClick={handleClearPending}
                  disabled={isLoading || isChatStreaming || pendingPositions.size === 0}
                  className="flex h-7 w-7 items-center justify-center rounded-md border-amber-100 bg-slate-300 text-[10.5px] font-medium text-slate-600 transition-colors hover:border-amber-500 hover:bg-amber-50 disabled:cursor-not-allowed disabled:opacity-50 sm:h-9 sm:w-9"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
      <div id="nla-chat-messages" className="relative min-h-0 flex-1">
        {isHydratingDemo ? (
          <div className="flex h-full min-h-0 flex-1 flex-col items-center justify-center gap-y-2 text-slate-400">
            <LoadingSquare size={20} />
            <span className="text-xs font-medium">Loading…</span>
          </div>
        ) : (
          <div
            ref={scrollRef}
            onMouseDown={() => setLockedPosition(null)}
            className="flex h-full max-h-full min-h-0 flex-1 flex-col gap-y-0 overflow-y-auto pt-1 sm:pb-3 sm:pr-1 sm:pt-4"
          >
            {visibleMessages.map(({ msg, originalIdx }, idx) => {
              const group = groupForIndex(idx);
              const showChips =
                !!group && (group.contentTokens.length > 0 || (showChatTokens && group.headerTokens.length > 0));
              const isUser = msg.role === 'user';
              const canEdit = isUser && msg.content.length > 0 && !isChatStreaming && !isLoading;
              return (
                <div key={originalIdx} className={`group flex w-full ${isUser ? 'justify-end' : 'justify-start'}`}>
                  <div
                    className={`flex max-w-[95%] flex-col gap-y-0 px-0.5 sm:max-w-[85%] ${isUser ? 'items-end' : 'items-start'}`}
                  >
                    {showChips ? (
                      <div
                        className={`select-none rounded-xl sm:px-3.5 ${isUser ? 'bg-white py-1 shadow' : 'bg-transparent py-0'}`}
                      >
                        {renderTokenGroup(group)}
                      </div>
                    ) : isUser && showChatTokens && !showChips && msg.content.length > 0 ? (
                      <div
                        className={`select-none rounded-xl sm:px-3.5 ${isUser ? 'bg-white py-1 shadow' : 'bg-transparent py-0'}`}
                      >
                        {renderTokenGroup(
                          buildSyntheticUserTurnPreviewGroup(
                            tokenizerFormat,
                            msg,
                            originalIdx,
                            chatMessages.length,
                            -1_000_000 - originalIdx * 16,
                          ),
                          { preview: true },
                        )}
                      </div>
                    ) : (
                      <div
                        className={`whitespace-pre-wrap rounded-xl py-2 font-serif text-[14px] font-medium leading-relaxed text-slate-400 sm:px-[18px] ${isUser ? 'bg-white shadow' : 'bg-transparent'}`}
                      >
                        {msg.content.length > 0 ? (
                          msg.content
                        ) : (
                          <span className="inline-flex items-center gap-1 text-slate-400">
                            <LoadingSquare size={14} />
                          </span>
                        )}
                      </div>
                    )}
                    {isUser && msg.content.length > 0 && (
                      <div className="mt-1 hidden items-center gap-x-1 sm:flex">
                        <button
                          type="button"
                          onClick={() => handleCopyMessage(originalIdx, msg.content)}
                          title="Copy message"
                          aria-label="Copy message"
                          className="flex h-5 w-5 items-center justify-center rounded-full text-slate-400 transition-colors hover:bg-slate-200 hover:text-slate-600"
                        >
                          {copiedMessageIdx === originalIdx ? (
                            <Check className="h-3 w-3 text-sky-600" />
                          ) : (
                            <Copy className="h-3 w-3" />
                          )}
                        </button>
                        <button
                          type="button"
                          onClick={() => handleEditUserMessage(originalIdx, msg.content)}
                          disabled={!canEdit}
                          title="Edit message (clears the rest of the chat)"
                          aria-label="Edit message"
                          className="flex h-5 w-5 items-center justify-center rounded-full text-slate-400 transition-colors hover:bg-slate-200 hover:text-slate-600 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-transparent disabled:hover:text-slate-400"
                        >
                          <Pencil className="h-3 w-3" />
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
            {!hasMessages && !isChatStreaming && (
              <>
                <div className="min-h-0 flex-1" aria-hidden />
                <div className="pointer-events-none flex shrink-0 flex-row justify-center">
                  <ArrowDown className="h-7 w-7 animate-bounce text-sky-600/70" strokeWidth={2.25} />
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {(error || providerError) && (
        <div className="mt-2 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-[11px] text-red-700">
          {error || providerError}
        </div>
      )}

      <div className="mt-0 flex flex-row items-end gap-x-2">
        <div className="relative flex flex-1 flex-col rounded-xl border border-sky-100 bg-white px-1.5 py-1.5 shadow-md">
          <ReactTextareaAutosize
            value={typedText}
            onChange={(e) => setTypedText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
              }
            }}
            minRows={3}
            maxRows={6}
            disabled={isBusy}
            placeholder="Type a message…"
            style={{ WebkitTapHighlightColor: 'transparent' }}
            className="w-full resize-none rounded-t-xl border-0 px-2.5 py-2 pr-11 text-[13px] leading-normal text-slate-800 placeholder-sky-600/60 outline-none ring-0 focus:outline-none focus:ring-0 disabled:placeholder-slate-300 sm:py-2.5"
          />
          <button
            type="button"
            onClick={() => {
              if (confirm('Are you sure you want to reset the chat?')) {
                setError(null);
                onClear();
              }
            }}
            disabled={isBusy || (!hasMessages && tokenList.length > 0)}
            title="Clear chat"
            className="absolute bottom-2 left-2 flex h-8 w-8 items-center justify-center rounded-md border-rose-200 bg-rose-100 text-rose-400 transition-colors hover:border-rose-600 hover:bg-rose-200 hover:text-rose-600 disabled:cursor-not-allowed disabled:border-slate-200 disabled:bg-slate-100 disabled:text-slate-400 disabled:opacity-50 disabled:hover:border-slate-200 disabled:hover:bg-slate-100 sm:hidden"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </button>
          <div className="mb-2 hidden w-full flex-wrap items-center justify-between gap-x-3 gap-y-2 px-2 sm:flex">
            <NLAInputChatControls
              isBusy={isBusy}
              hasMessages={hasMessages}
              hasTokens={tokenList.length > 0}
              onClearChat={() => {
                if (confirm('Are you sure you want to reset the chat?')) {
                  setError(null);
                  onClear();
                }
              }}
              showAdvanced={showAdvanced}
              setShowAdvanced={setShowAdvanced}
              topLevelMode={topLevelMode}
              onTopLevelModeChange={onTopLevelModeChange}
              showChatTokens={showChatTokens}
              setShowChatTokens={setShowChatTokens}
            />

            <div className="flex flex-col items-end gap-y-0.5">
              {/* <button
                type="button"
                onClick={onShare}
                disabled={isBusy || !hasMessages}
                title="Share this chat"
                className="flex h-8 items-center gap-x-1.5 rounded-md border border-slate-200 bg-white px-2.5 text-[11px] font-medium text-slate-500 transition-colors hover:border-sky-300 hover:bg-sky-50 hover:text-sky-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                <Share2 className="h-3.5 w-3.5" />
                Share
              </button> */}

              <button
                type="button"
                onClick={() => (isChatStreaming ? handleStop() : sendMessage())}
                disabled={isExplainInFlight || (!isChatStreaming && !typedText.trim())}
                className="flex h-7 w-7 items-center justify-center rounded-full transition-colors disabled:cursor-not-allowed disabled:opacity-40"
              >
                {isChatStreaming ? (
                  <X className="h-7 w-7 rounded-lg bg-red-400 p-1.5 text-white transition-all hover:bg-red-500" />
                ) : (
                  <ArrowUp className="h-7 w-7 rounded-lg bg-sky-700 p-1.5 text-white transition-all hover:bg-sky-800" />
                )}
              </button>
              {limitRemaining !== null &&
                (() => {
                  // Each send issues 2 requests to /api/nla/completion
                  // (streaming chat + canonical re-tokenize), so divide by 2
                  // to show user-perceived messages remaining. Floor so the
                  // counter never overstates what's actually left.
                  const messagesLeft = Math.floor(limitRemaining / 2);
                  return messagesLeft > 0 ? (
                    <div className="absolute bottom-6 right-12 text-[9px] leading-none text-slate-500">
                      Hourly Limit Left: {messagesLeft}
                    </div>
                  ) : (
                    <div className="absolute bottom-6 right-12 text-[9px] leading-none text-slate-500">
                      Limit reached. Try again later.
                    </div>
                  );
                })()}
            </div>
          </div>
        </div>
      </div>

      {showAdvanced && (
        <NLAInputChatAdvanced
          isBusy={isBusy}
          temperature={temperature}
          setTemperature={setTemperature}
          maxNewTokens={maxNewTokens}
          setMaxNewTokens={setMaxNewTokens}
        />
      )}
    </div>
  );
}
