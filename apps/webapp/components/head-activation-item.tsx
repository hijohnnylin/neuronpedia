'use client';

import {
  ACTIVATION_PRECISION,
  LINE_BREAK_REPLACEMENT_CHAR,
  makeActivationBackgroundColorWithDFA,
  replaceHtmlAnomalies,
} from '@/lib/utils/activations';
import { cn } from '@/lib/utils/ui';
import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

export type HeadSequenceData = {
  id: string;
  layer: number;
  headIndex: number;
  interval: number;
  tokens: string[];
  attentionIndices: number[];
  attentionValues: number[];
  maxActivation: number;
};

const REGULAR_TOKEN_CLASSNAME = 'border-2 border-transparent hover:bg-slate-200';
const HOVERED_QUERY_TOKEN_CLASSNAME = 'border-2 border-orange-400';

const ATTENTION_ORANGE_RGB = '251, 146, 60'; // orange-400

// Logarithmic opacity ramp used for token backgrounds. Mirrors the curve in
// `makeActivationBackgroundColorWithDFA` so cells visually match the rest of
// the codebase.
function tokenGradientOpacity(value: number, max: number) {
  const MINIMUM_OPACITY = 0.05;
  const MINIMUM_THRESHOLD = 0.00005;
  if (!max || max <= 0 || value <= MINIMUM_THRESHOLD) return 0;
  const ratio = value / max;
  const scale = 1 - MINIMUM_OPACITY;
  const op = MINIMUM_OPACITY + (Math.log10(1 + 9 * ratio) * scale) / Math.log10(10);
  return Math.max(0, Math.min(1, op));
}

// Escapes newlines for tooltip display, and swaps whitespace-only tokens with
// non-breaking spaces so the chip keeps a normal character line-height
// (mirrors the displayContent fix used in the inline token render).
function formatChipToken(raw: string | undefined) {
  const text = raw?.replaceAll('\n', '\\n') ?? 'N/A';
  if (text.length > 0 && text.trim() === '') {
    return text.replace(/\s/g, '\u00A0');
  }
  return text;
}

const START_TOKEN = ['<|start|>', '<|begin_of_text|>', '<|start_header_id|>', '<start_of_turn>', '<|im_start|>'];
const END_TOKEN = ['<|end|>', '<|eot_id|>', '<|im_end|>', '<end_of_turn>'];
const MESSAGE_TOKEN = ['<|message|>', '<|end_header_id|>'];
const CHANNEL_TOKEN = '<|channel|>';
const BOS_TOKEN = ['<bos>'];

const CHAT_ROLE_NAMES = ['system', 'user', 'assistant'];

export default function HeadActivationItem({
  sequence,
  modelId,
  overallMaxActivationValueInList = sequence.maxActivation,
  tokensToDisplayAroundMaxActToken = 10000,
  showLineBreaks = false,
  enableExpanding = true,
  overrideTextSize = 'text-xs',
  overrideLeading = 'leading-none sm:leading-none ',
  overrideTextColor = 'text-slate-700',
  className,
  showRawTokens = true,
  maxAttentionMode = 'all',
}: {
  sequence: HeadSequenceData;
  modelId?: string;
  overallMaxActivationValueInList?: number;
  tokensToDisplayAroundMaxActToken?: number;
  showLineBreaks?: boolean;
  enableExpanding?: boolean;
  overrideTextSize?: string;
  overrideLeading?: string;
  overrideTextColor?: string;
  className?: string;
  showRawTokens?: boolean;
  maxAttentionMode?: 'all' | 'keys' | 'queries';
}) {
  const [currentRange, setCurrentRange] = useState(tokensToDisplayAroundMaxActToken);
  const [isExpanded, setIsExpanded] = useState(false);
  const [isItemHovered, setIsItemHovered] = useState(false);

  // `attentionIndices` are flat COO indices in a seqLen x seqLen attention matrix
  // (idx = q * seqLen + k, where seqLen is this sequence's token count). We
  // compute three things in one pass:
  //  1. denseAttentionValues: per-token (per-key) max attention (column-max),
  //     skipping row 0 and column 0 (the BOS/position-0 attention sink).
  //     Matches headvis's getColMax behavior.
  //  2. outgoingByQuery: for each query q, list of { index: k, value } pairs
  //     (sorted by value desc). Used to highlight where a hovered query attends.
  //  3. incomingByKey: for each key k, list of { index: q, value } pairs
  //     (sorted by value desc). Used in the per-token tooltip.
  const { denseAttentionValues, queryMaxValues, outgoingByQuery, incomingByKey } = useMemo(() => {
    // `attentionIndices` were encoded as `q * actual_len + k`, where `actual_len`
    // is the number of decoded tokens for this sequence, so `tokens.length` is
    // exactly the decode stride.
    const tokenLen = sequence.tokens.length;
    const seqLen = tokenLen;
    // Per-key column max (max attention each token receives as a key).
    const dense = new Array(tokenLen).fill(0);
    // Per-query row max (max attention each token emits as a query).
    const queryMax = new Array(tokenLen).fill(0);
    const outgoing = new Map<number, Array<{ index: number; value: number }>>();
    const incoming = new Map<number, Array<{ index: number; value: number }>>();
    if (seqLen > 0) {
      for (let i = 0; i < sequence.attentionIndices.length; i += 1) {
        const flatIdx = sequence.attentionIndices[i];
        const val = sequence.attentionValues[i];
        if (val === undefined || !Number.isFinite(val)) continue;
        const q = Math.floor(flatIdx / seqLen);
        const k = flatIdx % seqLen;
        if (q === 0 || k === 0) continue;
        if (k >= 0 && k < tokenLen && val > dense[k]) dense[k] = val;
        if (q >= 0 && q < tokenLen && val > queryMax[q]) queryMax[q] = val;
        if (!outgoing.has(q)) outgoing.set(q, []);
        outgoing.get(q)!.push({ index: k, value: val });
        if (!incoming.has(k)) incoming.set(k, []);
        incoming.get(k)!.push({ index: q, value: val });
      }
      outgoing.forEach((list) => list.sort((a, b) => b.value - a.value));
      incoming.forEach((list) => list.sort((a, b) => b.value - a.value));
    }
    return {
      denseAttentionValues: dense,
      queryMaxValues: queryMax,
      outgoingByQuery: outgoing,
      incomingByKey: incoming,
    };
  }, [sequence.tokens.length, sequence.attentionIndices, sequence.attentionValues]);

  // Separate normalization maxes so the Keys/Queries views shade each token
  // relative to the strongest key (column max) or query (row max) respectively.
  const keyMaxOverall = useMemo(() => {
    let max = 0;
    for (let i = 0; i < denseAttentionValues.length; i += 1) {
      if (denseAttentionValues[i] > max) max = denseAttentionValues[i];
    }
    return max;
  }, [denseAttentionValues]);

  const queryMaxOverall = useMemo(() => {
    let max = 0;
    for (let i = 0; i < queryMaxValues.length; i += 1) {
      if (queryMaxValues[i] > max) max = queryMaxValues[i];
    }
    return max;
  }, [queryMaxValues]);

  const maxActivationTokenIndex = useMemo(() => {
    let maxIdx = 0;
    let maxValue = -Infinity;
    for (let i = 0; i < denseAttentionValues.length; i += 1) {
      if (denseAttentionValues[i] > maxValue) {
        maxValue = denseAttentionValues[i];
        maxIdx = i;
      }
    }
    return maxIdx;
  }, [denseAttentionValues]);

  const [hoveredTokenIndex, setHoveredTokenIndex] = useState<number | null>(null);

  // The popup is anchored above the sequence row and centered. We render it in a
  // portal with fixed positioning (measured from the row) so it isn't clipped by
  // the list's scroll container when hovering tokens in the top-most sequence.
  const sequenceRef = useRef<HTMLDivElement>(null);
  const [popupPos, setPopupPos] = useState<{ left: number; top: number } | null>(null);

  // Per-token DOM refs (only populated for tokens currently rendered/in view) so
  // we can draw connector lines from the hovered token to its key/query tokens.
  const tokenRefs = useRef<Map<number, HTMLSpanElement>>(new Map());
  const [attentionLines, setAttentionLines] = useState<
    Array<{ x1: number; y1: number; x2: number; y2: number; color: string }>
  >([]);

  const SLATE_RGB = '100, 116, 139'; // slate-500

  // When the user hovers a token, treat it as the QUERY and show where it
  // attends (outgoing). For each candidate key tokenIndex, look up the
  // attention value from the hovered query.
  const hoveredOutgoing = useMemo(() => {
    const map = new Map<number, number>();
    if (hoveredTokenIndex === null) return map;
    const list = outgoingByQuery.get(hoveredTokenIndex);
    if (!list) return map;
    list.forEach(({ index, value }) => {
      const existing = map.get(index);
      if (existing === undefined || value > existing) map.set(index, value);
    });
    return map;
  }, [hoveredTokenIndex, outgoingByQuery]);

  const hoveredOutgoingMax = useMemo(() => {
    let max = 0;
    hoveredOutgoing.forEach((v) => {
      if (v > max) max = v;
    });
    return max;
  }, [hoveredOutgoing]);

  // Treat the hovered token as the KEY: which queries attend to it (incoming).
  const hoveredIncoming = useMemo(() => {
    const map = new Map<number, number>();
    if (hoveredTokenIndex === null) return map;
    const list = incomingByKey.get(hoveredTokenIndex);
    if (!list) return map;
    list.forEach(({ index, value }) => {
      const existing = map.get(index);
      if (existing === undefined || value > existing) map.set(index, value);
    });
    return map;
  }, [hoveredTokenIndex, incomingByKey]);

  const hoveredIncomingMax = useMemo(() => {
    let max = 0;
    hoveredIncoming.forEach((v) => {
      if (v > max) max = v;
    });
    return max;
  }, [hoveredIncoming]);

  useEffect(() => {
    if (hoveredTokenIndex === null) {
      setPopupPos(null);
      setAttentionLines([]);
      return undefined;
    }
    const update = () => {
      const el = sequenceRef.current;
      if (el) {
        const rect = el.getBoundingClientRect();
        setPopupPos({ left: rect.left, top: rect.top });
      }
      const hoveredEl = tokenRefs.current.get(hoveredTokenIndex);
      if (!hoveredEl) {
        setAttentionLines([]);
        return;
      }
      const hr = hoveredEl.getBoundingClientRect();
      const hx = hr.left + hr.width / 2;
      const hy = hr.top + hr.height / 2;
      const hw = hr.width / 2;
      const hh = hr.height / 2;
      // Returns the point on a box's boundary (centered at cx,cy with half-extents
      // halfW,halfH) along the ray heading toward (towardX, towardY). Used to clip
      // each line so it starts/ends at the token boxes rather than overlapping them.
      const boxEdge = (cx: number, cy: number, halfW: number, halfH: number, towardX: number, towardY: number) => {
        const dx = towardX - cx;
        const dy = towardY - cy;
        if (dx === 0 && dy === 0) return { x: cx, y: cy };
        const tX = dx !== 0 ? halfW / Math.abs(dx) : Infinity;
        const tY = dy !== 0 ? halfH / Math.abs(dy) : Infinity;
        const t = Math.min(tX, tY, 1);
        return { x: cx + dx * t, y: cy + dy * t };
      };
      const lines: Array<{ x1: number; y1: number; x2: number; y2: number; color: string }> = [];
      const addLine = (idx: number, value: number, max: number, rgb: string) => {
        if (idx === hoveredTokenIndex) return;
        const target = tokenRefs.current.get(idx);
        if (!target) return; // token not in view (snippet collapsed) -> skip
        const r = target.getBoundingClientRect();
        const tx = r.left + r.width / 2;
        const ty = r.top + r.height / 2;
        const opacity = tokenGradientOpacity(value, max);
        const start = boxEdge(hx, hy, hw, hh, tx, ty);
        const end = boxEdge(tx, ty, r.width / 2, r.height / 2, hx, hy);
        lines.push({
          x1: start.x,
          y1: start.y,
          x2: end.x,
          y2: end.y,
          color: `rgba(${rgb}, ${Math.min(0.6 * opacity, 0.3).toFixed(3)})`,
        });
      };
      // Keys (tokens the hovered token attends to) -> light orange line.
      hoveredOutgoing.forEach((value, idx) => addLine(idx, value, hoveredOutgoingMax, ATTENTION_ORANGE_RGB));
      // Queries (tokens that attend to the hovered token) -> light slate line.
      hoveredIncoming.forEach((value, idx) => addLine(idx, value, hoveredIncomingMax, SLATE_RGB));
      setAttentionLines(lines);
    };
    update();
    window.addEventListener('scroll', update, true);
    window.addEventListener('resize', update);
    return () => {
      window.removeEventListener('scroll', update, true);
      window.removeEventListener('resize', update);
    };
  }, [hoveredTokenIndex, hoveredOutgoing, hoveredIncoming, hoveredOutgoingMax, hoveredIncomingMax, currentRange]);

  const hasImStartToken = sequence.tokens.some((t) => t === '<|im_start|>');

  useEffect(() => {
    setCurrentRange(isExpanded ? 10000 : tokensToDisplayAroundMaxActToken);
  }, [isExpanded, tokensToDisplayAroundMaxActToken]);

  function hasNextToken(tokenIndex: number) {
    return tokenIndex < sequence.tokens.length - 1;
  }

  function nextTokenIsMessageEndOrChannelToken(tokenIndex: number) {
    return (
      hasNextToken(tokenIndex) &&
      (MESSAGE_TOKEN.includes(sequence.tokens[tokenIndex + 1] || '') ||
        END_TOKEN.includes(sequence.tokens[tokenIndex + 1] || ''))
    );
  }

  function tokenIsRoleToken(tokenIndex: number) {
    const isGemmaInstruct = (modelId || '').startsWith('gemma-2-') || (modelId || '').startsWith('gemma-3-');
    if (
      hasImStartToken &&
      tokenIndex === 0 &&
      CHAT_ROLE_NAMES.includes(sequence.tokens[0] || '') &&
      sequence.tokens[1] === '\n'
    ) {
      return true;
    }
    return (
      tokenIndex > 0 &&
      START_TOKEN.includes(sequence.tokens[tokenIndex - 1] || '') &&
      (END_TOKEN.includes(sequence.tokens[tokenIndex + 1] || '') ||
        MESSAGE_TOKEN.includes(sequence.tokens[tokenIndex + 1] || '') ||
        (isGemmaInstruct && sequence.tokens[tokenIndex + 1] === '\n') ||
        (hasImStartToken && sequence.tokens[tokenIndex + 1] === '\n') ||
        CHANNEL_TOKEN === sequence.tokens[tokenIndex + 1])
    );
  }

  function prevTokenIsChannelToken(tokenIndex: number) {
    return tokenIndex > 0 && sequence.tokens[tokenIndex - 1] === CHANNEL_TOKEN;
  }

  function shouldShowToken(tokenIndex: number) {
    const isInMaxActBuffer =
      tokenIndex > maxActivationTokenIndex - currentRange && tokenIndex < maxActivationTokenIndex + currentRange;

    if (!showRawTokens) {
      const token = sequence.tokens[tokenIndex] || '';
      if (
        START_TOKEN.includes(token) ||
        END_TOKEN.includes(token) ||
        MESSAGE_TOKEN.includes(token) ||
        BOS_TOKEN.includes(token) ||
        token === CHANNEL_TOKEN
      ) {
        return false;
      }
    }

    return isInMaxActBuffer;
  }

  // Detect whether the snippet view is currently clipping tokens above or below
  // the anchor, so we can show clickable expand indicators with a token count.
  const firstShownIndex = Math.max(0, maxActivationTokenIndex - currentRange + 1);
  const lastShownIndex = Math.min(sequence.tokens.length - 1, maxActivationTokenIndex + currentRange - 1);
  const showExpandIndicator = enableExpanding && !isExpanded;
  const tokensHiddenAbove = showExpandIndicator ? firstShownIndex : 0;
  const tokensHiddenBelow = showExpandIndicator ? sequence.tokens.length - 1 - lastShownIndex : 0;
  const hasHiddenAbove = tokensHiddenAbove > 0;
  const hasHiddenBelow = tokensHiddenBelow > 0;
  const expandIndicatorClassName = `my-0.5 select-none py-1 px-1 rounded font-sans text-[10px] font-medium ${
    isItemHovered ? 'bg-sky-100 text-sky-700' : 'text-slate-400'
  }`;

  return (
    <div
      className={cn(
        `flex w-full flex-row ${isExpanded || tokensToDisplayAroundMaxActToken > 100 ? 'items-start' : 'items-center'} justify-start gap-x-2`,
        className,
      )}
      onMouseEnter={() => setIsItemHovered(true)}
      onMouseLeave={() => setIsItemHovered(false)}
    >
      <div
        ref={sequenceRef}
        className={`relative flex-1 ${!showRawTokens ? 'sm:ml-2.5' : ''} ${overrideLeading} ${
          showExpandIndicator ? 'cursor-pointer' : ''
        }`}
        onClick={() => {
          if (enableExpanding) {
            setIsExpanded(!isExpanded);
          }
        }}
      >
        {hoveredTokenIndex !== null &&
          attentionLines.length > 0 &&
          typeof document !== 'undefined' &&
          createPortal(
            <svg className="pointer-events-none fixed inset-0 z-[55] h-screen w-screen" aria-hidden>
              {attentionLines.map((l, i) => (
                <line
                  key={`line-${i}`}
                  x1={l.x1}
                  y1={l.y1}
                  x2={l.x2}
                  y2={l.y2}
                  stroke={l.color}
                  strokeWidth={1.5}
                  strokeLinecap="round"
                />
              ))}
            </svg>,
            document.body,
          )}
        {hoveredTokenIndex !== null &&
          popupPos !== null &&
          typeof document !== 'undefined' &&
          createPortal(
            (() => {
              const hToken = sequence.tokens[hoveredTokenIndex];
              const hAttn = Math.max(
                denseAttentionValues[hoveredTokenIndex] ?? 0,
                queryMaxValues[hoveredTokenIndex] ?? 0,
              );
              const hOutgoing = outgoingByQuery.get(hoveredTokenIndex) || [];
              const hIncoming = incomingByKey.get(hoveredTokenIndex) || [];
              return (
                <div
                  className="pointer-events-none fixed z-[60] flex justify-start"
                  style={{
                    left: popupPos.left,
                    top: popupPos.top,
                    transform: 'translate(0, calc(-100% - 4px))',
                  }}
                >
                  <div className="flex w-max max-w-[360px] flex-col items-center gap-y-1.5 rounded-lg border bg-white/95 px-3 py-2 text-center text-[11px] font-semibold text-slate-700 shadow backdrop-blur-[1px]">
                    <div className="whitespace-pre-wrap rounded bg-slate-300 px-1 font-mono">
                      {formatChipToken(hToken)}
                    </div>
                    <div className="font-base flex flex-row gap-x-2 font-normal leading-none">
                      <div>Max Attention</div>
                      <div className="font-mono">{hAttn.toFixed(ACTIVATION_PRECISION)}</div>
                    </div>
                    <div className="flex flex-row gap-x-3 border-t border-t-slate-200 pt-2">
                      {hOutgoing.length > 0 && (
                        <div className="font-base flex w-full flex-col gap-y-1 text-[11px] font-normal leading-tight">
                          <div className="whitespace-nowrap font-semibold">Attends To (Keys)</div>
                          {hOutgoing.slice(0, 5).map((item, i) => {
                            const itemTopOp = tokenGradientOpacity(
                              hoveredOutgoing.get(item.index) || 0,
                              hoveredOutgoingMax,
                            );
                            const itemBottomOp = tokenGradientOpacity(
                              hoveredIncoming.get(item.index) || 0,
                              hoveredIncomingMax,
                            );
                            const itemBg = `linear-gradient(to bottom, rgba(${ATTENTION_ORANGE_RGB}, ${itemTopOp}) 50%, rgba(${ATTENTION_ORANGE_RGB}, ${itemBottomOp}) 50%)`;
                            return (
                              <div className="flex flex-row items-center justify-center gap-x-1" key={`out-${i}`}>
                                <div
                                  className="rounded border border-slate-200 bg-origin-border px-1 font-mono font-bold"
                                  style={{ backgroundImage: itemBg }}
                                >
                                  {formatChipToken(sequence.tokens[item.index])}
                                </div>
                                <div className="font-mono">+{item.value.toFixed(ACTIVATION_PRECISION)}</div>
                              </div>
                            );
                          })}
                        </div>
                      )}
                      {hIncoming.length > 0 && (
                        <div className="font-base flex w-full flex-col gap-y-1 text-[11px] font-normal leading-tight">
                          <div className="whitespace-nowrap font-semibold">Attended By (Queries)</div>
                          {hIncoming.slice(0, 5).map((item, i) => {
                            const itemTopOp = tokenGradientOpacity(
                              hoveredOutgoing.get(item.index) || 0,
                              hoveredOutgoingMax,
                            );
                            const itemBottomOp = tokenGradientOpacity(
                              hoveredIncoming.get(item.index) || 0,
                              hoveredIncomingMax,
                            );
                            const itemBg = `linear-gradient(to bottom, rgba(${ATTENTION_ORANGE_RGB}, ${itemTopOp}) 50%, rgba(${ATTENTION_ORANGE_RGB}, ${itemBottomOp}) 50%)`;
                            return (
                              <div className="flex flex-row items-center justify-center gap-x-1" key={`in-${i}`}>
                                <div
                                  className="rounded border border-slate-200 bg-origin-border px-1 font-mono font-bold"
                                  style={{ backgroundImage: itemBg }}
                                >
                                  {formatChipToken(sequence.tokens[item.index])}
                                </div>
                                <div className="font-mono">+{item.value.toFixed(ACTIVATION_PRECISION)}</div>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              );
            })(),
            document.body,
          )}
        {showExpandIndicator && isItemHovered && (hasHiddenAbove || hasHiddenBelow) && (
          <div className="absolute top-0 flex w-full items-center justify-center py-0.5">
            <span className="rounded bg-sky-100 px-2 py-1 font-sans text-[9px] font-medium uppercase text-sky-600">
              Click to Show All Tokens
            </span>
          </div>
        )}
        {hasHiddenAbove ? (
          <div className={expandIndicatorClassName}>
            {tokensHiddenAbove} Token{tokensHiddenAbove === 1 ? '' : 's'} Before
          </div>
        ) : (
          <div className="h-5 w-full" />
        )}
        {sequence.tokens.map((token, tokenIndex) => {
          if (!shouldShowToken(tokenIndex)) {
            return '';
          }
          const tokenWithReplacedAnomalies = replaceHtmlAnomalies(token);
          const tokenEndsWithSpace = tokenWithReplacedAnomalies.endsWith(' ');
          const tokenStartsWithSpace = tokenWithReplacedAnomalies.startsWith(' ');
          // When the token is purely whitespace, swap each whitespace char with a
          // non-breaking space so the span keeps a normal character line-height
          // (otherwise the background gradient ends up shorter than neighboring tokens).
          const displayContent =
            tokenWithReplacedAnomalies.length > 0 && tokenWithReplacedAnomalies.trim() === ''
              ? tokenWithReplacedAnomalies.replace(/\s/g, '\u00A0')
              : tokenWithReplacedAnomalies;
          const attnValue = denseAttentionValues[tokenIndex] ?? 0;
          // "All" mode considers the token's strongest involvement in either
          // direction: as a key (column max) or as a query (row max).
          const combinedValue = Math.max(attnValue, queryMaxValues[tokenIndex] ?? 0);
          const isHoveredQuery = hoveredTokenIndex === tokenIndex;
          const isHovering = hoveredTokenIndex !== null;
          // While hovering: top half encodes outgoing attention FROM the hovered
          // token to this candidate; bottom half encodes incoming attention TO
          // the hovered token FROM this candidate. No green is shown.
          // While not hovering: both halves carry the default green column-max.
          let tokenBackgroundImage: string;
          if (isHovering) {
            const topOpacity = tokenGradientOpacity(hoveredOutgoing.get(tokenIndex) || 0, hoveredOutgoingMax);
            const bottomOpacity = tokenGradientOpacity(hoveredIncoming.get(tokenIndex) || 0, hoveredIncomingMax);
            const topColor = `rgba(${ATTENTION_ORANGE_RGB}, ${topOpacity})`;
            const bottomColor = `rgba(${ATTENTION_ORANGE_RGB}, ${bottomOpacity})`;
            tokenBackgroundImage = `linear-gradient(to bottom, ${topColor} 50%, ${bottomColor} 50%)`;
          } else if (maxAttentionMode === 'keys') {
            const op = tokenGradientOpacity(attnValue, keyMaxOverall);
            const color = `rgba(${ATTENTION_ORANGE_RGB}, ${op})`;
            tokenBackgroundImage = `linear-gradient(to bottom, ${color} 50%, rgba(${ATTENTION_ORANGE_RGB}, 0) 50%)`;
          } else if (maxAttentionMode === 'queries') {
            const op = tokenGradientOpacity(queryMaxValues[tokenIndex] ?? 0, queryMaxOverall);
            const color = `rgba(${ATTENTION_ORANGE_RGB}, ${op})`;
            tokenBackgroundImage = `linear-gradient(to bottom, rgba(${ATTENTION_ORANGE_RGB}, 0) 50%, ${color} 50%)`;
          } else {
            tokenBackgroundImage = makeActivationBackgroundColorWithDFA(
              overallMaxActivationValueInList || sequence.maxActivation || 1,
              combinedValue,
              ATTENTION_ORANGE_RGB,
            );
          }
          return (
            <span key={tokenIndex}>
              <span
                ref={(el) => {
                  if (el) tokenRefs.current.set(tokenIndex, el);
                  else tokenRefs.current.delete(tokenIndex);
                }}
                className={`inline-block cursor-default whitespace-nowrap bg-origin-border py-0.5 font-mono ${
                  isHoveredQuery ? HOVERED_QUERY_TOKEN_CLASSNAME : REGULAR_TOKEN_CLASSNAME
                } ${tokenEndsWithSpace ? 'pr-1' : ''} ${tokenStartsWithSpace ? 'pl-1' : ''} ${
                  !showRawTokens && tokenIsRoleToken(tokenIndex) ? '-ml-2 mr-1 mt-1 rounded bg-slate-300' : ''
                } ${
                  !showRawTokens && prevTokenIsChannelToken(tokenIndex) ? 'mt-1 rounded bg-slate-200' : ''
                } ${overrideTextColor} ${overrideTextSize}`}
                style={{
                  backgroundImage: tokenBackgroundImage,
                }}
                onMouseEnter={() => setHoveredTokenIndex(tokenIndex)}
                onMouseLeave={() => {
                  setHoveredTokenIndex((current) => (current === tokenIndex ? null : current));
                }}
              >
                {displayContent}
              </span>
              {((!showRawTokens && nextTokenIsMessageEndOrChannelToken(tokenIndex)) ||
                ((token.indexOf('\n') !== -1 || tokenWithReplacedAnomalies === LINE_BREAK_REPLACEMENT_CHAR) &&
                  showLineBreaks)) && <br />}
            </span>
          );
        })}
        {hasHiddenBelow && (
          <div className={expandIndicatorClassName}>
            {tokensHiddenBelow} Token{tokensHiddenBelow === 1 ? '' : 's'} After
          </div>
        )}
      </div>
    </div>
  );
}
