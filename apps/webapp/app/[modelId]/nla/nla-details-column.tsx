'use client';

import CustomTooltip from '@/components/custom-tooltip';
import { useNlaContext } from '@/components/provider/nla-provider';
import { LoadingSquare } from '@/components/svg/loading-square';
import { QuestionMarkCircledIcon } from '@radix-ui/react-icons';
import { Share2 } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { cleanPartialText, computeRelativeMse, confidenceLabel } from './nla-utils';

// Explanations are expected to come back as up to three labeled sections separated
// by `\n\n`: Message-Level, Phrase-Level, then Token-Level. Any additional `\n\n`
// after that stays inside the Token-Level block so nothing is truncated. During
// streaming the second/third labels appear as `\n\n` separators are emitted.
const PARAGRAPH_LABELS = ['Message-Level', 'Phrase-Level', 'Token-Level'];

// Replace single newline characters with the return glyph `⏎` so token-level
// newlines in the explanation render inline rather than breaking the line.
// Preserve `\n\n` pairs which are paragraph separators used to split labels.
// IMPORTANT: this is a 1:1 character substitution — it does not change the
// string length, so character offsets into the rendered paragraph map
// directly back to character offsets into the raw `description` blob (after
// adding the paragraph's start offset).
function formatExplanationContent(text: string): string {
  return text.replace(/(?<!\n)\n(?!\n)/g, '\u23CE');
}

function splitExplanationParagraphs(text: string): { label: string; content: string }[] {
  const parts = formatExplanationContent(text).split('\n\n');
  const out: { label: string; content: string }[] = [];
  if (parts.length >= 1) out.push({ label: PARAGRAPH_LABELS[0], content: parts[0] });
  if (parts.length >= 2) out.push({ label: PARAGRAPH_LABELS[1], content: parts[1] });
  // Everything after Phrase-Level is Token-Level: keep all remaining text, including
  // extra `\n\n` that would otherwise become truncated fourth+ "paragraphs".
  if (parts.length >= 3) out.push({ label: PARAGRAPH_LABELS[2], content: parts.slice(2).join('\n\n') });
  return out;
}

// Same as `splitExplanationParagraphs`, but also reports each paragraph's
// starting character index in the raw `description`. Used by the in-text
// selection / `?highlightStart=&highlightEnd=` machinery to map between
// per-paragraph DOM offsets and absolute description offsets. Exact offsets
// are preserved because `formatExplanationContent` only does 1:1 char
// substitutions and the `\n\n` separator is always two chars.
function splitExplanationParagraphsWithOffsets(
  text: string,
): { label: string; content: string; startOffset: number }[] {
  const parts = formatExplanationContent(text).split('\n\n');
  const out: { label: string; content: string; startOffset: number }[] = [];
  let offset = 0;
  if (parts.length >= 1) {
    out.push({ label: PARAGRAPH_LABELS[0], content: parts[0], startOffset: offset });
    offset += parts[0].length + 2;
  }
  if (parts.length >= 2) {
    out.push({ label: PARAGRAPH_LABELS[1], content: parts[1], startOffset: offset });
    offset += parts[1].length + 2;
  }
  if (parts.length >= 3) {
    const tokenContent = parts.slice(2).join('\n\n');
    out.push({ label: PARAGRAPH_LABELS[2], content: tokenContent, startOffset: offset });
  }
  return out;
}

// Maps a (DOM node, intra-node offset) pair anywhere inside `rootEl` to a
// flat character offset into the concatenated text content of `rootEl`,
// matching the order produced by a TreeWalker over text nodes. Returns
// null if `node` isn't reachable from `rootEl`.
function getCharOffsetWithinElement(rootEl: Element, node: Node, nodeOffset: number): number | null {
  // If the selection landed on the element itself (e.g. user double-
  // clicked then extended via keyboard) the offset is a child index;
  // resolve to "first text node inside that child" so the walker can
  // pick it up below.
  let targetNode = node;
  let targetOffset = nodeOffset;
  if (node.nodeType === Node.ELEMENT_NODE) {
    const child = node.childNodes[nodeOffset];
    if (child && child.nodeType === Node.TEXT_NODE) {
      targetNode = child;
      targetOffset = 0;
    } else if (child) {
      const inner = document.createTreeWalker(child, NodeFilter.SHOW_TEXT).nextNode();
      if (inner) {
        targetNode = inner;
        targetOffset = 0;
      }
    } else if (node === rootEl) {
      // Selection extends past the last child — return total length.
      return rootEl.textContent?.length ?? 0;
    }
  }
  const walker = document.createTreeWalker(rootEl, NodeFilter.SHOW_TEXT);
  let total = 0;
  let cur = walker.nextNode();
  while (cur) {
    if (cur === targetNode) return total + targetOffset;
    total += cur.nodeValue?.length ?? 0;
    cur = walker.nextNode();
  }
  return null;
}

// Renders paragraph content with an optional highlighted span. `localHl`
// is in paragraph-local coordinates (i.e. after subtracting the
// paragraph's `startOffset` from the absolute description range). Returns
// the raw string when no highlight applies, so React can use the cheaper
// text-node path in the common case.
function findCaseInsensitiveMatchRanges(text: string, needle: string): { start: number; end: number }[] {
  if (needle.length < 2) return [];
  const t = text.toLowerCase();
  const n = needle.toLowerCase();
  const ranges: { start: number; end: number }[] = [];
  let i = 0;
  while (i < t.length) {
    const j = t.indexOf(n, i);
    if (j === -1) break;
    ranges.push({ start: j, end: j + n.length });
    i = j + 1;
  }
  return ranges;
}

/** Paragraph body with optional share-range (amber) and explanation search (emerald) highlights. */
function renderParagraphWithHighlights(
  content: string,
  amberRange: { start: number; end: number } | null,
  searchNeedle: string,
): ReactNode {
  const emeraldRanges = searchNeedle.length >= 2 ? findCaseInsensitiveMatchRanges(content, searchNeedle) : [];
  if (!amberRange && emeraldRanges.length === 0) return content;
  const breakpoints = new Set<number>([0, content.length]);
  if (amberRange) {
    breakpoints.add(Math.max(0, amberRange.start));
    breakpoints.add(Math.min(content.length, amberRange.end));
  }
  for (const r of emeraldRanges) {
    breakpoints.add(r.start);
    breakpoints.add(r.end);
  }
  const sorted = [...breakpoints].sort((a, b) => a - b);
  const uniq: number[] = [];
  for (const x of sorted) {
    if (uniq.length === 0 || uniq[uniq.length - 1] !== x) uniq.push(x);
  }
  const parts: React.ReactNode[] = [];
  for (let i = 0; i < uniq.length - 1; i++) {
    const lo = uniq[i];
    const hi = uniq[i + 1];
    if (lo >= hi) continue;
    const mid = lo + (hi - lo) / 2;
    const inEmerald = emeraldRanges.some((r) => mid >= r.start && mid < r.end);
    const inAmber = amberRange !== null && mid >= amberRange.start && mid < amberRange.end;
    const slice = content.slice(lo, hi);
    let cls = '';
    if (inEmerald && inAmber) cls = 'rounded-sm bg-emerald-200 text-slate-700 ring-1 ring-amber-300';
    else if (inEmerald) cls = 'rounded-sm bg-emerald-200 text-slate-700';
    else if (inAmber) cls = 'rounded-sm bg-amber-200 text-slate-700';
    parts.push(
      cls ? (
        <span key={`${lo}-${hi}`} className={cls}>
          {slice}
        </span>
      ) : (
        slice
      ),
    );
  }
  return <>{parts}</>;
}

function renderTokenGlyph(token: string) {
  const newlineCount = (token.match(/\n/g) || []).length;
  if (token.trim() === '') {
    return newlineCount > 0 ? (
      <span className="opacity-30">{'\u21B5'.repeat(newlineCount)}</span>
    ) : (
      <span className="text-slate-300">{'\u00B7'}</span>
    );
  }
  return (
    <>
      {token.startsWith(' ') && <span className="text-slate-300">{'\u00B7'}</span>}
      {/* Replace \n with ↵ before trimming so trim() only strips edge
          spaces. Otherwise a content+trailing-newline token (e.g.
          Llama 3's "Hi!\n") would render without its return glyph. */}
      {token.replaceAll('\n', '\u21B5').trim()}
      {token.endsWith(' ') && <span className="text-slate-300">{'\u00B7'}</span>}
    </>
  );
}

export default function NLADetailsColumn() {
  const {
    tokenList,
    isLoading,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    isEmbed,
    selectedNlaSource,
    selectedPosition,
    lockedPosition,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    setLockedPosition,
    highlightedParagraph,
    setHighlightedParagraph,
    highlightedRange,
    setHighlightedRange,
    highlightComment,
    resultMap,
    partialMap,
    explanationSearchNeedle,
    selectedTokenPositions,
    detailsColumnRef,
    handleShareExplanation,
    isChatStreaming,
    isHydratingDemo,
  } = useNlaContext();

  // const [showRaw, setShowRaw] = useState(false);
  // Index of the paragraph whose mini share button is currently being
  // hovered. Takes precedence over the persistent `highlightedParagraph`
  // state so the user can preview exactly which paragraph their share
  // link will highlight — even if a different paragraph is already
  // highlighted from a prior URL load or share-button click.
  const [hoveredShareParagraph, setHoveredShareParagraph] = useState<number | null>(null);
  // Live-selection popover state. Shown above an active text selection
  // that fits within a single paragraph; offers a single Share button
  // that persists the selection as the highlighted range and copies a
  // share link with `?highlightStart=&highlightEnd=`.
  const [selectionPopover, setSelectionPopover] = useState<{
    range: { start: number; end: number };
    x: number;
    y: number;
  } | null>(null);
  const popoverRef = useRef<HTMLDivElement | null>(null);
  const explanationContainerRef = useRef<HTMLDivElement | null>(null);

  const sourceNorm = selectedNlaSource?.norm ?? 0;

  // Locked position pins the details panel; hovers don't override it.
  // When nothing is locked, fall back to the currently-hovered chip.
  const displayPosition = lockedPosition ?? selectedPosition;

  const selectedResult = displayPosition !== null ? resultMap.get(displayPosition) : undefined;
  const selectedPartial = displayPosition !== null ? partialMap.get(displayPosition) : undefined;
  const selectedToken = displayPosition !== null ? tokenList.find((t) => t.position === displayPosition) : undefined;
  const selectedIsGenerating = !selectedResult && selectedPartial !== undefined;
  const selectedIsPending =
    !selectedResult && !selectedIsGenerating && isLoading && selectedTokenPositions.has(displayPosition ?? -1);
  const isSelectedForExplanation = displayPosition !== null && selectedTokenPositions.has(displayPosition);
  const isLocked = lockedPosition !== null && lockedPosition === displayPosition;

  // The details column now also has content for plain "gray" hovers
  // (token without a result yet) — we just show the token glyph + a hint
  // about what to do next.
  const hasContent = !!selectedToken;
  const showEditorComment = isLocked && !!highlightComment && highlightComment.length > 0;

  // Compute paragraph offsets once per result. Used both for rendering
  // the persistent `highlightedRange` and for translating live-selection
  // DOM offsets into absolute description offsets.
  const paragraphsWithOffsets = useMemo(() => {
    if (!selectedResult) return [];
    return splitExplanationParagraphsWithOffsets(selectedResult.description);
  }, [selectedResult]);

  const rmsePanelData = useMemo(() => {
    if (!selectedResult) return null;
    const n = computeRelativeMse(selectedResult.mse, sourceNorm);
    if (n === null) return null;
    return { n, conf: confidenceLabel(n) };
  }, [selectedResult, sourceNorm]);

  // Reset the live-selection popover whenever the focus token changes —
  // a popover from a different token's selection would dangle otherwise.
  useEffect(() => {
    setSelectionPopover(null);
  }, [displayPosition]);

  // Capture text selection inside the explanation container. Fires on
  // mouseup (after the user finishes a drag) so the browser has a
  // settled, expanded Range to read. Selections that span paragraphs or
  // escape the explanation container are silently rejected.
  const handleExplanationMouseUp = useCallback(() => {
    if (!isLocked) return;
    const containerEl = explanationContainerRef.current;
    if (!containerEl) return;
    const sel = window.getSelection();
    if (!sel || sel.rangeCount === 0) {
      setSelectionPopover(null);
      return;
    }
    const range = sel.getRangeAt(0);
    if (range.collapsed) {
      setSelectionPopover(null);
      return;
    }
    if (!containerEl.contains(range.startContainer) || !containerEl.contains(range.endContainer)) {
      setSelectionPopover(null);
      return;
    }
    const startEl = (
      range.startContainer.nodeType === Node.TEXT_NODE
        ? range.startContainer.parentElement
        : (range.startContainer as Element)
    )?.closest('[data-paragraph-idx]');
    const endEl = (
      range.endContainer.nodeType === Node.TEXT_NODE
        ? range.endContainer.parentElement
        : (range.endContainer as Element)
    )?.closest('[data-paragraph-idx]');
    // Reject cross-paragraph selections — the user explicitly asked
    // that selections be confined to a single paragraph.
    if (!startEl || !endEl || startEl !== endEl) {
      setSelectionPopover(null);
      return;
    }
    const paraEl = startEl;
    const paraStart = Number(paraEl.getAttribute('data-paragraph-start'));
    if (!Number.isFinite(paraStart)) return;
    const localStart = getCharOffsetWithinElement(paraEl, range.startContainer, range.startOffset);
    const localEnd = getCharOffsetWithinElement(paraEl, range.endContainer, range.endOffset);
    if (localStart === null || localEnd === null) {
      setSelectionPopover(null);
      return;
    }
    const lo = Math.min(localStart, localEnd);
    const hi = Math.max(localStart, localEnd);
    if (hi - lo < 1) {
      setSelectionPopover(null);
      return;
    }
    const rect = range.getBoundingClientRect();
    setSelectionPopover({
      range: { start: paraStart + lo, end: paraStart + hi },
      // `getBoundingClientRect` returns viewport coords, which is what
      // `position: fixed` expects. Center the popover horizontally over
      // the selection and float it just above the top edge.
      x: rect.left + rect.width / 2,
      y: rect.top,
    });
  }, [isLocked]);

  // Clear any existing highlight (paragraph or range) the moment the
  // user starts a new selection inside the explanation. Without this,
  // the prior yellow span / paragraph tint would visually compete with
  // the in-progress browser selection (which is also yellow via the
  // `selection:bg-amber-100` variant) until the user clicks Share.
  // Listens at document scope and bails unless the selectstart target
  // is inside a paragraph body — this dodges the React-doesn't-have-
  // onSelectStart problem and survives the explanation div remounting
  // when the focus token changes.
  useEffect(() => {
    const handleSelectStart = (e: Event) => {
      const containerEl = explanationContainerRef.current;
      if (!containerEl) return;
      const target = e.target as Node | null;
      if (!target || !containerEl.contains(target)) return;
      const inPara = (target.nodeType === Node.TEXT_NODE ? target.parentElement : (target as Element))?.closest(
        '[data-paragraph-idx]',
      );
      if (!inPara) return;
      setHighlightedParagraph(null);
      setHighlightedRange(null);
    };
    document.addEventListener('selectstart', handleSelectStart);
    return () => document.removeEventListener('selectstart', handleSelectStart);
  }, [setHighlightedParagraph, setHighlightedRange]);

  // Dismiss popover on:
  //  - mousedown anywhere outside the popover (covers click-away both
  //    inside the details column and elsewhere on the page),
  //  - selectionchange that collapses the selection (covers caret moves
  //    via keyboard, etc.),
  //  - Escape key.
  useEffect(() => {
    if (!selectionPopover) return undefined;
    const handleDocMouseDown = (e: MouseEvent) => {
      if (popoverRef.current?.contains(e.target as Node)) return;
      setSelectionPopover(null);
    };
    const handleSelectionChange = () => {
      const sel = window.getSelection();
      if (!sel || sel.isCollapsed) setSelectionPopover(null);
    };
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setSelectionPopover(null);
    };
    document.addEventListener('mousedown', handleDocMouseDown);
    document.addEventListener('selectionchange', handleSelectionChange);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('mousedown', handleDocMouseDown);
      document.removeEventListener('selectionchange', handleSelectionChange);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [selectionPopover]);

  return (
    <div ref={detailsColumnRef} className="relative flex min-h-0 w-full flex-1 flex-col rounded-xl">
      <div
        className={`relative flex min-h-0 flex-1 flex-col overflow-hidden rounded-xl bg-white px-0 pt-0 transition-[border-color,box-shadow] ${
          isLocked && !isHydratingDemo ? 'border-2 border-sky-600 shadow' : 'border-2 border-slate-100 shadow'
        }`}
      >
        {isHydratingDemo ? (
          <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-y-2 px-6 py-8 text-slate-400">
            <LoadingSquare size={20} />
            <span className="text-xs font-medium">Loading…</span>
          </div>
        ) : (
          <>
            {hasContent && isLocked && selectedResult && (
              <div className="pointer-events-none absolute right-1.5 top-1.5 z-10">
                <button
                  type="button"
                  onClick={() => handleShareExplanation()}
                  disabled={isLoading || isChatStreaming}
                  title="Share this explanation"
                  className="pointer-events-auto flex h-6 items-center gap-x-1 rounded-md border border-slate-200 bg-white px-2 text-[10px] font-medium text-slate-500 shadow-sm transition-colors hover:border-sky-300 hover:bg-sky-50 hover:text-sky-700 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <Share2 className="h-3 w-3" />
                  Share
                </button>
              </div>
            )}
            {showEditorComment && (
              <div className="flex w-full shrink-0 flex-col gap-y-1 rounded-t-[calc(0.75rem-2px)] border-amber-200 bg-amber-50 px-3 py-2 sm:px-5 sm:py-4">
                <div className="-mt-0.5 mb-0.5 text-[11px] font-semibold uppercase tracking-wide text-amber-700/80">
                  Commentary by Human Editor
                </div>
                <div className="nla-markdown break-words font-serif text-slate-900">
                  <Markdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      a: ({ href, children, ...props }) => (
                        <a
                          {...props}
                          href={href}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="font-medium text-sky-700 underline decoration-sky-300 underline-offset-2 hover:text-sky-800"
                        >
                          {children}
                        </a>
                      ),
                    }}
                  >
                    {highlightComment}
                  </Markdown>
                </div>
              </div>
            )}
            <div className={`w-full shrink-0 bg-slate-100 ${showEditorComment ? '' : 'rounded-t-[calc(0.75rem-2px)]'}`}>
              <div className={`flex w-full items-start px-3 py-1.5 sm:py-3`}>
                <div className="shrink px-0 text-start text-[11px] font-semibold uppercase text-slate-500 sm:px-2">
                  Explanation by activation verbalizer
                </div>
              </div>
            </div>

            <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden overscroll-contain pb-3">
              {!hasContent ? (
                <div className="flex min-h-full flex-col items-start justify-start gap-y-1 px-8 py-8 text-center text-base text-slate-300">
                  <div className="w-full text-sm leading-relaxed text-slate-400/70">
                    Explanations generated by the activation verbalizer will appear here.
                  </div>
                </div>
              ) : (
                <div className="flex flex-col gap-y-1.5 px-3 pb-4 pt-2 text-xs text-slate-700 sm:gap-y-4 sm:px-5 sm:pt-5">
                  {selectedResult ? (
                    <>
                      <div className="mb-1 flex w-full items-center justify-between gap-x-3">
                        <span
                          className={`rounded-md border-[1.5px] border-sky-600 bg-sky-200 px-1 py-0.5 font-serif text-sm`}
                        >
                          {renderTokenGlyph(selectedResult.token)}
                        </span>
                        {rmsePanelData && (
                          <div id="nla-rmse" className="flex shrink-0 flex-col items-end gap-y-0.5 text-right">
                            <CustomTooltip
                              trigger={
                                <div className="flex cursor-pointer flex-row items-center gap-x-1 rounded-md p-1 text-slate-500 hover:bg-sky-200 hover:text-sky-700">
                                  <div className="flex flex-col items-center gap-y-0.5 font-mono text-[9px] leading-none">
                                    <span>RMSE</span>
                                    <span>{rmsePanelData.n.toFixed(2)}</span>
                                  </div>
                                  <QuestionMarkCircledIcon className="h-3 w-3" />
                                </div>
                              }
                            >
                              <div className="flex flex-col gap-y-3">
                                <p>
                                  <strong>Relative Mean Squared Error (RMSE)</strong> is a rough estimate of how good
                                  the AV&apos;s explanation is. Scores close to 0 are good. A score of ≈1 is bad.
                                </p>
                                <p>
                                  RMSE is calculated by passing the AV&apos;s explanation through the activation
                                  reconstructor (AR), which outputs activations that are compared to the model&apos;s
                                  original activations.
                                </p>
                                <div className="mt-0 flex flex-col gap-y-1">
                                  <code>RMSE = MSE(norm(pred), norm(target)) / var(dataset)</code>
                                  <div>
                                    <code>dataset</code> = a dataset of normed vectors
                                  </div>
                                  <div>
                                    <code>var(dataset)</code> = the mean MSE if we predict the mean of the dataset
                                  </div>
                                </div>
                              </div>
                            </CustomTooltip>
                          </div>
                        )}
                      </div>

                      <div
                        ref={explanationContainerRef}
                        className="flex flex-col gap-y-0 sm:gap-y-3"
                        onMouseUp={handleExplanationMouseUp}
                      >
                        {paragraphsWithOffsets.map((p, i) => {
                          // Hover preview wins over the persistent highlight: the
                          // hovered paragraph turns yellow and the previously-
                          // highlighted one (if any) temporarily reverts to white.
                          const effectiveParagraphHighlight =
                            hoveredShareParagraph !== null ? hoveredShareParagraph : highlightedParagraph;
                          const isParagraphHighlighted = isLocked && effectiveParagraphHighlight === i;
                          // When a character-specific range is set, dim every
                          // paragraph's body text so the yellow highlighted span
                          // visually leads. The hover-preview path still wins
                          // (turning the previewed paragraph fully yellow), so a
                          // user previewing a paragraph-share click never sees a
                          // dimmed preview.
                          const dimNonHighlightText = isLocked && highlightedRange !== null && !isParagraphHighlighted;
                          // Translate the absolute description-range into a
                          // paragraph-local range, but only if it fits entirely
                          // within this paragraph (cross-paragraph ranges aren't
                          // representable in the URL contract).
                          const paraEnd = p.startOffset + p.content.length;
                          const localRange =
                            isLocked &&
                            highlightedRange &&
                            highlightedRange.start >= p.startOffset &&
                            highlightedRange.end <= paraEnd
                              ? {
                                  start: highlightedRange.start - p.startOffset,
                                  end: highlightedRange.end - p.startOffset,
                                }
                              : null;
                          return (
                            <div key={p.label} className="relative flex flex-col gap-y-0.5">
                              <div className="absolute right-0 top-0 flex min-h-5 items-center justify-between gap-x-2 text-[10px] font-medium uppercase text-slate-400">
                                {/* <span className="flex items-center text-slate-400/70">{p.label}</span> */}
                                {isLocked && (
                                  <button
                                    type="button"
                                    onClick={() => {
                                      setHighlightedParagraph(i);
                                      handleShareExplanation({ paragraph: i, range: null });
                                    }}
                                    onMouseEnter={() => setHoveredShareParagraph(i)}
                                    onMouseLeave={() => setHoveredShareParagraph((prev) => (prev === i ? null : prev))}
                                    onFocus={() => setHoveredShareParagraph(i)}
                                    onBlur={() => setHoveredShareParagraph((prev) => (prev === i ? null : prev))}
                                    disabled={isLoading || isChatStreaming}
                                    title={`Share this paragraph`}
                                    aria-label={`Share this paragraph`}
                                    className="flex h-5 shrink-0 items-center gap-x-0.5 rounded border border-transparent px-1 text-slate-400 transition-colors hover:border-sky-300 hover:bg-sky-50 hover:text-sky-700 disabled:cursor-not-allowed disabled:opacity-50"
                                  >
                                    <Share2 className="h-3 w-3" />
                                  </button>
                                )}
                              </div>
                              <div
                                data-paragraph-idx={i}
                                data-paragraph-start={p.startOffset}
                                className={`whitespace-pre-wrap rounded-md py-0.5 pr-4 font-medium leading-snug transition-colors selection:bg-amber-200 selection:text-slate-700 sm:pr-8 ${
                                  isParagraphHighlighted
                                    ? 'bg-amber-100 text-slate-700'
                                    : dimNonHighlightText
                                      ? 'bg-white text-slate-400'
                                      : 'bg-white text-slate-800'
                                }`}
                              >
                                {renderParagraphWithHighlights(p.content, localRange, explanationSearchNeedle)}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </>
                  ) : selectedToken && (selectedIsGenerating || selectedIsPending) ? (
                    <>
                      <div className="mb-1 flex w-full items-center gap-x-4">
                        <span className="relative rounded-md border-[1.5px] border-transparent bg-amber-200 px-1 py-0.5 font-serif text-sm text-amber-700">
                          {renderTokenGlyph(selectedToken.token)}
                          <span
                            aria-hidden
                            className="pointer-events-none absolute -bottom-[1.5px] z-10 h-[1.5px] rounded-full bg-slate-500 [animation:chip-loading-sweep_1.6s_linear_infinite]"
                          />
                        </span>
                      </div>
                      {selectedIsGenerating &&
                        cleanPartialText(selectedPartial || '') &&
                        splitExplanationParagraphs(cleanPartialText(selectedPartial || '')).map((p) => (
                          <div key={p.label} className="flex flex-col gap-y-0">
                            {/* <div
                          className={`${i === 0 ? 'mt-0' : 'mt-1.5'} text-[10px] font-medium uppercase text-slate-400`}
                        >
                          {p.label}
                        </div> */}
                            <div className="whitespace-pre-wrap rounded-md bg-white py-0.5 pr-4 font-medium leading-snug text-slate-600 sm:pr-8">
                              {renderParagraphWithHighlights(p.content, null, explanationSearchNeedle)}
                            </div>
                          </div>
                        ))}
                    </>
                  ) : selectedToken ? (
                    <>
                      <div className="mb-1 flex w-full items-center gap-x-4">
                        <span
                          className={`rounded-md border-[1.5px] px-1 py-0.5 font-serif text-sm ${
                            isSelectedForExplanation
                              ? 'border-amber-300 bg-amber-200 text-amber-700'
                              : 'border-transparent bg-white text-slate-500'
                          }`}
                        >
                          {renderTokenGlyph(selectedToken.token)}
                        </span>
                      </div>
                      <div className="mt-4 flex w-full flex-col items-center justify-center gap-y-1 text-center text-xs text-slate-400">
                        <span className="text-sm font-semibold text-slate-500">
                          {isSelectedForExplanation
                            ? 'This token is selected to be explained.'
                            : 'This token has not been explained.'}
                        </span>
                        <span>
                          {isSelectedForExplanation
                            ? 'Click "Explain" at the top of the chat to explain it.'
                            : 'Click the token to select it for explanation, then click "Explain" at the top of the chat.'}
                        </span>
                      </div>
                    </>
                  ) : null}
                </div>
              )}
            </div>
          </>
        )}
      </div>
      <div className="absolute bottom-0 left-1.5 right-1.5 mb-[2px] rounded-b-xl bg-white/30 pb-1 pt-1 text-center text-[9px] text-rose-700 backdrop-blur-sm sm:pb-1.5 sm:pt-1.5">
        NLAs can produce unexpected or incorrect explanations. See{' '}
        <a
          href="https://transformer-circuits.pub/2026/nla/index.html#discussion-and-limitations"
          target="_blank"
          rel="noopener noreferrer"
          className="text-rose-700 underline"
        >
          limitations
        </a>
        .
      </div>
      {selectionPopover && (
        <div
          ref={popoverRef}
          // `position: fixed` so the popover floats above the selection
          // in viewport coordinates (matching `getBoundingClientRect`'s
          // output). 36px above accommodates the popover height + a 4px
          // gap; clamps to the top edge to avoid clipping at very top.
          style={{
            position: 'fixed',
            top: Math.max(8, selectionPopover.y - 36),
            left: selectionPopover.x,
            transform: 'translateX(-50%)',
            zIndex: 100,
          }}
          // Stop mousedown from collapsing the selection (and from
          // triggering our own outside-click dismissal).
          onMouseDown={(e) => e.preventDefault()}
          className="flex items-center gap-x-1 rounded-md border border-slate-200 bg-white px-1.5 py-1 shadow-lg"
        >
          <button
            type="button"
            onClick={() => {
              const range = selectionPopover.range;
              setHighlightedRange(range);
              handleShareExplanation({ range, paragraph: null });
              window.getSelection()?.removeAllRanges();
              setSelectionPopover(null);
            }}
            disabled={isLoading || isChatStreaming}
            title="Share this selection"
            aria-label="Share this selection"
            className="flex h-6 items-center gap-x-1 rounded px-2 text-[10px] font-medium text-slate-600 transition-colors hover:bg-sky-50 hover:text-sky-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Share2 className="h-3 w-3" />
            Share Highlight
          </button>
        </div>
      )}
    </div>
  );
}
