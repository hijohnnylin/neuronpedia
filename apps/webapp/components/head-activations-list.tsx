'use client';

import * as ToggleGroup from '@radix-ui/react-toggle-group';
import copy from 'copy-to-clipboard';
import { Check, Copy, Share, XIcon } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import ReactTextareaAutosize from 'react-textarea-autosize';
import HeadActivationItem, { HeadSequenceData } from './head-activation-item';
import { useGlobalContext } from './provider/global-provider';
import { LoadingSquare } from './svg/loading-square';

export const ACTIVATION_DISPLAY_DEFAULT_CONTEXT_TOKENS = [
  { text: 'Short', size: 12 },
  { text: 'Snippet', size: 64 },
  { text: 'Full', size: 512 },
  // { text: "Max", size: 100000 },
];

export default function HeadActivationsList({
  sequences,
  modelId,
  layer,
  headIndex,
  isLoading = false,
  errorMessage,
  emptyMessage = 'No sequences for this head.',
  defaultRange = 1,
  defaultShowLineBreaks = true,
  defaultShowRawTokens = true,
  unbounded = false,
  inferenceEnabled = false,
  defaultCustomText,
}: {
  sequences: HeadSequenceData[];
  modelId?: string;
  layer?: number;
  headIndex?: number;
  isLoading?: boolean;
  errorMessage?: string | null;
  emptyMessage?: string;
  defaultRange?: number;
  defaultShowLineBreaks?: boolean;
  defaultShowRawTokens?: boolean;
  // When true, the list grows to its full height (no internal scroll) and lets the page scroll.
  unbounded?: boolean;
  // When true (model has an inference host), show a custom-text box to run
  // attention on the user's own input for this head.
  inferenceEnabled?: boolean;
  // Custom text from the `defaulttesttext` URL param, auto-run once on mount to make results linkable.
  defaultCustomText?: string;
}) {
  const { showToastMessage } = useGlobalContext();
  const [selectedRange, setSelectedRange] = useState(defaultRange);
  const [showLineBreaks, setShowLineBreaks] = useState(defaultShowLineBreaks);
  const [showRawTokens, setShowRawTokens] = useState(defaultShowRawTokens);
  const [maxAttentionMode, setMaxAttentionMode] = useState<'all' | 'keys' | 'queries'>('all');

  // Custom-text attention state. Only used when `inferenceEnabled` and we have a
  // concrete (modelId, layer, headIndex) to run against.
  const [customText, setCustomText] = useState('');
  const [customResult, setCustomResult] = useState<HeadSequenceData | null>(null);
  const [isRunningCustom, setIsRunningCustom] = useState(false);
  const [customError, setCustomError] = useState<string | null>(null);
  const [copyClicked, setCopyClicked] = useState(false);
  const canRunCustom = inferenceEnabled && modelId !== undefined && layer !== undefined && headIndex !== undefined;
  // Ensures the `defaulttesttext` URL param is only auto-run once (for the initial head), not again
  // when the user edits or switches heads.
  const didAutoRunRef = useRef(false);

  // A head change (new layer/head/model) invalidates any prior custom result.
  useEffect(() => {
    setCustomResult(null);
    setCustomError(null);
    setCopyClicked(false);
  }, [modelId, layer, headIndex]);

  // Write the tested text into the URL (as `defaulttesttext`) so the result is linkable/shareable,
  // preserving any other existing query params (e.g. `headFinder`).
  const writeTestTextToUrl = (text: string) => {
    if (typeof window === 'undefined') return;
    const url = new URL(window.location.href);
    url.searchParams.set('defaulttesttext', text);
    window.history.replaceState(window.history.state, '', url.toString());
  };

  const clearTestTextFromUrl = () => {
    if (typeof window === 'undefined') return;
    const url = new URL(window.location.href);
    url.searchParams.delete('defaulttesttext');
    window.history.replaceState(window.history.state, '', url.toString());
  };

  const runCustom = (textArg?: string) => {
    if (!canRunCustom) return;
    const text = textArg !== undefined ? textArg : customText;
    if (text.trim().length === 0) {
      setCustomError('Please enter some text.');
      return;
    }
    setIsRunningCustom(true);
    setCustomError(null);
    setCopyClicked(false);
    fetch('/api/model/head-attention/get', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ modelId, layer, headIndex, text }),
    })
      .then(async (res) => {
        if (!res.ok) {
          const errorBody = await res.json().catch(() => null);
          throw new Error(errorBody?.error || `Failed to run attention (${res.status})`);
        }
        return res.json();
      })
      .then((data: HeadSequenceData) => {
        setCustomResult(data);
        setIsRunningCustom(false);
        writeTestTextToUrl(data.tokens.join(''));
      })
      .catch((error) => {
        setCustomResult(null);
        setCustomError(error?.message || 'Failed to run attention');
        setIsRunningCustom(false);
      });
  };

  // Auto-run the custom text passed via the `defaulttesttext` URL param, once, so shared links
  // reproduce the tested result on load.
  useEffect(() => {
    if (didAutoRunRef.current) return;
    if (!canRunCustom) return;
    if (!defaultCustomText || defaultCustomText.trim().length === 0) return;
    didAutoRunRef.current = true;
    setCustomText(defaultCustomText);
    runCustom(defaultCustomText);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [canRunCustom, defaultCustomText]);

  // Drop a top-sequence's text into the custom field and immediately run it.
  const copyRemix = (text: string) => {
    if (!canRunCustom) return;
    setCustomText(text);
    runCustom(text);
    if (typeof window !== 'undefined') {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  // Discover the intervals present in the data, sorted highest-first.
  // The first (highest) interval is labeled "Top"; the rest use their numeric value.
  const intervals = useMemo(() => {
    const set = new Set<number>();
    sequences.forEach((s) => set.add(s.interval));
    return Array.from(set).sort((a, b) => b - a);
  }, [sequences]);

  const maxInterval = intervals[0] ?? null;
  const [selectedInterval, setSelectedInterval] = useState<number | null>(null);

  // Reset to "Top" whenever the dataset (and therefore its max interval) changes.
  useEffect(() => {
    setSelectedInterval(maxInterval);
  }, [maxInterval]);

  const sortedSequences = useMemo(
    () =>
      [...sequences].sort((a, b) => {
        if (a.interval !== b.interval) return b.interval - a.interval;
        return b.maxActivation - a.maxActivation;
      }),
    [sequences],
  );

  const filteredSequences = useMemo(() => {
    if (selectedInterval === null) return sortedSequences;
    return sortedSequences.filter((s) => s.interval === selectedInterval);
  }, [sortedSequences, selectedInterval]);

  // Color scale uses the full set so per-interval views stay comparable.
  const overallMaxActivation = useMemo(() => {
    let max = 0;
    sortedSequences.forEach((s) => {
      if (s.maxActivation > max) max = s.maxActivation;
    });
    return max;
  }, [sortedSequences]);

  useEffect(() => {
    setSelectedRange(defaultRange);
  }, [defaultRange]);

  useEffect(() => {
    setShowLineBreaks(defaultShowLineBreaks);
  }, [defaultShowLineBreaks]);

  useEffect(() => {
    setShowRawTokens(defaultShowRawTokens);
  }, [defaultShowRawTokens]);

  const hasLayerHead = layer !== undefined && headIndex !== undefined;
  const ATTENTION_ORANGE_RGB = '251, 146, 60';

  return (
    <div className={`flex w-full flex-col ${unbounded ? '' : 'max-h-[600px] overflow-y-auto overscroll-contain'}`}>
      {canRunCustom && (
        <div className="flex w-full flex-col gap-y-1.5 border-b border-slate-200 bg-white px-3 py-2 sm:px-4">
          <div className="text-[9px] font-medium uppercase text-slate-400">Test Custom Text</div>
          <div className="flex w-full flex-row items-stretch gap-x-1.5">
            <ReactTextareaAutosize
              value={customText}
              onChange={(e) => setCustomText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                  e.preventDefault();
                  runCustom();
                }
              }}
              minRows={2}
              placeholder="Enter custom text to see this head's attention pattern."
              className="form-input min-h-[38px] flex-1 resize-none rounded border border-slate-300 px-2.5 py-2 font-mono text-[11px] leading-tight text-slate-700 placeholder-slate-400 focus:border-sky-500 focus:outline-0 focus:ring-0"
            />
            <button
              type="button"
              onClick={() => runCustom()}
              disabled={isRunningCustom}
              className="flex w-[54px] min-w-[54px] flex-col items-center justify-center gap-y-0.5 rounded bg-sky-700 px-2.5 py-1.5 text-[11px] font-medium text-white hover:bg-sky-600 disabled:bg-slate-300 disabled:text-slate-400"
            >
              {isRunningCustom ? <LoadingSquare size={16} className="text-white" /> : 'Test'}
            </button>
          </div>
          {customError && <div className="text-[10px] font-medium text-rose-500">{customError}</div>}
          {customResult && (
            <div className="mt-1 flex w-full flex-col rounded-md border border-slate-200 bg-slate-50 px-3 pb-3 pt-0 sm:px-4">
              <HeadActivationItem
                key={`custom-${modelId}-${layer}-${headIndex}-${showLineBreaks}-${showRawTokens}-${maxAttentionMode}`}
                sequence={customResult}
                modelId={modelId}
                tokensToDisplayAroundMaxActToken={100000}
                showLineBreaks={showLineBreaks}
                showRawTokens={showRawTokens}
                enableExpanding={false}
                overallMaxActivationValueInList={customResult.maxActivation}
                overrideTextSize="text-[9.5px] sm:text-[11px]"
                maxAttentionMode={maxAttentionMode}
              />
              <div className="mt-0 flex flex-row items-center justify-start">
                <button
                  type="button"
                  className="my-1 flex w-[62px] cursor-pointer flex-row items-center justify-center gap-x-0.5 whitespace-pre rounded bg-slate-200 px-1.5 py-1.5 text-[9px] font-medium text-slate-600 hover:bg-slate-300 sm:px-2 sm:py-1.5 sm:text-[10.5px]"
                  title="Clear Result"
                  onClick={() => {
                    setCustomResult(null);
                    setCustomText('');
                    setCopyClicked(false);
                    clearTestTextFromUrl();
                  }}
                >
                  <XIcon className="h-3 w-3" /> Reset
                </button>
                <button
                  type="button"
                  className="my-1 ml-1.5 flex w-[62px] cursor-pointer flex-row items-center justify-center gap-x-0.5 whitespace-pre rounded bg-slate-200 px-1.5 py-1.5 text-[9px] font-medium text-slate-600 hover:bg-slate-300 sm:px-2 sm:py-1.5 sm:text-[10.5px]"
                  title="Share Custom Attention Test Result"
                  onClick={() => {
                    const url = `${window.location.origin}/${modelId}/head/${layer}/${headIndex}?defaulttesttext=${encodeURIComponent(
                      customResult.tokens.join(''),
                    )}`;
                    copy(url);
                    setCopyClicked(true);
                    showToastMessage(
                      <div className="flex flex-col items-center justify-center gap-y-1">
                        <div className="flex flex-row items-center justify-center gap-x-1 font-semibold">
                          <Copy className="h-4 w-4" /> Copied!
                        </div>
                        <div className="mt-1 text-xs">
                          The link to this head, including this attention test result, has been copied to your clipboard.
                        </div>
                      </div>,
                    );
                  }}
                >
                  {copyClicked ? (
                    <>
                      <Check className="h-3 w-3" /> Copied
                    </>
                  ) : (
                    <>
                      <Share className="h-3 w-3" /> Share
                    </>
                  )}
                </button>
              </div>
            </div>
          )}
        </div>
      )}
      {hasLayerHead && (
        <div
          className={`${unbounded ? '' : 'sticky top-0'} z-10 flex flex-row items-center justify-between border-b border-slate-200 bg-slate-50 px-3 py-1 pt-1.5 text-[11px] font-medium text-slate-600`}
        >
          <div className="flex flex-col">
            {/* <div className="flex flex-row items-center gap-x-2 font-bold">
              <span className="rounded px-0 font-mono text-[10px] uppercase text-sky-700">Layer {layer}</span> -
              <span className="rounded px-0 font-mono text-[10px] uppercase text-sky-700">Head {headIndex}</span>
            </div> */}
            <div className="text-[9px] font-medium uppercase text-slate-500">
              Top Sequences by <span className="rounded bg-orange-300 px-1 text-slate-600">Max Attention</span>
              <span className="hidden sm:inline">
                . Hover for{' '}
                <span
                  className="inline-block rounded bg-origin-border px-1 font-mono font-bold text-slate-700"
                  style={{
                    backgroundImage: `linear-gradient(to bottom, rgba(${ATTENTION_ORANGE_RGB}, 0.7) 50%, rgba(${ATTENTION_ORANGE_RGB}, 0) 50%)`,
                  }}
                >
                  Keys
                </span>{' '}
                and{' '}
                <span
                  className="inline-block rounded bg-origin-border px-1 font-mono font-bold text-slate-700"
                  style={{
                    backgroundImage: `linear-gradient(to bottom, rgba(${ATTENTION_ORANGE_RGB}, 0) 50%, rgba(${ATTENTION_ORANGE_RGB}, 0.7) 50%)`,
                  }}
                >
                  Queries
                </span>
                .
              </span>
            </div>
          </div>
          <div className="flex flex-row items-center gap-x-4">
            <div className="hidden flex-row items-center gap-x-2 sm:flex">
              <div className="mr-0 text-[9px] font-medium uppercase text-slate-400">Max Attention</div>
              <ToggleGroup.Root
                className="inline-flex overflow-hidden rounded border bg-white"
                type="single"
                value={maxAttentionMode}
                onValueChange={(value) => {
                  if (value) setMaxAttentionMode(value as 'all' | 'keys' | 'queries');
                }}
                aria-label="Max attention display mode"
              >
                {(
                  [
                    { value: 'all', label: 'All', bg: `rgba(${ATTENTION_ORANGE_RGB}, 0.7)` },
                    {
                      value: 'keys',
                      label: 'Keys',
                      bg: `linear-gradient(to bottom, rgba(${ATTENTION_ORANGE_RGB}, 0.7) 50%, rgba(${ATTENTION_ORANGE_RGB}, 0) 50%)`,
                    },
                    {
                      value: 'queries',
                      label: 'Queries',
                      bg: `linear-gradient(to bottom, rgba(${ATTENTION_ORANGE_RGB}, 0) 50%, rgba(${ATTENTION_ORANGE_RGB}, 0.7) 50%)`,
                    },
                  ] as const
                ).map((item) => (
                  <ToggleGroup.Item
                    key={item.value}
                    value={item.value}
                    aria-label={item.label}
                    className="flex items-center gap-x-1 rounded px-2 py-0.5 text-[10px] font-medium text-slate-400 transition-all hover:bg-slate-100 data-[state=on]:bg-slate-200 data-[state=on]:text-slate-600"
                  >
                    <span
                      className="inline-block h-2.5 w-2.5 rounded-sm border border-slate-200 bg-origin-border"
                      style={item.value === 'all' ? { backgroundColor: item.bg } : { backgroundImage: item.bg }}
                    />
                    {item.label}
                  </ToggleGroup.Item>
                ))}
              </ToggleGroup.Root>
            </div>
            {intervals.length > 0 && (
              <div className="flex flex-row items-center gap-x-2">
                <div className="mr-0 text-[9px] font-medium uppercase text-slate-400">Interval</div>
                <ToggleGroup.Root
                  className="inline-flex overflow-hidden rounded border bg-white"
                  type="single"
                  value={selectedInterval !== null ? selectedInterval.toString() : ''}
                  onValueChange={(value) => {
                    if (value) setSelectedInterval(Number(value));
                  }}
                  aria-label="Filter by interval"
                >
                  {intervals.map((interval) => (
                    <ToggleGroup.Item
                      key={interval}
                      value={interval.toString()}
                      aria-label={interval === maxInterval ? 'Top' : `Interval ${interval}`}
                      className="items-center rounded px-2 py-0.5 text-[10px] font-medium text-slate-400 transition-all hover:bg-slate-100 data-[state=on]:bg-slate-200 data-[state=on]:text-slate-600"
                    >
                      {interval === maxInterval ? 'Top' : interval}
                    </ToggleGroup.Item>
                  ))}
                </ToggleGroup.Root>
              </div>
            )}
          </div>
        </div>
      )}
      {isLoading ? (
        <div className="flex h-48 w-full flex-col items-center justify-center gap-y-2">
          <LoadingSquare size={24} className="text-sky-700" />
          <h1 className="text-sm font-medium text-slate-400">Loading sequences…</h1>
        </div>
      ) : errorMessage ? (
        <div className="flex h-48 w-full items-center justify-center px-4 text-center">
          <h1 className="text-sm font-medium text-rose-500">{errorMessage}</h1>
        </div>
      ) : filteredSequences.length === 0 ? (
        <div className="flex h-48 w-full items-center justify-center">
          <h1 className="text-sm font-medium text-slate-400">{emptyMessage}</h1>
        </div>
      ) : (
        <div className="flex flex-col">
          {filteredSequences.map((sequence, idx) => (
            <div
              key={`sequence-${sequence.id}`}
              className={`relative border-slate-200 px-3 py-1 sm:px-4 [&:not(:last-child)]:border-b ${
                selectedRange > 0 ? 'sm:py-2.5' : ''
              }`}
            >
              <div className="flex w-full flex-row items-center justify-center">
                <div className="flex w-full flex-auto flex-col text-left text-sm">
                  <HeadActivationItem
                    key={`${ACTIVATION_DISPLAY_DEFAULT_CONTEXT_TOKENS[selectedRange].size}-${idx}-${showLineBreaks}`}
                    sequence={sequence}
                    modelId={modelId}
                    tokensToDisplayAroundMaxActToken={ACTIVATION_DISPLAY_DEFAULT_CONTEXT_TOKENS[selectedRange].size}
                    showLineBreaks={showLineBreaks}
                    showRawTokens={showRawTokens}
                    overallMaxActivationValueInList={overallMaxActivation}
                    overrideTextSize="text-[9.5px] sm:text-[11px]"
                    maxAttentionMode={maxAttentionMode}
                    onCopyRemix={canRunCustom ? copyRemix : undefined}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
