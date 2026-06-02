'use client';

import * as ToggleGroup from '@radix-ui/react-toggle-group';
import { useEffect, useMemo, useState } from 'react';
import HeadActivationItem, { HeadSequenceData } from './head-activation-item';
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
}) {
  const [selectedRange, setSelectedRange] = useState(defaultRange);
  const [showLineBreaks, setShowLineBreaks] = useState(defaultShowLineBreaks);
  const [showRawTokens, setShowRawTokens] = useState(defaultShowRawTokens);

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
      {hasLayerHead && (
        <div
          className={`${unbounded ? '' : 'sticky top-0'} z-10 flex flex-row items-center justify-between border-b border-slate-200 bg-slate-50 px-3 py-1 pt-1.5 text-[11px] font-medium text-slate-600`}
        >
          <div className="flex flex-col">
            {/* <div className="flex flex-row items-center gap-x-2 font-bold">
              <span className="rounded px-0 font-mono text-[10px] uppercase text-sky-700">Layer {layer}</span> -
              <span className="rounded px-0 font-mono text-[10px] uppercase text-sky-700">Head {headIndex}</span>
            </div> */}
            <div className="text-[9px] font-medium uppercase text-slate-400">
              Top Sequences by <span className="rounded bg-orange-300 px-1 text-slate-600">Max Attention</span>. Hover
              for{' '}
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
            </div>
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
