'use client';

import CustomTooltip from '@/components/custom-tooltip';
import { HeadSequenceData } from '@/components/head-activation-item';
import HeadActivationsList from '@/components/head-activations-list';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/shadcn/card';
import { Label } from '@/components/shadcn/label';
import { QuestionMarkCircledIcon } from '@radix-ui/react-icons';
import * as RadixSlider from '@radix-ui/react-slider';
import * as ToggleGroup from '@radix-ui/react-toggle-group';
import dynamic from 'next/dynamic';
import { useEffect, useMemo, useRef, useState } from 'react';
import createPlotlyComponent from 'react-plotly.js/factory';

const Plot = dynamic(
  () => Promise.resolve(import('plotly.js-dist-min').then((Plotly) => createPlotlyComponent(Plotly))),
  {
    ssr: false,
  },
);

type Histogram = { bin_edges: number[]; bin_values: number[] };
type TopToken = { token: string; weight: number };

type ModelHeadMetricsRow = {
  layer: number;
  headIndex: number;
  inductionScore: number | null;
  prevTokenScore: number | null;
  patternEntropy: number | null;
  selfAttentionScore?: number | null;
  qkDistance?: number | null;
  qkDistanceVariance?: number | null;
  activationHistogram?: unknown;
  qkDistanceHistogram?: unknown;
  topQueryTokens?: unknown;
  topKeyTokens?: unknown;
};

const METRIC_OPTIONS = [
  {
    key: 'inductionScore',
    label: 'Induction Score',
  },
  {
    key: 'prevTokenScore',
    label: 'Previous Token Score',
  },
  {
    key: 'patternEntropy',
    label: 'Attention Entropy',
  },
] as const;

type MetricKey = (typeof METRIC_OPTIONS)[number]['key'];

// Extra metrics shown alongside the selected head's distribution (not filterable).
const EXTRA_METRIC_OPTIONS = [
  { key: 'selfAttentionScore', label: 'Self Attention Score' },
  { key: 'qkDistance', label: 'Q-K Distance' },
  { key: 'qkDistanceVariance', label: 'Q-K Distance Variance' },
] as const;

type ExtraMetricKey = (typeof EXTRA_METRIC_OPTIONS)[number]['key'];

type LayerDisplayRow = { type: 'layer'; layer: number } | { type: 'collapsed'; startLayer: number; endLayer: number };

function getCellBackground(value: number | null | undefined, min: number, max: number) {
  if (value == null || !Number.isFinite(value)) {
    return '#ffffff';
  }
  if (max === min) {
    return 'rgb(56 189 248)';
  }

  const opacity = (value - min) / (max - min);
  return `rgb(56 189 248 / ${Math.max(0, Math.min(1, opacity))})`;
}

function formatTooltipValue(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value)) {
    return '—';
  }
  return value.toFixed(3);
}

function parseHistogram(raw: unknown): Histogram | null {
  if (!raw || typeof raw !== 'object') {
    return null;
  }
  const binEdges = (raw as { bin_edges?: unknown }).bin_edges;
  const binValues = (raw as { bin_values?: unknown }).bin_values;
  if (
    !Array.isArray(binEdges) ||
    !Array.isArray(binValues) ||
    binValues.length === 0 ||
    binEdges.length < binValues.length + 1
  ) {
    return null;
  }
  return { bin_edges: binEdges as number[], bin_values: binValues as number[] };
}

function parseTopTokens(raw: unknown, limit: number): TopToken[] {
  if (!Array.isArray(raw)) {
    return [];
  }
  return raw
    .filter(
      (entry): entry is TopToken =>
        !!entry &&
        typeof entry === 'object' &&
        typeof (entry as TopToken).token === 'string' &&
        Number.isFinite((entry as TopToken).weight),
    )
    .slice(0, limit);
}

// Renders whitespace-only tokens with a visible placeholder so chips keep height.
function formatTokenLabel(token: string) {
  const escaped = token.replaceAll('\n', '\\n');
  if (escaped.length > 0 && escaped.trim() === '') {
    return escaped.replace(/\s/g, '\u00A0');
  }
  return escaped;
}

export default function ModelHeadMetricsPane({
  modelId,
  metrics,
}: {
  modelId: string;
  metrics: ModelHeadMetricsRow[];
}) {
  const [selectedMetric, setSelectedMetric] = useState<MetricKey>('inductionScore');
  const [showTopN, setShowTopN] = useState(8);
  const [forceShownLayers, setForceShownLayers] = useState<Set<number>>(new Set());
  const [selectedHead, setSelectedHead] = useState<{ layer: number; headIndex: number } | null>(null);
  const [sequences, setSequences] = useState<HeadSequenceData[]>([]);
  const [isLoadingSequences, setIsLoadingSequences] = useState(false);
  const [sequencesError, setSequencesError] = useState<string | null>(null);
  const sequencesRequestId = useRef(0);

  useEffect(() => {
    if (!selectedHead) {
      setSequences([]);
      setSequencesError(null);
      setIsLoadingSequences(false);
      return;
    }
    const requestId = sequencesRequestId.current + 1;
    sequencesRequestId.current = requestId;
    const controller = new AbortController();
    setIsLoadingSequences(true);
    setSequencesError(null);
    fetch('/api/model/head-sequences/get', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ modelId, layer: selectedHead.layer, headIndex: selectedHead.headIndex }),
      signal: controller.signal,
    })
      .then(async (res) => {
        if (!res.ok) {
          const errorBody = await res.json().catch(() => null);
          throw new Error(errorBody?.error || `Failed to load sequences (${res.status})`);
        }
        return res.json();
      })
      .then((data: HeadSequenceData[]) => {
        if (requestId !== sequencesRequestId.current) return;
        setSequences(Array.isArray(data) ? data : []);
        setIsLoadingSequences(false);
      })
      .catch((error) => {
        if (controller.signal.aborted) return;
        if (requestId !== sequencesRequestId.current) return;
        setSequences([]);
        setSequencesError(error?.message || 'Failed to load sequences');
        setIsLoadingSequences(false);
      });
    return () => {
      controller.abort();
    };
  }, [modelId, selectedHead]);

  const maxHeads = Math.max(1, metrics.length);

  const ranksByMetric = useMemo(() => {
    const result = new Map<MetricKey, Map<string, { rank: number; total: number }>>();
    METRIC_OPTIONS.forEach((opt) => {
      const entries = metrics
        .map((m) => ({ key: `${m.layer}:${m.headIndex}`, value: m[opt.key] }))
        .filter((entry): entry is { key: string; value: number } => entry.value != null && Number.isFinite(entry.value))
        .sort((a, b) => b.value - a.value);
      const rankMap = new Map<string, { rank: number; total: number }>();
      entries.forEach((entry, index) => {
        rankMap.set(entry.key, { rank: index + 1, total: entries.length });
      });
      result.set(opt.key, rankMap);
    });
    return result;
  }, [metrics]);

  const { displayRows, topHeadsByLayer, metricByLayerAndHead, min, max } = useMemo(() => {
    const layersSet = new Set<number>();
    const headIndexesSet = new Set<number>();
    const metricMap = new Map<string, ModelHeadMetricsRow>();

    metrics.forEach((metric) => {
      layersSet.add(metric.layer);
      headIndexesSet.add(metric.headIndex);
      const key = `${metric.layer}:${metric.headIndex}`;
      if (!metricMap.has(key)) {
        metricMap.set(key, metric);
      }
    });

    const values = metrics
      .map((metric) => metric[selectedMetric])
      .filter((value): value is number => value != null && Number.isFinite(value));
    const sortedValues = [...values].sort((a, b) => b - a);
    const topValueCount = Math.max(1, Math.min(showTopN, sortedValues.length));
    const topThreshold = sortedValues[Math.min(topValueCount - 1, sortedValues.length - 1)] ?? Number.POSITIVE_INFINITY;
    const layers = Array.from(layersSet).sort((a, b) => a - b);
    const sortedHeadIndexes = Array.from(headIndexesSet).sort((a, b) => a - b);

    const topHeadsMap = new Map<number, number[]>();
    layers.forEach((layer) => {
      const topHeads = sortedHeadIndexes.filter((headIndex) => {
        const value = metricMap.get(`${layer}:${headIndex}`)?.[selectedMetric];
        return value != null && Number.isFinite(value) && value >= topThreshold;
      });
      if (topHeads.length > 0) {
        topHeadsMap.set(layer, topHeads);
      }
    });

    const rows: LayerDisplayRow[] = [];
    for (let index = 0; index < layers.length; index += 1) {
      const layer = layers[index];
      if (topHeadsMap.has(layer) || forceShownLayers.has(layer)) {
        rows.push({ type: 'layer', layer });
        continue;
      }

      const startLayer = layer;
      let endLayer = layer;
      while (
        index + 1 < layers.length &&
        !topHeadsMap.has(layers[index + 1]) &&
        !forceShownLayers.has(layers[index + 1])
      ) {
        index += 1;
        endLayer = layers[index];
      }
      rows.push({ type: 'collapsed', startLayer, endLayer });
    }

    return {
      displayRows: rows,
      topHeadsByLayer: topHeadsMap,
      metricByLayerAndHead: metricMap,
      min: values.length > 0 ? Math.min(...values) : 0,
      max: values.length > 0 ? Math.max(...values) : 0,
    };
  }, [forceShownLayers, metrics, selectedMetric, showTopN]);

  const selectedHeadRow = selectedHead
    ? (metricByLayerAndHead.get(`${selectedHead.layer}:${selectedHead.headIndex}`) ?? null)
    : null;

  const selectedHistogram = useMemo(() => {
    const parsed = parseHistogram(selectedHeadRow?.activationHistogram);
    if (!parsed) {
      return null;
    }
    const centers = parsed.bin_values.map((_, i) => {
      const lo = parsed.bin_edges[i] ?? 0;
      const hi = parsed.bin_edges[i + 1] ?? lo;
      return (lo + hi) / 2;
    });
    return { centers, binEdges: parsed.bin_edges, binValues: parsed.bin_values };
  }, [selectedHeadRow]);

  const qkDistanceHistogram = useMemo(() => {
    const parsed = parseHistogram(selectedHeadRow?.qkDistanceHistogram);
    if (!parsed) {
      return null;
    }
    // QK distance bins are log-scaled (0,1,2,4,8,...), so use the bin's lower edge
    // as an evenly-spaced category label instead of a numeric axis.
    const labels = parsed.bin_values.map((_, i) => `${parsed.bin_edges[i]}`);
    return { labels, binEdges: parsed.bin_edges, binValues: parsed.bin_values };
  }, [selectedHeadRow]);

  const topQueryTokens = useMemo(() => parseTopTokens(selectedHeadRow?.topQueryTokens, 5), [selectedHeadRow]);
  const topKeyTokens = useMemo(() => parseTopTokens(selectedHeadRow?.topKeyTokens, 5), [selectedHeadRow]);

  const updateShowTopN = (nextValue: number) => {
    setShowTopN(Math.max(1, Math.min(maxHeads, nextValue)));
    setForceShownLayers(new Set());
  };

  const SLIDER_RESOLUTION = 1000;
  const logMaxHeads = Math.log(Math.max(2, maxHeads));

  const nToSliderValue = (n: number) => {
    if (maxHeads <= 1) return 0;
    const clamped = Math.max(1, Math.min(maxHeads, n));
    return Math.round((Math.log(clamped) / logMaxHeads) * SLIDER_RESOLUTION);
  };

  const sliderValueToN = (sliderPos: number) => {
    if (maxHeads <= 1) return 1;
    const fraction = sliderPos / SLIDER_RESOLUTION;
    const exact = Math.exp(fraction * logMaxHeads);
    if (exact < 4) {
      return Math.max(1, Math.min(maxHeads, Math.round(exact)));
    }
    return Math.max(1, Math.min(maxHeads, Math.round(exact / 4) * 4));
  };

  return (
    <Card className="w-full bg-white">
      <CardHeader className="pb-3">
        <div className="flex flex-row items-end justify-between gap-x-2 gap-y-1">
          <CardTitle>Attention Visualizer</CardTitle>
          <CustomTooltip
            trigger={
              <div className="flex flex-row items-center gap-x-1 text-xs font-medium text-slate-400">
                <a
                  href="https://transformer-circuits.pub/2026/headvis/index.html"
                  target="_blank"
                  rel="noreferrer"
                  className="text-sky-700 hover:underline"
                >
                  HeadVis
                </a>{' '}
                (Luger, Kamath et al.) <QuestionMarkCircledIcon className="h-4 w-4" />
              </div>
            }
          >
            <div className="flex flex-col">
              <p>
                The attention visualizer is based on
                <a
                  href="https://transformer-circuits.pub/2026/headvis/index.html"
                  target="_blank"
                  rel="noreferrer"
                  className="text-sky-700 hover:underline"
                >
                  HeadVis
                </a>
                , an interactive tool for investigating attention heads in a model.
              </p>
              <p className="mt-1.5">
                Raw model metrics and attention sequences are available in our{' '}
                <a
                  href="https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/index.html?prefix=v1/"
                  target="_blank"
                  rel="noreferrer"
                  className="text-sky-700 hover:underline"
                >
                  exports
                </a>{' '}
                under the model&apos;s <code>headvis</code> folder.
              </p>
              <ul className="mt-1.5 list-inside list-disc text-[11px] font-normal text-slate-600">
                <li>
                  <a
                    href="https://transformer-circuits.pub/2026/headvis/index.html"
                    target="_blank"
                    rel="noreferrer"
                    className="text-sky-700 hover:underline"
                  >
                    Paper
                  </a>
                </li>
                <li>
                  <a
                    href="https://github.com/anthropics/headvis"
                    target="_blank"
                    rel="noreferrer"
                    className="text-sky-700 hover:underline"
                  >
                    Reference Spec
                  </a>
                </li>
                <li>
                  <a
                    href="https://transformer-circuits.pub/2026/headvis/gemma3/index.html"
                    target="_blank"
                    rel="noreferrer"
                    className="text-sky-700 hover:underline"
                  >
                    Reference Demo
                  </a>
                </li>
              </ul>
            </div>
          </CustomTooltip>
        </div>
      </CardHeader>
      <CardContent className="flex w-full flex-col gap-x-2 gap-y-0 overflow-x-auto">
        <div className="flex w-full flex-1 flex-col gap-2">
          <div className="mb-0 flex flex-1 flex-row gap-x-3 gap-y-1.5">
            <div className="mt-1 flex flex-1 flex-col">
              <div className="mb-1 text-[9px] font-medium uppercase text-slate-400">1 - Filter Heads By Metric</div>
              <ToggleGroup.Root
                className="inline-flex flex-1 overflow-hidden rounded bg-slate-100 px-0 py-0 sm:rounded-md"
                type="single"
                value={selectedMetric}
                onValueChange={(value) => {
                  if (value) setSelectedMetric(value as MetricKey);
                }}
                aria-label="Head metric"
              >
                {METRIC_OPTIONS.map((option) => (
                  <ToggleGroup.Item
                    key={option.key}
                    value={option.key}
                    aria-label={option.label}
                    className="flex-1 items-center rounded px-0 py-1 text-[10px] font-medium text-slate-400 transition-all hover:bg-slate-100 data-[state=on]:bg-slate-200 data-[state=on]:text-slate-600 sm:rounded-md sm:py-1.5 sm:text-[11px]"
                  >
                    {option.label}
                  </ToggleGroup.Item>
                ))}
              </ToggleGroup.Root>
            </div>
            <div className="flex flex-1 flex-col">
              <div className="mb-1 text-[9px] font-medium uppercase text-slate-400">&nbsp;</div>
              <div className="flex flex-1 flex-row items-center rounded px-2 py-0.5">
                <Label
                  htmlFor="showTopN"
                  className="mr-2.5 min-w-[72px] max-w-[72px] whitespace-nowrap text-center text-[9px] leading-[10px] text-slate-400"
                >
                  {showTopN} Top Head{showTopN === 1 ? '' : 's'}
                </Label>
                <RadixSlider.Root
                  id="showTopN"
                  name="showTopN"
                  value={[nToSliderValue(showTopN)]}
                  onValueChange={(newVal) => updateShowTopN(sliderValueToN(newVal[0]))}
                  min={0}
                  max={SLIDER_RESOLUTION}
                  step={1}
                  className="relative flex h-4 w-20 min-w-20 flex-1 touch-none select-none items-center"
                >
                  <RadixSlider.Track className="relative h-1 w-full flex-grow overflow-hidden rounded-full bg-slate-300">
                    <RadixSlider.Range className="absolute h-full rounded-full bg-sky-600" />
                  </RadixSlider.Track>
                  <RadixSlider.Thumb className="block h-3 w-3 rounded-full border border-sky-600 bg-white shadow transition-colors focus:outline-none focus:ring-0 disabled:pointer-events-none disabled:opacity-50" />
                </RadixSlider.Root>
              </div>
            </div>
          </div>
          <div className="flex w-full flex-col items-stretch gap-x-1">
            <div className="mb-1 mt-1.5 text-[9px] font-medium uppercase text-slate-400">
              2 - Select an Attention Head
            </div>
            <div className="flex w-full flex-row items-stretch gap-3">
              <div className="flex flex-1 basis-0 flex-col items-start justify-start gap-1">
                {displayRows.map((displayRow) => {
                  if (displayRow.type === 'collapsed') {
                    return <></>;
                  }

                  const topHeads = topHeadsByLayer.get(displayRow.layer) ?? [];
                  const currentMetricOption = METRIC_OPTIONS.find((opt) => opt.key === selectedMetric);
                  const otherMetricOptions = METRIC_OPTIONS.filter((opt) => opt.key !== selectedMetric);
                  return (
                    <div key={displayRow.layer} className="flex w-full flex-row items-start gap-1">
                      <div className="flex h-5 w-16 min-w-16 items-center justify-start pl-2.5 font-mono text-[9.5px] font-bold uppercase text-sky-700">
                        Layer {displayRow.layer}
                      </div>
                      <div className="flex flex-1 flex-row flex-wrap gap-1.5">
                        {topHeads.map((headIndex) => {
                          const cellKey = `${displayRow.layer}:${headIndex}`;
                          const row = metricByLayerAndHead.get(cellKey);
                          const value = row?.[selectedMetric] ?? null;
                          const renderMetricValue = (metricKey: MetricKey) => {
                            const metricValue = row?.[metricKey];
                            const rankInfo = ranksByMetric.get(metricKey)?.get(cellKey);
                            if (metricValue == null || !Number.isFinite(metricValue) || !rankInfo) {
                              return formatTooltipValue(metricValue);
                            }
                            return `${formatTooltipValue(metricValue)} (${rankInfo.rank} of ${rankInfo.total})`;
                          };

                          const isSelectedHead =
                            selectedHead?.layer === displayRow.layer && selectedHead?.headIndex === headIndex;
                          return (
                            <CustomTooltip
                              key={`${displayRow.layer}-${headIndex}`}
                              side="top"
                              minMargin
                              trigger={
                                <button
                                  type="button"
                                  onClick={() => {
                                    setSelectedHead({ layer: displayRow.layer, headIndex });
                                  }}
                                  className={`flex h-5 w-[68px] items-center justify-center rounded px-1 text-center font-mono text-[9.5px] font-bold uppercase text-sky-800 outline-none hover:outline hover:outline-2 hover:outline-sky-600 focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-600 ${
                                    isSelectedHead ? 'outline outline-2 outline-sky-600' : ''
                                  }`}
                                  style={{ backgroundColor: getCellBackground(value, min, max) }}
                                >
                                  Head {headIndex}
                                </button>
                              }
                            >
                              <div className="flex min-w-[200px] flex-col gap-y-1 text-[11px]">
                                <div className="flex w-full flex-row justify-between">
                                  <div className="font-mono text-slate-600">
                                    Layer {displayRow.layer} - Head {headIndex}
                                  </div>
                                </div>

                                <div className="mb-1 mt-1 flex flex-row items-center justify-between gap-x-4 rounded-md border border-sky-600 px-2 py-1 font-semibold text-sky-700">
                                  <span>{currentMetricOption?.label}</span>
                                  <span className="font-mono">{renderMetricValue(selectedMetric)}</span>
                                </div>
                                {otherMetricOptions.map((opt) => (
                                  <div
                                    key={opt.key}
                                    className="flex flex-row items-center justify-between gap-x-4 text-slate-500"
                                  >
                                    <span>{opt.label}</span>
                                    <span className="font-mono">{renderMetricValue(opt.key)}</span>
                                  </div>
                                ))}
                              </div>
                            </CustomTooltip>
                          );
                        })}
                      </div>
                    </div>
                  );
                })}
              </div>
              <div className="flex flex-1 basis-0 flex-col rounded bg-white p-0 pt-0">
                {selectedHead && selectedHeadRow ? (
                  <>
                    <div className="mb-1 flex flex-row items-center justify-center gap-x-1 text-center text-[10px] font-medium uppercase text-slate-400">
                      Max Activation Distribution
                    </div>
                    {selectedHistogram ? (
                      <Plot
                        className="w-full"
                        useResizeHandler
                        data={[
                          {
                            x: selectedHistogram.centers,
                            y: selectedHistogram.binValues,
                            type: 'bar',
                            marker: { color: 'rgba(249, 115, 22, 0.85)' },
                            hovertemplate: selectedHistogram.centers.map((_, i) => {
                              const lo = selectedHistogram.binEdges[i];
                              const hi = selectedHistogram.binEdges[i + 1];
                              const count = selectedHistogram.binValues[i];
                              return `<b>Activation Range (x)</b>: ${lo?.toFixed(2)} – ${hi?.toFixed(2)}<br><b># Sequences (y)</b>: ${count.toLocaleString()}<extra></extra>`;
                            }),
                          },
                        ]}
                        layout={{
                          height: 80,
                          xaxis: {
                            showgrid: false,
                            zeroline: false,
                            fixedrange: true,
                            tickfont: { size: 9 },
                          },
                          yaxis: {
                            showgrid: false,
                            zeroline: false,
                            showticklabels: false,
                            fixedrange: true,
                          },
                          barmode: 'relative',
                          bargap: 0.05,
                          showlegend: false,
                          margin: { l: 6, r: 6, b: 18, t: 2, pad: 2 },
                          paper_bgcolor: 'rgba(0,0,0,0)',
                          plot_bgcolor: 'rgba(0,0,0,0)',
                        }}
                        config={{
                          responsive: true,
                          displayModeBar: false,
                          editable: false,
                          scrollZoom: false,
                        }}
                      />
                    ) : (
                      <div className="flex h-[130px] w-full items-center justify-center text-center">
                        <p className="text-[11px] font-medium text-slate-400">
                          No activation histogram available for this head.
                        </p>
                      </div>
                    )}
                    <div className="mt-2 flex flex-col gap-y-1 border-t border-slate-100 pt-2">
                      {METRIC_OPTIONS.map((opt) => (
                        <div
                          key={opt.key}
                          className="flex flex-row items-center justify-between gap-x-4 px-1 text-[10px] text-slate-500"
                        >
                          <span>{opt.label}</span>
                          <span className="font-mono text-slate-700">
                            {formatTooltipValue(selectedHeadRow[opt.key])}
                          </span>
                        </div>
                      ))}
                      {EXTRA_METRIC_OPTIONS.map((opt) => (
                        <div
                          key={opt.key}
                          className="flex flex-row items-center justify-between gap-x-4 px-1 text-[10px] text-slate-500"
                        >
                          <span>{opt.label}</span>
                          <span className="font-mono text-slate-700">
                            {formatTooltipValue(selectedHeadRow[opt.key as ExtraMetricKey])}
                          </span>
                        </div>
                      ))}
                    </div>
                  </>
                ) : (
                  <div className="flex h-full min-h-[12rem] w-full items-center justify-center px-4 text-center">
                    <p className="text-xs font-bold text-slate-300">Activation Distribution & Metrics</p>
                  </div>
                )}
              </div>
              <div className="flex flex-1 basis-0 flex-col rounded bg-white p-0 pt-0">
                {selectedHead && selectedHeadRow ? (
                  <>
                    <div className="mb-1 flex flex-row items-center justify-center gap-x-1 text-center text-[10px] font-medium uppercase text-slate-400">
                      Q-K Distance Distribution
                    </div>
                    {qkDistanceHistogram ? (
                      <Plot
                        className="w-full"
                        useResizeHandler
                        data={[
                          {
                            x: qkDistanceHistogram.labels,
                            y: qkDistanceHistogram.binValues,
                            type: 'scatter',
                            mode: 'lines+markers',
                            line: { color: 'rgba(14, 165, 233, 0.9)', width: 2 },
                            marker: { color: 'rgba(14, 165, 233, 0.9)', size: 4 },
                            hovertemplate: qkDistanceHistogram.binValues.map((_, i) => {
                              const lo = qkDistanceHistogram.binEdges[i];
                              const hi = qkDistanceHistogram.binEdges[i + 1];
                              const weight = qkDistanceHistogram.binValues[i];
                              return `<b>|q − k| Range (x)</b>: ${lo} – ${hi}<br><b>Attention Mass (y)</b>: ${weight.toLocaleString()}<extra></extra>`;
                            }),
                          },
                        ]}
                        layout={{
                          height: 80,
                          xaxis: {
                            type: 'category',
                            showgrid: false,
                            zeroline: false,
                            fixedrange: true,
                            tickfont: { size: 9 },
                          },
                          yaxis: {
                            showgrid: false,
                            zeroline: false,
                            showticklabels: false,
                            fixedrange: true,
                          },
                          showlegend: false,
                          margin: { l: 6, r: 6, b: 18, t: 2, pad: 2 },
                          paper_bgcolor: 'rgba(0,0,0,0)',
                          plot_bgcolor: 'rgba(0,0,0,0)',
                        }}
                        config={{
                          responsive: true,
                          displayModeBar: false,
                          editable: false,
                          scrollZoom: false,
                        }}
                      />
                    ) : (
                      <div className="flex h-[100px] w-full items-center justify-center text-center">
                        <p className="text-[11px] font-medium text-slate-400">
                          No Q-K distance distribution available for this head.
                        </p>
                      </div>
                    )}
                    <div className="mt-2 flex flex-row gap-x-3 border-t border-slate-100 pt-2">
                      <div className="flex flex-1 basis-0 flex-col">
                        <div className="mb-1 text-center text-[9px] font-medium uppercase text-slate-400">
                          Top Query Tokens
                        </div>
                        {topQueryTokens.length > 0 ? (
                          topQueryTokens.map((entry, i) => (
                            <div
                              key={`query-${i}`}
                              className="flex flex-row items-center justify-between gap-x-2 px-1 py-0.5 text-[9px]"
                            >
                              <span className="max-w-[70%] truncate rounded bg-slate-100 px-1 font-mono text-slate-700">
                                {formatTokenLabel(entry.token)}
                              </span>
                              <span className="font-mono text-slate-500">{entry.weight.toFixed(2)}</span>
                            </div>
                          ))
                        ) : (
                          <div className="px-1 text-center text-[10px] text-slate-400">—</div>
                        )}
                      </div>
                      <div className="flex flex-1 basis-0 flex-col">
                        <div className="mb-1 text-center text-[9px] font-medium uppercase text-slate-400">
                          Top Key Tokens
                        </div>
                        {topKeyTokens.length > 0 ? (
                          topKeyTokens.map((entry, i) => (
                            <div
                              key={`key-${i}`}
                              className="flex flex-row items-center justify-between gap-x-2 px-1 py-0.5 text-[9px]"
                            >
                              <span className="max-w-[70%] truncate rounded bg-slate-100 px-1 font-mono text-slate-700">
                                {formatTokenLabel(entry.token)}
                              </span>
                              <span className="font-mono text-slate-500">{entry.weight.toFixed(2)}</span>
                            </div>
                          ))
                        ) : (
                          <div className="px-1 text-center text-[10px] text-slate-400">—</div>
                        )}
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="flex h-full min-h-[12rem] w-full items-center justify-center px-4 text-center">
                    <p className="text-xs font-bold text-slate-300">Q-K Distance Distribution & Top Tokens</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
        <div className="mb-1 mt-4 text-[9px] font-medium uppercase text-slate-400">3 - Explore Top Sequences</div>
        <div className="mt-1 flex min-w-0 flex-1 flex-col overflow-hidden rounded border border-slate-200 bg-white">
          {selectedHead ? (
            <HeadActivationsList
              sequences={sequences}
              modelId={modelId}
              layer={selectedHead.layer}
              headIndex={selectedHead.headIndex}
              isLoading={isLoadingSequences}
              errorMessage={sequencesError}
            />
          ) : (
            <div className="flex h-full min-h-[12rem] w-full items-center justify-center px-4 text-center">
              <p className="text-xs font-medium text-slate-400">Click a head to view its top activating sequences.</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
