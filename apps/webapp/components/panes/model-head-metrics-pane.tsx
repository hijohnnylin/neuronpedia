'use client';

import CustomTooltip from '@/components/custom-tooltip';
import FeatureSelector from '@/components/feature-selector/feature-selector';
import { HeadSequenceData } from '@/components/head-activation-item';
import HeadActivationsList from '@/components/head-activations-list';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/shadcn/card';
import { LoadingSquare } from '@/components/svg/loading-square';
import { useRouter } from '@bprogress/next';
import { Cross2Icon, ExternalLinkIcon, QuestionMarkCircledIcon } from '@radix-ui/react-icons';
import * as RadixSlider from '@radix-ui/react-slider';
import * as ToggleGroup from '@radix-ui/react-toggle-group';
import dynamic from 'next/dynamic';
import { useSearchParams } from 'next/navigation';
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

// Lightweight per-head metrics loaded up front for every head on model load.
export type ModelHeadMetricsRow = {
  layer: number;
  headIndex: number;
  inductionScore: number | null;
  prevTokenScore: number | null;
  patternEntropy: number | null;
  selfAttentionScore: number | null;
};

// Heavy per-head detail fetched on demand when a head is selected.
type ModelHeadDetail = {
  layer: number;
  headIndex: number;
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
    label: 'Prev Token Score',
  },
  {
    key: 'patternEntropy',
    label: 'Attention Entropy',
  },
  { key: 'selfAttentionScore', label: 'Self Attention' },
] as const;

type MetricKey = (typeof METRIC_OPTIONS)[number]['key'];

// Extra metrics shown alongside the selected head's distribution (not filterable).
const EXTRA_METRIC_OPTIONS = [
  // { key: 'selfAttentionScore', label: 'Self Attention Score' },
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

// Scrolls `container` (only) so `child` is visible. Adjusts the container's own
// scrollTop instead of using scrollIntoView, so parent/page scroll is untouched.
function scrollContainerToChild(container: HTMLElement | null, child: HTMLElement | null) {
  if (!container || !child) {
    return;
  }
  const containerRect = container.getBoundingClientRect();
  const childRect = child.getBoundingClientRect();
  if (childRect.top < containerRect.top) {
    container.scrollTop += childRect.top - containerRect.top;
  } else if (childRect.bottom > containerRect.bottom) {
    container.scrollTop += childRect.bottom - containerRect.bottom;
  }
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

export function getHeadMetricsTooltip(showFirstParagraph = true) {
  return (
    <div className="flex flex-col gap-y-2">
      {showFirstParagraph && (
        <p>
          Models have many layers, each with many heads. Find Head By Metric finds the top heads in a model by a given
          metric, which are often associated with important heads. Click to select a metric, and adjust the slider to
          change the top N heads to display.
        </p>
      )}
      <ul className="mb-1 ml-4 list-outside list-disc text-[11px] font-normal text-slate-600">
        <li>
          <strong>Induction Score</strong> - An induction head lets the model repeat previously seen token patterns. For
          example, if the model saw &quot;Jane Smith said, my name is Jane...&quot;, the induction head causes the model
          to complete the next token with &quot;Smith&quot;, in the pattern [A][B] ... [A]-&gt;[B]. The
          <strong>induction score</strong> is the average of the attention from second-A positions back to first-B
          positions across many sequences. A high induction score indicates a head is likely an induction head.
        </li>
        <li>
          <strong>Previous Token Score</strong> - This measures how reliably a head&apos;s attention concentrates on
          position-1 (one position back) across the dataset. A high previous token score indicates a head is likely a
          previous-token head.
        </li>
        <li>
          <strong>Pattern Entropy</strong> - This measures how concentrated the attention of a head. Low entropy =
          sharp, high = diffuse/sink-like.
        </li>
        <li>
          <strong>Self-Attention Score</strong> - This measures how much attention a head directs to itself. A high
          self-attention score indicates a head is likely a self-attention head.
        </li>
      </ul>

      <p>
        The implementations are based on{' '}
        <a
          href="https://transformer-circuits.pub/2026/headvis/index.html"
          target="_blank"
          rel="noreferrer"
          className="font-bold text-sky-700 hover:underline"
        >
          HeadVis
        </a>{' '}
        by Luger, Kamath et al (Anthropic 2026).
      </p>
      <p>
        For the specifications of how these are calculated, check out the{' '}
        <a
          href="https://github.com/anthropics/headvis/blob/304acba075bee56b80f8bc60e18896c8d597b4cd/data_pipeline.py#L59-L81"
          target="_blank"
          rel="noreferrer"
          className="font-bold text-sky-700 hover:underline"
        >
          HeadVis repository
        </a>
        .
      </p>
      <p>
        For our specific implementation for Neuronpedia, check out our{' '}
        <a
          href="https://github.com/neuronpedia/neuronpedia/blob/main/utils/neuronpedia-utils/neuronpedia_utils/headvis/compute-head-metrics.py"
          target="_blank"
          rel="noreferrer"
          className="font-bold text-sky-700 hover:underline"
        >
          generation script
        </a>
        .
      </p>
      <p>
        All data is downloadable from our{' '}
        <a
          href="https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/index.html?prefix=v1/"
          target="_blank"
          rel="noreferrer"
          className="font-bold text-sky-700 hover:underline"
        >
          exports bucket
        </a>{' '}
        under the model name, under the &quot;HeadVis&quot; folder.
      </p>
    </div>
  );
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
  showCard = true,
  initialLayer,
  initialHeadIndex,
  onHeadChange,
  inferenceEnabled = false,
}: {
  modelId: string;
  metrics: ModelHeadMetricsRow[];
  // When false, renders just the card body content without the surrounding Card/CardHeader wrapper.
  showCard?: boolean;
  initialLayer?: number;
  initialHeadIndex?: number;
  // Notified when the selected head changes (head-page mode), e.g. so breadcrumbs can stay in sync.
  onHeadChange?: (head: { modelId: string; layer: number; headIndex: number }) => void;
  // When true (model has an inference host), enable the per-head custom-text box.
  inferenceEnabled?: boolean;
}) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [selectedMetric, setSelectedMetric] = useState<MetricKey>('inductionScore');
  const [showTopN, setShowTopN] = useState(8);
  const [forceShownLayers, setForceShownLayers] = useState<Set<number>>(new Set());
  // On the head page (showCard=false), the in-pane head finder is hidden behind a toggle. Its
  // open state is mirrored in the `headFinder` query param so it persists across navigation.
  const [headFinderOpen, setHeadFinderOpen] = useState(searchParams.get('headFinder') === 'true');
  // The finder can target a different model than the page (when the user switches models in the
  // selector), so it loads that model's metrics on demand instead of reusing the page's `metrics`.
  const [finderModelId, setFinderModelId] = useState(modelId);
  const [finderMetrics, setFinderMetrics] = useState<ModelHeadMetricsRow[]>(metrics);
  const [isLoadingFinderMetrics, setIsLoadingFinderMetrics] = useState(false);
  const finderMetricsRequestId = useRef(0);
  const [selectedHead, setSelectedHead] = useState<{ layer: number; headIndex: number } | null>(
    initialLayer != null && initialHeadIndex != null ? { layer: initialLayer, headIndex: initialHeadIndex } : null,
  );
  const [sequences, setSequences] = useState<HeadSequenceData[]>([]);
  const [isLoadingSequences, setIsLoadingSequences] = useState(false);
  const [sequencesError, setSequencesError] = useState<string | null>(null);
  const sequencesRequestId = useRef(0);
  const [headDetail, setHeadDetail] = useState<ModelHeadDetail | null>(null);
  const [isLoadingDetail, setIsLoadingDetail] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const detailRequestId = useRef(0);
  const layerScrollRef = useRef<HTMLDivElement>(null);
  const headScrollRef = useRef<HTMLDivElement>(null);

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

  useEffect(() => {
    if (!selectedHead) {
      setHeadDetail(null);
      setDetailError(null);
      setIsLoadingDetail(false);
      return;
    }
    const requestId = detailRequestId.current + 1;
    detailRequestId.current = requestId;
    const controller = new AbortController();
    setHeadDetail(null);
    setIsLoadingDetail(true);
    setDetailError(null);
    fetch('/api/model/head-metrics/get', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ modelId, layer: selectedHead.layer, headIndex: selectedHead.headIndex }),
      signal: controller.signal,
    })
      .then(async (res) => {
        if (!res.ok) {
          const errorBody = await res.json().catch(() => null);
          throw new Error(errorBody?.error || `Failed to load head metrics (${res.status})`);
        }
        return res.json();
      })
      .then((data: ModelHeadDetail) => {
        if (requestId !== detailRequestId.current) return;
        setHeadDetail(data ?? null);
        setIsLoadingDetail(false);
      })
      .catch((error) => {
        if (controller.signal.aborted) return;
        if (requestId !== detailRequestId.current) return;
        setHeadDetail(null);
        setDetailError(error?.message || 'Failed to load head metrics');
        setIsLoadingDetail(false);
      });
    return () => {
      controller.abort();
    };
  }, [modelId, selectedHead]);

  const maxHeads = Math.min(256, Math.max(1, finderMetrics.length));

  const ranksByMetric = useMemo(() => {
    const result = new Map<MetricKey, Map<string, { rank: number; total: number }>>();
    METRIC_OPTIONS.forEach((opt) => {
      const entries = finderMetrics
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
  }, [finderMetrics]);

  const { displayRows, topHeadsByLayer, metricByLayerAndHead, min, max } = useMemo(() => {
    const layersSet = new Set<number>();
    const headIndexesSet = new Set<number>();
    const metricMap = new Map<string, ModelHeadMetricsRow>();

    finderMetrics.forEach((metric) => {
      layersSet.add(metric.layer);
      headIndexesSet.add(metric.headIndex);
      const key = `${metric.layer}:${metric.headIndex}`;
      if (!metricMap.has(key)) {
        metricMap.set(key, metric);
      }
    });

    const values = finderMetrics
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
  }, [forceShownLayers, finderMetrics, selectedMetric, showTopN]);

  // The selected head belongs to the page's model, so resolve its row from the page `metrics`
  // (not the finder metrics, which may target a different model the user is browsing).
  const pageMetricByLayerAndHead = useMemo(() => {
    const metricMap = new Map<string, ModelHeadMetricsRow>();
    metrics.forEach((metric) => {
      const key = `${metric.layer}:${metric.headIndex}`;
      if (!metricMap.has(key)) {
        metricMap.set(key, metric);
      }
    });
    return metricMap;
  }, [metrics]);

  const selectedHeadRow = selectedHead
    ? (pageMetricByLayerAndHead.get(`${selectedHead.layer}:${selectedHead.headIndex}`) ?? null)
    : null;

  const { layerOptions, headIndexOptions } = useMemo(() => {
    let maxLayer = 0;
    let maxHeadIndex = 0;
    finderMetrics.forEach((metric) => {
      if (metric.layer > maxLayer) maxLayer = metric.layer;
      if (metric.headIndex > maxHeadIndex) maxHeadIndex = metric.headIndex;
    });
    return {
      layerOptions: Array.from({ length: maxLayer + 1 }, (_, i) => i),
      headIndexOptions: Array.from({ length: maxHeadIndex + 1 }, (_, i) => i),
    };
  }, [finderMetrics]);

  const selectLayer = (layer: number) => {
    setSelectedHead((prev) => ({ layer, headIndex: prev?.headIndex ?? 0 }));
  };

  const selectHeadIndex = (headIndex: number) => {
    setSelectedHead((prev) => ({ layer: prev?.layer ?? 0, headIndex }));
  };

  // Switching models on the head page invalidates the current selection: the new model may not
  // have the same layer/head, and the displayed metrics/sequences belong to the previous model.
  // Clear the head selection (which clears the metrics, histograms, and sequences via the effects
  // above) so we never show stale, model-mismatched data.
  const handleModelChange = (newModelId: string) => {
    setFinderModelId(newModelId);
    setSelectedHead(null);
  };

  // Keep the manual Layer/Head grids scrolled so the active selection is visible,
  // without scrolling the page/card (e.g. when selecting via Find a Head by Score).
  useEffect(() => {
    if (!selectedHead) {
      return;
    }
    const layerCell = layerScrollRef.current?.querySelector<HTMLElement>(`[data-layer="${selectedHead.layer}"]`);
    const headCell = headScrollRef.current?.querySelector<HTMLElement>(`[data-head="${selectedHead.headIndex}"]`);
    scrollContainerToChild(layerScrollRef.current, layerCell ?? null);
    scrollContainerToChild(headScrollRef.current, headCell ?? null);
  }, [selectedHead]);

  // Keep the latest finder model in a ref so selecting a head navigates to the model the user is
  // browsing in the selector, without auto-navigating merely because the model dropdown changed.
  const finderModelIdRef = useRef(finderModelId);
  useEffect(() => {
    finderModelIdRef.current = finderModelId;
  }, [finderModelId]);

  // On the standalone head page (showCard=false), keep the URL in sync with the selected head so
  // it's shareable. When the head belongs to the page's model we update the URL client-side
  // (replaceState) instead of navigating — the pane's async effects already load the new head's
  // data, so a full navigation would just reload everything a second time. When the head belongs
  // to a different model (the user switched models in the selector), the page's server-loaded
  // metrics are model-specific, so we navigate to load that model's data.
  useEffect(() => {
    if (showCard || !selectedHead || typeof window === 'undefined') {
      return;
    }
    const targetModelId = finderModelIdRef.current;
    const basePath = `/${targetModelId}/head/${selectedHead.layer}/${selectedHead.headIndex}`;
    if (window.location.pathname === basePath) {
      return;
    }
    const url = `${basePath}${headFinderOpen ? '?headFinder=true' : ''}`;
    if (targetModelId === modelId) {
      // Same model: update the URL and keep the (client) breadcrumbs in sync without navigating.
      window.history.replaceState(window.history.state, '', url);
      onHeadChange?.({ modelId: targetModelId, layer: selectedHead.layer, headIndex: selectedHead.headIndex });
    } else {
      router.push(url);
    }
  }, [showCard, modelId, selectedHead, headFinderOpen, router, onHeadChange]);

  // Load the finder's metrics for whichever model is currently selected in the selector. When it
  // matches the page's model we reuse the `metrics` prop; otherwise we fetch that model's metrics
  // so the finder grid, layer count, and head count reflect the chosen model.
  useEffect(() => {
    if (finderModelId === modelId) {
      setFinderMetrics(metrics);
      setIsLoadingFinderMetrics(false);
      return;
    }
    const requestId = finderMetricsRequestId.current + 1;
    finderMetricsRequestId.current = requestId;
    const controller = new AbortController();
    setFinderMetrics([]);
    setIsLoadingFinderMetrics(true);
    fetch('/api/model/head-metrics/list', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ modelId: finderModelId }),
      signal: controller.signal,
    })
      .then(async (res) => {
        if (!res.ok) {
          const errorBody = await res.json().catch(() => null);
          throw new Error(errorBody?.error || `Failed to load head metrics (${res.status})`);
        }
        return res.json();
      })
      .then((data: ModelHeadMetricsRow[]) => {
        if (requestId !== finderMetricsRequestId.current) return;
        setFinderMetrics(Array.isArray(data) ? data : []);
        setIsLoadingFinderMetrics(false);
      })
      .catch(() => {
        if (controller.signal.aborted) return;
        if (requestId !== finderMetricsRequestId.current) return;
        setFinderMetrics([]);
        setIsLoadingFinderMetrics(false);
      });
    return () => {
      controller.abort();
    };
  }, [finderModelId, modelId, metrics]);

  // Set the in-pane head finder open state, keeping the `headFinder` query param in sync (without
  // a navigation) so a refresh or subsequent navigation preserves the open/closed state.
  const applyHeadFinderOpen = (next: boolean) => {
    setHeadFinderOpen(next);
    if (typeof window !== 'undefined') {
      const url = new URL(window.location.href);
      if (next) {
        url.searchParams.set('headFinder', 'true');
      } else {
        url.searchParams.delete('headFinder');
      }
      window.history.replaceState(window.history.state, '', url.toString());
    }
  };

  const toggleHeadFinder = () => applyHeadFinderOpen(!headFinderOpen);

  // Called when the selector leaves attention-head mode (a non-head release/source is picked).
  // The head finder is head-specific, so dismiss it.
  const handleExitHeadMode = () => {
    if (headFinderOpen) {
      applyHeadFinderOpen(false);
    }
  };

  const selectedHistogram = useMemo(() => {
    const parsed = parseHistogram(headDetail?.activationHistogram);
    if (!parsed) {
      return null;
    }
    const centers = parsed.bin_values.map((_, i) => {
      const lo = parsed.bin_edges[i] ?? 0;
      const hi = parsed.bin_edges[i + 1] ?? lo;
      return (lo + hi) / 2;
    });
    return { centers, binEdges: parsed.bin_edges, binValues: parsed.bin_values };
  }, [headDetail]);

  const qkDistanceHistogram = useMemo(() => {
    const parsed = parseHistogram(headDetail?.qkDistanceHistogram);
    if (!parsed) {
      return null;
    }
    // Trim trailing zero bins from the end only (keeps leading/interior zeros).
    let lastNonZero = parsed.bin_values.length - 1;
    while (lastNonZero >= 0 && parsed.bin_values[lastNonZero] === 0) {
      lastNonZero -= 1;
    }
    if (lastNonZero < 0) {
      return null;
    }
    const binValues = parsed.bin_values.slice(0, lastNonZero + 1);
    // QK distance bins are log-scaled (0,1,2,4,8,...), so use the bin's lower edge
    // as an evenly-spaced category label instead of a numeric axis.
    const labels = binValues.map((_, i) => `${parsed.bin_edges[i]}`);
    return { labels, binEdges: parsed.bin_edges, binValues };
  }, [headDetail]);

  const topQueryTokens = useMemo(() => parseTopTokens(headDetail?.topQueryTokens, 8), [headDetail]);
  const topKeyTokens = useMemo(() => parseTopTokens(headDetail?.topKeyTokens, 8), [headDetail]);

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

  const cardHeader = (
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
            <span className="hidden text-slate-400 sm:inline">(Luger, Kamath et al.)</span>{' '}
            <QuestionMarkCircledIcon className="h-4 w-4" />
          </div>
        }
      >
        <div className="flex flex-col">
          <p>
            The attention visualizer is based on{' '}
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
  );

  const cardContent = (
    <CardContent className="flex w-full flex-col gap-x-2 gap-y-0 overflow-x-auto px-2 sm:px-6">
      <div className="flex w-full flex-1 flex-col items-center justify-center gap-2">
        {!showCard && (
          <div className="flex w-full flex-row items-center justify-center py-1">
            <FeatureSelector
              defaultModelId={modelId}
              includeHeads
              numHeadLayers={layerOptions.length}
              numHeadIndexes={headIndexOptions.length}
              defaultHeadLayer={selectedHead?.layer}
              defaultIndex={selectedHead?.headIndex?.toString()}
              openInNewTab={false}
              showHeadFinderToggle
              headFinderActive={headFinderOpen}
              onHeadFinderToggle={toggleHeadFinder}
              onModelChange={handleModelChange}
              onExitHeadMode={handleExitHeadMode}
            />
          </div>
        )}
        {(showCard || headFinderOpen) && (
          <div
            className={`relative flex flex-1 flex-row gap-x-3 gap-y-1.5 ${showCard ? '' : 'mb-2 max-w-screen-lg rounded-xl border bg-white px-5 py-2 shadow-sm'}`}
          >
            {!showCard && (
              <button
                type="button"
                onClick={toggleHeadFinder}
                aria-label="Close head finder"
                className="absolute right-1.5 top-1.5 z-20 flex h-5 w-5 items-center justify-center rounded text-slate-400 hover:bg-slate-100 hover:text-slate-600 focus:outline-none"
              >
                <Cross2Icon className="h-3.5 w-3.5" />
              </button>
            )}
            {isLoadingFinderMetrics && (
              <div className="absolute inset-0 z-10 flex items-center justify-center rounded-xl bg-white/70">
                <LoadingSquare size={24} className="text-sky-700" />
              </div>
            )}
            <div className="mt-1 flex flex-1 flex-col">
              <div className="flex w-full flex-col items-center justify-center gap-x-3 px-0 pt-0 sm:h-[240px] sm:flex-row">
                <div className="mb-2 flex h-full w-full min-w-0 flex-1 flex-col items-start justify-start border-b border-slate-200 px-3 py-2 pb-4 sm:mb-0 sm:border-b-0 sm:border-r sm:pb-2">
                  <div className="mb-2 text-xs font-bold text-slate-400">
                    Find Head By Metric{' '}
                    <CustomTooltip side="right" wide trigger={<QuestionMarkCircledIcon className="h-3 w-3" />}>
                      {getHeadMetricsTooltip(true)}
                    </CustomTooltip>
                  </div>
                  <div className="mb-1 text-center text-[9px] font-medium uppercase text-slate-400">
                    Metric & Number of Heads
                  </div>
                  <div className="w-full sm:px-2">
                    <ToggleGroup.Root
                      className="inline-flex h-7 w-full rounded bg-slate-100 px-0 py-0 sm:rounded-md"
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
                          className="flex h-7 flex-1 items-center justify-center rounded px-0 text-[10px] font-medium leading-none text-slate-400 transition-all hover:bg-slate-100 hover:text-slate-600 data-[state=on]:bg-slate-300 data-[state=on]:text-slate-600 sm:rounded-md sm:text-[10px]"
                        >
                          {option.label}
                        </ToggleGroup.Item>
                      ))}
                    </ToggleGroup.Root>
                    <RadixSlider.Root
                      id="showTopN"
                      name="showTopN"
                      value={[nToSliderValue(showTopN)]}
                      onValueChange={(newVal) => updateShowTopN(sliderValueToN(newVal[0]))}
                      min={0}
                      max={SLIDER_RESOLUTION}
                      step={1}
                      className="relative my-2.5 mt-1 flex h-4 w-full touch-none select-none items-center px-0"
                    >
                      <RadixSlider.Track className="relative h-2 w-full flex-grow overflow-hidden rounded-full bg-slate-300">
                        <RadixSlider.Range className="absolute h-full rounded-full bg-sky-600" />
                      </RadixSlider.Track>
                      <RadixSlider.Thumb className="flex h-4 w-16 items-center justify-center rounded-full border border-sky-600 bg-white text-center text-[8.5px] font-bold text-sky-700 shadow transition-colors focus:outline-none focus:ring-0 disabled:pointer-events-none disabled:opacity-50">
                        Top {showTopN}
                      </RadixSlider.Thumb>
                    </RadixSlider.Root>
                  </div>
                  <div className="mb-1 text-center text-[9px] font-medium uppercase text-slate-400">
                    Click head to select
                  </div>
                  <div className="flex min-h-0 w-full flex-1 flex-col pl-2 pr-2">
                    <div className="forceShowScrollBar flex min-h-0 w-full flex-1 flex-col items-start justify-start gap-y-0.5 overflow-y-auto overscroll-contain rounded-md bg-slate-50 py-1">
                      {displayRows.map((displayRow) => {
                        if (displayRow.type === 'collapsed') {
                          return <></>;
                        }

                        const topHeads = topHeadsByLayer.get(displayRow.layer) ?? [];
                        const currentMetricOption = METRIC_OPTIONS.find((opt) => opt.key === selectedMetric);
                        return (
                          <div key={displayRow.layer} className="flex w-full flex-row items-start gap-1">
                            <div className="flex h-5 w-16 min-w-16 items-center justify-start pl-2.5 font-mono text-[9.5px] font-bold uppercase text-slate-400">
                              Layer {displayRow.layer}
                            </div>
                            <div className="flex min-w-0 flex-1 flex-row flex-wrap gap-0.5">
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
                                        className={`flex h-[22px] w-[68px] items-center justify-center rounded border-[1.5px] px-1 text-center font-mono text-[9.5px] font-bold uppercase text-sky-800 outline-none ${
                                          isSelectedHead ? 'border-sky-800' : 'border-slate-50'
                                        } hover:border-sky-600 focus-visible:border-sky-600`}
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

                                      <div className="mb-0 mt-0 flex flex-row items-center justify-between gap-x-4 font-semibold text-sky-700">
                                        <span>{currentMetricOption?.label}</span>
                                        <span className="font-mono">{renderMetricValue(selectedMetric)}</span>
                                      </div>
                                      {/* {otherMetricOptions.map((opt) => (
                                      <div
                                        key={opt.key}
                                        className="flex flex-row items-center justify-between gap-x-4 text-slate-500"
                                      >
                                        <span>{opt.label}</span>
                                        <span className="font-mono">{renderMetricValue(opt.key)}</span>
                                      </div>
                                    ))} */}
                                    </div>
                                  </CustomTooltip>
                                );
                              })}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
                <div className="flex h-full min-w-0 flex-1 flex-col px-3 py-2">
                  <div className="mb-2 text-xs font-bold text-slate-400">Select Head Manually</div>
                  <div className="flex min-h-0 w-full flex-1 flex-row gap-x-1 px-0 pr-5">
                    <div className="flex h-full min-h-0 flex-1 flex-col">
                      <div className="mb-1 text-center text-[9px] font-medium uppercase text-slate-400">Layer</div>
                      <div
                        ref={layerScrollRef}
                        className="flex min-h-0 w-full flex-1 flex-row flex-wrap content-start gap-0 overflow-y-auto overscroll-contain"
                      >
                        {layerOptions.map((layer) => {
                          const isSelected = selectedHead?.layer === layer;
                          return (
                            <button
                              key={layer}
                              type="button"
                              data-layer={layer}
                              onClick={() => selectLayer(layer)}
                              className={`flex h-5 w-5 min-w-5 items-center justify-center text-[9px] font-bold outline-none transition-colors first:rounded-tl last:rounded-br focus-visible:outline focus-visible:outline-1 focus-visible:outline-sky-600 ${
                                isSelected
                                  ? 'bg-sky-300 text-sky-800 hover:bg-sky-400 hover:text-sky-900'
                                  : 'border-slate-200 bg-slate-100 text-slate-400 hover:bg-sky-200 hover:text-sky-600'
                              }`}
                            >
                              {layer}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                    <div className="flex h-full min-h-0 flex-1 flex-col">
                      <div className="mb-1 text-center text-[9px] font-medium uppercase text-slate-400">Head Index</div>
                      <div
                        ref={headScrollRef}
                        className="flex min-h-0 w-full flex-1 flex-row flex-wrap content-start gap-0 overflow-y-auto overscroll-contain"
                      >
                        {headIndexOptions.map((headIndex) => {
                          const isSelected = selectedHead?.headIndex === headIndex;
                          return (
                            <button
                              key={headIndex}
                              type="button"
                              data-head={headIndex}
                              onClick={() => selectHeadIndex(headIndex)}
                              className={`flex h-5 w-5 min-w-5 items-center justify-center text-[9px] font-bold outline-none transition-colors first:rounded-tl last:rounded-br focus-visible:outline focus-visible:outline-1 focus-visible:outline-sky-600 ${
                                isSelected
                                  ? 'bg-sky-300 text-sky-800 hover:bg-sky-400 hover:text-sky-800'
                                  : 'border-slate-200 bg-slate-100 text-slate-400 hover:bg-sky-200 hover:text-sky-600'
                              }`}
                            >
                              {headIndex}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
        {showCard && selectedHead && (
          <div className="relative mb-3 mt-1.5 flex w-full flex-row items-center justify-center">
            <div className="absolute left-0 top-4 h-[1px] w-full bg-sky-600"></div>
            {selectedHead && (
              <a
                href={`/${modelId}/head/${selectedHead.layer}/${selectedHead.headIndex}`}
                className="group z-10 flex flex-row items-center justify-center overflow-hidden rounded border border-sky-600 bg-slate-50 px-0 py-0 text-[10px] font-semibold leading-none text-sky-700 transition-colors"
              >
                <div className="px-4 py-1.5 pr-2 font-mono transition-colors">{modelId}</div>
                <div className="px-4 py-1.5 pl-2 pr-2 transition-colors">Layer {selectedHead?.layer} </div>
                <div className="px-4 py-1.5 pl-2 pr-4 transition-colors">Head {selectedHead?.headIndex}</div>
                <div className="flex flex-row items-center justify-center gap-x-1 border-l border-sky-600 bg-sky-100 px-4 py-1.5 transition-colors group-hover:bg-sky-200 group-hover:text-sky-800">
                  Open Head Details <ExternalLinkIcon className="h-4 w-4" />
                </div>
              </a>
            )}
          </div>
        )}
      </div>
      {(!showCard || selectedHead) && (
        <div className={`flex w-full ${showCard ? 'flex-col' : 'flex-col gap-x-3 gap-y-3 sm:flex-row sm:items-start'}`}>
          <div
            className={`mt-1.5 flex flex-col items-stretch gap-x-1 ${showCard ? 'w-full' : 'w-full rounded-xl border bg-white px-5 py-2 sm:w-1/4'}`}
          >
            <div
              className={`flex w-full items-stretch gap-3 ${showCard ? 'flex-col gap-y-3 sm:flex-row' : 'flex-col gap-y-7 py-2'}`}
            >
              <div className="flex flex-1 basis-0 flex-col rounded bg-white p-0 pt-0">
                {selectedHead && selectedHeadRow ? (
                  <>
                    <div className="mb-0 flex flex-row items-center justify-center text-center text-[10px] font-medium uppercase text-slate-400">
                      Head Metrics
                      <CustomTooltip
                        side="right"
                        wide
                        trigger={<QuestionMarkCircledIcon className="-mt-0.5 ml-1 h-3 w-3" />}
                      >
                        {getHeadMetricsTooltip(false)}
                      </CustomTooltip>
                    </div>
                    <div className="flex flex-col gap-y-1 divide-y divide-slate-100">
                      {METRIC_OPTIONS.map((opt) => (
                        <div
                          key={opt.key}
                          className="flex flex-row items-center justify-between gap-x-4 px-1 pt-1 text-[10px] text-slate-500"
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
                          className="flex flex-row items-center justify-between gap-x-4 px-1 pt-1 text-[10px] text-slate-500"
                        >
                          <span>{opt.label}</span>
                          <span className="font-mono text-slate-700">
                            {isLoadingDetail ? '…' : formatTooltipValue(headDetail?.[opt.key as ExtraMetricKey])}
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
                      Max Attention Distribution
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
                              return `<b>Attention Weight Range (x)</b>: ${lo?.toFixed(2)} – ${hi?.toFixed(2)}<br><b># Sequences (y)</b>: ${count.toLocaleString()}<extra></extra>`;
                            }),
                          },
                        ]}
                        layout={{
                          height: 60,
                          xaxis: {
                            showgrid: false,
                            zeroline: false,
                            fixedrange: true,
                            tickfont: { size: 9, color: 'lightgrey' },
                          },
                          yaxis: {
                            showgrid: false,
                            zeroline: true,
                            zerolinecolor: 'lightgrey',
                            zerolinewidth: 1,
                            showticklabels: false,
                            fixedrange: true,
                          },
                          barmode: 'relative',
                          bargap: 0.05,
                          showlegend: false,
                          margin: { l: 16, r: 16, b: 18, t: 2, pad: 2 },
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
                    ) : isLoadingDetail ? (
                      <div className="flex h-[60px] w-full items-center justify-center text-center">
                        <LoadingSquare size={20} className="text-sky-700" />
                      </div>
                    ) : (
                      <div className="flex h-[130px] w-full items-center justify-center text-center">
                        <p className="text-[11px] font-medium text-slate-400">
                          {detailError || 'No attention distribution available for this head.'}
                        </p>
                      </div>
                    )}
                    <div className="mb-1 mt-1.5 flex flex-row items-center justify-center gap-x-1 text-center text-[10px] font-medium uppercase text-slate-400">
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
                          height: 60,
                          xaxis: {
                            type: 'category',
                            showgrid: false,
                            zeroline: false,
                            fixedrange: true,
                            tickfont: { size: 9, color: 'lightgrey' },
                          },
                          yaxis: {
                            showgrid: false,
                            zeroline: false,
                            showticklabels: false,
                            fixedrange: true,
                          },
                          showlegend: false,
                          margin: { l: 0, r: 0, b: 18, t: 2, pad: 2 },
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
                    ) : isLoadingDetail ? (
                      <div className="flex h-[60px] w-full items-center justify-center text-center">
                        <LoadingSquare size={20} className="text-sky-700" />
                      </div>
                    ) : (
                      <div className="flex h-[100px] w-full items-center justify-center text-center">
                        <p className="text-[11px] font-medium text-slate-400">
                          {detailError || 'No Q-K distance distribution available for this head.'}
                        </p>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="flex h-full min-h-[12rem] w-full items-center justify-center px-4 text-center">
                    <p className="text-xs font-bold text-slate-300">Q-K Distance Distribution</p>
                  </div>
                )}
              </div>
              <div className="flex flex-1 basis-0 flex-col rounded bg-white p-0 pt-0">
                {selectedHead && isLoadingDetail ? (
                  <div className="flex h-24 w-full items-center justify-center px-4 text-center">
                    <LoadingSquare size={20} className="text-sky-700" />
                  </div>
                ) : (
                  <div className="flex flex-row gap-x-3">
                    {topQueryTokens.length > 0 ? (
                      <div className="flex flex-1 basis-0 flex-col">
                        <div className="mb-1 text-center text-[10px] font-medium uppercase text-slate-400">
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
                    ) : (
                      <div className="flex h-full min-h-[12rem] w-full items-center justify-center px-4 text-center">
                        <p className="text-xs font-bold text-slate-300">Top Query Tokens</p>
                      </div>
                    )}
                    {topKeyTokens.length > 0 ? (
                      <div className="flex flex-1 basis-0 flex-col">
                        <div className="mb-1 text-center text-[10px] font-medium uppercase text-slate-400">
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
                    ) : (
                      <div className="flex h-full min-h-[12rem] w-full items-center justify-center px-4 text-center">
                        <p className="text-xs font-bold text-slate-300">Top Key Tokens</p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
          <div
            className={`flex min-w-0 flex-1 flex-col overflow-hidden rounded-xl border-slate-200 bg-white ${showCard ? 'mt-4' : 'mt-1.5 w-full sm:w-3/4'} ${selectedHead ? 'border' : 'border-none'}`}
          >
            {selectedHead ? (
              <HeadActivationsList
                sequences={sequences}
                modelId={modelId}
                layer={selectedHead.layer}
                headIndex={selectedHead.headIndex}
                isLoading={isLoadingSequences}
                errorMessage={sequencesError}
                unbounded={!showCard}
                inferenceEnabled={inferenceEnabled}
              />
            ) : (
              <div className="flex h-full min-h-[12rem] w-full items-center justify-center px-4 text-center">
                <p className="text-xs font-bold text-slate-300">Top Activating Sequences</p>
              </div>
            )}
          </div>
        </div>
      )}
    </CardContent>
  );

  if (!showCard) {
    return cardContent;
  }

  return (
    <Card className="w-full bg-white">
      <CardHeader className="pb-3">{cardHeader}</CardHeader>
      {cardContent}
    </Card>
  );
}
