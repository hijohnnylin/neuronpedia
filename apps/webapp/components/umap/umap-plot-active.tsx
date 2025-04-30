import Plotly from 'plotly.js-dist-min';
// import { Data } from "plotly.js";
import {
  SPARSITY_COLOR_MAX,
  SPARSITY_COLOR_MIN,
  SPARSITY_COLORS,
  UMAP_HEIGHT,
  UMAP_INITIAL_COLORS,
  useUmapContext,
} from '@/components/provider/umap-provider';
import { NeuronIdentifier } from '@/lib/utils/neuron-identifier';
import createPlotlyComponent from 'react-plotly.js/factory';

const Plot = createPlotlyComponent(Plotly);

export const SEARCH_NOT_MATCHED_COLOR = '#e3e3e3';
const SEARCH_NOT_MATCHED_SIZE = 3.5;
const INITIAL_SIZE = 3.5;

export function getLogSparsityColorFromValue(value: number): string {
  // Define the value range
  const minValue = SPARSITY_COLOR_MIN;
  const maxValue = SPARSITY_COLOR_MAX;

  // Normalize the value to a 0-1 scale based on the value range
  let normalizedValue = (value - minValue) / (maxValue - minValue);

  // Ensure the normalized value is clamped between 0 and 1
  normalizedValue = Math.min(Math.max(normalizedValue, 0), 1);

  // Scale the normalized value to the range of indices
  const index = Math.round(normalizedValue * (SPARSITY_COLORS.length - 1));

  return SPARSITY_COLORS[index];
}

export default function UmapPlotActive({ modelId, showLists }: { modelId: string; showLists: boolean }) {
  const {
    searchText,
    setGraphRanges,
    graphRanges,
    plotActiveRef,
    visibleUmapExplanations,
    loadFeature,
    // addAnnotationForExp,
    showLogSparsity,
    layerToInitialColor,
  } = useUmapContext();

  function makeData(): any[] {
    let x = [];
    let y = [];
    let color: string | string[] = [];
    let size: number | number[] = [];

    const isSearching = searchText.length > 0;

    x = visibleUmapExplanations.map((e) => e.umap_x);
    y = visibleUmapExplanations.map((e) => e.umap_y);

    if (isSearching) {
      color = Array(x.length).fill(SEARCH_NOT_MATCHED_COLOR);
      size = Array(x.length).fill(SEARCH_NOT_MATCHED_SIZE);
    } else {
      color = showLogSparsity
        ? visibleUmapExplanations.map((e) => getLogSparsityColorFromValue(e.umap_log_feature_sparsity || 0))
        : visibleUmapExplanations.map((e) => layerToInitialColor[e.layer] || UMAP_INITIAL_COLORS[0]);
      size = INITIAL_SIZE;
    }

    return [
      {
        showlegend: false,
        x,
        y,
        name: '',
        opacity: 1,
        hoverinfo: 'none',
        marker: {
          color,
          size,
          opacity: Array.isArray(size)
            ? size.map((s) => {
                if (s === SEARCH_NOT_MATCHED_SIZE) {
                  return 0.7;
                }
                return 1;
              })
            : INITIAL_SIZE,
          line: {
            width: 0,
          },
        },
        mode: 'markers',
        type: 'scatter',
      },
    ];
  }

  return (
    <Plot
      // @ts-ignore
      ref={plotActiveRef}
      className={`col-start-1 row-start-1 mb-0 ${showLists ? '' : 'h-full'}`}
      onClick={(event) => {
        const pn = event.points[0].pointNumber;
        loadFeature(
          new NeuronIdentifier(modelId, visibleUmapExplanations[pn].layer, visibleUmapExplanations[pn].index),
        );
      }}
      onHover={() => {
        // const pn = event.points[0].pointNumber;
        // addAnnotationForExp(visibleUmapExplanations[pn]);
      }}
      onUpdate={(figure) => {
        const { layout } = figure;
        if (!layout.xaxis?.range || !layout.yaxis?.range) {
          return;
        }
        if (
          layout.xaxis.range[0] !== graphRanges.minX ||
          layout.xaxis.range[1] !== graphRanges.maxX ||
          layout.yaxis.range[0] !== graphRanges.minY ||
          layout.yaxis.range[1] !== graphRanges.maxY
        ) {
          setGraphRanges({
            minX: layout.xaxis.range[0],
            maxX: layout.xaxis.range[1],
            minY: layout.yaxis.range[0],
            maxY: layout.yaxis.range[1],
          });
        }
      }}
      data={makeData()}
      layout={{
        margin: {
          l: 0,
          r: 0,
          b: 0,
          t: 0,
        },
        height: showLists ? UMAP_HEIGHT : undefined,
        xaxis: {
          range: [graphRanges.minX, graphRanges.maxX],
          showgrid: true,
          showticklabels: false,
          tickmode: 'array',
          tickvals: [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30],
          visible: true,
          color: '#ffffff',
          gridcolor: '#ffffff',
          zeroline: false,
        },
        yaxis: {
          range: [graphRanges.minY, graphRanges.maxY],
          showgrid: true,
          showticklabels: false,
          tickmode: 'array',
          tickvals: [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30],
          visible: true,
          color: '#ffffff',
          gridcolor: '#ffffff',
          zeroline: false,
        },
        plot_bgcolor: '#e1e5e9',
      }}
      config={{
        responsive: false,
        displayModeBar: false,
        editable: false,
        scrollZoom: false,
      }}
    />
  );
}
