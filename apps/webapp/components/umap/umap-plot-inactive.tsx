import Plotly from 'plotly.js-dist-min';
// import { Data } from "plotly.js";
import { UMAP_HEIGHT, UMAP_INITIAL_COLORS, useUmapContext } from '@/components/provider/umap-provider';
import createPlotlyComponent from 'react-plotly.js/factory';
import { getLogSparsityColorFromValue } from './umap-constants';

const Plot = createPlotlyComponent(Plotly);

// export const SEARCH_MATCHED_COLOR = colors.amber[400];
const SEARCH_MATCHED_SIZE = 3.5;

export default function UmapPlotInactive({ showLists }: { showLists: boolean }) {
  const {
    plotInactiveRef,
    searchText,
    graphRanges,
    showLogSparsity,
    layerToInitialColor,
    highlightedUmapExplanations,
  } = useUmapContext();

  function makeData(): any[] {
    let x: number[] = [];
    let y: number[] = [];
    let color: string | string[] = [];
    let size: number | number[] = [];

    const isSearching = searchText.length > 0;

    if (isSearching) {
      x = highlightedUmapExplanations.map((e) => e.umap_x as number);
      y = highlightedUmapExplanations.map((e) => e.umap_y as number);
      color = showLogSparsity
        ? highlightedUmapExplanations.map((e) => getLogSparsityColorFromValue(e.umap_log_feature_sparsity || 0))
        : highlightedUmapExplanations.map((e) => layerToInitialColor[e.layer] || UMAP_INITIAL_COLORS[0]);
      size = SEARCH_MATCHED_SIZE;
    } else {
      color = [];
      size = [];
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
          opacity: 1,
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
      ref={plotInactiveRef}
      divId="plotNoPointerEvents"
      className={`pointer-events-none col-start-1 row-start-1 mb-0 ${showLists ? '' : 'h-full'}`}
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
          showgrid: false,
          showticklabels: false,
          visible: true,
          color: '#ffffff',
          gridcolor: '#ffffff',
          zeroline: false,
        },
        yaxis: {
          range: [graphRanges.minY, graphRanges.maxY],
          showgrid: false,
          showticklabels: false,
          visible: true,
          color: '#ffffff',
          gridcolor: '#ffffff',
          zeroline: false,
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(255,255,255,0)',
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
