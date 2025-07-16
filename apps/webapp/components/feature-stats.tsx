'use client';

import { useGlobalContext } from '@/components/provider/global-provider';
import useWindowSize from '@/lib/hooks/use-window-size';
import { replaceHtmlAnomalies } from '@/lib/utils/activations';
import { getLayerNumFromSource, NEURONS_SOURCESET } from '@/lib/utils/source';
import * as Tooltip from '@radix-ui/react-tooltip';
import { HelpCircle } from 'lucide-react';
// import Plotly from 'plotly.js-dist-min';
import { NeuronWithPartialRelations } from 'prisma/generated/zod';
import { useEffect, useState } from 'react';
import createPlotlyComponent from 'react-plotly.js/factory';

// // https://github.com/plotly/react-plotly.js/issues/273
import dynamic from 'next/dynamic';

const Plot = dynamic(
  () => Promise.resolve(import('plotly.js-dist-min').then((Plotly) => createPlotlyComponent(Plotly))),
  {
    ssr: false,
  },
);
// const Plot = createPlotlyComponent(Plotly);
const MAX_EMBED_TOP_LOGITS = 5;

export default function FeatureStats({
  currentNeuron,
  vertical = false,
  smallText = false,
  embed = false,
  embedPlots = true,
  forceMiniStats = false,
  activationMarkerValue,
}: {
  currentNeuron: NeuronWithPartialRelations;
  vertical?: boolean;
  smallText?: boolean;
  embed?: boolean;
  embedPlots?: boolean;
  forceMiniStats?: boolean;
  activationMarkerValue?: number;
}) {
  const { getSourceSet, globalModels } = useGlobalContext();
  const [maxLogit, setMaxLogit] = useState<number | undefined>();
  const { windowSize } = useWindowSize();

  const modelHasNeuronsSourceSet = (modelId: string) => {
    const model = globalModels[modelId];
    if (model.sourceSets && model.sourceSets?.filter((ss) => ss.name === NEURONS_SOURCESET).length > 0) {
      return true;
    }
    return false;
  };

  function makeColorRangeForFrequencyHistogram(numColors: number) {
    const colors = Array.from({ length: numColors }, (_, i) => {
      const ratio = i / (numColors - 1);
      const hex = (color1: string, color2: string, ratioCalc: number) => {
        const r = Math.round(
          parseInt(color1.substring(1, 3), 16) * ratioCalc + parseInt(color2.substring(1, 3), 16) * (1 - ratioCalc),
        ).toString(16);
        const g = Math.round(
          parseInt(color1.substring(3, 5), 16) * ratioCalc + parseInt(color2.substring(3, 5), 16) * (1 - ratioCalc),
        ).toString(16);
        const b = Math.round(
          parseInt(color1.substring(5, 7), 16) * ratioCalc + parseInt(color2.substring(5, 7), 16) * (1 - ratioCalc),
        ).toString(16);
        return `#${r.padStart(2, '0')}${g.padStart(2, '0')}${b.padStart(2, '0')}`;
      };
      return hex('#ff8c00', '#ffd097', ratio);
    });
    return colors;
  }

  useEffect(() => {
    if (currentNeuron) {
      const negMax = Math.max(...(currentNeuron?.neg_values || []).map((value) => Math.abs(value)));
      const posMax = Math.max(...(currentNeuron?.pos_values || []).map((value) => Math.abs(value)));
      const max = Math.max(negMax, posMax);
      setMaxLogit(max);
    }
  }, [currentNeuron]);

  return (
    <div className="w-full flex-row pt-0 text-[11px] text-slate-900 sm:flex">
      <div
        className={`mb-0 flex w-full flex-row gap-x-3 gap-y-2.5 px-0 pb-0 sm:gap-x-5 ${vertical ? '' : 'sm:flex-row'}`}
      >
        {getSourceSet(currentNeuron?.modelId, currentNeuron?.sourceSetName || '')?.showCorrelated && !vertical && (
          <div className="hidden basis-1/4 flex-col gap-y-3">
            <div className="flex w-full flex-col">
              <div className="mb-1 flex flex-row items-center gap-x-1 font-mono text-[11px] font-bold uppercase text-slate-500">
                <div>Neuron Alignment</div>
                <Tooltip.Provider delayDuration={0} skipDelayDuration={0}>
                  <Tooltip.Root>
                    <Tooltip.Trigger asChild>
                      <button type="button">
                        <HelpCircle className="h-3 w-3" />
                      </button>
                    </Tooltip.Trigger>
                    <Tooltip.Portal>
                      <Tooltip.Content className="rounded bg-slate-500 px-3 py-2 text-xs text-white" sideOffset={5}>
                        Top neurons by how much this feature activates them.
                        <Tooltip.Arrow className="fill-slate-500" />
                      </Tooltip.Content>
                    </Tooltip.Portal>
                  </Tooltip.Root>
                </Tooltip.Provider>
              </div>
              <div className="relative flex w-full flex-col">
                <div className="mx-0 flex w-full flex-row border-b">
                  <div className="basis-1/4">Index</div>
                  <div className="flex-1 text-right">Value</div>
                  <div className="flex-1 text-right">% of L&#8321;</div>
                </div>
                {currentNeuron?.neuron_alignment_indices.map((item, j) => (
                  <div key={j} className="mx-0 flex w-full flex-row border-b py-0.5">
                    <div className="basis-1/4">
                      {modelHasNeuronsSourceSet(currentNeuron?.modelId) ? (
                        <a
                          href={`/${currentNeuron?.modelId}/${getLayerNumFromSource(currentNeuron?.layer)}/${item}`}
                          target="_blank"
                          rel="noreferrer"
                          className="text-sky-700"
                        >
                          {item}
                        </a>
                      ) : (
                        item
                      )}
                    </div>
                    <div className="flex-1 text-right">
                      {currentNeuron?.neuron_alignment_values[j] > 0 ? '+' : '-'}
                      {currentNeuron?.neuron_alignment_values[j].toFixed(2)}
                    </div>
                    <div className="flex-1 text-right">
                      {((currentNeuron?.neuron_alignment_l1?.[j] || 0) * 100).toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div className="flex w-full flex-col">
              <div className="mb-1 flex flex-row items-center gap-x-1 font-mono text-[11px] font-bold uppercase text-slate-500">
                <div>Correlated Neurons</div>
                <Tooltip.Provider delayDuration={0} skipDelayDuration={0}>
                  <Tooltip.Root>
                    <Tooltip.Trigger asChild>
                      <button type="button">
                        <HelpCircle className="h-3 w-3" />
                      </button>
                    </Tooltip.Trigger>
                    <Tooltip.Portal>
                      <Tooltip.Content className="rounded bg-slate-500 px-3 py-2 text-xs text-white" sideOffset={5}>
                        Top neurons by token correlation.
                        <Tooltip.Arrow className="fill-slate-500" />
                      </Tooltip.Content>
                    </Tooltip.Portal>
                  </Tooltip.Root>
                </Tooltip.Provider>
              </div>
              <div className="relative flex w-full flex-col">
                <div className="mx-0 flex w-full flex-row border-b">
                  <div className="basis-1/4">Index</div>
                  <div className="flex-1 text-right">P. Corr.</div>
                  <div className="flex-1 text-right">Cos Sim.</div>
                </div>
                {currentNeuron?.correlated_neurons_indices.map((item, j) => (
                  <div key={j} className="mx-0 flex w-full flex-row py-0.5">
                    <div className="basis-1/4">
                      {modelHasNeuronsSourceSet(currentNeuron?.modelId) ? (
                        <a
                          href={`/${currentNeuron?.modelId}/${getLayerNumFromSource(currentNeuron?.layer)}/${item}`}
                          target="_blank"
                          rel="noreferrer"
                          className="text-sky-700"
                        >
                          {item}
                        </a>
                      ) : (
                        item
                      )}
                    </div>
                    <div className="flex-1 text-right">
                      {currentNeuron?.correlated_neurons_pearson[j] > 0 ? '+' : '-'}
                      {currentNeuron?.neuron_alignment_values[j].toFixed(2)}
                    </div>
                    <div className="flex-1 text-right">{currentNeuron?.correlated_neurons_l1[j].toFixed(2)}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        <div className="flex flex-1 flex-col">
          <div className={`flex flex-row gap-x-3 font-mono ${smallText ? 'text-[9.5px]' : 'text-xs'}`}>
            {getSourceSet(currentNeuron?.modelId, currentNeuron?.sourceSetName || '')?.showHeadAttribution &&
              !vertical &&
              !embed && (
                <div className="flex min-w-[105px] max-w-[105px] flex-1 flex-col sm:min-w-[138px] sm:max-w-[138px]">
                  <div className="mb-0 flex flex-row items-center gap-x-1 whitespace-pre text-[9px] font-bold uppercase text-slate-500 sm:mb-1 sm:text-[11px]">
                    <div className="mb-0 flex flex-row items-center justify-center gap-x-1 font-sans text-[10px] font-normal uppercase text-slate-400">
                      Head Attr Weights
                    </div>
                    <Tooltip.Provider delayDuration={0} skipDelayDuration={0}>
                      <Tooltip.Root>
                        <Tooltip.Trigger asChild>
                          <button type="button">
                            <HelpCircle className="h-2 w-2 text-slate-400 sm:h-2.5 sm:w-2.5" />
                          </button>
                        </Tooltip.Trigger>
                        <Tooltip.Portal>
                          <Tooltip.Content className="rounded bg-slate-500 px-3 py-2 text-xs text-white" sideOffset={5}>
                            Weights-based Head Attribution
                            <Tooltip.Arrow className="fill-slate-500" />
                          </Tooltip.Content>
                        </Tooltip.Portal>
                      </Tooltip.Root>
                    </Tooltip.Provider>
                  </div>
                  {currentNeuron.decoder_weights_dist.map((value, ind) => (
                    <div
                      key={ind}
                      className={`mb-[0.5px] flex flex-row gap-x-1 font-mono font-medium sm:mb-[1px] ${
                        embed ? 'text-[7px] leading-none sm:text-[10px] sm:leading-[1.4]' : 'text-[10px] leading-[1.4]'
                      } text-slate-500`}
                    >
                      <span className="w-4">{ind}:</span>
                      <span className="rounded-sm bg-slate-300" style={{ width: `${100 * value}px` }} />
                      <span className="text-slate-600">{(Math.floor(value * 100) / 100).toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              )}
            {currentNeuron?.neg_str.length > 0 && (
              <div className="flex-1">
                {!vertical && (
                  <div className="mb-0 flex flex-row items-center gap-x-1 whitespace-pre text-[9px] font-bold uppercase text-slate-500 sm:mb-1 sm:text-[11px]">
                    <div className="mb-0.5 flex flex-row items-center justify-center gap-x-1 font-sans text-[10px] font-normal uppercase text-slate-400 sm:mb-0">
                      Negative Logits
                    </div>
                    <Tooltip.Provider delayDuration={0} skipDelayDuration={0}>
                      <Tooltip.Root>
                        <Tooltip.Trigger asChild>
                          <button type="button">
                            <HelpCircle className="h-2.5 w-2.5 text-slate-400 sm:h-2.5 sm:w-2.5" />
                          </button>
                        </Tooltip.Trigger>
                        <Tooltip.Portal>
                          <Tooltip.Content className="rounded bg-slate-500 px-3 py-2 text-xs text-white" sideOffset={5}>
                            Top negative logits of the feature.
                            <Tooltip.Arrow className="fill-slate-500" />
                          </Tooltip.Content>
                        </Tooltip.Portal>
                      </Tooltip.Root>
                    </Tooltip.Provider>
                  </div>
                )}
                <div className="flex flex-col gap-y-0.5 text-[9px] sm:text-[11px]">
                  {currentNeuron?.neg_str.map((s, i) => (
                    <div
                      className={`flex-row justify-between gap-x-1 ${
                        ((embed && !embedPlots) ||
                          (windowSize.width && windowSize.width < 640) ||
                          vertical ||
                          forceMiniStats) &&
                        i >= MAX_EMBED_TOP_LOGITS
                          ? 'hidden'
                          : 'flex'
                      }`}
                      key={i}
                    >
                      <pre
                        style={{
                          backgroundColor: maxLogit
                            ? `rgba(255, calc(255 * (1 - ${
                                Math.abs(currentNeuron?.neg_values[i]) / maxLogit
                              })), calc(255 * (1 - ${Math.abs(currentNeuron?.neg_values[i]) / maxLogit})), 0.5)`
                            : undefined,
                        }}
                        className="max-w-[80px] overflow-x-hidden rounded-sm font-bold sm:max-w-[100px] sm:rounded"
                      >
                        {replaceHtmlAnomalies(s)}
                      </pre>
                      <div>{currentNeuron?.neg_values[i].toFixed(2)}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {currentNeuron?.pos_str.length > 0 && (
              <div className="flex-1">
                {!vertical && (
                  <div className="mb-0 flex flex-row items-center gap-x-1 whitespace-pre text-[9px] font-bold uppercase text-slate-500 sm:mb-1 sm:text-[11px]">
                    <div className="mb-0.5 flex flex-row items-center justify-center gap-x-1 font-sans text-[10px] font-normal uppercase text-slate-400 sm:mb-0">
                      POSITIVE LOGITS
                    </div>
                    <Tooltip.Provider delayDuration={0} skipDelayDuration={0}>
                      <Tooltip.Root>
                        <Tooltip.Trigger asChild>
                          <button type="button">
                            <HelpCircle className="h-2.5 w-2.5 text-slate-400 sm:h-2.5 sm:w-2.5" />
                          </button>
                        </Tooltip.Trigger>
                        <Tooltip.Portal>
                          <Tooltip.Content className="rounded bg-slate-500 px-3 py-2 text-xs text-white" sideOffset={5}>
                            Top positive logits of the feature.
                            <Tooltip.Arrow className="fill-slate-500" />
                          </Tooltip.Content>
                        </Tooltip.Portal>
                      </Tooltip.Root>
                    </Tooltip.Provider>
                  </div>
                )}
                <div className="flex flex-col gap-y-0.5 text-[9px] sm:text-[11px]">
                  {currentNeuron?.pos_str.map((s, i) => (
                    <div
                      className={`flex-row justify-between gap-x-1 ${
                        ((embed && !embedPlots) ||
                          (windowSize.width && windowSize.width < 640) ||
                          vertical ||
                          forceMiniStats) &&
                        i >= MAX_EMBED_TOP_LOGITS
                          ? 'hidden'
                          : 'flex'
                      }`}
                      key={i}
                    >
                      <pre
                        style={{
                          backgroundColor: maxLogit
                            ? `rgba(calc(255 * (1 - ${
                                Math.abs(currentNeuron?.pos_values[i]) / maxLogit
                              })), calc(255 * (1 - ${Math.abs(currentNeuron?.pos_values[i]) / maxLogit})), 255, 0.5)`
                            : undefined,
                        }}
                        className="max-w-[80px] overflow-x-hidden rounded-sm font-bold sm:max-w-[100px] sm:rounded"
                      >
                        {replaceHtmlAnomalies(s)}
                      </pre>
                      <div>{currentNeuron?.pos_values[i].toFixed(2)}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        <div className={`flex-1 flex-col ${(embed && !embedPlots) || vertical ? 'hidden' : 'flex'}`}>
          <div
            className={`mb-0 flex flex-row items-center gap-x-1 font-sans font-medium uppercase text-slate-500 ${
              smallText ? 'text-[9.5px]' : 'text-[10px] sm:text-[11px]'
            }`}
          >
            <div className="mt-[1px] w-full text-center sm:mt-0">
              Act<span className="hidden sm:inline-block">ivations</span> Density{' '}
              {((currentNeuron?.frac_nonzero || 0) * 100).toFixed(3)}%
            </div>
            <Tooltip.Provider delayDuration={0} skipDelayDuration={0}>
              <Tooltip.Root>
                <Tooltip.Trigger asChild>
                  <button type="button">
                    <HelpCircle className="h-2.5 w-2.5 text-slate-400 sm:h-3 sm:w-3" />
                  </button>
                </Tooltip.Trigger>
                <Tooltip.Portal>
                  <Tooltip.Content
                    className="rounded bg-slate-500 px-3 py-2 text-center text-xs text-white"
                    sideOffset={5}
                  >
                    (First) Histogram of randomly sampled non-zero activations.
                    <br />
                    (Second) Logit density histogram.
                    <Tooltip.Arrow className="fill-slate-500" />
                  </Tooltip.Content>
                </Tooltip.Portal>
              </Tooltip.Root>
            </Tooltip.Provider>
          </div>
          <div className="flex flex-col sm:flex-col sm:gap-y-1 sm:pt-1">
            {currentNeuron?.freq_hist_data_bar_values.length > 0 && (
              <Plot
                data={[
                  {
                    x: currentNeuron?.freq_hist_data_bar_values,
                    y: currentNeuron?.freq_hist_data_bar_heights,
                    type: 'bar',
                    marker: {
                      color: makeColorRangeForFrequencyHistogram(currentNeuron?.freq_hist_data_bar_values.length),
                    },
                  },
                  ...(activationMarkerValue !== undefined
                    ? [
                        {
                          x: [activationMarkerValue, activationMarkerValue],
                          y: [0, Math.max(...(currentNeuron?.freq_hist_data_bar_heights || []))],
                          type: 'scatter' as const,
                          mode: 'lines' as const,
                          line: {
                            color: 'green',
                            width: 2,
                          },
                          showlegend: false,
                        },
                      ]
                    : []),
                ]}
                layout={{
                  height: (windowSize.width && windowSize.width < 640) || forceMiniStats ? 35 : 70,
                  xaxis: {
                    gridcolor: 'lightgrey',
                    zerolinecolor: 'lightgrey',
                    fixedrange: true,
                    // tickvals: width < 640 ? [Math.max(...currentNeuron?.freq_hist_data_bar_values)] : undefined
                    tickfont: {
                      size: (windowSize.width && windowSize.width < 640) || vertical || forceMiniStats ? 7 : 10,
                    },
                  },
                  yaxis: {
                    gridcolor: 'lightgrey',
                    zerolinecolor: 'lightgrey',
                    // tickvals: width < 640 ? [Math.max(...currentNeuron?.freq_hist_data_bar_values)] : undefined
                    fixedrange: true,
                    tickfont: {
                      size: (windowSize.width && windowSize.width < 640) || forceMiniStats ? 7 : 10,
                    },
                  },
                  barmode: 'relative',
                  bargap: 0.05,
                  showlegend: false,
                  margin: {
                    l: 0,
                    r: 0,
                    b: 0,
                    t: 0,
                    pad: (windowSize.width && windowSize.width < 640) || forceMiniStats ? 1 : 4,
                  },
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(0,0,0,0)',
                }}
                config={{
                  responsive: false,
                  displayModeBar: false,
                  editable: false,
                  scrollZoom: false,
                }}
                className={`mb-3 ml-3.5 mr-0 mt-[1px] h-[35px] min-h-[35px] flex-1 ${
                  forceMiniStats ? '' : 'sm:mb-4 sm:ml-7 sm:h-16 sm:min-h-16'
                }`}
              />
            )}
            {currentNeuron?.logits_hist_data_bar_values.length > 0 && (
              <Plot
                className={`mb-3 ml-3.5 mr-0 h-[35px] min-h-[35px] flex-1 ${
                  forceMiniStats ? '' : 'sm:ml-7 sm:mt-1 sm:h-[70px] sm:min-h-[70px]'
                }`}
                data={[
                  {
                    x: currentNeuron?.logits_hist_data_bar_values.filter((value) => value >= 0),
                    y: currentNeuron?.logits_hist_data_bar_heights.filter(
                      (_, i) => currentNeuron?.logits_hist_data_bar_values[i] >= 0,
                    ),
                    type: 'bar',
                    mode: 'lines+markers',
                    name: '',
                    marker: { color: 'rgba(128, 128, 250, 1.0)' },
                  },
                  {
                    x: currentNeuron?.logits_hist_data_bar_values.filter((value) => value < 0),
                    y: currentNeuron?.logits_hist_data_bar_heights.filter(
                      (_, i) => currentNeuron?.logits_hist_data_bar_values[i] < 0,
                    ),
                    type: 'bar',
                    name: '',
                    mode: 'lines+markers',
                    marker: { color: 'rgba(250, 128, 128, 1.0)' },
                  },
                ]}
                layout={{
                  height: (windowSize.width && windowSize.width < 640) || forceMiniStats ? 35 : 70,
                  xaxis: {
                    gridcolor: 'lightgrey',
                    zerolinecolor: 'lightgrey',
                    // tickvals: [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                    fixedrange: true,
                    tickfont: {
                      size: (windowSize.width && windowSize.width < 640) || forceMiniStats ? 7 : 10,
                    },
                  },
                  yaxis: {
                    tickformat: '.1s',
                    gridcolor: 'lightgrey',
                    zerolinecolor: 'lightgrey',
                    fixedrange: true,
                    tickfont: {
                      size: (windowSize.width && windowSize.width < 640) || forceMiniStats ? 7 : 10,
                    },
                  },
                  barmode: 'relative',
                  bargap: 0.05,
                  showlegend: false,
                  margin: {
                    l: 0,
                    r: 0,
                    b: 0,
                    t: 0,
                    pad: (windowSize.width && windowSize.width < 640) || forceMiniStats ? 1 : 4,
                  },
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(0,0,0,0)',
                }}
                config={{
                  responsive: false,
                  displayModeBar: false,
                  editable: false,
                  scrollZoom: false,
                }}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
