'use client';

import { getEvalOutputSchemaFromPrismaJson } from '@/app/sae-bench/evals-table';
import CustomTooltip from '@/components/custom-tooltip';
import { useGlobalContext } from '@/components/provider/global-provider';
import { Accordion, AccordionContent, AccordionItem } from '@/components/shadcn/accordion';
import { Button } from '@/components/shadcn/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/shadcn/card';
import {
  EvalWithPartialRelations,
  NeuronWithPartialRelations,
  SourceWithPartialRelations,
} from '@/prisma/generated/zod';
import { Absorption, AbsorptionResultDetail } from '@/types/eval_output_schema_absorption_first_letter';
import { Core, CoreFeatureMetric } from '@/types/eval_output_schema_core';
import { Prisma } from '@prisma/client';
import { ExternalLinkIcon, QuestionMarkCircledIcon } from '@radix-ui/react-icons';
import { JSONSchema } from 'json-schema-to-typescript';
import dynamic from 'next/dynamic';
import createPlotlyComponent from 'react-plotly.js/factory';
import SaeEvalAccordionTrigger from './accordion-trigger';

const Plot = dynamic(
  () => Promise.resolve(import('plotly.js-dist-min').then((Plotly) => createPlotlyComponent(Plotly))),
  { ssr: false },
);
import SaeEvalMetricsAccordion from './metrics-accordion';

/*
converting plotly express to plotly graph objects: 
  f = fig.to_dict()
  print(f['layout'])
  print(f['data'])
*/

export default function SAEEvalsPane({
  sae,
  evals,
}: {
  sae: SourceWithPartialRelations;
  evals: EvalWithPartialRelations[];
}) {
  const { setFeatureModalFeature, setFeatureModalOpen } = useGlobalContext();

  return (
    <Card className="w-full bg-white">
      <CardHeader className="w-full pb-0 pt-5">
        <div className="flex w-full flex-row items-center justify-between">
          <CardTitle>SAE Evaluations</CardTitle>
          <a
            href="/sae-bench"
            target="_blank"
            onClick={(e) => {
              e.stopPropagation();
            }}
            rel="noreferrer"
            className="mx-3 flex h-5 flex-row items-center justify-center rounded-full bg-slate-400 px-3 py-[1px] text-[9px] font-medium uppercase text-white hover:bg-sky-600"
          >
            SAE Bench
            <ExternalLinkIcon className="ml-1 h-2.5 w-2.5" />
          </a>
        </div>
      </CardHeader>
      <CardContent className="flex flex-col items-start justify-between pb-6 pt-4">
        <Accordion type="multiple" defaultValue={evals.map((evalItem) => evalItem.id)} className="w-full">
          {evals.length === 0 ? (
            <div className="flex w-full flex-col items-center justify-center">
              <div className="mb-3 font-bold text-slate-400">No Evals Found</div>
              <a
                href={`mailto:team@neuronpedia.org?subject=Request%20Evals%20for%20${sae.modelId}%2C%20${sae.id}&body=I'd%20like%20to%20request%20evals%20for%20${sae.modelId}%2C%20${sae.id}%20on%20Neuronpedia.`}
                target="_blank"
                rel="noreferrer"
              >
                <Button variant="outline">Request</Button>
              </a>
            </div>
          ) : (
            evals
              .sort((a, b) => {
                if (a.typeName === 'core') return -1;
                if (b.typeName === 'core') return 1;
                return a.id.localeCompare(b.id);
              })
              .map(async (evalItem: EvalWithPartialRelations) => {
                const evalSchema = await getEvalOutputSchemaFromPrismaJson(
                  evalItem.type?.outputSchema as Prisma.JsonValue,
                );

                if (evalItem.typeName === 'core') {
                  const output = evalItem.output as Core;

                  const evalConfigSchema = evalSchema.properties?.eval_config?.properties;
                  const evalMetricsSchema = evalSchema.properties?.eval_result_metrics?.properties;

                  const featureMetricsRaw = evalItem.detailedMetrics
                    ? (evalItem.detailedMetrics as Prisma.JsonArray)
                    : [];

                  const featureMetrics = featureMetricsRaw.map((item: Prisma.JsonValue) => item as CoreFeatureMetric);

                  const featureDensityLog10 = featureMetrics.map((item: CoreFeatureMetric) =>
                    typeof item.feature_density === 'number' ? Math.log10(item.feature_density + 1e-10) : 1e-10,
                  );
                  const consistentActivationHeuristicLog10 = featureMetrics.map((item: CoreFeatureMetric) =>
                    typeof item.consistent_activation_heuristic === 'number'
                      ? Math.log10(item.consistent_activation_heuristic + 1e-10)
                      : 1e-10,
                  );

                  const densityVsCAHDomain = 3 / 4;

                  const encoderBias = featureMetrics.map((item: CoreFeatureMetric) => item.encoder_bias);
                  const encoderNorm = featureMetrics.map((item: CoreFeatureMetric) => item.encoder_norm);
                  const encoderDecoderCosineSim = featureMetrics.map(
                    (item: CoreFeatureMetric) => item.encoder_decoder_cosine_sim,
                  );

                  return (
                    <AccordionItem
                      value={evalItem.id}
                      key={evalItem.id}
                      className="mb-4 mt-1 flex w-full flex-col text-sm font-normal text-slate-600"
                    >
                      <SaeEvalAccordionTrigger evalSchema={evalSchema} evalItem={evalItem} />
                      <AccordionContent className="flex w-full flex-row gap-x-5 px-2">
                        <SaeEvalMetricsAccordion
                          output={output}
                          evalConfigSchema={evalConfigSchema || {}}
                          evalMetricsSchema={evalMetricsSchema || {}}
                          showBothWide={featureMetricsRaw.length === 0}
                        />
                        {featureMetricsRaw.length > 0 && (
                          <div className="grid basis-2/3 grid-cols-2 flex-col items-start justify-center gap-x-6 gap-y-5 px-0 pt-1">
                            <div className="col-span-2">
                              <div className="mb-2 flex cursor-default flex-row items-center gap-x-1 text-[13px] font-medium text-slate-600">
                                Feature Density{' '}
                                <CustomTooltip wide trigger={<QuestionMarkCircledIcon className="h-3.5 w-3.5" />}>
                                  <div className="flex flex-col">
                                    <div>Feature density is initially calculated as:</div>
                                    <div className="ml-2 mt-1">
                                      <code>feature_density = total_feature_acts / total_tokens</code>
                                    </div>
                                    <div className="mt-3">The log10 transformation is then applied:</div>
                                    <div className="ml-2 mt-1">
                                      <code>log10_feature_density = np.log10(feature_density + 1e-10)</code>
                                    </div>
                                    <div className="mt-3">
                                      Significance
                                      <ul className="mt-1 list-outside list-disc pl-6">
                                        <li>
                                          <b>Compression of Range</b>: The log transformation compresses the range of
                                          values, making it easier to visualize and compare features with very different
                                          activation rates.
                                        </li>
                                        <li>
                                          <b>Pattern Identification</b>: It helps in identifying patterns across orders
                                          of magnitude, which is useful when dealing with sparse activations.
                                        </li>
                                        <li>
                                          <b>Error Prevention</b>: The small constant (1e-10) is added to avoid log(0)
                                          errors for features that never activate.
                                        </li>
                                      </ul>
                                    </div>
                                    <div className="mt-3">
                                      Interpretation
                                      <ul className="mt-1 list-outside list-disc pl-6">
                                        <li>
                                          Values close to 0: Features that activate for nearly every token. These are
                                          highly dense features and can be difficult to interpret (but this is not
                                          necessarily the case). Context features tend to be more dense than other
                                          features.
                                        </li>
                                        <li>
                                          {`More negative values: Increasingly sparse
                                features. Dead features will show up as a bin of
                                maximally negative features on the left side of
                                the histogram. However, keep in mind that these
                                are mere features that did not activate over the
                                course of our evaluation--if too few prompts
                                were included, more features will appear
                                artificially "dead".`}
                                        </li>
                                        <li>
                                          The distribution of log10 feature density reveals the overall sparsity pattern
                                          of your SAE.
                                        </li>
                                      </ul>
                                    </div>
                                  </div>
                                </CustomTooltip>
                              </div>
                              <Plot
                                className="w-full"
                                data={[
                                  {
                                    x: featureDensityLog10,
                                    nbinsx: 100,
                                    showlegend: false,
                                    type: 'histogram',
                                  },
                                ]}
                                layout={{
                                  hovermode: 'closest',
                                  height: 120,
                                  xaxis: {
                                    fixedrange: true,
                                    tickfont: {
                                      size: 8,
                                    },
                                  },
                                  yaxis: {
                                    tickformat: '.1s',
                                    gridcolor: 'rgba(226,232,240,1)',
                                    zerolinecolor: 'rgba(226,232,240,1)',
                                    fixedrange: true,
                                    tickfont: {
                                      size: 8,
                                    },
                                  },
                                  barmode: 'relative',
                                  bargap: 0.0,
                                  showlegend: false,
                                  margin: {
                                    l: 30,
                                    r: 10,
                                    b: 20,
                                    t: 10,
                                    pad: 8,
                                  },
                                  paper_bgcolor: 'rgba(0,0,0,0)',
                                  plot_bgcolor: 'rgba(248,250,252,1)',
                                }}
                                config={{
                                  responsive: false,
                                  displayModeBar: false,
                                  editable: false,
                                  scrollZoom: false,
                                }}
                              />
                            </div>
                            <div className="col-span-2">
                              <div className="mb-2 flex cursor-default flex-row items-center gap-x-1 text-[13px] font-medium text-slate-600">
                                Feature Density (x) vs Consistent Activation Heuristic (y) [Log10]
                                <CustomTooltip wide trigger={<QuestionMarkCircledIcon className="h-3.5 w-3.5" />}>
                                  <div className="flex flex-col">
                                    <div>The Consistent Activation Heuristic is calculated as:</div>
                                    <div className="ml-2 mt-1">
                                      <code>
                                        log10_consistent_activation_heuristic = np.log10(consistent_activation_heuristic
                                        + 1e-10)
                                      </code>
                                    </div>
                                    <div className="mt-3">Where:</div>
                                    <ul className="mt-1 list-outside list-disc pl-6">
                                      <li>
                                        <code>total_feature_acts</code>: The total number of times each feature was
                                        activated across all tokens.
                                      </li>
                                      <li>
                                        <code>total_feature_prompts</code>: The number of prompts in which each feature
                                        was activated at least once.
                                      </li>
                                    </ul>
                                    <div className="mt-3">
                                      Significance
                                      <ul className="mt-1 list-outside list-disc pl-6">
                                        <li>
                                          <b>Measure of Consistency</b>
                                          {`: CAH indicates how
                                consistently a feature activates when it's
                                present in a prompt.`}
                                        </li>
                                        <li>
                                          <b>Feature Importance</b>
                                          {`: Higher values suggest
                                features that are consistently important when
                                relevant. For example, multi-token features and
                                context features (e.g., the "French" feature)
                                will have higher CAH scores.`}
                                        </li>
                                        <li>
                                          <b>Pattern Identification</b>: It can help identify features that capture
                                          specific, consistent patterns or concepts in the data.
                                        </li>
                                        <li>
                                          <b>Feature Categorization</b>
                                          {`: Helps distinguish
                                between "specialist" features (high CAH,
                                activated consistently when relevant) and more
                                general or noise-like features (low CAH,
                                activated inconsistently).`}
                                        </li>
                                      </ul>
                                    </div>
                                    <div className="mt-3">
                                      Combining Log10 CAH and Log10 Feature Density
                                      <div>
                                        When visualizing these metrics together (e.g., in a scatter plot), you can
                                        identify:
                                      </div>
                                      <ul className="mt-1 list-outside list-disc pl-6">
                                        <li>Dense, consistent features (high log10 density, high log10 CAH)</li>
                                        <li>Sparse, consistent features (low log10 density, high log10 CAH)</li>
                                        <li>Dense, inconsistent features (high log10 density, low log10 CAH)</li>
                                        <li>Sparse, inconsistent features (low log10 density, low log10 CAH)</li>
                                      </ul>
                                    </div>
                                  </div>
                                </CustomTooltip>
                              </div>
                              <Plot
                                className="w-full"
                                onClick={(event) => {
                                  if (sae.hasDashboards) {
                                    const pn = event.points[0].pointNumber;
                                    setFeatureModalFeature({
                                      modelId: evalItem.modelId,
                                      layer: evalItem.sourceId,
                                      index: pn.toString(),
                                    } as NeuronWithPartialRelations);
                                    setFeatureModalOpen(true);
                                  }
                                }}
                                data={[
                                  {
                                    marker: {
                                      size: 2,
                                    },
                                    mode: 'markers',
                                    showlegend: false,
                                    x: featureDensityLog10,
                                    xaxis: 'x',
                                    y: consistentActivationHeuristicLog10,
                                    yaxis: 'y',
                                    type: 'scattergl',
                                    hovertemplate: `Feature Index %{pointNumber}${
                                      sae.hasDashboards ? '<br>Click for Details' : ''
                                    }<extra></extra>`,
                                  },
                                  {
                                    alignmentgroup: true,
                                    bingroup: 'x',
                                    hovertemplate: 'log10_feature_density=%{x}<br>count=%{y}<extra></extra>',
                                    marker: {
                                      size: 2,
                                    },
                                    showlegend: false,
                                    x: featureDensityLog10,
                                    xaxis: 'x3',
                                    yaxis: 'y3',
                                    type: 'histogram',
                                  },
                                  {
                                    alignmentgroup: true,
                                    bingroup: 'y',
                                    hovertemplate:
                                      'log10_consistent_activation_heuristic=%{y}<br>count=%{x}<extra></extra>',
                                    marker: {
                                      size: 2,
                                    },
                                    showlegend: false,
                                    xaxis: 'x2',
                                    y: consistentActivationHeuristicLog10,
                                    yaxis: 'y2',
                                    type: 'histogram',
                                  },
                                ]}
                                layout={{
                                  template: {
                                    layout: {
                                      font: { color: '#2a3f5f' },
                                      hovermode: 'closest',
                                      hoverlabel: { align: 'auto' },
                                      paper_bgcolor: 'rgba(0,0,0,0)',
                                      plot_bgcolor: 'rgba(248,250,252,1)',
                                      xaxis: {
                                        fixedrange: true,
                                        tickfont: {
                                          size: 8,
                                        },
                                        gridcolor: 'transparent',
                                        zerolinecolor: 'rgba(226,232,240,1)',
                                        zerolinewidth: 1,
                                      },
                                      yaxis: {
                                        fixedrange: true,
                                        tickfont: {
                                          size: 8,
                                        },
                                        gridcolor: 'rgba(226,232,240,1)',
                                        zerolinecolor: 'rgba(226,232,240,1)',
                                        zerolinewidth: 1,
                                      },
                                    },
                                  },
                                  xaxis: {
                                    anchor: 'y',
                                    domain: [0.0, densityVsCAHDomain],
                                  },
                                  yaxis: {
                                    anchor: 'x',
                                    domain: [0.0, densityVsCAHDomain],
                                  },
                                  xaxis2: {
                                    anchor: 'y2',
                                    domain: [densityVsCAHDomain, 1.0],
                                    // matches: "x2",
                                    showticklabels: false,
                                    showline: false,
                                    showgrid: true,
                                  },
                                  yaxis2: {
                                    anchor: 'x2',
                                    domain: [0.0, densityVsCAHDomain],
                                    matches: 'y',
                                    showticklabels: false,
                                    showgrid: true,
                                  },
                                  xaxis3: {
                                    anchor: 'y3',
                                    matches: 'x',
                                    domain: [0.0, densityVsCAHDomain],
                                    showticklabels: false,
                                    showgrid: true,
                                  },
                                  yaxis3: {
                                    anchor: 'x3',
                                    domain: [densityVsCAHDomain, 1.0],
                                    // matches: "y3",
                                    showticklabels: false,
                                    showline: false,
                                    showgrid: true,
                                  },
                                  xaxis4: {
                                    anchor: 'y4',
                                    domain: [densityVsCAHDomain, 1.0],
                                    matches: 'x2',
                                    showticklabels: false,
                                    showgrid: true,
                                    showline: false,
                                  },
                                  yaxis4: {
                                    anchor: 'x4',
                                    domain: [densityVsCAHDomain, 1.0],
                                    showticklabels: false,
                                    matches: 'y3',
                                    showline: false,
                                    showgrid: true,
                                  },
                                  height: 250,
                                  showlegend: false,
                                  margin: {
                                    l: 30,
                                    r: 5,
                                    b: 17,
                                    t: 5,
                                    pad: 4,
                                  },
                                  paper_bgcolor: 'rgba(0,0,0,0)',
                                  plot_bgcolor: 'rgba(248,250,252,1)',
                                }}
                                config={{
                                  responsive: false,
                                  displayModeBar: false,
                                  editable: false,
                                  scrollZoom: false,
                                }}
                              />
                            </div>
                            <div className="col-span-2">
                              <div className="mb-2 text-[13px] font-medium text-slate-600">Feature Scatter Matrix</div>
                              <Plot
                                className="w-full"
                                onClick={(event) => {
                                  if (sae.hasDashboards) {
                                    const pn = event.points[0].pointNumber;
                                    setFeatureModalFeature({
                                      modelId: evalItem.modelId,
                                      layer: evalItem.sourceId,
                                      index: pn.toString(),
                                    } as NeuronWithPartialRelations);
                                    setFeatureModalOpen(true);
                                  }
                                }}
                                data={[
                                  {
                                    hovertemplate: `Feature Index %{pointNumber}${
                                      sae.hasDashboards ? '<br>Click for Details' : ''
                                    }<extra></extra>`,
                                    dimensions: [
                                      {
                                        values: encoderBias,
                                        label: 'Encoder Bias',
                                      },
                                      {
                                        values: encoderNorm,
                                        label: 'Encoder Norm',
                                      },
                                      {
                                        values: encoderDecoderCosineSim,
                                        label: 'Encoder Decoder Cosine Sim',
                                      },
                                    ],
                                    type: 'splom',
                                    marker: {
                                      size: 2,
                                    },
                                  },
                                ]}
                                layout={{
                                  height: 400,
                                  template: {
                                    layout: {
                                      hovermode: 'closest',
                                      xaxis: {
                                        zerolinecolor: 'rgba(203, 213, 225,1)',
                                        fixedrange: true,
                                        zerolinewidth: 1.5,
                                      },
                                      yaxis: {
                                        fixedrange: true,
                                        zerolinecolor: 'rgba(203, 213, 225,1)',
                                        zerolinewidth: 1.5,
                                      },
                                    },
                                  },
                                  xaxis: {
                                    fixedrange: true,
                                    zerolinecolor: 'rgba(226,232,240,1)',
                                    gridcolor: 'rgba(226,232,240,1)',
                                    tickformat: '.1s',
                                    tickfont: {
                                      size: 8,
                                    },
                                  },
                                  font: {
                                    size: 7,
                                  },
                                  yaxis: {
                                    tickformat: '.1s',
                                    gridcolor: 'rgba(226,232,240,1)',
                                    zerolinecolor: 'rgba(226,232,240,1)',
                                    fixedrange: true,
                                    tickfont: {
                                      size: 8,
                                    },
                                  },
                                  showlegend: false,
                                  margin: {
                                    l: 30,
                                    r: 0,
                                    b: 30,
                                    t: 10,
                                    pad: 2,
                                  },
                                  paper_bgcolor: 'rgba(0,0,0,0)',
                                  plot_bgcolor: 'rgba(248,250,252,1)',
                                }}
                                config={{
                                  responsive: false,
                                  displayModeBar: false,
                                  editable: false,
                                  scrollZoom: false,
                                }}
                              />
                            </div>
                            <div className="col-span-2">
                              <div className="mb-2 text-[13px] font-medium text-slate-600">
                                Encoder-Decoder Cosine Similarity
                              </div>
                              <Plot
                                className="w-full"
                                data={[
                                  {
                                    x: encoderDecoderCosineSim,
                                    // nbinsx: 100,
                                    showlegend: false,
                                    type: 'histogram',
                                  },
                                ]}
                                layout={{
                                  hovermode: 'closest',
                                  height: 120,
                                  xaxis: {
                                    fixedrange: true,
                                    tickfont: {
                                      size: 8,
                                    },
                                  },
                                  yaxis: {
                                    tickformat: '.1s',
                                    gridcolor: 'rgba(226,232,240,1)',
                                    zerolinecolor: 'rgba(226,232,240,1)',
                                    fixedrange: true,
                                    tickfont: {
                                      size: 8,
                                    },
                                  },
                                  barmode: 'relative',
                                  bargap: 0.0,
                                  showlegend: false,
                                  margin: {
                                    l: 30,
                                    r: 10,
                                    b: 20,
                                    t: 10,
                                    pad: 8,
                                  },
                                  paper_bgcolor: 'rgba(0,0,0,0)',
                                  plot_bgcolor: 'rgba(248,250,252,1)',
                                }}
                                config={{
                                  responsive: false,
                                  displayModeBar: false,
                                  editable: false,
                                  scrollZoom: false,
                                }}
                              />
                            </div>
                          </div>
                        )}
                      </AccordionContent>
                    </AccordionItem>
                  );
                }
                if (evalItem.typeName === 'absorption_first_letter') {
                  const output = evalItem.output as Absorption;
                  const evalConfigSchema = evalSchema.properties?.eval_config?.properties;
                  const evalMetricsSchema = evalSchema.properties?.eval_result_metrics?.properties;

                  if (evalItem.detailedMetrics) {
                    output.eval_result_details = evalItem.detailedMetrics as AbsorptionResultDetail[];
                  }

                  return (
                    <AccordionItem
                      value={evalItem.id}
                      key={evalItem.id}
                      className="mb-4 mt-1 flex w-full flex-col text-sm font-normal text-slate-600"
                    >
                      <SaeEvalAccordionTrigger evalSchema={evalSchema} evalItem={evalItem} />
                      <AccordionContent className="flex w-full flex-row gap-x-5 px-2">
                        <SaeEvalMetricsAccordion
                          output={output}
                          evalConfigSchema={evalConfigSchema || {}}
                          evalMetricsSchema={evalMetricsSchema || {}}
                        />
                        <div className="grid basis-2/3 grid-cols-3 flex-col items-start justify-center gap-x-6 gap-y-3 px-0 pt-1">
                          <div className="col-span-3 mb-4">
                            <div className="mb-2 text-[13px] font-medium text-slate-600">
                              Absorption Rate by First Letter
                            </div>
                            <Plot
                              className="w-full"
                              data={[
                                {
                                  x: output.eval_result_details?.map((d) => d.first_letter.toUpperCase()),
                                  y: output.eval_result_details?.map((d) => d.full_absorption_rate),
                                  type: 'bar',
                                  marker: {
                                    color: 'rgb(99, 102, 241)',
                                  },
                                  hovertext: output.eval_result_details?.map(
                                    (d) =>
                                      `${d.first_letter.toUpperCase()}<br>Absorption Rate: ${(
                                        d.full_absorption_rate * 100
                                      ).toFixed(0)}%` +
                                      `<br># Absorbed: ${d.num_full_absorption.toLocaleString()}<br># True Positives: ${d.num_probe_true_positives.toLocaleString()}<br># Split Features: ${d.num_split_features.toLocaleString()}`,
                                  ),
                                  name: '',
                                  hoverlabel: {
                                    bgcolor: '#f1f5f9',
                                    font: {
                                      color: 'black',
                                      size: 10,
                                    },
                                    align: 'left',
                                    bordercolor: 'lightgray',
                                  },
                                  hovertemplate: '%{hovertext}',
                                },
                              ]}
                              layout={{
                                height: 140,
                                xaxis: {
                                  fixedrange: true,
                                  tickfont: {
                                    size: 10,
                                  },
                                },
                                yaxis: {
                                  tickfont: {
                                    size: 10,
                                  },
                                  fixedrange: true,
                                },
                                barmode: 'relative',
                                bargap: 0.0,
                                showlegend: false,
                                margin: {
                                  l: 30,
                                  r: 10,
                                  b: 25,
                                  t: 5,
                                  pad: 8,
                                },
                                paper_bgcolor: 'rgba(0,0,0,0)',
                                plot_bgcolor: 'rgba(248,250,252,1)',
                              }}
                              config={{
                                responsive: false,
                                displayModeBar: false,
                                editable: false,
                                scrollZoom: false,
                              }}
                            />
                          </div>
                        </div>
                      </AccordionContent>
                    </AccordionItem>
                  );
                }
                const evalConfigSchema = evalSchema.properties?.eval_config?.properties;
                const evalMetricsSchema = evalSchema.properties?.eval_result_metrics?.properties;

                return (
                  <AccordionItem
                    value={evalItem.id}
                    key={evalItem.id}
                    className="mb-4 mt-1 flex w-full flex-col text-sm font-normal text-slate-600"
                  >
                    <SaeEvalAccordionTrigger evalSchema={evalSchema} evalItem={evalItem} />
                    <AccordionContent className="flex w-full flex-row gap-x-5 px-2">
                      <SaeEvalMetricsAccordion
                        output={evalItem.output as JSONSchema}
                        evalConfigSchema={evalConfigSchema || {}}
                        evalMetricsSchema={evalMetricsSchema || {}}
                        showBothWide
                      />
                    </AccordionContent>
                  </AccordionItem>
                );
              })
          )}
        </Accordion>
      </CardContent>
    </Card>
  );
}
