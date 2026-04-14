'use client';

import { Button } from '@/components/shadcn/button';
import * as Select from '@radix-ui/react-select';
import { JSONSchema } from 'json-schema-to-typescript';
import { ChevronDownIcon, ExternalLinkIcon } from 'lucide-react';
import { useSearchParams } from 'next/navigation';
import Plotly from 'plotly.js-dist-min';
import { DataTableFilterMeta } from 'primereact/datatable';
import { forwardRef, useEffect, useImperativeHandle, useState } from 'react';
import createPlotlyComponent from 'react-plotly.js/factory';
import CustomTooltip from '../../components/custom-tooltip';
import { PCA_BASELINES } from './pca-baselines-data';

import { convertEvalTypeToHash, convertNumToAbbr, MetricColumn } from './evals-table';

// const Plot = dynamic(
//   () => Promise.resolve(import('plotly.js-dist-min').then((Plotly) => createPlotlyComponent(Plotly))),
//   {
//     ssr: false,
//   },
// );

const Plot = createPlotlyComponent(Plotly);

export function getSaeBenchDisplayString(v: string) {
  return v === 'sae_bench'
    ? 'SAE Bench - Aug 24'
    : v === 'sae_bench_0125'
      ? 'SAE Bench - Jan 25'
      : v === 'gemmascope'
        ? 'Gemma Scope'
        : v === 'llamascope'
          ? 'Llama Scope'
          : v === 'deepseek-r1-distill-llama-8b'
            ? 'DeepSeek-R1-Distill-Llama-8B'
            : v === 'gemma-2-2b'
              ? 'Gemma-2-2B'
              : v === 'gemma-2-9b'
                ? 'Gemma-2-9B'
                : v === 'pythia-70m-deduped'
                  ? 'Pythia-70M-Deduped'
                  : v === 'pythia-160m-deduped'
                    ? 'Pythia-160M-Deduped'
                    : v === 'jumprelu'
                      ? 'JumpReLU'
                      : v === 'standard'
                        ? 'ReLU'
                        : v === 'topk'
                          ? 'TopK'
                          : v === 'gated'
                            ? 'Gated'
                            : v === 'batch_topk'
                              ? 'Batch TopK'
                              : v === 'matryoshka_batch_topk'
                                ? 'Matryoshka'
                                : v === 'p-anneal' || v === 'p_anneal'
                                  ? 'P-Anneal'
                                  : v;
}

const UNIQUE_COLORS = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7'];
const UNIQUE_SHAPES = ['circle', 'square', 'triangle-up', 'cross', 'diamond', 'star', 'x'];

const WIDTH_TO_COLOR: Record<string, string> = {
  '4096': UNIQUE_COLORS[0],
  '16384': UNIQUE_COLORS[1],
  '65536': UNIQUE_COLORS[2],
  '131072': UNIQUE_COLORS[3],
  '1048576': UNIQUE_COLORS[4],
};

const WIDTH_TO_SHAPE: Record<string, string> = {
  '4096': UNIQUE_SHAPES[0],
  '16384': UNIQUE_SHAPES[1],
  '65536': UNIQUE_SHAPES[2],
  '131072': UNIQUE_SHAPES[3],
  '1048576': UNIQUE_SHAPES[4],
};

const SAE_CLASS_TO_COLOR: Record<string, string> = {
  jumprelu: UNIQUE_COLORS[0],
  standard: UNIQUE_COLORS[1],
  topk: UNIQUE_COLORS[2],
  gated: UNIQUE_COLORS[3],
  batch_topk: UNIQUE_COLORS[4],
  matryoshka_batch_topk: UNIQUE_COLORS[5],
  p_anneal: UNIQUE_COLORS[6],
  'p-anneal': UNIQUE_COLORS[6],
};

const SAE_CLASS_TO_SHAPE: Record<string, string> = {
  jumprelu: UNIQUE_SHAPES[0],
  standard: UNIQUE_SHAPES[1],
  topk: UNIQUE_SHAPES[2],
  gated: UNIQUE_SHAPES[3],
  batch_topk: UNIQUE_SHAPES[4],
  matryoshka_batch_topk: UNIQUE_SHAPES[5],
  p_anneal: UNIQUE_SHAPES[6],
  'p-anneal': UNIQUE_SHAPES[6],
};

export interface EvalsPlotMethods {
  setPlotValuesX: (metricColumn: MetricColumn, values: number[]) => void;
  setPlotValuesY: (metricColumn: MetricColumn, values: number[]) => void;
  metricColumnX: MetricColumn | null;
  metricColumnY: MetricColumn | null;
  setMetricColumnX: (metricColumn: MetricColumn | null) => void;
  setMetricColumnY: (metricColumn: MetricColumn | null) => void;
  setVisibleRows: (rows: Record<string, any>[]) => void;
  logX: boolean;
  logY: boolean;
  setLogX: (logX: boolean) => void;
  setLogY: (logY: boolean) => void;
  groupBy: string | undefined;
  setGroupBy: (groupBy: string | undefined) => void;
  filterMeta: DataTableFilterMeta;
  setFilterMeta: (filterMeta: DataTableFilterMeta) => void;
}

export interface EvalsPlotProps {
  anonymized: boolean;
  evalTypeToOutputSchema: Map<string, JSONSchema>;
  metricColumns: MetricColumn[];
  setDefaultMetricColumnNamesToDisplay: (names: string[]) => void;
  setGroupByParent: (groupBy: string | undefined) => void;
  rerenderFilteredRows: (forceSortByMetricColumn: string | undefined) => void;
}

const groupByToDisplayName: Record<string, string> = {
  saeClass: 'SAE Class',
  dSae: 'Width',
  layer: 'Layer',
  trainingTokens: 'Training Tokens',
  modelId: 'Model',
  release: 'Release',
};

export const specialMetricColumnToDictKey: Record<string, string> = {
  Width: 'dSae',
  Layer: 'layer',
  'Training Tokens': 'trainingTokens',
};

// eslint-disable-next-line react/display-name
export default forwardRef<EvalsPlotMethods, EvalsPlotProps>(
  (
    {
      evalTypeToOutputSchema,
      metricColumns,
      setDefaultMetricColumnNamesToDisplay,
      setGroupByParent,
      rerenderFilteredRows,
      anonymized,
    },
    ref,
  ) => {
    const [plotValuesX, setPlotValuesX] = useState<number[]>([]);
    const [plotValuesY, setPlotValuesY] = useState<number[]>([]);
    const [metricColumnX, setMetricColumnX] = useState<MetricColumn | null>(null);
    const [metricColumnY, setMetricColumnY] = useState<MetricColumn | null>(null);
    const [groupBy, setGroupBy] = useState<string | undefined>(Object.keys(groupByToDisplayName)[0]);
    const [visibleRows, setVisibleRows] = useState<Record<string, any>[]>([]);

    function getEvalTypeDisplayName(metricColumn: MetricColumn) {
      return evalTypeToOutputSchema.get(metricColumn.evalType)?.title;
    }

    function getEvalTypeDescription(metricColumn: MetricColumn) {
      return evalTypeToOutputSchema.get(metricColumn.evalType)?.description;
    }
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const [showShapes, setShowShapes] = useState(false);
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const [showBaseline, setShowBaseline] = useState(true);

    const [logX, setLogX] = useState(false);
    const [logY, setLogY] = useState(false);
    const [filterMeta, setFilterMeta] = useState<DataTableFilterMeta>({});
    useImperativeHandle(ref, () => ({
      setPlotValuesX: (metricColumn: MetricColumn, values: number[]) => {
        setPlotValuesX(values);
        setMetricColumnX(metricColumn);
      },
      setPlotValuesY: (metricColumn: MetricColumn, values: number[]) => {
        setPlotValuesY(values);
        setMetricColumnY(metricColumn);
      },
      metricColumnX,
      metricColumnY,
      setMetricColumnX,
      setMetricColumnY,
      logX,
      logY,
      setLogX,
      setLogY,
      setVisibleRows,
      groupBy,
      setGroupBy,
      filterMeta,
      setFilterMeta,
    }));

    function getGroupedData() {
      if (!groupBy) {
        return [];
      }

      // split visibleRows by groupBy
      let groupedRows = visibleRows.reduce(
        (acc, row) => {
          const key = row[groupBy];
          acc[key] = acc[key] || [];
          acc[key].push(row);
          return acc;
        },
        {} as Record<string, Record<string, any>[]>,
      );

      // sort groupedRows by key
      const sortedEntries = Object.entries(groupedRows).sort(([keyA], [keyB]) => {
        if (groupBy === 'dSae') {
          // Sort numerically for dSae
          return Number(keyA) - Number(keyB);
        }
        // Default alphabetical sort
        return keyA.localeCompare(keyB);
      });
      groupedRows = Object.fromEntries(sortedEntries);

      const toReturn = Object.entries(groupedRows).map(([key, rows]) => ({
        x: rows.map((flattenedEval: Record<string, any>) => {
          if (metricColumnX && metricColumnX.getName() in flattenedEval) {
            return flattenedEval[metricColumnX.getName()];
          }
          if (metricColumnX && metricColumnX.metricName in specialMetricColumnToDictKey) {
            if (metricColumnX.metricName === 'Training Tokens' && flattenedEval.trainingTokens === -1) {
              return null;
            }
            return flattenedEval[specialMetricColumnToDictKey[metricColumnX.metricName]];
          }
          return null;
        }),
        y: rows.map((flattenedEval: Record<string, any>) => {
          if (metricColumnY && metricColumnY.getName() in flattenedEval) {
            return flattenedEval[metricColumnY.getName()];
          }
          if (metricColumnY && metricColumnY.metricName in specialMetricColumnToDictKey) {
            if (metricColumnY.metricName === 'Training Tokens' && flattenedEval.trainingTokens === -1) {
              return null;
            }
            return flattenedEval[specialMetricColumnToDictKey[metricColumnY.metricName]];
          }
          return null;
        }),
        hovertext: rows.map(
          (row: Record<string, any>) =>
            `${row.modelId.toUpperCase()}<br>Layer ${row.layer}<br>${row.sourceId}<br><br>${
              anonymized ? '' : '➡️ Click to View SAE'
            }`,
        ),
        type: 'scatter',
        name:
          groupBy === 'dSae'
            ? `${convertNumToAbbr(Number(key))} Width`
            : groupBy === 'trainingTokens'
              ? Number(key) === -1
                ? 'N/A Tokens'
                : `${convertNumToAbbr(Number(key))} Tokens`
              : getSaeBenchDisplayString(key),
        mode: 'markers',
        marker: {
          color:
            groupBy === 'saeClass' ? SAE_CLASS_TO_COLOR[key] : groupBy === 'dSae' ? WIDTH_TO_COLOR[key] : undefined,
          symbol:
            showShapes &&
            (groupBy === 'saeClass' ? SAE_CLASS_TO_SHAPE[key] : groupBy === 'dSae' ? WIDTH_TO_SHAPE[key] : undefined),
          size: 6,
        },
        hoverlabel: {
          bgcolor: '#f1f5f9',
          font: {
            color: 'black',
            size: 10,
          },
          align: 'left',
          bordercolor: 'lightgray',
        },
        hovertemplate: `${metricColumnX?.metricSchema?.title || metricColumnX?.metricName}: %{x:.2f}<br>${
          metricColumnY?.metricSchema?.title || metricColumnY?.metricName
        }: %{y:.2f}<br><br>%{hovertext}`,
      }));
      return toReturn;
    }

    function capitalizeFirstLetter(string: string) {
      return string.charAt(0).toUpperCase() + string.slice(1);
    }

    function getMetricTooltip(metricColumn: MetricColumn) {
      return (
        <CustomTooltip
          trigger={
            <div className="cursor-pointer rounded-full bg-sky-50 px-3 py-1 text-xs text-sky-700">
              {/* {getEvalTypeDisplayName(metricColumn)
              ? getEvalTypeDisplayName(metricColumn) + ": "
              : ""} */}
              {metricColumn.metricSchema?.title || metricColumn.metricName}
            </div>
          }
        >
          <div className="flex flex-col gap-y-1 whitespace-pre-wrap">
            <div className="mb-3 flex flex-col">
              <b className="mb-1">
                {getEvalTypeDisplayName(metricColumn) ? `Eval Type: ${getEvalTypeDisplayName(metricColumn)}` : ''}
              </b>
              <div>{getEvalTypeDescription(metricColumn)}</div>
            </div>
            <b>Eval Metric: {metricColumn.metricSchema?.title || metricColumn.metricName}</b>
            {metricColumn.metricSchema?.description}
            <Button
              variant="outline"
              size="sm"
              onClick={(e) => {
                e.stopPropagation();
                e.preventDefault();
                window.open(`/sae-bench/info#${convertEvalTypeToHash(metricColumn.evalType)}`, '_blank');
              }}
              className={`mt-1.5 ${
                anonymized ? 'hidden' : 'flex'
              } flex-row items-center gap-x-1 border-none bg-sky-700 text-xs font-medium text-white hover:bg-sky-600 hover:text-white`}
            >
              Eval Details <ExternalLinkIcon className="h-3 w-3" />
            </Button>
          </div>
        </CustomTooltip>
      );
    }

    function getBaselineValue() {
      // show baseline if the current visibleRows has: no more than one unique modelId, no more than one unique layer, and the y metric exists is not a special metric
      const uniqueModelIds = new Set(visibleRows.map((row) => row.modelId));
      const uniqueLayers = new Set(visibleRows.map((row) => row.layer));
      const yMetricExists = metricColumnY !== null;
      const yMetricIsNotSpecial = !metricColumnY || !(metricColumnY.metricName in specialMetricColumnToDictKey);
      const shouldShowBaseline =
        showBaseline && uniqueModelIds.size === 1 && uniqueLayers.size === 1 && yMetricExists && yMetricIsNotSpecial;

      let baselineValue;
      if (shouldShowBaseline) {
        const uniqueModelId = uniqueModelIds.values().next().value as string;
        const uniqueLayer = (uniqueLayers.values().next().value as number).toString();
        // look up the baseline value in baselineValues
        // check if modelId is in baselines
        if (uniqueModelId in PCA_BASELINES) {
          // check if layer is in baselines[uniqueModelId]
          if (uniqueLayer in PCA_BASELINES[uniqueModelId]) {
            // check if evalType is in baselines[uniqueModelId][uniqueLayer]
            if (metricColumnY?.evalType in PCA_BASELINES[uniqueModelId][uniqueLayer]) {
              // check if category + "." + metricName are in baselines[uniqueModelId][uniqueLayer][evalType]
              if (
                `${metricColumnY?.category}.${metricColumnY?.metricName}` in
                PCA_BASELINES[uniqueModelId][uniqueLayer][metricColumnY?.evalType]
              ) {
                baselineValue =
                  PCA_BASELINES[uniqueModelId][uniqueLayer][metricColumnY?.evalType][
                    `${metricColumnY?.category}.${metricColumnY?.metricName}`
                  ];
                if (baselineValue === null) {
                  baselineValue = undefined;
                }
              }
            }
          }
        }
      }
      return baselineValue;
    }

    function getMetricColumnFromName(name: string) {
      return metricColumns.find((metricColumn) => metricColumn.getName() === name);
    }

    function getBaseline() {
      const baselineValue = getBaselineValue();
      if (baselineValue === undefined) {
        return [];
      }
      return [
        {
          type: 'line',
          xref: 'paper', // Makes the line span the full width
          yref: 'y', // Keeps the line fixed to the y-axis values
          x0: 0, // Start at the left edge of the plot
          x1: 1, // End at the right edge of the plot
          y0: baselineValue, // Y-axis value where the line starts
          y1: baselineValue, // Y-axis value where the line ends
          line: {
            color: 'red',
            width: 1,
            dash: 'dot', // Dotted line
          },
        },
      ];
    }

    const searchParams = useSearchParams();
    const isEmbed = searchParams.get('embed') === 'true';

    function getSpecialMetricColumns() {
      const specialMetricColumns: MetricColumn[] = [];

      Object.keys(specialMetricColumnToDictKey).forEach((key) => {
        specialMetricColumns.push(new MetricColumn('none', 'none', key, {}));
      });

      return specialMetricColumns;
    }

    useEffect(() => {
      if (groupBy) {
        setGroupByParent(groupBy);
      }
    }, [groupBy, setGroupByParent]);

    function getMetricTooltipBodyForMetricColumn(metricColumn: MetricColumn) {
      if (metricColumn.evalType !== 'none') {
        return (
          <div className="flex flex-col gap-y-1 whitespace-pre-wrap">
            <div className="mb-3 flex flex-col">
              <b className="mb-1">
                {getEvalTypeDisplayName(metricColumn) ? `Eval Type: ${getEvalTypeDisplayName(metricColumn)}` : ''}
              </b>
              <div>{getEvalTypeDescription(metricColumn)}</div>
            </div>
            <b>Eval Metric: {metricColumn.metricSchema?.title || metricColumn.metricName}</b>
            {metricColumn.metricSchema?.description}

            <Button
              variant="outline"
              size="sm"
              onClick={(e) => {
                e.stopPropagation();
                e.preventDefault();
                window.open(`/sae-bench/info#${convertEvalTypeToHash(metricColumn.evalType)}`, '_blank');
              }}
              className={`mt-1.5 ${
                anonymized ? 'hidden' : 'flex'
              } flex-row items-center gap-x-1 bg-sky-700 text-xs font-medium text-white hover:bg-sky-600 hover:text-white`}
            >
              {`${getEvalTypeDisplayName(metricColumn)} `}Eval Details <ExternalLinkIcon className="h-3 w-3" />
            </Button>
          </div>
        );
      }
      return <div className="flex flex-col gap-y-1 whitespace-pre-wrap">{metricColumn.metricName}</div>;
    }

    useEffect(() => {
      if (metricColumnX && metricColumnY) {
        setDefaultMetricColumnNamesToDisplay([metricColumnX?.getName(), metricColumnY?.getName()]);
      }
    }, [metricColumnX, metricColumnY, setDefaultMetricColumnNamesToDisplay]);

    return (
      <div className="flex w-full flex-col items-center justify-center gap-y-0 bg-white pt-0 sm:items-start sm:gap-y-0 md:gap-y-5 lg:flex-row">
        {!isEmbed && (
          <div className="order-2 mb-2 flex flex-col gap-y-2 pl-5 pr-5 text-xs sm:order-1">
            <div className="mb-2 flex w-full flex-col items-start justify-start gap-x-2">
              <div className="mb-1 flex w-full flex-row items-center justify-between gap-x-2">
                <b className="text-[11px] text-slate-500">Plot X</b>
                <div className="flex flex-row items-center gap-x-1 text-[10px] font-medium text-slate-500">
                  <input
                    onChange={(e) => {
                      setLogX(e.target.checked);
                    }}
                    type="checkbox"
                    checked={logX}
                    className="h-3.5 w-3.5 cursor-pointer rounded border-slate-300 bg-slate-100 py-1 text-center text-xs text-sky-700 checked:border-sky-500 checked:bg-sky-500"
                  />{' '}
                  Log
                </div>
              </div>

              <div className="flex flex-row items-center gap-x-2">
                <Select.Root
                  defaultValue={metricColumnX?.getName()}
                  value={metricColumnX?.getName()}
                  onValueChange={(newVal) => {
                    const metricColumn = getMetricColumnFromName(newVal);
                    if (metricColumn) {
                      setMetricColumnX(metricColumn);
                    } else if (newVal.split('||').length === 3) {
                      const specialMetricColumn = new MetricColumn('none', 'none', newVal.split('||')[2], {});
                      setMetricColumnX(specialMetricColumn);
                    }
                  }}
                >
                  <Select.Trigger className="group flex max-h-[32px] min-h-[32px] w-[320px] min-w-[320px] max-w-[320px] flex-row items-center justify-start gap-x-1 whitespace-pre rounded-full border-slate-300 bg-slate-200 px-2 text-[10px] font-medium text-slate-700 hover:bg-slate-100 focus:outline-none sm:text-xs">
                    <div className="flex w-full flex-row items-center justify-center gap-x-1">
                      <div className="flex flex-col gap-y-0 sm:gap-y-0">
                        <Select.Value>
                          {metricColumnX && getEvalTypeDisplayName(metricColumnX)
                            ? `${getEvalTypeDisplayName(metricColumnX)}: `
                            : ''}
                          {metricColumnX?.metricSchema?.title || metricColumnX?.metricName}
                        </Select.Value>
                      </div>
                    </div>
                    <Select.Icon>
                      <ChevronDownIcon className="ml-0 w-4 leading-none sm:w-4" />
                    </Select.Icon>
                  </Select.Trigger>
                  <Select.Portal>
                    <Select.Content
                      position="popper"
                      align="center"
                      sideOffset={3}
                      className="z-30 max-h-[400px] cursor-pointer overflow-hidden rounded-md border-slate-300 bg-white text-xs font-medium text-sky-700 shadow-[0px_10px_38px_-10px_rgba(22,_23,_24,_0.35),_0px_10px_20px_-15px_rgba(22,_23,_24,_0.2)]"
                    >
                      <Select.Viewport className="text-left">
                        {metricColumns.concat(getSpecialMetricColumns()).map((metricColumn) => (
                          <div
                            key={metricColumn.getName()}
                            className={`flex w-full flex-row items-center justify-center border-b border-b-slate-100 px-3 ${
                              metricColumn.getName() === metricColumnX?.getName()
                                ? 'bg-sky-200 text-sky-700'
                                : 'text-slate-600 hover:bg-slate-100'
                            }`}
                          >
                            <CustomTooltip
                              trigger={
                                <div className="flex h-4 w-4 cursor-pointer items-center justify-center rounded-full bg-slate-200 text-center text-[10px] text-slate-600">
                                  ?
                                </div>
                              }
                            >
                              {metricColumn ? getMetricTooltipBodyForMetricColumn(metricColumn) : ''}
                            </CustomTooltip>
                            <Select.Item
                              key={metricColumn.getName()}
                              value={metricColumn.getName()}
                              className="flex flex-1 flex-col items-start gap-y-0.5 overflow-hidden px-3 py-2.5 text-xs focus:outline-none"
                            >
                              <div className="flex w-full flex-row items-center justify-start gap-x-3">
                                <Select.ItemText>
                                  {getEvalTypeDisplayName(metricColumn)
                                    ? `${getEvalTypeDisplayName(metricColumn)}: `
                                    : ''}
                                  <span className="">
                                    {metricColumn.metricSchema?.title || metricColumn.metricName}
                                  </span>
                                </Select.ItemText>
                              </div>
                            </Select.Item>
                          </div>
                        ))}
                      </Select.Viewport>
                      <Select.Arrow className="fill-white" />
                    </Select.Content>
                  </Select.Portal>
                </Select.Root>
              </div>
            </div>
            <div className="mb-2 flex w-full flex-col items-start justify-start gap-x-2">
              <div className="mb-1 flex w-full flex-row items-center justify-between gap-x-2">
                <b className="text-[11px] text-slate-500">Plot Y</b>
                <div className="flex flex-row items-center gap-x-1 text-[10px] font-medium text-slate-500">
                  <input
                    onChange={(e) => {
                      setLogY(e.target.checked);
                    }}
                    type="checkbox"
                    checked={logY}
                    className="h-3.5 w-3.5 cursor-pointer rounded border-slate-300 bg-slate-100 py-1 text-center text-xs text-sky-700 checked:border-sky-500 checked:bg-sky-500"
                  />{' '}
                  Log
                </div>
              </div>
              <div className="flex flex-row items-center gap-x-2">
                <Select.Root
                  defaultValue={metricColumnY?.getName()}
                  value={metricColumnY?.getName()}
                  onValueChange={(newVal) => {
                    const metricColumn = getMetricColumnFromName(newVal);
                    if (metricColumn) {
                      setMetricColumnY(metricColumn);
                      rerenderFilteredRows(metricColumn.getName());
                    } else if (newVal.split('||').length === 3) {
                      const specialMetricColumn = new MetricColumn('none', 'none', newVal.split('||')[2], {});
                      setMetricColumnY(specialMetricColumn);
                      rerenderFilteredRows(metricColumnY?.getName());
                    }
                  }}
                >
                  <Select.Trigger className="group flex max-h-[32px] min-h-[32px] w-[320px] min-w-[320px] max-w-[320px] flex-row items-center justify-start gap-x-1 whitespace-pre rounded-full border-slate-300 bg-slate-200 px-2 text-[10px] font-medium text-slate-700 hover:bg-slate-100 focus:outline-none sm:text-xs">
                    <div className="flex w-full flex-row items-center justify-center gap-x-1">
                      <div className="flex flex-col gap-y-0 sm:gap-y-0">
                        <Select.Value>
                          {metricColumnY && getEvalTypeDisplayName(metricColumnY)
                            ? `${getEvalTypeDisplayName(metricColumnY)}: `
                            : ''}
                          {metricColumnY?.metricSchema?.title || metricColumnY?.metricName}
                        </Select.Value>
                      </div>
                    </div>
                    <Select.Icon>
                      <ChevronDownIcon className="ml-0 w-4 leading-none sm:w-4" />
                    </Select.Icon>
                  </Select.Trigger>
                  <Select.Portal>
                    <Select.Content
                      position="popper"
                      align="center"
                      sideOffset={3}
                      className="z-30 max-h-[400px] cursor-pointer overflow-hidden rounded-md border-slate-300 bg-white text-xs font-medium text-sky-700 shadow-[0px_10px_38px_-10px_rgba(22,_23,_24,_0.35),_0px_10px_20px_-15px_rgba(22,_23,_24,_0.2)]"
                    >
                      <Select.Viewport className="text-left">
                        {metricColumns.concat(getSpecialMetricColumns()).map((metricColumn) => (
                          <div
                            key={metricColumn.getName()}
                            className={`flex w-full flex-row items-center justify-center border-b border-b-slate-100 px-3 ${
                              metricColumn.getName() === metricColumnY?.getName()
                                ? 'bg-sky-200 text-sky-700'
                                : 'text-slate-600 hover:bg-slate-100'
                            }`}
                          >
                            <CustomTooltip
                              trigger={
                                <div className="flex h-4 w-4 cursor-pointer items-center justify-center rounded-full bg-slate-200 text-center text-[10px] text-slate-600">
                                  ?
                                </div>
                              }
                            >
                              {metricColumn ? getMetricTooltipBodyForMetricColumn(metricColumn) : ''}
                            </CustomTooltip>
                            <Select.Item
                              key={metricColumn.getName()}
                              value={metricColumn.getName()}
                              className="flex flex-1 flex-col items-start gap-y-0.5 overflow-hidden px-3 py-2.5 text-xs focus:outline-none"
                            >
                              <div className="flex w-full flex-row items-center justify-start gap-x-3">
                                <Select.ItemText>
                                  {getEvalTypeDisplayName(metricColumn)
                                    ? `${getEvalTypeDisplayName(metricColumn)}: `
                                    : ''}
                                  <span className="">
                                    {metricColumn.metricSchema?.title || metricColumn.metricName}
                                  </span>
                                </Select.ItemText>
                              </div>
                            </Select.Item>
                          </div>
                        ))}
                      </Select.Viewport>
                      <Select.Arrow className="fill-white" />
                    </Select.Content>
                  </Select.Portal>
                </Select.Root>
              </div>
            </div>

            <div className="flex w-full flex-row items-center justify-between">
              <div className="flex flex-col items-center gap-x-2">
                <div className="mb-1 flex w-full flex-row items-center justify-between gap-x-2">
                  <b className="text-[11px] text-slate-500">Group By</b>
                </div>
                <Select.Root
                  defaultValue={groupBy}
                  value={groupBy}
                  onValueChange={(newVal) => {
                    setGroupBy(newVal);
                  }}
                >
                  <Select.Trigger className="group flex max-h-[32px] min-h-[32px] w-[320px] min-w-[320px] max-w-[320px] flex-row items-center justify-between gap-x-1 whitespace-pre rounded-full border-slate-300 bg-slate-200 px-2 pl-5 text-[10px] font-medium text-slate-700 hover:bg-slate-100 focus:outline-none sm:text-xs">
                    <div className="flex flex-1 flex-row justify-center">
                      <Select.Value />
                    </div>
                    <Select.Icon>
                      <ChevronDownIcon className="ml-0 w-4 leading-none sm:w-4" />
                    </Select.Icon>
                  </Select.Trigger>
                  <Select.Portal>
                    <Select.Content
                      position="popper"
                      align="center"
                      sideOffset={3}
                      className="z-30 max-h-[400px] cursor-pointer overflow-hidden rounded-md border-slate-300 bg-white text-xs font-medium text-sky-700 shadow-[0px_10px_38px_-10px_rgba(22,_23,_24,_0.35),_0px_10px_20px_-15px_rgba(22,_23,_24,_0.2)]"
                    >
                      <Select.Viewport className="text-left">
                        {Object.keys(groupByToDisplayName).map((groupById) => (
                          <Select.Item
                            key={groupById}
                            value={groupById}
                            className={`flex flex-col items-start gap-y-0.5 overflow-hidden border-b border-b-slate-100 px-3 py-2.5 text-xs ${
                              groupById === groupBy ? 'bg-sky-100 text-sky-700' : 'text-slate-600'
                            } hover:bg-slate-100 focus:outline-none`}
                          >
                            <div className="flex w-full flex-row items-center justify-between gap-x-3">
                              <Select.ItemText>
                                <span className="">{groupByToDisplayName[groupById]}</span>
                              </Select.ItemText>
                            </div>
                          </Select.Item>
                        ))}
                      </Select.Viewport>
                      <Select.Arrow className="fill-white" />
                    </Select.Content>
                  </Select.Portal>
                </Select.Root>
              </div>

              {/* <div className="flex flex-row items-center gap-x-1">
              <input
                onChange={(e) => {
                  setShowBaseline(e.target.checked);
                }}
                type="checkbox"
                checked={showBaseline}
                className="h-4 w-4 cursor-pointer rounded border-slate-300 bg-slate-100 py-1 text-center text-xs text-slate-700 checked:border-sky-600 checked:bg-sky-600 checked:text-white"
              />
              <div className="text-[11px] text-slate-400">
                <b>Show Baseline</b>
              </div>
            </div> */}
            </div>
          </div>
        )}

        <div className="order-1 flex w-full flex-col items-center justify-center sm:order-2">
          {/* title for when we have a filter */}
          <div className={`${isEmbed ? 'mt-3' : 'mt-0'} mb-1 text-[15px] font-medium text-slate-600`}>
            {Object.keys(filterMeta)
              .sort((a, b) => {
                if (a === 'release') return -1;
                if (b === 'release') return 1;
                if (a === 'modelId') return -1;
                if (b === 'modelId') return 1;
                if (a === 'dSae') return -1;
                if (b === 'dSae') return 1;
                if (a === 'layer') return -1;
                if (b === 'layer') return 1;
                return a.localeCompare(b);
              })
              .map((filter) => {
                // @ts-ignore
                let value = filterMeta[filter]?.value as any[];
                if (value && value.length > 0) {
                  value = value.map((v) => getSaeBenchDisplayString(v));
                  return value ? (
                    <span>
                      {filter === 'dSae'
                        ? `${value.map((v) => convertNumToAbbr(v)).join(', ')} Width`
                        : filter === 'trainingTokens'
                          ? ''
                          : filter === 'layer'
                            ? `Layer ${value.join(', ')}`
                            : typeof value[0] === 'string'
                              ? value.map((v) => capitalizeFirstLetter(v)).join(', ')
                              : value.join(', ')}{' '}
                    </span>
                  ) : (
                    ''
                  );
                }
                return '';
              })}
          </div>
          <div className="mb-2 mt-0 text-xs font-medium text-slate-500">
            {metricColumnX ? getMetricTooltip(metricColumnX) : ''}
            {' vs '}
            {metricColumnY ? getMetricTooltip(metricColumnY) : ''}
          </div>
          {/* @ts-ignore */}
          <Plot
            onClick={(event) => {
              if (anonymized) return;
              const pn = event.points[0].pointNumber;
              window.open(`/${visibleRows[pn].modelId}/${visibleRows[pn].sourceId}`, '_blank');
            }}
            // @ts-ignore
            data={
              groupBy
                ? getGroupedData()
                : [
                    {
                      x: plotValuesX,
                      mode: 'markers',
                      y: plotValuesY,
                      type: 'scatter',
                      hovertext: visibleRows.map(
                        (row) => `${row.modelId.toUpperCase()}<br>Layer ${row.layer}<br>${row.sourceId}`,
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
                      hovertemplate: `${metricColumnX?.metricSchema?.title || metricColumnX?.metricName}: %{x:.2f}<br>${
                        metricColumnY?.metricSchema?.title || metricColumnY?.metricName
                      }: %{y:.2f}<br><br>%{hovertext}`,
                    },
                  ]
            }
            layout={{
              height: 320,
              // width: 440,
              // @ts-ignore
              shapes: getBaseline(),
              annotations:
                getBaselineValue() !== undefined
                  ? [
                      {
                        x: 1, // x position of the label (moved to right)
                        y: getBaselineValue() || 0, // y position (same as line)
                        text: logY ? '' : 'PCA', // text is in wrong place if logY is true
                        showarrow: false,
                        yshift: 8, // Shift label slightly above the line
                        xref: 'x',
                        font: {
                          size: 8,
                          color: 'red',
                        },
                        yref: 'y',
                        xanchor: 'right', // Align text to the right
                      },
                    ]
                  : [],
              xaxis: {
                title: {
                  text: metricColumnX?.metricSchema?.title || metricColumnX?.metricName,
                  font: {
                    size: 10,
                    color: 'darkgrey',
                  },
                  standoff: 40,
                },
                type: logX ? 'log' : 'linear',
                gridcolor: 'lightgrey',
                zerolinecolor: 'lightgrey',
                fixedrange: true,
                tickfont: {
                  size: 10,
                },
              },
              yaxis: {
                title: {
                  text: metricColumnY?.metricSchema?.title || metricColumnY?.metricName,
                  font: {
                    size: 10,
                    color: 'darkgrey',
                  },
                  standoff: 40,
                },
                type: logY ? 'log' : 'linear',
                gridcolor: 'lightgrey',
                zerolinecolor: 'lightgrey',
                fixedrange: true,
                tickfont: {
                  size: 10,
                },
              },
              barmode: 'relative',
              bargap: 0.05,
              showlegend: !!groupBy,
              legend: {
                orientation: 'h',
                yanchor: 'bottom',
                y: 1.02,
                xanchor: 'center',
                x: 0.5,
              },
              margin: {
                l: 50,
                r: 0,
                b: 38,
                t: 0,
                pad: 4,
              },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
            }}
            config={{
              responsive: true,
              displayModeBar: false,
              editable: false,
              scrollZoom: false,
            }}
            className={`ml-0 mr-1 mt-0 h-[320px] min-h-[320px] w-full ${isEmbed ? 'mb-[10px]' : 'mb-5 mt-2'}`}
          />
          {/* 
          <div className="flex flex-row items-center gap-x-1">
            <input
              onChange={(e) => {
                setShowShapes(e.target.checked);
              }}
              type="checkbox"
              checked={showShapes}
              className="h-3.5 w-3.5 cursor-pointer rounded border-slate-300 bg-slate-100 py-1 text-center text-xs text-sky-700 checked:border-sky-500 checked:bg-sky-500"
            />
            Shapes
          </div> */}
        </div>
      </div>
    );
  },
);
