'use client';

// Obviously, need to be cleaned up.
// Mostly one-off code for SAE Bench / SAE Evals

import CustomTooltip from '@/components/custom-tooltip';
import { Button } from '@/components/shadcn/button';
import { getLayerNumFromSource } from '@/lib/utils/source';
import { EvalTypeWithPartialRelations } from '@/prisma/generated/zod';
import $RefParser from '@apidevtools/json-schema-ref-parser';
import { Prisma } from '@prisma/client';
import copy from 'copy-to-clipboard';
import { JSONSchema } from 'json-schema-to-typescript';
import { ChartScatter, DownloadCloud, ExternalLinkIcon, Info, NotebookText } from 'lucide-react';
import { useSearchParams } from 'next/navigation';
import { FilterMatchMode } from 'primereact/api';
import { Column, ColumnFilterElementTemplateOptions } from 'primereact/column';
import { DataTable, DataTableFilterMeta, DataTableValueArray } from 'primereact/datatable';
import { MultiSelect, MultiSelectChangeEvent } from 'primereact/multiselect';
import 'primereact/resources/themes/nano/theme.css';
import { useEffect, useRef, useState } from 'react';

import { Absorption } from '@/types/eval_output_schema_absorption_first_letter';
import { AutoInterp } from '@/types/eval_output_schema_autointerp';
import { Core } from '@/types/eval_output_schema_core';
import { RAVEL } from '@/types/eval_output_schema_ravel';
import { SCR } from '@/types/eval_output_schema_scr';
import { SparseProbing } from '@/types/eval_output_schema_sparse_probing';
import { TPP } from '@/types/eval_output_schema_tpp';
import { Unlearning } from '@/types/eval_output_schema_unlearning';

import EvalsPlot, { EvalsPlotMethods, getSaeBenchDisplayString, specialMetricColumnToDictKey } from './evals-plot';

import { EVALS_TABLE_PRESETS } from './evals-table-presets';

const DEFAULT_SORT_ORDER = -1;
const DEFAULT_NUMERIC_PRECISION = 2;

export class MetricColumn {
  evalType: string;

  category: string;

  metricName: string;

  metricSchema: JSONSchema;

  constructor(evalType: string, category: string, metricName: string, metricSchema: JSONSchema) {
    this.evalType = evalType;
    this.category = category;
    this.metricName = metricName;
    this.metricSchema = metricSchema;
  }

  getName(): string {
    return `${this.evalType}||${this.category}||${this.metricName}`;
  }
}

export function convertNumToAbbr(dSae: number) {
  return dSae >= 1000000 ? `${(dSae / 1000000).toFixed(0)}M` : dSae >= 1000 ? `${Math.floor(dSae / 1000)}k` : dSae;
}

export function convertEvalTypeToHash(evalType: string) {
  if (evalType === 'core') {
    return 'unsupervised-metrics-core';
  }
  if (evalType === 'absorption_first_letter') {
    return 'feature-absorption';
  }
  if (evalType === 'unlearning') {
    return 'unlearning';
  }
  if (evalType === 'scr') {
    return 'spurious-correlation-removal-scr';
  }
  if (evalType === 'tpp') {
    return 'targeted-probe-perturbation-tpp';
  }
  if (evalType === 'autointerp') {
    return 'automated-interpretability';
  }
  if (evalType === 'ravel') {
    return 'ravel';
  }
  if (evalType === 'sparse_probing') {
    return 'sparse-probing';
  }
  return evalType;
}

export type EvalsTableParams = {
  metricX?: string;
  metricY?: string;
  logX?: boolean;
  logY?: boolean;
  groupBy?: string;
  modelFilter?: string[];
  releaseFilter?: string[];
  saeClassFilter?: string[];
  widthFilter?: number[];
  layerFilter?: number[];
  trainingTokensFilter?: number[];
};

export const EVAL_UI_DEFAULT_DISPLAY = 'ui_default_display';

export async function getEvalOutputSchemaFromPrismaJson(json: Prisma.JsonValue): Promise<JSONSchema> {
  const toReturn = JSON.parse(JSON.stringify(json)) as JSONSchema;
  await $RefParser.dereference(toReturn);
  return toReturn;
}

export function getEvalMetricsCategorizedSchemaFromOutputSchema(outputSchema: JSONSchema) {
  return outputSchema.properties?.eval_result_metrics?.properties;
}

export function getEvalCategory(
  evalTypeName: string,
  evaluation: any,
  categoryName: string,
): Record<string, number> | null {
  let output;
  if (evalTypeName === 'core') {
    output = evaluation.output as Core;
  } else if (evalTypeName === 'absorption_first_letter') {
    output = evaluation.output as Absorption;
  } else if (evalTypeName === 'scr') {
    output = evaluation.output as SCR;
  } else if (evalTypeName === 'tpp') {
    output = evaluation.output as TPP;
  } else if (evalTypeName === 'ravel') {
    output = evaluation.output as RAVEL;
  } else if (evalTypeName === 'autointerp') {
    output = evaluation.output as AutoInterp;
  } else if (evalTypeName === 'unlearning') {
    output = evaluation.output as Unlearning;
  } else if (evalTypeName === 'sparse_probing') {
    output = evaluation.output as SparseProbing;
  }

  if (!output) {
    return null;
  }
  return output.eval_result_metrics[categoryName] as Record<string, number>;
}

export default function EvalsTable({
  evalTypes,
  anonymized = true,
}: {
  evalTypes: EvalTypeWithPartialRelations[];
  anonymized?: boolean;
}) {
  const searchParams = useSearchParams();
  const evalsPlotRef = useRef<EvalsPlotMethods>(null);
  const isEmbed = searchParams.get('embed') === 'true';
  const primaryKeys: string[] = ['sourceId', 'modelId', 'release', 'saeClass', 'dSae', 'layer', 'trainingTokens'];
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [visibleEvalTypes, setVisibleEvalTypes] = useState<EvalTypeWithPartialRelations[]>(
    evalTypes.filter((evalType) => evalType.evals && evalType.evals.length > 0),
  );
  const [evalTypeToOutputSchema, setEvalTypeToOutputSchema] = useState<Map<string, JSONSchema>>(new Map());
  const [evalsFlattened, setEvalsFlattened] = useState<Record<string, any>[]>([]);
  const [metricColumns, setMetricColumns] = useState<MetricColumn[]>([]);
  const [filteredRows, setFilteredRows] = useState<Record<string, any>[]>([]);
  const [metricColumnNamesToDisplay, setMetricColumnNamesToDisplay] = useState<string[]>([]);
  const dataTableRef = useRef<DataTable<DataTableValueArray>>(null);
  const [modelIdFilter, setModelIdFilter] = useState<string[]>([]);
  const [sourceIdFilter, setSourceIdFilter] = useState<string[]>([]);
  const [layerFilter, setLayerFilter] = useState<number[]>([]);
  const [trainingTokensFilter, setTrainingTokensFilter] = useState<number[]>([]);
  const [saeClassFilter, setSaeClassFilter] = useState<string[]>([]);
  const [dSaeFilter, setDsaeFilter] = useState<number[]>([]);
  const [releaseFilter, setReleaseFilter] = useState<string[]>([]);
  const [sortByMetricColumnName, setSortByMetricColumnName] = useState<string | undefined>();
  const visibleRows = useRef<Record<string, any>[]>([]);

  function getEvalTypeDisplayName(metricColumn: MetricColumn) {
    return evalTypeToOutputSchema.get(metricColumn.evalType)?.title;
  }

  const getValuesForPlot = (metricColumn: MetricColumn) => {
    if (metricColumn.metricName in specialMetricColumnToDictKey) {
      return visibleRows.current.map(
        (flattenedEval) => flattenedEval[specialMetricColumnToDictKey[metricColumn.metricName]] || 0,
      );
    }
    return visibleRows.current.map((flattenedEval) => {
      if (metricColumn.getName() in flattenedEval) {
        return flattenedEval[metricColumn.getName()];
      }
      return 0;
    });
  };

  function searchParamsToEvalsTableParams(searchParamsToUse: URLSearchParams): EvalsTableParams {
    const toReturn: EvalsTableParams = {
      metricX: searchParamsToUse.get('metricX') || undefined,
      metricY: searchParamsToUse.get('metricY') || undefined,
      logX: searchParamsToUse.get('logX') === 'true' || false,
      logY: searchParamsToUse.get('logY') === 'true' || false,
      groupBy: searchParamsToUse.get('groupBy') || undefined,
      modelFilter: searchParamsToUse.get('modelId')?.split(',') || undefined,
      releaseFilter: searchParamsToUse.get('release')?.split(',') || undefined,
      saeClassFilter: searchParamsToUse.get('saeClass')?.split(',') || undefined,
      widthFilter: searchParamsToUse.get('dSae')?.split(',').map(Number) || undefined,
      layerFilter: searchParamsToUse.get('layer')?.split(',').map(Number) || undefined,
      trainingTokensFilter: searchParamsToUse.get('trainingTokens')?.split(',').map(Number) || undefined,
    };

    return toReturn;
  }

  const [defaultEvalsTableParams, setDefaultEvalsTableParams] = useState<EvalsTableParams>(
    searchParamsToEvalsTableParams(searchParams as unknown as URLSearchParams),
  );

  function clearFilters() {
    setModelIdFilter([]);
    setSourceIdFilter([]);
    setLayerFilter([]);
    setTrainingTokensFilter([]);
    setSaeClassFilter([]);
    setDsaeFilter([]);
    setReleaseFilter([]);
    setDefaultEvalsTableParams(searchParamsToEvalsTableParams(new URLSearchParams()));
  }

  function getFilteredRows(
    ignoreFilter: string | undefined = undefined,
    forceSortByMetricColumn: string | undefined = undefined,
  ) {
    let rows = evalsFlattened;
    // sort the evalsFlattened by the sortByMetricColumn
    const sortByMetricColName = forceSortByMetricColumn || evalsPlotRef.current?.metricColumnY?.getName();
    rows = rows.sort((a, b) => {
      if (sortByMetricColName === undefined) {
        return 0;
      }
      if (
        a.hasOwnProperty(sortByMetricColName) &&
        b.hasOwnProperty(sortByMetricColName) &&
        typeof a[sortByMetricColName] === 'number' &&
        typeof b[sortByMetricColName] === 'number'
      ) {
        return b[sortByMetricColName] - a[sortByMetricColName];
      }
      return 0;
    });
    // filter out if the sortByMetricColName is undefined
    if (sortByMetricColName) {
      rows = rows.filter((evalItem) => evalItem[sortByMetricColName] !== undefined);
    }
    if (modelIdFilter.length > 0 && ignoreFilter !== 'modelId') {
      rows = rows.filter((evalItem) => modelIdFilter.includes(evalItem.modelId as string));
    }
    if (sourceIdFilter.length > 0 && ignoreFilter !== 'sourceId') {
      rows = rows.filter((evalItem) => sourceIdFilter.includes(evalItem.sourceId as string));
    }
    if (layerFilter.length > 0 && ignoreFilter !== 'layer') {
      rows = rows.filter((evalItem) => layerFilter.includes(evalItem.layer));
    }
    if (saeClassFilter.length > 0 && ignoreFilter !== 'saeClass') {
      rows = rows.filter((evalItem) => saeClassFilter.includes(evalItem.saeClass as string));
    }
    if (dSaeFilter.length > 0 && ignoreFilter !== 'dSae') {
      rows = rows.filter((evalItem) => dSaeFilter.includes(evalItem.dSae));
    }
    if (releaseFilter.length > 0 && ignoreFilter !== 'release') {
      rows = rows.filter((evalItem) => releaseFilter.includes(evalItem.release as string));
    }
    if (trainingTokensFilter.length > 0 && ignoreFilter !== 'trainingTokens') {
      rows = rows.filter((evalItem) => trainingTokensFilter.includes(evalItem.trainingTokens));
    }
    return rows;
  }

  useEffect(() => {
    let flattenedEvalsToSet: Record<string, any>[] = [];
    // make it a copy of evalsFlattened
    for (const evaluation of evalsFlattened) {
      flattenedEvalsToSet.push(evaluation);
    }
    const metricColumnsToSet: MetricColumn[] = [];
    for (const evalType of visibleEvalTypes) {
      if (evalTypeToOutputSchema.get(evalType.name)) {
        const evalMetricsCategorizedSchema = getEvalMetricsCategorizedSchemaFromOutputSchema(
          evalTypeToOutputSchema.get(evalType.name) as JSONSchema,
        );
        // print each of the keys in each category
        for (const category in evalMetricsCategorizedSchema) {
          for (const metricKey in evalMetricsCategorizedSchema[category].properties) {
            // check if the UI_DISPLAY is true
            const metricSchema = evalMetricsCategorizedSchema[category].properties[metricKey];
            const uiDisplay = metricSchema[EVAL_UI_DEFAULT_DISPLAY];
            if (uiDisplay === true) {
              const metricColumn = new MetricColumn(evalType.name, category, metricKey, metricSchema);
              metricColumnsToSet.push(metricColumn);
              if (evalType.evals) {
                // find the metric in each of the evals, set the value
                for (const evaluation of evalType.evals) {
                  const evalCategory = getEvalCategory(evalType.name, evaluation, metricColumn.category);
                  if (evalCategory) {
                    const metric = evalCategory[metricColumn.metricName];

                    const foundEval = flattenedEvalsToSet.find(
                      (evaluationToMatch) =>
                        evaluationToMatch.modelId === evaluation.modelId &&
                        evaluationToMatch.sourceId === evaluation.sourceId,
                    );

                    if (foundEval) {
                      foundEval[metricColumn.getName()] = metric;
                    } else {
                      const release = evaluation.sourceId?.split('-')?.[1];
                      const output = evaluation.output as any;
                      const trainingTokens = output?.sae_cfg_dict?.training_tokens as number;
                      let saeClass = output?.sae_cfg_dict?.architecture as string;
                      const saelensReleaseId = output?.sae_lens_release_id as string;
                      // for some SAEs the architecture is not reliable and it says "Standard" instead of p-anneal in the architecture field
                      // so we have to check the saelensReleaseId
                      if (
                        saelensReleaseId &&
                        (saelensReleaseId.includes('sae_bench') || saelensReleaseId.includes('saebench')) &&
                        (saelensReleaseId.toLowerCase().includes('panneal') ||
                          saelensReleaseId.toLowerCase().includes('p_anneal'))
                      ) {
                        saeClass = 'p-anneal';
                      } else if (output?.sae_cfg_dict?.activation_fn_str === 'topk') {
                        saeClass = 'topk';
                      } else if (output?.sae_cfg_dict?.architecture === 'standard_april_update') {
                        saeClass = 'standard';
                      }
                      const dSae = output?.sae_cfg_dict?.d_sae as number;
                      flattenedEvalsToSet.push({
                        modelId: evaluation.modelId,
                        sourceId: evaluation.sourceId,
                        release,
                        layer: getLayerNumFromSource(evaluation.sourceId || ''),
                        saeClass,
                        dSae,
                        trainingTokens: trainingTokens !== undefined ? trainingTokens : -1,
                        [metricColumn.getName()]: metric,
                      });
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    // hide flattened evals that are missing core
    flattenedEvalsToSet = flattenedEvalsToSet.filter((ev) => {
      // check keys for this eval
      const keys = Object.keys(ev);
      // iterate over the keys
      let shouldReturn = false;
      for (const key of keys) {
        if (key.startsWith('core||')) {
          shouldReturn = true;
          break;
        }
      }
      return shouldReturn;
    });

    setEvalsFlattened(flattenedEvalsToSet);
    setMetricColumns(metricColumnsToSet);
  }, [evalTypeToOutputSchema]);

  function getMetricSchema(evalType: string, category: string, metricName: string) {
    return evalTypeToOutputSchema.get(evalType)?.properties?.eval_result_metrics?.properties?.[category]?.properties?.[
      metricName
    ] as JSONSchema;
  }

  function rerenderFilteredRows(forceSortByMetricColumn: string | undefined = undefined) {
    setFilteredRows(getFilteredRows(undefined, forceSortByMetricColumn));
  }

  const [groupBy, setGroupBy] = useState<string | undefined>(undefined);

  function setPreset(evalTableParams: EvalsTableParams) {
    const toReturn = evalTableParams;
    const colsToDisplay: string[] = [];
    if (toReturn.metricX) {
      const split = toReturn.metricX.split('||');
      // get the metric schema
      const metricSchema = getMetricSchema(split[0], split[1], split[2]);
      evalsPlotRef.current?.setMetricColumnX(new MetricColumn(split[0], split[1], split[2], metricSchema));
      colsToDisplay.push(toReturn.metricX);
    }
    if (toReturn.metricY) {
      const split = toReturn.metricY.split('||');
      const metricSchema = getMetricSchema(split[0], split[1], split[2]);
      evalsPlotRef.current?.setMetricColumnY(new MetricColumn(split[0], split[1], split[2], metricSchema));
      colsToDisplay.push(toReturn.metricY);
    }
    setMetricColumnNamesToDisplay(colsToDisplay);

    setSortByMetricColumnName(evalsPlotRef.current?.metricColumnY?.getName() || undefined);
    if (toReturn.groupBy) {
      evalsPlotRef.current?.setGroupBy(toReturn.groupBy);
      setGroupBy(toReturn.groupBy);
    }
    if (toReturn.logX) {
      evalsPlotRef.current?.setLogX(toReturn.logX);
    } else {
      evalsPlotRef.current?.setLogX(false);
    }
    if (toReturn.logY) {
      evalsPlotRef.current?.setLogY(toReturn.logY);
    } else {
      evalsPlotRef.current?.setLogY(false);
    }

    // if there are params or it's embed, then use those. otherwise set our primary preset
    if (
      isEmbed ||
      evalTableParams.metricX ||
      evalTableParams.metricY ||
      evalTableParams.groupBy ||
      (evalTableParams.modelFilter && evalTableParams.modelFilter.length > 0) ||
      (evalTableParams.releaseFilter && evalTableParams.releaseFilter.length > 0) ||
      (evalTableParams.saeClassFilter && evalTableParams.saeClassFilter.length > 0) ||
      (evalTableParams.widthFilter && evalTableParams.widthFilter.length > 0) ||
      (evalTableParams.layerFilter && evalTableParams.layerFilter.length > 0) ||
      (evalTableParams.trainingTokensFilter && evalTableParams.trainingTokensFilter.length > 0)
    ) {
      setModelIdFilter(evalTableParams.modelFilter || []);
      setReleaseFilter(evalTableParams.releaseFilter || []);
      setSaeClassFilter(evalTableParams.saeClassFilter || []);
      setDsaeFilter(evalTableParams.widthFilter || []);
      setLayerFilter(evalTableParams.layerFilter || []);
      setTrainingTokensFilter(evalTableParams.trainingTokensFilter || []);
    } else {
      // first preset
      const firstPreset = EVALS_TABLE_PRESETS[0];
      setModelIdFilter(firstPreset.params.modelFilter || []);
      setReleaseFilter(firstPreset.params.releaseFilter || []);
      setSaeClassFilter(firstPreset.params.saeClassFilter || []);
      setDsaeFilter(firstPreset.params.widthFilter || []);
      setLayerFilter(firstPreset.params.layerFilter || []);
      setTrainingTokensFilter(firstPreset.params.trainingTokensFilter || []);
      const metricXsplit = firstPreset.params.metricX?.split('||');
      const metricCols = [];
      if (metricXsplit && metricXsplit.length === 3) {
        const metricSchemaX = getMetricSchema(metricXsplit[0], metricXsplit[1], metricXsplit[2]);
        const metricX = new MetricColumn(metricXsplit[0], metricXsplit[1], metricXsplit[2], metricSchemaX);
        evalsPlotRef.current?.setPlotValuesX(metricX, getValuesForPlot(metricX));
        evalsPlotRef.current?.setLogX(firstPreset.params.logX || false);
        metricCols.push(metricX);
      }
      const metricYsplit = firstPreset.params.metricY?.split('||');
      if (metricYsplit && metricYsplit.length === 3) {
        const metricSchemaY = getMetricSchema(metricYsplit[0], metricYsplit[1], metricYsplit[2]);
        const metricY = new MetricColumn(metricYsplit[0], metricYsplit[1], metricYsplit[2], metricSchemaY);
        evalsPlotRef.current?.setPlotValuesY(metricY, getValuesForPlot(metricY));
        evalsPlotRef.current?.setLogY(firstPreset.params.logY || false);
        metricCols.push(metricY);
      }
      setMetricColumnNamesToDisplay(metricCols.map((metricCol) => metricCol?.getName() || ''));
    }
  }

  useEffect(() => {
    if (evalsFlattened.length > 0 && evalTypeToOutputSchema) {
      setPreset(defaultEvalsTableParams);
    }
  }, [evalsFlattened, evalTypeToOutputSchema]);

  useEffect(() => {
    async function updateMetricsToShow() {
      const newSchemaMap = new Map(evalTypeToOutputSchema);
      for (const evalType of visibleEvalTypes) {
        newSchemaMap.set(
          evalType.name,
          // eslint-disable-next-line no-await-in-loop
          await getEvalOutputSchemaFromPrismaJson(evalType.outputSchema as Prisma.JsonValue),
        );
      }
      setEvalTypeToOutputSchema(newSchemaMap);
    }
    updateMetricsToShow();
  }, [visibleEvalTypes]);

  function getEvalTypeDescription(metricColumn: MetricColumn) {
    return evalTypeToOutputSchema.get(metricColumn.evalType)?.description;
  }

  function getHeaderForMetricColumn(metricColumn: MetricColumn) {
    const evalTypeDisplayName = getEvalTypeDisplayName(metricColumn);
    if (metricColumn.metricSchema) {
      return (
        <div className="flex flex-col items-center justify-center leading-tight">
          <CustomTooltip
            trigger={
              <div className="flex select-none flex-col items-center justify-start">
                <div className="text-[9px] font-semibold text-slate-400">{evalTypeDisplayName}</div>
                <div className="mb-0.5 mt-0.5 flex items-center justify-center text-center text-[10px]">
                  {metricColumn.metricSchema.title}
                </div>
              </div>
            }
          >
            <div className="flex flex-col gap-y-1 whitespace-pre-wrap">
              <div className="mb-3 flex flex-col">
                <b className="mb-1">Eval Type: {evalTypeDisplayName}</b>
                <div>{getEvalTypeDescription(metricColumn)}</div>
              </div>
              <b>Eval Metric: {metricColumn.metricSchema.title}</b>
              {metricColumn.metricSchema.description}
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
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
        </div>
      );
    }
    return <div className="whitespace-pre">{metricColumn.metricName}</div>;
  }

  function ClearFiltersButton() {
    return (
      <Button
        size="sm"
        onClick={() => {
          clearFilters();
        }}
        variant="outline"
        className="h-6 flex-1 gap-x-1 rounded border-none bg-slate-200 px-2 py-1.5 text-[10px] font-medium text-slate-500 hover:bg-slate-300 hover:text-slate-600"
      >
        Clear Filters
      </Button>
    );
  }

  const formatCell = (data: any, field: string) => {
    let metric = data[field];

    if (typeof metric === 'number' || (typeof metric === 'string' && !isNaN(parseFloat(metric)))) {
      const numericValue = typeof metric === 'number' ? metric : parseFloat(metric);
      if (Number.isInteger(numericValue)) {
        metric = numericValue;
      } else {
        metric = numericValue.toFixed(DEFAULT_NUMERIC_PRECISION);
      }
    }

    return (
      <div className="px-2 py-0 text-center text-[11px] font-medium">
        {typeof metric === 'number' ? metric.toLocaleString() : metric}
      </div>
    );
  };

  function getFilterElement(options: ColumnFilterElementTemplateOptions) {
    if (primaryKeys.includes(options.field) && evalsFlattened) {
      let filterOptions: string[] = [];
      if (options.field === 'modelId') {
        filterOptions = Array.from(new Set(evalsFlattened.map((evalItem) => evalItem.modelId as string))).sort();
      } else if (options.field === 'sourceId') {
        filterOptions = Array.from(new Set(evalsFlattened.map((evalItem) => evalItem.sourceId as string))).sort();
      } else if (options.field === 'layer') {
        filterOptions = Array.from(
          new Set(
            evalsFlattened
              .map((evalItem) => evalItem.layer as string)
              .filter((layer) => layer !== undefined && layer !== '' && layer !== null),
          ),
        ).sort((a, b) => Number(a) - Number(b));
      } else if (options.field === 'trainingTokens') {
        filterOptions = Array.from(
          new Set(
            evalsFlattened
              .map((evalItem) => evalItem.trainingTokens)
              .filter((trainingTokens) => trainingTokens !== undefined),
          ),
        ).sort((a, b) => Number(a) - Number(b));
      } else if (options.field === 'saeClass') {
        filterOptions = Array.from(
          new Set(
            evalsFlattened
              .map((evalItem) => evalItem.saeClass as string)
              .filter((saeClass) => saeClass !== undefined && saeClass !== ''),
          ),
        ).sort();
      } else if (options.field === 'dSae') {
        filterOptions = Array.from(new Set(evalsFlattened.map((evalItem) => evalItem.dSae)))
          .filter((dSae) => dSae !== undefined && dSae !== '')
          .sort((a, b) => Number(a) - Number(b));
      } else if (options.field === 'release') {
        filterOptions = Array.from(new Set(evalsFlattened.map((evalItem) => evalItem.release as string))).sort();
      }
      return (
        <MultiSelect
          showSelectAll={false}
          value={options.value}
          options={filterOptions}
          showClear
          selectedItemTemplate={(item) => (
            <div className="h-4 select-none text-[10px] text-slate-600">
              {item === undefined || item === null || item === '' ? (
                <span className="text-slate-400" key={item}>
                  Any
                </span>
              ) : ['trainingTokens', 'dSae'].includes(options.field) ? (
                item === -1 ? (
                  'N/A'
                ) : (
                  convertNumToAbbr(item)
                )
              ) : item === 'pythia-70m-deduped' ? (
                'pythia-70m-d'
              ) : (
                item
              )}
            </div>
          )}
          selectedItemsLabel="{0} ☑️"
          itemTemplate={(item) => (
            <div className="select-none text-[12.5px] leading-none text-slate-600">
              {options.field === 'trainingTokens' ? (
                <div className="flex w-full flex-row items-start justify-center gap-x-0.5 text-center">
                  {item === 0 ? '0' : item === -1 ? 'N/A' : convertNumToAbbr(item)}
                </div>
              ) : options.field === 'dSae' ? (
                <div className="flex w-full flex-row items-start justify-center gap-x-0.5 text-center">
                  {convertNumToAbbr(item)}
                </div>
              ) : (
                item
              )}
            </div>
          )}
          onChange={(e: MultiSelectChangeEvent) => {
            options.filterApplyCallback(e.value, options.index);
          }}
          filterMatchMode="equals"
          className=""
          placeholder="Any"
          maxSelectedLabels={1}
          style={{
            minWidth:
              options.field === 'modelId'
                ? '115px'
                : options.field === 'release'
                  ? '90px'
                  : options.field === 'saeClass'
                    ? '90px'
                    : undefined,
            maxWidth: options.field === 'modelId' ? '126px' : undefined,
          }}
        />
      );
    }
    return <div />;
  }

  function makeExportStringFromFilters() {
    let exportString = '';
    if (modelIdFilter.length > 0) {
      exportString += `models-${modelIdFilter.join('-')}`;
    }
    if (releaseFilter.length > 0) {
      exportString += `__releases-${releaseFilter.join('-')}`;
    }
    if (saeClassFilter.length > 0) {
      exportString += `__sae-classes-${saeClassFilter.join('-')}`;
    }
    if (dSaeFilter.length > 0) {
      exportString += `__widths-${dSaeFilter.join('-')}`;
    }
    if (trainingTokensFilter.length > 0) {
      exportString += `__training-tokens-${trainingTokensFilter.join('-')}`;
    }
    if (layerFilter.length > 0) {
      exportString += `__layers-${layerFilter.join('-')}`;
    }
    if (sourceIdFilter.length > 0) {
      exportString += `__sources-${sourceIdFilter.join('-')}`;
    }
    return exportString;
  }

  function makeUrlFromFilters(filters: DataTableFilterMeta, embed: boolean) {
    const urlParams = new URLSearchParams();
    if (modelIdFilter.length > 0) {
      urlParams.set('modelId', modelIdFilter.join(','));
    }
    if (releaseFilter.length > 0) {
      urlParams.set('release', releaseFilter.join(','));
    }
    if (saeClassFilter.length > 0) {
      urlParams.set('saeClass', saeClassFilter.join(','));
    }
    if (dSaeFilter.length > 0) {
      urlParams.set('dSae', dSaeFilter.join(','));
    }
    if (trainingTokensFilter.length > 0) {
      urlParams.set('trainingTokens', trainingTokensFilter.join(','));
    }
    if (layerFilter.length > 0) {
      urlParams.set('layer', layerFilter.join(','));
    }
    if (sourceIdFilter.length > 0) {
      urlParams.set('sourceId', sourceIdFilter.join(','));
    }

    // add this evalsPlotRef.current?.metricColumnX?.getName()
    if (evalsPlotRef.current?.metricColumnX) {
      urlParams.set('metricX', evalsPlotRef.current.metricColumnX.getName());
    }
    if (evalsPlotRef.current?.metricColumnY) {
      urlParams.set('metricY', evalsPlotRef.current.metricColumnY.getName());
    }
    if (evalsPlotRef.current?.logX) {
      urlParams.set('logX', evalsPlotRef.current.logX.toString());
    }
    if (evalsPlotRef.current?.logY) {
      urlParams.set('logY', evalsPlotRef.current.logY.toString());
    }
    if (evalsPlotRef.current?.groupBy) {
      urlParams.set('groupBy', evalsPlotRef.current.groupBy);
    }

    const url = new URL(window.location.href);
    url.search = urlParams.toString();
    if (embed) {
      url.searchParams.set('embed', 'true');
    }
    return url;
  }

  useEffect(() => {
    const filterMetasToSet: DataTableFilterMeta = {};
    if (modelIdFilter.length > 0) {
      filterMetasToSet.modelId = {
        value: modelIdFilter,
        matchMode: FilterMatchMode.IN,
      };
    }
    if (sourceIdFilter.length > 0) {
      filterMetasToSet.sourceId = {
        value: sourceIdFilter,
        matchMode: FilterMatchMode.IN,
      };
    }
    if (layerFilter.length > 0) {
      filterMetasToSet.layer = {
        value: layerFilter,
        matchMode: FilterMatchMode.IN,
      };
    }
    if (trainingTokensFilter.length > 0) {
      filterMetasToSet.trainingTokens = {
        value: trainingTokensFilter,
        matchMode: FilterMatchMode.IN,
      };
    }
    if (saeClassFilter.length > 0) {
      filterMetasToSet.saeClass = {
        value: saeClassFilter,
        matchMode: FilterMatchMode.IN,
      };
    }
    if (dSaeFilter.length > 0) {
      filterMetasToSet.dSae = {
        value: dSaeFilter,
        matchMode: FilterMatchMode.IN,
      };
    }
    if (releaseFilter.length > 0) {
      filterMetasToSet.release = {
        value: releaseFilter,
        matchMode: FilterMatchMode.IN,
      };
    }
    evalsPlotRef.current?.setFilterMeta(filterMetasToSet);
    setFilteredRows(getFilteredRows());
  }, [modelIdFilter, sourceIdFilter, layerFilter, trainingTokensFilter, saeClassFilter, dSaeFilter, releaseFilter]);

  function getFilteredRowsForJustThisFilter(filter: string, item: string) {
    let rows = evalsFlattened;
    if (modelIdFilter.length > 0 && filter === 'modelId') {
      rows = rows.filter((evalItem) => evalItem.modelId === item);
    } else if (sourceIdFilter.length > 0 && filter === 'sourceId') {
      rows = rows.filter((evalItem) => evalItem.sourceId === item);
    } else if (layerFilter.length > 0 && filter === 'layer') {
      rows = rows.filter((evalItem) => evalItem.layer === item);
    } else if (saeClassFilter.length > 0 && filter === 'saeClass') {
      rows = rows.filter((evalItem) => evalItem.saeClass === item);
    } else if (dSaeFilter.length > 0 && filter === 'dSae') {
      rows = rows.filter((evalItem) => evalItem.dSae === item);
    } else if (releaseFilter.length > 0 && filter === 'release') {
      rows = rows.filter((evalItem) => evalItem.release === item);
    } else if (trainingTokensFilter.length > 0 && filter === 'trainingTokens') {
      rows = rows.filter((evalItem) => evalItem.trainingTokens === item);
    }
    return rows;
  }

  function makeFilterElement(fieldName: string) {
    const field = fieldName;
    let options = Array.from(new Set(evalsFlattened.map((evalItem) => evalItem[field] as string)));
    if (['dSae', 'layer', 'trainingTokens'].includes(field)) {
      options = options.filter((f) => f !== undefined && f !== '' && f !== null).sort((a, b) => Number(a) - Number(b));
    } else if (field === 'saeClass') {
      options = options.filter((f) => f !== undefined && f !== '' && f !== null).sort();
    } else {
      options = options.sort();
    }

    let filter: string[] | number[] = modelIdFilter;
    if (field === 'sourceId') {
      filter = sourceIdFilter;
    } else if (field === 'layer') {
      filter = layerFilter;
    } else if (field === 'trainingTokens') {
      filter = trainingTokensFilter;
    } else if (field === 'saeClass') {
      filter = saeClassFilter;
    } else if (field === 'dSae') {
      filter = dSaeFilter;
    } else if (field === 'release') {
      filter = releaseFilter;
    }

    return (
      <MultiSelect
        showSelectAll={false}
        value={filter}
        options={options}
        showClear={false}
        selectedItemTemplate={(item) => (
          <div className="h-5 select-none px-2 text-[12px] text-slate-600">
            {item === undefined || item === null || item === '' ? (
              <span className="text-slate-400" key={item}>
                Any
              </span>
            ) : ['trainingTokens', 'dSae'].includes(field) ? (
              item === -1 ? (
                'N/A'
              ) : (
                convertNumToAbbr(item)
              )
            ) : item === 'pythia-70m-deduped' ? (
              'Pythia-70M-D'
            ) : (
              getSaeBenchDisplayString(item)
            )}
          </div>
        )}
        selectedItemsLabel="{0} Selected"
        itemTemplate={(item) => {
          let numberOfSaesThatWouldMatchFilter = 0;
          if (item !== undefined) {
            if (field === 'modelId') {
              numberOfSaesThatWouldMatchFilter = getFilteredRows('modelId').filter(
                (row) => row.modelId === item,
              ).length;
            } else if (field === 'saeClass') {
              numberOfSaesThatWouldMatchFilter = getFilteredRows('saeClass').filter(
                (row) => row.saeClass === item,
              ).length;
            } else if (field === 'layer') {
              numberOfSaesThatWouldMatchFilter = getFilteredRows('layer').filter((row) => row.layer === item).length;
            } else if (field === 'trainingTokens') {
              numberOfSaesThatWouldMatchFilter = getFilteredRows('trainingTokens').filter(
                (row) => row.trainingTokens === item,
              ).length;
            } else if (field === 'dSae') {
              numberOfSaesThatWouldMatchFilter = getFilteredRows('dSae').filter((row) => row.dSae === item).length;
            } else if (field === 'release') {
              numberOfSaesThatWouldMatchFilter = getFilteredRows('release').filter(
                (row) => row.release === item,
              ).length;
            }
          }

          return (
            <div
              className={`select-none text-[12.5px] leading-none ${
                numberOfSaesThatWouldMatchFilter > 0 ? 'text-slate-600' : 'text-slate-400'
              }`}
            >
              {field === 'trainingTokens'
                ? item === 0
                  ? '0'
                  : item === -1
                    ? 'N/A'
                    : convertNumToAbbr(item)
                : field === 'dSae'
                  ? convertNumToAbbr(item)
                  : getSaeBenchDisplayString(item)}
              {` ` +
                ` (${numberOfSaesThatWouldMatchFilter} of ${getFilteredRowsForJustThisFilter(field, item).length})`}
            </div>
          );
        }}
        onChange={(e: MultiSelectChangeEvent) => {
          if (field === 'modelId') {
            setModelIdFilter(e.value);
          } else if (field === 'sourceId') {
            setSourceIdFilter(e.value);
          } else if (field === 'layer') {
            setLayerFilter(e.value);
          } else if (field === 'trainingTokens') {
            setTrainingTokensFilter(e.value);
          } else if (field === 'saeClass') {
            setSaeClassFilter(e.value);
          } else if (field === 'dSae') {
            setDsaeFilter(e.value);
          } else if (field === 'release') {
            setReleaseFilter(e.value);
          }
        }}
        filterMatchMode="equals"
        className=""
        placeholder="Any"
        maxSelectedLabels={1}
        style={{
          minWidth:
            field === 'modelId' ? '115px' : field === 'release' ? '90px' : field === 'saeClass' ? '90px' : undefined,
          maxWidth: field === 'modelId' ? '126px' : undefined,
        }}
      />
    );
  }

  function arraysEqual(a: any[] | undefined, b: any[] | undefined) {
    if (a === undefined || b === undefined) {
      return false;
    }
    return a.length === b.length && a.every((val, i) => val === b[i]);
  }

  function presetIsSelected(preset: { name: string; params: EvalsTableParams }) {
    return (
      arraysEqual(preset.params.modelFilter, modelIdFilter) &&
      arraysEqual(preset.params.widthFilter, dSaeFilter) &&
      arraysEqual(preset.params.layerFilter, layerFilter) &&
      arraysEqual(preset.params.saeClassFilter, saeClassFilter) &&
      arraysEqual(preset.params.releaseFilter, releaseFilter) &&
      arraysEqual(preset.params.trainingTokensFilter, trainingTokensFilter) &&
      preset.params.metricX === evalsPlotRef.current?.metricColumnX?.getName() &&
      preset.params.metricY === evalsPlotRef.current?.metricColumnY?.getName() &&
      preset.params.groupBy === groupBy &&
      preset.params.logX === evalsPlotRef.current?.logX &&
      preset.params.logY === evalsPlotRef.current?.logY
    );
  }

  return (
    <div className="flex h-full w-full flex-col overflow-hidden bg-white">
      {!isEmbed && (
        <div className="w-full items-center justify-center gap-y-2 bg-red-100 px-2 py-2 text-center font-sans text-xs font-medium text-red-600 sm:hidden">
          Warning: This page is best viewed on a larger screen.
        </div>
      )}
      {anonymized && (
        <div className="flex flex-col items-center justify-center gap-y-1 bg-slate-100 px-5 py-0 pb-3 text-[12px] text-slate-500">
          <div className="max-w-screen-md">
            We evaluate a suite of open-source sparse autoencoders on each SAEBench metric. Use the filters to select a
            subset of the SAE Suite. Hover over metric names in the plot title to see a detailed metric description.
            Choose which metric to plot on x and y axis.
          </div>
        </div>
      )}
      {!isEmbed && (
        <div className="flex w-full flex-col items-center justify-center gap-y-1 bg-slate-200 py-3">
          <div className="text-[11px] font-medium text-slate-500">Choose a Preset</div>
          <div className="flex max-w-[320px] flex-row flex-wrap gap-x-2 gap-y-2 sm:max-w-none">
            {EVALS_TABLE_PRESETS.map((preset) => (
              <Button
                key={preset.name}
                size="sm"
                onClick={() => {
                  setPreset(preset.params);
                }}
                variant="outline"
                className={`h-8 flex-1 gap-x-1 rounded-full border border-slate-300 px-3.5 py-2 text-[12px] font-medium ${
                  presetIsSelected(preset)
                    ? 'border-sky-700 bg-sky-200 text-sky-700 hover:bg-sky-200 hover:text-sky-700'
                    : 'border-slate-300 bg-slate-100 text-slate-500 hover:bg-sky-100 hover:text-sky-600'
                }`}
              >
                {preset.name}
              </Button>
            ))}
          </div>
        </div>
      )}

      {!isEmbed && (
        <div className="order-2 mb-5 mt-5 flex w-full flex-col items-center justify-center bg-slate-50 py-3 sm:order-1 sm:mb-0 sm:mt-0">
          {/* <div className=" flex flex-col items-center gap-x-3 pb-2 text-center text-[11px] font-medium text-slate-500 sm:flex-row">
            SAEs to Compare
          </div> */}
          <div className="flex w-full flex-col justify-center gap-x-2.5 gap-y-2 px-2 sm:flex-row sm:items-center">
            <div className="flex flex-col items-center justify-center gap-y-0.5">
              <div className="text-[11px] font-medium text-slate-500">Model</div>
              {makeFilterElement('modelId')}
            </div>
            <div className="flex flex-col items-center justify-center gap-y-0.5">
              <div className="text-[11px] font-medium text-slate-500">Width</div>
              {makeFilterElement('dSae')}
            </div>
            <div className="flex flex-col items-center justify-center gap-y-0.5">
              <div className="text-[11px] font-medium text-slate-500">Layer</div>
              {makeFilterElement('layer')}
            </div>
            <div className="flex flex-col items-center justify-center gap-y-0.5">
              <div className="text-[11px] font-medium text-slate-500">SAE Class</div>
              {makeFilterElement('saeClass')}
            </div>
            <div className="flex flex-col items-center justify-center gap-y-0.5">
              <div className="text-[11px] font-medium text-slate-500">Release</div>
              {makeFilterElement('release')}
            </div>
            <div className="flex flex-col items-center justify-center gap-y-0.5">
              <div className="text-[11px] font-medium text-slate-500">Training Tokens</div>
              {makeFilterElement('trainingTokens')}
            </div>

            <div className="mx-2 flex h-7 flex-col items-center justify-center gap-y-0.5">
              <ClearFiltersButton />
            </div>
          </div>
        </div>
      )}

      <div
        className={`flex ${
          isEmbed
            ? 'relative w-full items-center justify-center'
            : 'w-full items-start justify-center gap-x-5 px-5 pb-0 pt-5 sm:gap-x-7 sm:px-7'
        } order-1 flex-col items-center bg-white sm:order-2 sm:flex-row sm:items-start`}
      >
        <div className={`flex flex-col items-center justify-center gap-x-5 ${isEmbed ? 'w-full px-3' : 'flex-1'}`}>
          <EvalsPlot
            ref={evalsPlotRef}
            evalTypeToOutputSchema={evalTypeToOutputSchema}
            metricColumns={metricColumns}
            setDefaultMetricColumnNamesToDisplay={setMetricColumnNamesToDisplay}
            setGroupByParent={setGroupBy}
            rerenderFilteredRows={rerenderFilteredRows}
            anonymized={anonymized}
          />
        </div>
        {isEmbed && (
          <div className="absolute right-0 top-0 mb-3 flex flex-col items-center justify-end">
            <Button
              size="sm"
              variant="outline"
              className="h-9 flex-1 gap-x-1 rounded border-none bg-slate-100 px-1.5 py-1.5 text-[10px] font-medium text-slate-500 hover:bg-slate-200"
              onClick={() => {
                const dt = dataTableRef.current as unknown as DataTable<DataTableValueArray>;
                const filters = dt.getFilterMeta();
                if (filters) {
                  const url = makeUrlFromFilters(filters, false);
                  window.open(url.toString(), '_blank');
                }
              }}
            >
              <ExternalLinkIcon className="h-3.5 w-3.5 text-slate-500" />
            </Button>
          </div>
        )}
        <DataTable
          ref={dataTableRef}
          scrollable
          removableSort
          virtualScrollerOptions={{ itemSize: 36 }} // this has to match .p-datatable .p-datatable-tbody > tr in the css
          sortField={sortByMetricColumnName}
          onValueChange={(e) => {
            visibleRows.current = e;
            evalsPlotRef.current?.setVisibleRows(e);
          }}
          rowHover
          sortOrder={DEFAULT_SORT_ORDER}
          scrollHeight="350px" // "calc(100vh - 280px)"
          className={`${
            isEmbed ? 'hidden sm:block' : ''
          } mt-5 min-w-[360px] overflow-hidden overscroll-none rounded-md bg-slate-50 sm:mt-0`}
          stripedRows
          filters={{
            modelId: {
              value: defaultEvalsTableParams.modelFilter,
              matchMode: FilterMatchMode.IN,
            },
            layer: {
              value: defaultEvalsTableParams.layerFilter,
              matchMode: FilterMatchMode.IN,
            },
            saeClass: {
              value: defaultEvalsTableParams.saeClassFilter,
              matchMode: FilterMatchMode.IN,
            },
            dSae: {
              value: defaultEvalsTableParams.widthFilter,
              matchMode: FilterMatchMode.IN,
            },
            trainingTokens: {
              value: defaultEvalsTableParams.trainingTokensFilter,
              matchMode: FilterMatchMode.IN,
            },
            release: {
              value: defaultEvalsTableParams.releaseFilter,
              matchMode: FilterMatchMode.IN,
            },
          }}
          size="small"
          value={filteredRows}
          emptyMessage={
            evalTypes.length === 0 || evalsFlattened.length === 0 ? (
              <div className="flex w-full flex-col items-center justify-center px-5 py-5">
                <div className="mb-2 text-sm text-slate-500">Loading...</div>
              </div>
            ) : (
              <div className="flex w-full flex-col items-center justify-center px-5 py-5">
                <div className="mb-2 text-sm font-medium text-red-500">No matching evals found.</div>
                <div className="mb-2 text-xs text-slate-500">
                  Change the &quot;Plot Y&quot; metric or clear the filters.
                </div>
                <ClearFiltersButton />
              </div>
            )
          }
        >
          {primaryKeys
            .filter((key) => {
              if (key === 'sourceId') {
                return true;
              }
              const label = Object.keys(specialMetricColumnToDictKey).find(
                (k) => specialMetricColumnToDictKey[k] === key,
              );
              if (label) {
                const met = new MetricColumn('none', 'none', label, {});
                return metricColumnNamesToDisplay.includes(met.getName());
              }

              return false;
            })
            .map((key) => (
              <Column
                body={(data, columnOptions) => {
                  if (key === 'sourceId') {
                    return (
                      <div className="flex w-full flex-row items-center justify-start gap-x-2">
                        <CustomTooltip wide trigger={<Info className="h-3.5 w-3.5 text-slate-400" />}>
                          <div className="flex flex-col gap-y-1 text-xs">
                            <div>
                              <b>Model:</b> {getSaeBenchDisplayString(data.modelId)}
                            </div>
                            <div>
                              <b>SAE ID:</b> {data.sourceId}
                            </div>
                            <div>
                              <b>Layer:</b> {data.layer}
                            </div>
                            <div>
                              <b>SAE Class:</b> {getSaeBenchDisplayString(data.saeClass)}
                            </div>
                            <div>
                              <b>Width:</b> {convertNumToAbbr(data.dSae)}
                            </div>
                            <div>
                              <b>Release:</b> {getSaeBenchDisplayString(data.release)}
                            </div>
                            <div>
                              <b>Training Tokens:</b>{' '}
                              {data.trainingTokens !== -1 ? convertNumToAbbr(data.trainingTokens) : 'N/A'}
                            </div>
                          </div>
                          <Button
                            size="sm"
                            onClick={() => {
                              window.open(`/${data.modelId}/${data.sourceId}`, '_blank');
                            }}
                            variant="outline"
                            className={`mt-2 h-6 flex-1 gap-x-1 rounded border-none bg-slate-100 px-3 py-2 text-[10px] font-medium text-slate-600 hover:bg-slate-200 hover:text-slate-600 ${
                              anonymized ? 'hidden' : ''
                            }`}
                          >
                            <ExternalLinkIcon className="h-3 w-3" />
                            Open SAE Details
                          </Button>
                        </CustomTooltip>
                        <div className="flex flex-col items-start gap-y-0.5">
                          {groupBy === 'modelId' && (
                            <div className="whitespace-pre px-1.5 py-0.5 text-[11px] font-medium text-slate-700">
                              {getSaeBenchDisplayString(data.modelId)}
                            </div>
                          )}
                          {groupBy === 'layer' && (
                            <div className="whitespace-pre px-1.5 py-0.5 text-[11px] font-medium text-slate-700">
                              Layer {data.layer}
                            </div>
                          )}
                          {groupBy === 'saeClass' && (
                            <div className="whitespace-pre px-1.5 py-0.5 text-[11px] font-medium text-slate-700">
                              {getSaeBenchDisplayString(data.saeClass)}
                            </div>
                          )}
                          {groupBy === 'dSae' && (
                            <div className="whitespace-pre px-1.5 py-0.5 text-[11px] font-medium text-slate-700">
                              {convertNumToAbbr(data.dSae)} Width
                            </div>
                          )}
                          {groupBy === 'release' && (
                            <div className="whitespace-pre px-1.5 py-0.5 text-[11px] font-medium text-slate-700">
                              {getSaeBenchDisplayString(data.release)}
                            </div>
                          )}
                          {groupBy === 'trainingTokens' && (
                            <div className="whitespace-pre px-1.5 py-0.5 text-[11px] font-medium text-slate-700">
                              {data.trainingTokens === -1
                                ? 'N/A Tokens'
                                : `${convertNumToAbbr(data.trainingTokens)} Tokens`}
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  }
                  if (key === 'layer') {
                    return (
                      <div className="px-0 py-0 text-center text-[11px] uppercase">{data[columnOptions.field]}</div>
                    );
                  }
                  if (key === 'modelId') {
                    return (
                      <div className="px-0 py-0 text-center text-[11px]">
                        <a
                          href={`/${data.modelId}`}
                          target="_blank"
                          rel="noreferrer"
                          className="whitespace-pre text-sky-800 hover:underline"
                        >
                          {data[columnOptions.field] === 'pythia-70m-deduped'
                            ? 'pythia-70m-d'
                            : data[columnOptions.field]}
                        </a>
                      </div>
                    );
                  }
                  if (key === 'trainingTokens') {
                    return (
                      <div className="px-0 py-0 text-center text-[11px] uppercase">
                        {data[columnOptions.field] === 0
                          ? '0'
                          : data[columnOptions.field] === -1
                            ? 'N/A'
                            : data[columnOptions.field]
                              ? convertNumToAbbr(data[columnOptions.field])
                              : '-'}
                      </div>
                    );
                  }
                  if (key === 'saeClass') {
                    return <div className="px-0 py-0 text-center text-[11px]">{data[columnOptions.field]}</div>;
                  }
                  if (key === 'dSae') {
                    return (
                      <div className="px-0 py-0 text-center text-[11px]">
                        {convertNumToAbbr(data[columnOptions.field])}
                      </div>
                    );
                  }
                  if (key === 'release') {
                    return <div className="px-0 py-0 text-center text-[11px]">{data[columnOptions.field]}</div>;
                  }
                  return '';
                }}
                key={key}
                field={key}
                filter={false}
                frozen
                sortable={key !== 'sourceId'}
                headerClassName="text-slate-600 px-0"
                showFilterMenu={false}
                showClearButton
                filterMenuClassName="rounded-md text-slate-600 text-xs"
                filterMatchModeOptions={
                  key === 'layer'
                    ? [
                        {
                          label: 'Equals',
                          value: FilterMatchMode.EQUALS,
                        },
                      ]
                    : key === 'trainingTokens'
                      ? [
                          {
                            label: 'Equals',
                            value: FilterMatchMode.EQUALS,
                          },
                        ]
                      : key === 'saeClass'
                        ? [
                            {
                              label: 'Equals',
                              value: FilterMatchMode.EQUALS,
                            },
                          ]
                        : key === 'dSae'
                          ? [
                              {
                                label: 'Equals',
                                value: FilterMatchMode.EQUALS,
                              },
                            ]
                          : key === 'release'
                            ? [
                                {
                                  label: 'Equals',
                                  value: FilterMatchMode.EQUALS,
                                },
                              ]
                            : undefined
                }
                filterElement={key === 'sourceId' ? undefined : getFilterElement}
                header={() => {
                  if (key === 'modelId') {
                    return (
                      <div className="flex flex-col px-0">
                        <div className="flex select-none flex-col items-center justify-center text-[11px] font-semibold">
                          Model
                        </div>
                      </div>
                    );
                  }
                  if (key === 'release') {
                    return (
                      <div className="flex flex-col px-0">
                        <div className="flex select-none flex-col items-center justify-center text-[11px] font-semibold">
                          Release
                        </div>
                      </div>
                    );
                  }
                  if (key === 'sourceId') {
                    return (
                      <div className="flex flex-col px-0">
                        <div className="flex select-none flex-col items-center justify-center text-[11px] font-semibold">
                          {/* SAE */}
                        </div>
                      </div>
                    );
                  }
                  if (key === 'saeClass') {
                    return (
                      <div className="flex flex-col">
                        <div className="flex h-[45px] select-none flex-col items-center justify-center text-[11px] font-semibold">
                          SAE Class
                        </div>
                      </div>
                    );
                  }
                  if (key === 'dSae' || key === 'trainingTokens' || key === 'layer') {
                    // find the key in specialMetricColumnToDictKey that has the value "trainingTokens"
                    const specialKey =
                      Object.keys(specialMetricColumnToDictKey).find(
                        (keyToIterate) => specialMetricColumnToDictKey[keyToIterate] === key,
                      ) || '';
                    return (
                      <div className="flex flex-col px-0">
                        <div className="flex h-[45px] select-none flex-col items-center justify-center text-[11px] font-semibold">
                          {specialKey}
                        </div>
                      </div>
                    );
                  }
                  return '';
                }}
              />
            ))}
          {metricColumns
            .filter((metricColumn) => metricColumnNamesToDisplay.includes(metricColumn.getName()))
            .map((metricColumn) => (
              <Column
                sortable
                headerClassName=" px-0 text-slate-600"
                header={getHeaderForMetricColumn(metricColumn)}
                key={metricColumn.getName()}
                field={metricColumn.getName()}
                body={(data, columnOptions) => formatCell(data, columnOptions.field)}
              />
            ))}
        </DataTable>
      </div>

      <div
        className={`order-3 flex-row items-center justify-center gap-x-2 px-5 pb-5 pt-0 text-xs ${
          isEmbed || anonymized ? 'hidden' : 'flex'
        }`}
      >
        <Button
          size="sm"
          variant="outline"
          className="w-[120px] gap-x-1.5 bg-slate-100 text-[11px] text-slate-600 hover:bg-slate-200"
          onClick={() => {
            const dt = dataTableRef.current as unknown as DataTable<DataTableValueArray>;
            const filters = dt.getFilterMeta();
            if (filters) {
              const url = makeUrlFromFilters(filters, false);
              copy(url.toString());
              alert("Copied this page's link to clipboard.");
            }
          }}
        >
          <NotebookText className="h-4 w-4 text-slate-500" /> Share Page
        </Button>
        <Button
          size="sm"
          variant="outline"
          className="w-[120px] gap-x-1.5 bg-slate-100 text-[11px] text-slate-600 hover:bg-slate-200"
          onClick={() => {
            const dt = dataTableRef.current as unknown as DataTable<DataTableValueArray>;
            const filters = dt.getFilterMeta();
            if (filters) {
              const url = makeUrlFromFilters(filters, true);
              copy(url.toString());
              alert('Copied an embed of only the table to clipboard.');
            }
          }}
        >
          <ChartScatter className="h-4 w-4 text-slate-500" /> Share Graph
        </Button>
        <Button
          size="sm"
          variant="outline"
          className="w-[120px] gap-x-1.5 bg-slate-100 text-[11px] text-slate-600 hover:bg-slate-200"
          onClick={() => {
            const dt = dataTableRef.current as unknown as DataTable<DataTableValueArray>;
            const filters = dt.getFilterMeta();
            if (filters) {
              const jsonString = JSON.stringify(visibleRows?.current, null, 2);
              const blob = new Blob([jsonString], { type: 'application/json' });
              const url = URL.createObjectURL(blob);
              const link = document.createElement('a');
              link.href = url;
              link.download = `saebench_${makeExportStringFromFilters()}.json`;
              document.body.appendChild(link);
              link.click();
              document.body.removeChild(link);
              URL.revokeObjectURL(url);
            }
          }}
        >
          <DownloadCloud className="h-4 w-4 text-slate-500" /> Export JSON
        </Button>
      </div>
    </div>
  );
}
