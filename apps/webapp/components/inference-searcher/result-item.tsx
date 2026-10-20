import ActivationItem from '@/components/activation-item';
import FeatureStats from '@/components/feature-stats';
import { useGlobalContext } from '@/components/provider/global-provider';
import { InferenceActivationAllResult } from '@/components/provider/inference-activation-all-provider';
import useNeuronDashboard from '@/lib/hooks/use-neuron-dashboard';
import { replaceHtmlAnomalies } from '@/lib/utils/activations';
import { signedLogHistogram } from '@/lib/utils/neuron-dashboard-histogram';
import {
  applyNeuronDashboard,
  emptyNeuron,
  NEURON_DASHBOARD_BOTTOM_DATA_SOURCE,
  neuronDashboardExtremes,
} from '@/lib/utils/neuron-dashboards';
import { getSourceSetNameFromSource } from '@/lib/utils/source';
import { ActivationPartialWithRelations, ExplanationPartialWithRelations } from '@/prisma/generated/zod';
import { useMemo } from 'react';

function valueColor(value: number) {
  return value < 0 ? 'text-red-700' : 'text-emerald-700';
}

function MiniActivation({
  label,
  activation,
  valueClassName,
  dfa,
}: {
  label: string;
  activation: ActivationPartialWithRelations;
  valueClassName: string;
  dfa: boolean | undefined;
}) {
  const peakIndex = activation.maxValueTokenIndex || 0;
  return (
    <>
      <div className="mb-0.5 mt-3 text-left text-[12px] font-bold uppercase text-slate-600">{label}</div>
      <div className="mt-1 flex flex-row gap-x-2">
        {activation.tokens && activation.values && (
          <div className="flex flex-col items-center gap-y-0.5 px-2 text-xs text-slate-700">
            <span className="whitespace-pre rounded bg-slate-200 px-1 font-mono font-medium">
              {replaceHtmlAnomalies(activation.tokens[peakIndex])}
            </span>
            <span className={`mt-0 text-[11px] ${valueClassName}`}>{activation.values[peakIndex].toFixed(1)}</span>
          </div>
        )}
        <ActivationItem
          showLineBreaks={false}
          activation={{
            values: activation.values,
            tokens: activation.tokens,
            maxValueTokenIndex: activation.maxValueTokenIndex,
            maxValue: activation.maxValue,
            // A bottom row centers on, and colors by, its most negative value.
            minValue: activation.minValue,
            dataSource: activation.dataSource,
            dfaValues: activation.dfaValues,
            dfaMaxValue: activation.dfaMaxValue,
            dfaTargetIndex: activation.dfaTargetIndex,
          }}
          enableExpanding={false}
          tokensToDisplayAroundMaxActToken={10}
          overrideLeading="leading-none"
          overrideTextSize="text-xs"
          dfa={dfa}
        />
      </div>
    </>
  );
}

export default function ResultItem({
  result,
  tokens,
  topExplanation,
  searchSortIndexes,
  showDashboards,
}: {
  result: InferenceActivationAllResult;
  tokens: string[];
  topExplanation: ExplanationPartialWithRelations | undefined;
  searchSortIndexes: number[];
  showDashboards: boolean;
}) {
  const { getSourceSet } = useGlobalContext();
  const sourceSetName = getSourceSetNameFromSource(result.layer);
  const externalDashboardsUrl = getSourceSet(result.modelId, sourceSetName)?.externalDashboardsUrl;
  // Loads after the results show, and only while dashboards are shown.
  const externalDashboard = useNeuronDashboard(
    showDashboards ? externalDashboardsUrl : null,
    result.modelId,
    result.layer,
    result.index,
  );
  const shownNeuron = useMemo(
    () =>
      externalDashboard.status === 'ready'
        ? applyNeuronDashboard(
            result.neuron ?? emptyNeuron(result.modelId, result.layer, result.index, sourceSetName),
            externalDashboard.dashboard,
          )
        : result.neuron,
    [externalDashboard, result, sourceSetName],
  );
  const logHistogram = useMemo(
    () => (externalDashboard.status === 'ready' ? signedLogHistogram(externalDashboard.dashboard.record) : undefined),
    [externalDashboard],
  );
  const extremes = useMemo(
    () =>
      externalDashboard.status === 'ready'
        ? neuronDashboardExtremes(externalDashboard.dashboard, result)
        : { top: result.neuron?.activations?.[0], bottom: undefined },
    [externalDashboard, result],
  );
  const dfa = getSourceSet(result.modelId, sourceSetName)?.showDfa;
  const lowestIndex = useMemo(
    () => result.values.reduce((lowest, value, i) => (value < result.values[lowest] ? i : lowest), 0),
    [result.values],
  );
  // A negative match shows its lowest token, not its highest.
  const shownIndex = result.rankedByLowest ? lowestIndex : result.maxValueIndex;
  const sortSum = searchSortIndexes.reduce((sum, index) => sum + result.values[index], 0);

  return (
    <a
      href={`/${result.modelId}/${result.layer}/${result.index}`}
      target="_blank"
      rel="noreferrer"
      key={result.index}
      className="group flex w-full cursor-pointer flex-col items-center justify-center border-b border-slate-100 bg-white px-3 py-5 sm:px-5"
    >
      <div className="flex w-full max-w-screen-lg flex-col items-start justify-center gap-x-2 sm:flex-row sm:gap-x-5">
        <div className="flex basis-4/12 flex-col items-center justify-start font-mono text-[11px] font-medium text-slate-500 sm:basis-3/12">
          <span
            className={`mb-1 mt-0 flex flex-col font-sans leading-tight ${
              topExplanation ? 'text-[12px] text-slate-700 sm:text-sm' : 'text-[12px] text-slate-500 sm:text-xs'
            }`}
          >
            {topExplanation ? (
              topExplanation.description
            ) : (
              <div className="flex flex-col items-center justify-center gap-y-0.5 text-center">
                <div>No Explanation</div>
              </div>
            )}
          </span>
          <span className="mb-2.5 rounded-md px-2 py-1 text-[10px] font-bold uppercase text-slate-600 group-hover:bg-sky-200 sm:mb-1 sm:text-xs">
            {result.layer}:{result.index}
          </span>
          {result.rankedByLowest && (
            <span
              className="rounded bg-red-50 px-1.5 py-0.5 font-sans text-[10px] font-bold uppercase text-red-600"
              title="This neuron matched by how rarely it goes this negative, not by a high activation."
            >
              Negative Match
            </span>
          )}
        </div>
        <div className="flex basis-8/12 flex-col sm:basis-9/12">
          <div className={`mt-0 flex ${searchSortIndexes.length > 0 ? 'flex-col' : 'flex-row items-center'} gap-x-2`}>
            {searchSortIndexes.length > 0 ? (
              <div className="mb-1.5 flex max-w-screen-md flex-row gap-x-1 overflow-x-scroll">
                {searchSortIndexes.length > 1 && (
                  <div className="sticky left-0 flex flex-col items-center justify-center gap-y-0.5 bg-white pr-1">
                    <div className="px-1 text-[10px] font-bold uppercase text-slate-400">Sum</div>
                    <span className={`mt-0 text-[11px] ${valueColor(sortSum)}`}>{sortSum.toFixed(1)}</span>
                  </div>
                )}
                {searchSortIndexes.map((searchSortIndex) => (
                  <div key={searchSortIndex} className="flex flex-col items-center justify-center gap-y-0.5">
                    <div className="whitespace-pre rounded bg-slate-200 px-1 font-mono text-xs font-medium">
                      {replaceHtmlAnomalies(tokens[searchSortIndex])}
                    </div>
                    <span className={`mt-0 text-[11px] ${valueColor(result.values[searchSortIndex])}`}>
                      {result.values[searchSortIndex].toFixed(1)}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex flex-col items-center gap-y-0.5 px-2 text-xs text-slate-700">
                <div className="whitespace-pre rounded bg-slate-200 px-1 font-mono font-medium">
                  {replaceHtmlAnomalies(tokens[shownIndex])}
                </div>
                <span className={`mt-0 text-[11px] ${valueColor(result.values[shownIndex])}`}>
                  {result.values[shownIndex].toFixed(1)}
                </span>
              </div>
            )}
            <ActivationItem
              showLineBreaks={false}
              activation={{
                values: result.values,
                tokens,
                maxValueTokenIndex: result.maxValueIndex,
                maxValue: result.maxValue,
                ...(result.rankedByLowest && {
                  minValue: result.values[lowestIndex],
                  dataSource: NEURON_DASHBOARD_BOTTOM_DATA_SOURCE,
                }),
                dfaValues: result.dfaValues,
                dfaMaxValue: result.dfaMaxValue,
                dfaTargetIndex: result.dfaTargetIndex,
              }}
              enableExpanding={false}
              overrideLeading="leading-none"
              overrideTextSize="text-xs"
              dfa={dfa}
            />
          </div>

          {showDashboards && externalDashboard.status === 'loading' && (
            <div className="mt-3 flex h-16 items-center justify-center rounded-md border border-slate-200 text-xs font-medium text-slate-400">
              Loading dashboard...
            </div>
          )}
          {showDashboards && externalDashboard.status === 'error' && (
            <div className="mt-3 rounded-md border border-slate-200 px-3 py-2 text-xs text-red-600">
              Could not load this dashboard.
            </div>
          )}
          {showDashboards && externalDashboard.status !== 'loading' && shownNeuron && (
            <div className="mt-3 flex flex-col rounded-md border border-slate-200 px-3 pb-3">
              {extremes.top && (
                <MiniActivation
                  label="Top Activation"
                  activation={extremes.top}
                  valueClassName="text-emerald-700"
                  dfa={dfa}
                />
              )}
              {extremes.bottom && (
                <MiniActivation
                  label="Bottom Activation"
                  activation={extremes.bottom}
                  valueClassName="text-red-700"
                  dfa={dfa}
                />
              )}

              {shownNeuron.pos_str && shownNeuron.pos_str.length > 0 && (
                <div className="pointer-events-none mb-0 mt-2 pt-2">
                  {/* Records have more logits than fit here, and the row is a link, so "Show More" cannot be clicked */}
                  <FeatureStats
                    currentNeuron={shownNeuron}
                    signedLogHistogram={logHistogram}
                    forceMiniStats={externalDashboard.status === 'ready'}
                  />
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </a>
  );
}
