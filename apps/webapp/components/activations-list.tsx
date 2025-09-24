'use client';

import ActivationSingleForm from '@/components/activation-single-form';
import { ACTIVATION_DISPLAY_DEFAULT_CONTEXT_TOKENS } from '@/lib/utils/activations';
import * as ToggleGroup from '@radix-ui/react-toggle-group';
import * as Tooltip from '@radix-ui/react-tooltip';
import { HelpCircle } from 'lucide-react';
import { Activation, ActivationPartialWithRelations, NeuronWithPartialRelations } from 'prisma/generated/zod';
import { useEffect, useState } from 'react';
import 'react-loading-skeleton/dist/skeleton.css';
import ActivationItem from './activation-item';
import { useGlobalContext } from './provider/global-provider';

export default function ActivationsList({
  feature,
  activations,
  defaultRange = 1,
  defaultShowLineBreaks = true,
  showControls = true,
  showDatasource = true,
  showTopActivationToken = true,
  showTest = false,
  showCopy = false,
  defaultActivationTestText,
  activationTestTextCallback,
  embed = false,
  overrideLeading,
  overrideTextSize,
  activationItemClassName,
  // by default we show the actual tokens instead of hiding special tokens (eg for chat / reasoning)
  defaultShowRawTokens = true,
}: {
  feature?: NeuronWithPartialRelations;
  activations: ActivationPartialWithRelations[] | void;
  defaultRange?: number;
  defaultShowLineBreaks?: boolean;
  showControls?: boolean;
  showDatasource?: boolean;
  showTopActivationToken?: boolean;
  showTest?: boolean;
  showCopy?: boolean;
  defaultActivationTestText?: string;
  activationTestTextCallback?: (activation?: Activation) => void;
  embed?: boolean;
  overrideLeading?: string | undefined;
  overrideTextSize?: string | undefined;
  activationItemClassName?: string;
  defaultShowRawTokens?: boolean;
}) {
  const { getSourceSet, user } = useGlobalContext();
  // used for showing correct snippet
  const [selectedRange, setSelectedRange] = useState(defaultRange);
  const [showLineBreaks, setShowLineBreaks] = useState(defaultShowLineBreaks);
  const [showRawTokens, setShowRawTokens] = useState(defaultShowRawTokens);
  const [dfaSplit, setDfaSplit] = useState(true);
  // used to determine colors
  const [overallMaxValue, setOverallMaxValue] = useState<number>(-100);
  const [items, setItems] = useState<ActivationPartialWithRelations[]>([]);
  const [activationTestText, setActivationTestText] = useState('');
  const [showHidden, setShowHidden] = useState(false);

  function getMaxValueOfActivations(activationsToCheck: ActivationPartialWithRelations[]) {
    let tempOverallMaxValue = -100;
    activationsToCheck.forEach((activation) => {
      if (activation.maxValue && activation.maxValue > tempOverallMaxValue) {
        tempOverallMaxValue = activation.maxValue;
      }
    });
    return tempOverallMaxValue;
  }

  // set highest index of activations, sort by max activation
  // this useEffect fixes actiavtions where they are in the wrong "bin" due to some bug in SAEDashboard
  useEffect(() => {
    if (activations) {
      setOverallMaxValue(getMaxValueOfActivations(activations));
      let sortedActs = activations;
      if (getSourceSet(feature?.modelId || '', feature?.sourceSetName || '')?.showDfa) {
        // remove all where the dfaTargetIndex !== maxActIndex
        if (!showHidden) {
          sortedActs = activations.filter((act) => act.dfaTargetIndex === act.maxValueTokenIndex);
          // remove duplicate sortedacts where act.tokens are the same
          const uniqueActsMap = new Map<string, ActivationPartialWithRelations>();
          sortedActs.forEach((act) => {
            if (act.tokens) {
              const key = JSON.stringify(act.tokens.toSorted());
              if (!uniqueActsMap.has(key) && act.maxValue !== undefined && act.maxValue > 0) {
                uniqueActsMap.set(key, act);
              }
            }
          });
          sortedActs = Array.from(uniqueActsMap.values());
        }
        // correct all incorrect quantiles
        // find all quantiles
        const quantilesSet = new Set<number>();
        sortedActs.forEach((act) => {
          if (act.binMax !== undefined && act.binMax !== null) {
            quantilesSet.add(act.binMax);
          }
        });
        if (quantilesSet.size > 1) {
          const quantiles = new Array<number>(...quantilesSet).sort((a, b) => a - b);
          sortedActs.forEach((act, index) => {
            if (act.maxValue !== undefined) {
              if (act.maxValue >= (act.binMax || 0) || act.binMin === -1) {
                // find the correct quantile
                const correctQuantile = quantiles.find((q) => q >= (act.maxValue || 0));
                if (correctQuantile !== undefined) {
                  const correctQuantileIndex = quantiles.indexOf(correctQuantile);
                  if (correctQuantileIndex > 0) {
                    sortedActs[index].binMin = quantiles[correctQuantileIndex - 1];
                    sortedActs[index].binMax = correctQuantile;
                  }
                }
              }
            }
          });
        }
        sortedActs = sortedActs.toSorted((a, b) => {
          if (
            a.values === undefined ||
            b.values === undefined ||
            a.dfaTargetIndex === undefined ||
            a.dfaTargetIndex === null ||
            b.dfaTargetIndex === undefined ||
            b.dfaTargetIndex === null
          ) {
            return 0;
          }
          if (a.binMax && a.maxValue !== 0 && !b.binMax) {
            return 1;
          }
          if (!a.binMax && b.binMax && b.maxValue !== 0) {
            return -1;
          }
          // sort by activation of the targetDFA token
          const targetDFAactA = a.values[a.dfaTargetIndex];
          const targetDFAactB = b.values[b.dfaTargetIndex];

          if (targetDFAactA > targetDFAactB) {
            return -1;
          }
          if (targetDFAactA < targetDFAactB) {
            return 1;
          }
          return 0;
        });
      } else {
        if (!showHidden) {
          // remove duplicate sortedacts where act.tokens are the same
          const uniqueActsMap = new Map<string, ActivationPartialWithRelations>();
          activations.forEach((act) => {
            if (act.tokens) {
              const key = JSON.stringify(act.tokens.toSorted());
              if (!uniqueActsMap.has(key) && act.maxValue !== undefined && act.maxValue > 0) {
                uniqueActsMap.set(key, act);
              }
            }
          });
          // eslint-disable-next-line no-param-reassign
          activations = Array.from(uniqueActsMap.values());
        }
        sortedActs = activations;
        // correct all incorrect quantiles
        // find all quantiles
        const quantilesSet = new Set<number>();
        sortedActs.forEach((act) => {
          if (act.binMax !== undefined && act.binMax !== null) {
            quantilesSet.add(act.binMax);
          }
        });
        if (quantilesSet.size > 1) {
          const quantiles = new Array<number>(...quantilesSet).sort((a, b) => a - b);
          sortedActs.forEach((act, index) => {
            if (act.maxValue !== undefined) {
              if (act.maxValue >= (act.binMax || 0)) {
                // find the correct quantile
                const correctQuantile = quantiles.find((q) => q >= (act.maxValue || 0));
                if (correctQuantile !== undefined) {
                  const correctQuantileIndex = quantiles.indexOf(correctQuantile);
                  if (correctQuantileIndex > 0) {
                    sortedActs[index].binMin = quantiles[correctQuantileIndex - 1];
                    sortedActs[index].binMax = correctQuantile;
                  }
                }
              }
            }
          });
        }
        sortedActs = sortedActs.toSorted((a, b) => {
          if (a.maxValue !== undefined && b.maxValue !== undefined) {
            if (a.binMax && b.binMax) {
              if (a.binMax === b.binMax) {
                return b.maxValue - a.maxValue;
              }
              return b.binMax - a.binMax;
            }
            return b.maxValue - a.maxValue;
          }
          return 0;
        });
      }
      setItems(sortedActs);
    }
  }, [activations, user, showHidden]);

  useEffect(() => {
    setSelectedRange(defaultRange);
  }, [defaultRange]);

  useEffect(() => {
    setShowLineBreaks(defaultShowLineBreaks);
  }, [defaultShowLineBreaks]);

  useEffect(() => {
    setActivationTestText(defaultActivationTestText || '');
  }, [defaultActivationTestText]);

  return (
    <>
      {feature && showTest && (
        <div className="bg-white pt-1 sm:pt-0">
          <ActivationSingleForm
            key={feature.modelId + feature.layer + feature.index}
            neuron={feature}
            hideBos
            overallMaxValue={overallMaxValue}
            formValue={activationTestText}
            callback={activationTestTextCallback}
            embed={embed}
          />
        </div>
      )}
      {feature && feature.activations && feature?.activations.length > 0 && (
        <div
          className={`${
            showControls ? 'flex' : 'hidden'
          } flex-row gap-x-2 overflow-hidden border-b bg-white px-2 pb-1 sm:px-3 sm:pb-2`}
        >
          <ToggleGroup.Root
            className="mb-1 ml-auto mt-1 inline-flex flex-1 overflow-hidden rounded bg-slate-100 px-0 py-0 sm:rounded-md"
            type="single"
            defaultValue={ACTIVATION_DISPLAY_DEFAULT_CONTEXT_TOKENS[selectedRange].size.toString()}
            value={ACTIVATION_DISPLAY_DEFAULT_CONTEXT_TOKENS[selectedRange].size.toString()}
            onValueChange={(value) => {
              if (value) {
                ACTIVATION_DISPLAY_DEFAULT_CONTEXT_TOKENS.forEach((r, rIndex) => {
                  if (r.size.toString() === value) {
                    setSelectedRange(rIndex);
                  }
                });
              }
            }}
            aria-label="Range of text to display before and after the highest token"
          >
            {ACTIVATION_DISPLAY_DEFAULT_CONTEXT_TOKENS.map((range) => (
              <ToggleGroup.Item
                key={`range-${range.size}`}
                className="flex-1 items-center rounded px-0 py-1 text-[10px] font-medium text-slate-400 transition-all hover:bg-slate-100 data-[state=on]:bg-slate-200 data-[state=on]:text-slate-600 sm:rounded-md sm:py-1.5 sm:text-[11px]"
                value={range.size.toString()}
                aria-label={range.text.toString()}
              >
                {range.text}
              </ToggleGroup.Item>
            ))}
          </ToggleGroup.Root>
          {getSourceSet(feature?.modelId || '', feature?.sourceSetName || '')?.showDfa && (
            <ToggleGroup.Root
              className="mb-1 ml-auto mt-1 inline-flex flex-1 overflow-hidden rounded bg-slate-100 px-0 py-0 sm:rounded-md"
              type="single"
              defaultValue={dfaSplit ? 'split' : 'notSplit'}
              value={dfaSplit ? 'split' : 'notSplit'}
              onValueChange={(value) => {
                setDfaSplit(value === 'split');
              }}
              aria-label="Range of text to display before and after the highest token"
            >
              <ToggleGroup.Item
                key="split"
                className="flex-1 items-center rounded px-1 py-1 text-[10px] font-medium text-slate-400 transition-all hover:bg-slate-100 data-[state=on]:bg-slate-200 data-[state=on]:text-slate-600 sm:rounded-md sm:px-4 sm:py-1.5 sm:text-[11px]"
                value="split"
                aria-label="split"
              >
                Split DFA
              </ToggleGroup.Item>
              <ToggleGroup.Item
                key="notSplit"
                className="flex-1 items-center rounded-md px-1 py-1 text-[10px] font-medium text-slate-400 transition-all hover:bg-slate-100 data-[state=on]:bg-slate-200 data-[state=on]:text-slate-600 sm:px-4 sm:py-1.5 sm:text-[11px]"
                value="notSplit"
                aria-label="notSplit"
              >
                Combine DFA
              </ToggleGroup.Item>
            </ToggleGroup.Root>
          )}
          <ToggleGroup.Root
            className="mb-1 ml-auto mt-1 inline-flex flex-1 overflow-hidden rounded bg-slate-100 px-0 py-0 sm:rounded-md"
            type="single"
            defaultValue={showRawTokens ? 'showRawTokens' : 'hideRawTokens'}
            value={showRawTokens ? 'showRawTokens' : 'hideRawTokens'}
            onValueChange={(value) => {
              setShowRawTokens(value === 'showRawTokens');
            }}
            aria-label="Show raw or formatted tokens"
          >
            <ToggleGroup.Item
              key="showRawTokens"
              className="flex-1 items-center rounded px-1 py-1 text-[10px] font-medium text-slate-400 transition-all hover:bg-slate-100 data-[state=on]:bg-slate-200 data-[state=on]:text-slate-600 sm:rounded-md sm:px-4 sm:py-1.5 sm:text-[11px]"
              value="showRawTokens"
              aria-label="showRawTokens"
            >
              Show Raw Tokens
            </ToggleGroup.Item>

            <ToggleGroup.Item
              key="hideRawTokens"
              className="flex-1 items-center rounded-md px-1 py-1 text-[10px] font-medium text-slate-400 transition-all hover:bg-slate-100 data-[state=on]:bg-slate-200 data-[state=on]:text-slate-600 sm:px-4 sm:py-1.5 sm:text-[11px]"
              value="hideRawTokens"
              aria-label="hideRawTokens"
            >
              Show Formatted
            </ToggleGroup.Item>
          </ToggleGroup.Root>
          <ToggleGroup.Root
            className="mb-1 ml-auto mt-1 inline-flex flex-1 overflow-hidden rounded bg-slate-100 px-0 py-0 sm:rounded-md"
            type="single"
            defaultValue={showLineBreaks ? 'showLineBreaks' : 'hideLineBreaks'}
            value={showLineBreaks ? 'showLineBreaks' : 'hideLineBreaks'}
            onValueChange={(value) => {
              setShowLineBreaks(value === 'showLineBreaks');
            }}
            aria-label="Range of text to display before and after the highest token"
          >
            <ToggleGroup.Item
              key="showLineBreaks"
              className="flex-1 items-center rounded px-1 py-1 text-[10px] font-medium text-slate-400 transition-all hover:bg-slate-100 data-[state=on]:bg-slate-200 data-[state=on]:text-slate-600 sm:rounded-md sm:px-4 sm:py-1.5 sm:text-[11px]"
              value="showLineBreaks"
              aria-label="showLineBreaks"
            >
              Show Breaks
            </ToggleGroup.Item>

            <ToggleGroup.Item
              key="hideLineBreaks"
              className="flex-1 items-center rounded-md px-1 py-1 text-[10px] font-medium text-slate-400 transition-all hover:bg-slate-100 data-[state=on]:bg-slate-200 data-[state=on]:text-slate-600 sm:px-4 sm:py-1.5 sm:text-[11px]"
              value="hideLineBreaks"
              aria-label="hideLineBreaks"
            >
              Hide Breaks
            </ToggleGroup.Item>
          </ToggleGroup.Root>
        </div>
      )}
      {!activations || activations.length === 0 || items.length === 0 ? (
        <div className="flex h-48 w-full flex-col items-center justify-center">
          <h1 className="text-lg font-bold text-slate-300">No Known Activations</h1>
        </div>
      ) : (
        <>
          {items.map((activation, activationIndex) => (
            <div
              key={`activation-${activation.id}`}
              id={activationIndex === 0 ? 'firstActivation' : undefined}
              className={`relative border-slate-100 px-3 py-1 sm:px-5 [&:not(:last-child)]:border-b ${
                selectedRange > 0 && 'sm:py-2.5'
              }`}
            >
              {(activation.binMin === -1 &&
                activationIndex > 0 &&
                items[activationIndex - 1].binMax !== activation.binMax) ||
              activationIndex === 0 ? (
                <div className="absolute left-0 top-0 hidden h-6 w-full flex-row items-center justify-between bg-slate-100 px-3 font-mono text-[8px] font-bold uppercase text-slate-600 sm:flex">
                  <div className="flex flex-row items-center gap-x-1 leading-normal">
                    <div className="py-[1px]">Top</div>{' '}
                    {activation?.dfaTargetIndex !== null && activation?.dfaTargetIndex !== -1 && (
                      <>
                        <div className="rounded-sm bg-orange-400 px-1 py-[1px]">DFA</div>
                        <div className="py-[1px]">and</div>
                      </>
                    )}
                    <div className="rounded-sm bg-emerald-400 px-1 py-[1px]">Activations</div>
                  </div>
                  {activation?.dfaTargetIndex !== null && activation?.dfaTargetIndex !== -1 && (
                    <div className="flex flex-row gap-x-1.5">
                      <Tooltip.Provider delayDuration={0} skipDelayDuration={0}>
                        <Tooltip.Root>
                          <Tooltip.Trigger asChild>
                            <button type="button">
                              <HelpCircle className="h-3 w-3" />
                            </button>
                          </Tooltip.Trigger>
                          <Tooltip.Portal>
                            <Tooltip.Content
                              className="rounded bg-slate-500 px-3 py-2 text-xs text-white"
                              sideOffset={5}
                            >
                              DFA is Direct Feature Attribution:
                              <br />
                              The top source position that the attention layer uses to compute this feature.
                              <Tooltip.Arrow className="fill-slate-500" />
                            </Tooltip.Content>
                          </Tooltip.Portal>
                        </Tooltip.Root>
                      </Tooltip.Provider>
                      <div className="border-[2px] border-emerald-400 px-1 py-[1px]">DFA Source</div>
                      <div className="border-[2px] border-orange-400 px-1 py-[1px]">DFA Target</div>
                    </div>
                  )}
                </div>
              ) : (
                activation.binMax &&
                activationIndex > 0 &&
                items[activationIndex - 1].binMax !== activation.binMax && (
                  <div className="absolute left-0 top-0 hidden h-6 w-full flex-row items-center justify-start bg-slate-100 px-3 font-mono text-[8px] font-bold uppercase text-slate-600 sm:flex">
                    Interval {activation.binMin?.toFixed(3)} - {activation.binMax?.toFixed(3)} (Contains{' '}
                    {((activation.binContains || 0) * 100).toFixed(3)}%)
                  </div>
                )
              )}
              <div
                className={`flex w-full flex-row items-center justify-center ${
                  (activation.binMin === -1 &&
                    activationIndex > 0 &&
                    items[activationIndex - 1].binMax !== activation.binMax) ||
                  activationIndex === 0 ||
                  (activation.binMax && activationIndex > 0 && items[activationIndex - 1].binMax !== activation.binMax)
                    ? 'sm:mt-6'
                    : ''
                }`}
              >
                <div className="flex w-full flex-auto flex-col text-left text-sm">
                  {activation.tokens && (
                    <ActivationItem
                      key={`${ACTIVATION_DISPLAY_DEFAULT_CONTEXT_TOKENS[selectedRange].size}-${activationIndex}${showLineBreaks}`}
                      activation={activation}
                      tokensToDisplayAroundMaxActToken={ACTIVATION_DISPLAY_DEFAULT_CONTEXT_TOKENS[selectedRange].size}
                      showLineBreaks={showLineBreaks}
                      setActivationTestText={setActivationTestText}
                      showTopActivationToken={showTopActivationToken}
                      dfa={getSourceSet(feature?.modelId || '', feature?.sourceSetName || '')?.showDfa}
                      dfaSplit={dfaSplit}
                      showCopy={showCopy}
                      overallMaxActivationValueInList={overallMaxValue}
                      overrideLeading={overrideLeading}
                      overrideTextSize={overrideTextSize || 'text-[9.5px] sm:text-xs'}
                      className={activationItemClassName}
                      showRawTokens={showRawTokens}
                    />
                  )}
                  {showDatasource && activation.dataSource && (
                    <div className="flex w-full flex-row items-center justify-end pt-0.5 text-right text-[8px] font-medium leading-none text-slate-400 sm:pt-0">
                      {activation.dataSource}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
          {!showHidden && items.length < activations.length && (
            <div className="flex w-full flex-row items-center justify-center py-3">
              <button
                type="button"
                onClick={() => {
                  setShowHidden(true);
                }}
                className="rounded bg-slate-200 px-3.5 py-2 text-center text-xs font-medium text-slate-500 hover:bg-slate-300"
              >
                Show {activations.length - items.length} Duplicates
              </button>
            </div>
          )}
        </>
      )}
    </>
  );
}
