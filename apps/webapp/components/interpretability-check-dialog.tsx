'use client';

import CustomTooltip from '@/components/custom-tooltip';
import type { InterpretabilityCheckResult, IntruderGroupResult } from '@/lib/external/interpretability-jev';
import * as Checkbox from '@radix-ui/react-checkbox';
import { Check, Info } from 'lucide-react';
import { useState } from 'react';
import ActivationItem from './activation-item';
import { Dialog, DialogContent, DialogTitle } from './shadcn/dialog';
import { LoadingSquare } from './svg/loading-square';

// Result of the on-demand "Check If Interpretable" run. Same layout as the explanation score
// detail dialog, but there is no explanation: the check is about the feature itself.

const KIND_LABEL: Record<string, string> = {
  top: 'Top Activation',
  zero: 'Zero Activation',
  decoy: 'Decoy',
};

function scoreColor(score: number): string {
  if (score >= 80) return 'bg-emerald-500';
  if (score >= 60) return 'bg-amber-500';
  return 'bg-red-500';
}

function GroupRows({ group, showPromptText }: { group: IntruderGroupResult; showPromptText: boolean }) {
  return (
    <>
      {group.examples.map((example, i) => {
        const isPicked = i === group.picked_index;
        const max = Math.max(...example.values);
        return (
          <tr
            key={i}
            className={
              example.is_intruder && isPicked ? 'bg-emerald-50' : example.is_intruder || isPicked ? 'bg-red-50' : ''
            }
          >
            <td className="whitespace-pre px-2 text-center font-mono text-[10px] text-slate-400">{i}</td>
            <td className="px-2 py-1.5 font-mono text-[10px]">
              {showPromptText ? (
                <div className="whitespace-pre-wrap text-[10.5px] leading-snug text-slate-700">
                  {example.marked_text}
                </div>
              ) : (
                <ActivationItem
                  tokensToDisplayAroundMaxActToken={64}
                  activation={{
                    tokens: example.tokens,
                    values: example.values,
                    maxValueTokenIndex: example.values.indexOf(max),
                  }}
                  overallMaxActivationValueInList={max}
                  overrideTextSize="text-[10.5px]"
                />
              )}
            </td>
            <td className="whitespace-pre px-2 text-center">{KIND_LABEL[example.kind] || example.kind}</td>
            <td className="whitespace-pre px-2 text-center">{example.is_intruder ? 'Intruder' : ''}</td>
            <td className="whitespace-pre px-2 text-center">
              {(group.probabilities[i] * 100).toFixed(0)}%{isPicked ? ' · Picked' : ''}
            </td>
          </tr>
        );
      })}
    </>
  );
}

export default function InterpretabilityCheckDialog({
  open,
  onOpenChange,
  loading,
  error,
  result,
  featureLabel,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  loading: boolean;
  error: string | null;
  result: InterpretabilityCheckResult | null;
  featureLabel: string;
}) {
  const [showPromptText, setShowPromptText] = useState(false);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="fixed left-[50%] top-[50%] z-50 flex max-h-[92vh] w-[95vw] max-w-[95vw] translate-x-[-50%] translate-y-[-50%] flex-col items-center justify-center rounded-md bg-slate-50 shadow-xl transition-all focus:outline-none sm:top-[50%] sm:w-[90vw] md:rounded-md md:border md:border-slate-200">
        {loading || (!result && !error) ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-y-4 py-5">
            <DialogTitle className="text-center">
              <div className="text-sm font-medium text-slate-400">Performing Intruder Detection Check...</div>
            </DialogTitle>
            <LoadingSquare size={32} className="text-sky-700" />
          </div>
        ) : error ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-y-2 px-6 py-8">
            <DialogTitle className="text-xs font-bold uppercase text-slate-400">Check Failed</DialogTitle>
            <div className="text-center text-sm text-red-600">{error}</div>
          </div>
        ) : (
          result && (
            <div className="z-50 flex w-full flex-col gap-y-0 overflow-scroll px-0 text-xs">
              <div className="top-0 flex w-full flex-col items-start justify-center border-b bg-slate-50 pt-4 sm:sticky">
                <div className="relative mb-4 flex w-full flex-col gap-x-5 gap-y-4 px-2 text-left text-sm text-slate-600 sm:flex-row sm:px-6 sm:text-base">
                  <div className="flex flex-col items-start sm:basis-2/6">
                    <DialogTitle className="mb-0 flex flex-row items-center gap-x-1 text-[10px] font-medium uppercase text-slate-400">
                      Interpretability
                      <CustomTooltip trigger={<Info className="h-3 w-3" />}>
                        How many of the {result.total} groups the model answered correctly, as a percentage. Chance is{' '}
                        {Math.round(100 / (result.groups[0]?.examples.length || 5))}%.
                      </CustomTooltip>
                    </DialogTitle>
                    <div className={`rounded-xl p-3 text-lg font-bold text-white ${scoreColor(result.score)}`}>
                      {result.score}
                    </div>
                    <div className="mt-2 text-xs text-slate-600">
                      {result.correct} of {result.total} intruders found for {featureLabel}
                    </div>
                  </div>
                  <div className="flex flex-col sm:basis-1/3">
                    <div className="mb-0 flex flex-row items-center gap-x-1 text-[10px] font-medium uppercase text-slate-400">
                      What This Checks
                    </div>
                    <div className="text-xs leading-normal text-slate-600">
                      Intruder detection, with no explanation involved. Each group has four texts where this feature
                      fires, with the firing tokens marked {'<< >>'}, and one intruder where it does not, with random
                      tokens marked. The model picks the odd one out. A feature that is interpretable from its contexts
                      alone should score well here even if its explanations score poorly. Based on{' '}
                      <a
                        href="https://arxiv.org/pdf/2507.08473"
                        target="_blank"
                        rel="noreferrer"
                        className="text-sky-700 underline hover:text-sky-900"
                      >
                        Paulo &amp; Belrose (2025)
                      </a>
                      .
                    </div>
                  </div>
                  <div className="flex flex-col sm:basis-2/6">
                    <div className="mb-0 flex flex-row items-center gap-x-1 text-[10px] font-medium uppercase text-slate-400">
                      Model
                    </div>
                    <div className="font-mono text-sm">{result.model}</div>
                    <label
                      className="mt-3 flex cursor-pointer flex-row items-center gap-x-1.5 text-xs leading-none text-slate-600"
                      htmlFor="interpretable-show-prompt"
                    >
                      <Checkbox.Root
                        id="interpretable-show-prompt"
                        className="flex h-4 w-4 appearance-none items-center justify-center rounded-[3px] border border-slate-300 bg-white outline-none"
                        checked={showPromptText}
                        onCheckedChange={(checked) => setShowPromptText(checked === true)}
                      >
                        <Checkbox.Indicator className="text-sky-700">
                          <Check className="h-4 w-4" />
                        </Checkbox.Indicator>
                      </Checkbox.Root>
                      Show text as sent to the model
                    </label>
                  </div>
                </div>
              </div>

              <div className="flex w-full flex-col divide-y divide-slate-200 px-5 pb-10 text-slate-600">
                {result.groups.map((group, g) => (
                  <div key={g} className="flex flex-col gap-y-1 py-4">
                    <div className="flex flex-row flex-wrap items-center gap-x-3 text-[10px] font-medium uppercase text-slate-400">
                      <span>Group {g + 1}</span>
                      <span>Intruder at {group.intruder_index}</span>
                      <span>
                        Picked {group.picked_index} ({(group.probabilities[group.picked_index] * 100).toFixed(0)}%)
                      </span>
                      <span className={group.correct ? 'font-bold text-emerald-600' : 'font-bold text-red-600'}>
                        {group.correct ? 'Correct' : 'Incorrect'}
                      </span>
                    </div>
                    <table>
                      <thead>
                        <tr>
                          <th className="whitespace-pre px-2 py-1.5 text-[10px] text-slate-400">#</th>
                          <th className="gap-x-1 whitespace-pre px-2 py-1.5 text-left">Text</th>
                          <th className="gap-x-1 whitespace-pre px-2 py-1.5">
                            Set{' '}
                            <CustomTooltip trigger={<Info className="h-3 w-3" />}>
                              Top activations fire this feature. The intruder is a zero-activation text for this
                              feature, or a decoy when none is stored.
                            </CustomTooltip>
                          </th>
                          <th className="gap-x-1 whitespace-pre px-2 py-1.5">Truth</th>
                          <th className="gap-x-1 whitespace-pre px-2 py-1.5">
                            Probability{' '}
                            <CustomTooltip trigger={<Info className="h-3 w-3" />}>
                              {`The model's`} probability that this text is the intruder. The five sum to 100%.
                            </CustomTooltip>
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        <GroupRows group={group} showPromptText={showPromptText} />
                      </tbody>
                    </table>
                  </div>
                ))}
              </div>
            </div>
          )
        )}
      </DialogContent>
    </Dialog>
  );
}
