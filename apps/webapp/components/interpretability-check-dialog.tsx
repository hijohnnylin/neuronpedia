'use client';

import CustomTooltip from '@/components/custom-tooltip';
import type { InterpretabilityCheckResult, IntruderGroupResult } from '@/lib/external/interpretability-jev';
import { Info } from 'lucide-react';
import { ExplanationPartialWithRelations } from 'prisma/generated/zod';
import ActivationItem from './activation-item';
import { Dialog, DialogContent, DialogTitle } from './shadcn/dialog';
import { LoadingSquare } from './svg/loading-square';

// Result of the on-demand "Check If Interpretable" run. Same layout as the explanation score
// detail dialog, but there is no explanation: the check is about the feature itself.

const KIND_LABEL: Record<string, string> = {
  top: 'Top Activation',
  zero: 'Intruder (Zero Activation)',
  decoy: 'Intruder',
};

function scoreColor(score: number): string {
  if (score >= 80) return 'bg-emerald-500';
  if (score >= 60) return 'bg-amber-500';
  return 'bg-red-500';
}

function GroupRows({ group }: { group: IntruderGroupResult }) {
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
            </td>
            <td className="whitespace-pre px-2 text-center">{KIND_LABEL[example.kind] || example.kind}</td>
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
  explanations,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  loading: boolean;
  error: string | null;
  result: InterpretabilityCheckResult | null;
  featureLabel: string;
  explanations: ExplanationPartialWithRelations[];
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        aria-describedby={undefined}
        className="fixed left-[50%] top-[50%] z-50 flex max-h-[92vh] w-[95vw] max-w-[95vw] translate-x-[-50%] translate-y-[-50%] flex-col items-center justify-center rounded-md bg-slate-50 shadow-xl transition-all focus:outline-none sm:top-[50%] sm:w-[90vw] md:rounded-md md:border md:border-slate-200"
      >
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
                <div className="mb-4 flex w-full flex-col items-start gap-y-1 px-2 pr-10 sm:flex-row sm:items-center sm:justify-between sm:gap-x-4 sm:px-6 sm:pr-12">
                  <DialogTitle className="flex flex-row items-center gap-x-1.5 text-sm font-medium text-slate-700">
                    Interpretable Score (Intrusion Detection)
                    <CustomTooltip
                      trigger={
                        <span className="ml-1 cursor-pointer rounded border border-sky-600 bg-white px-1.5 py-1 text-[9px] font-bold uppercase tracking-wide text-sky-700 transition-colors hover:bg-sky-600 hover:text-white">
                          How It Works
                        </span>
                      }
                      side="bottom"
                      wide
                    >
                      <p className="leading-relaxed">
                        We score how interpretable a latent is by seeing if a judge model is able to distinguish the{' '}
                        {`latent's`} true top activations from fake activations (
                        <a
                          href="https://arxiv.org/pdf/2507.08473"
                          target="_blank"
                          rel="noreferrer"
                          className="text-sky-700 underline hover:text-sky-900"
                        >
                          Paulo &amp; Belrose, 2025
                        </a>
                        ). We show the judge model groups of five activation samples. In each group, there are four true
                        top activations, and one {'"intruder"'} (a zero or made up activation). The judge model is asked
                        to identify the intruder. The overall score is the percentage of times that the judge correctly
                        picked the intruder.
                      </p>
                    </CustomTooltip>
                  </DialogTitle>
                  <div className="truncate text-xs font-medium text-slate-500 sm:text-right">{featureLabel}</div>
                </div>
                <div className="relative mb-4 flex w-full flex-col gap-x-5 gap-y-4 px-2 text-left text-sm text-slate-600 sm:flex-row sm:px-6 sm:text-base">
                  <div className="flex flex-col items-start sm:basis-2/6">
                    <div className="mb-0 text-[10px] font-medium uppercase text-slate-400">Score</div>
                    <div className="flex flex-row items-center gap-x-3">
                      <div className={`rounded-xl p-3 text-lg font-bold text-white ${scoreColor(result.score)}`}>
                        {result.score}
                      </div>
                      <div className="text-xs text-slate-600">
                        {result.correct} of {result.total} Intruders Detected
                      </div>
                    </div>
                  </div>
                  <div className="flex flex-col sm:basis-1/3">
                    <div className="mb-0 flex flex-row items-center gap-x-1 text-[10px] font-medium uppercase text-slate-400">
                      Current Explanations
                    </div>
                    {explanations.length === 0 ? (
                      <div className="text-xs text-slate-400">No explanations yet.</div>
                    ) : (
                      <div className="flex max-h-36 flex-col gap-y-1.5 overflow-y-auto pr-1">
                        {explanations.map((explanation) => (
                          <div key={explanation.id} className="flex flex-col">
                            <div className="text-xs leading-snug text-slate-700">{explanation.description}</div>
                            <div className="font-sans text-[10px] font-medium text-slate-400">
                              {explanation.typeName
                                ? `${explanation.typeName} · ${explanation.explanationModelName}`
                                : 'human'}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                  <div className="flex flex-col sm:basis-2/6">
                    <div className="mb-0 text-[10px] font-medium uppercase text-slate-400">Judge Model</div>
                    <div className="font-mono text-xs text-slate-600">{result.model}</div>
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
                            Truth{' '}
                            <CustomTooltip trigger={<Info className="h-3 w-3" />}>
                              Top activations fire this feature. The intruder is a zero-activation text for this
                              feature, or a made up one when none is stored.
                            </CustomTooltip>
                          </th>
                          <th className="gap-x-1 whitespace-pre px-2 py-1.5">
                            {`Model's`} Pick{' '}
                            <CustomTooltip trigger={<Info className="h-3 w-3" />}>
                              {`The judge model's`} probability that this text is the intruder. The five sum to 100%.
                            </CustomTooltip>
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        <GroupRows group={group} />
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
