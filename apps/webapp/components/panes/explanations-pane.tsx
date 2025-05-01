import CustomTooltip from '@/components/custom-tooltip';
import { useExplanationScoreDetailContext } from '@/components/provider/explanation-score-detail-provider';
import { useGlobalContext } from '@/components/provider/global-provider';
import { Button } from '@/components/shadcn/button';
import {
  ERROR_NO_AUTOINTERP_KEY,
  ERROR_RECALL_ALT_FAILED,
  ERROR_REQUIRES_OPENROUTER,
  EXPLANATIONTYPE_HUMAN,
} from '@/lib/utils/autointerp';
import {
  ExplanationScoreWithPartialRelations,
  ExplanationWithPartialRelations,
  NeuronWithPartialRelations,
} from '@/prisma/generated/zod';
import {
  Explanation,
  ExplanationModelType,
  ExplanationScore,
  ExplanationScoreModel,
  ExplanationScoreType,
  ExplanationType,
} from '@prisma/client';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import { ChevronDownIcon, HelpCircle, Info, X } from 'lucide-react';
import Link from 'next/link';
import router from 'next/router';
import { useEffect, useState } from 'react';
import { LoadingSpinner } from '../svg/loading-spinner';

export default function ExplanationsPane({
  currentNeuron,
  setCurrentNeuron,
}: {
  currentNeuron: NeuronWithPartialRelations | undefined;
  setCurrentNeuron: (neuron: NeuronWithPartialRelations) => void;
}) {
  const {
    getSourceSet,
    explanationTypes,
    explanationModels,
    explanationScoreTypes,
    explanationScoreModelTypes,
    setSignInModalOpen,
    showToastServerError,
    showToastMessage,
    user,
  } = useGlobalContext();
  const [explanationTypesFiltered, setExplanationTypesFiltered] = useState<ExplanationType[]>([]);
  const [isAutoInterping, setIsAutoInterping] = useState(false);
  const [isScoringId, setIsScoringId] = useState<string | undefined>();
  const { loadExplanationScoreDetail } = useExplanationScoreDetailContext();
  const [selectedExplanationType, setSelectedExplanationType] = useState<ExplanationType | undefined>(undefined);
  const [selectedExplanationModel, setSelectedExplanationModel] = useState<ExplanationModelType | undefined>(
    explanationModels.length > 0 ? explanationModels[0] : undefined,
  );
  const [selectedExplanationScoreType, setSelectedExplanationScoreType] = useState<ExplanationScoreType | undefined>(
    undefined,
  );
  const [selectedExplanationScoreModelType, setSelectedExplanationScoreModelType] = useState<
    ExplanationScoreModel | undefined
  >(explanationScoreModelTypes.length > 0 ? explanationScoreModelTypes[0] : undefined);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [isExplaining, setIsExplaining] = useState(false);

  useEffect(() => {
    if (explanationTypesFiltered.length > 0) {
      setSelectedExplanationType(explanationTypesFiltered[0]);
    }
  }, [explanationTypesFiltered]);

  useEffect(() => {
    if (explanationScoreTypes.length > 0) {
      setSelectedExplanationScoreType(explanationScoreTypes.filter((type) => type.featured).reverse()[0]);
    }
  }, [explanationScoreTypes]);

  useEffect(() => {
    const filtered = explanationTypes
      .filter((expType) => {
        const sourceSet = getSourceSet(currentNeuron?.modelId || '', currentNeuron?.sourceSetName || '');
        if (sourceSet && sourceSet.showDfa) {
          return expType.isAttention;
        }
        return !expType.isAttention;
      })
      .sort((a, b) => (a.createdAt?.getTime() ?? 0) - (b.createdAt?.getTime() ?? 0));
    setExplanationTypesFiltered(filtered);
  }, [explanationTypes, currentNeuron, getSourceSet]);

  return (
    <div className="relative mb-0 flex flex-col gap-x-2 overflow-hidden border bg-white px-0 pb-0 pt-2 text-xs transition-all sm:rounded-lg sm:shadow">
      <div className="mb-0 flex w-full flex-row items-center justify-center gap-x-1 text-[10px] font-normal uppercase text-slate-400">
        Explanations
        <CustomTooltip trigger={<HelpCircle className="h-2.5 w-2.5" />}>
          <div className="flex flex-col gap-2">
            <div className="flex flex-col text-xs font-medium text-slate-500">
              <div className="mt-0">
                &bull; <strong>Auto-Interp explanations</strong> are created by showing an AI model (like GPT4o-mini or
                Claude 3.5 Sonnet) the top activating texts, then asking it to guess what this feature is about. Various
                prompting methods are used, including{' '}
                <a
                  target="_blank"
                  rel="noreferrer"
                  href="https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html#sec-algorithm-explain"
                  className="font-bold text-sky-700 hover:underline"
                >
                  token activation pairs
                </a>
                .
              </div>
              <div className="mt-2">
                &bull; <strong>Scoring of explanations</strong> is done with various methods, including{' '}
                <a
                  target="_blank"
                  rel="noreferrer"
                  href="https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html#sec-algorithm-simulate"
                  className="font-bold text-sky-700 hover:underline"
                >
                  simulation
                </a>
                .
              </div>
              <div className="mt-2">&bull; Click the explanation&#39;s score to see scoring details.</div>
            </div>
          </div>
        </CustomTooltip>
      </div>

      {!currentNeuron || !currentNeuron.explanations || currentNeuron.explanations.length === 0 ? (
        <div className="flex h-10 flex-col items-center justify-center text-xs font-bold text-slate-300 sm:h-16 sm:text-sm">
          <div>No Explanations Found</div>
        </div>
      ) : (
        <div className="divide-y-slate-100 my-0 flex flex-col divide-y">
          {currentNeuron?.explanations?.map((explanation) => (
            <div key={explanation.id} className="transition-all">
              <div className="py-2">
                <div className="relative flex w-full flex-row items-center gap-3 px-2 sm:px-3">
                  {/* <div className="flex basis-1/6 flex-col items-center justify-center">
                                <VoteWidget
                                  explanation={explanation}
                                  mini={true}
                                />
                              </div> */}
                  <div className="h-fill flex-1">
                    <div className="flex-inline relative flex items-center">
                      <h1
                        className="cursor-text select-text py-0 text-xs font-medium leading-snug text-slate-700"
                        title={explanation.description}
                      >
                        {explanation.description}
                      </h1>
                    </div>
                    <div className="mt-1.5 flex flex-row items-center justify-between font-mono text-xs text-slate-400">
                      {(() => {
                        // CASE: autointerped explanation
                        if (explanation.typeName && explanation.typeName !== EXPLANATIONTYPE_HUMAN) {
                          return (
                            <div className="flex flex-col items-start justify-center gap-y-0.5">
                              <a
                                className="font-sans text-[11px] font-medium text-slate-500 hover:text-sky-600"
                                href={`/explanation-type/${explanation.typeName}`}
                                target="_blank"
                                rel="noreferrer"
                              >
                                {explanation.typeName} · {explanation.explanationModelName}
                              </a>
                              {explanation.triggeredByUser && (
                                <a
                                  className="cursor-pointer font-sans text-[10px] font-medium leading-none text-slate-300 hover:text-sky-600"
                                  href={`/user/${explanation.triggeredByUser?.name}`}
                                  target="_blank"
                                  rel="noreferrer"
                                >
                                  Triggered by{' @'}
                                  {explanation.triggeredByUser.name}
                                </a>
                              )}
                            </div>
                          );
                        }
                        // CASE: human explanation and we have the user info
                        if (explanation.triggeredByUser?.name) {
                          return (
                            <button
                              type="button"
                              className="hidden cursor-pointer font-sans text-[11px] font-medium text-slate-400/70 hover:text-sky-600 sm:inline-block"
                              onClick={(e) => {
                                e.preventDefault();
                                e.stopPropagation();
                                router.push(`/user/${explanation.author?.name}`);
                              }}
                            >
                              {explanation.triggeredByUser && (
                                <a
                                  className="cursor-pointer font-sans text-[10px] font-medium leading-none text-slate-300 hover:text-sky-600"
                                  href={`/user/${explanation.triggeredByUser?.name}`}
                                  target="_blank"
                                  rel="noreferrer"
                                >
                                  Triggered by{' @'}
                                  {explanation.triggeredByUser.name}
                                </a>
                              )}
                            </button>
                          );
                        }
                        return '';
                      })()}
                      {/* we don't have DFA scoring yet */}
                      {!currentNeuron?.sourceSet?.showDfa && (
                        <DropdownMenu.Root>
                          <DropdownMenu.Trigger asChild>
                            <Button
                              variant="outline"
                              size="xs"
                              className="flex flex-row items-center justify-center gap-x-1 whitespace-pre rounded px-2 font-medium text-slate-500 sm:pl-2 sm:pr-1.5"
                            >
                              <div className="flex select-none flex-col items-center justify-center gap-y-1 py-1 font-sans text-[10px] leading-none">
                                {explanation.scores && explanation.scores.length > 0 ? (
                                  <>
                                    Avg Score:{' '}
                                    {(
                                      (explanation.scores.reduce((sum, score) => sum + (score.value || 0), 0) /
                                        explanation.scores.length) *
                                      100
                                    ).toFixed(0)}
                                  </>
                                ) : (
                                  <>No Scores</>
                                )}
                              </div>
                              <ChevronDownIcon className="-mr-1 ml-0 h-4 w-4 leading-none text-slate-400" />
                            </Button>
                          </DropdownMenu.Trigger>
                          <DropdownMenu.Portal>
                            <DropdownMenu.Content
                              className="z-30 max-h-[900px] overflow-hidden rounded-md border border-slate-200 bg-white text-xs font-medium text-slate-500 shadow-lg"
                              sideOffset={2}
                              align="center"
                            >
                              <div className="flex h-full flex-col items-stretch">
                                {explanation.scores && explanation.scores.length > 0 && (
                                  <div className="flex flex-col pb-0.5">
                                    <div className="flex items-center justify-center pt-2 text-center text-[10px] uppercase text-slate-400">
                                      Current Scores
                                    </div>
                                    <div className="flex w-full flex-col gap-x-3 gap-y-1 divide-y divide-slate-100 pb-2 pt-1">
                                      {explanation.scores
                                        ?.sort((a, b) =>
                                          (a.explanationScoreTypeName || '').localeCompare(
                                            b.explanationScoreTypeName || '',
                                          ),
                                        )
                                        .map((score) => (
                                          <div
                                            key={score.id}
                                            className="flex w-full flex-row items-center justify-center gap-x-2.5 px-3 pt-1 text-center text-slate-500"
                                          >
                                            <Button
                                              onClick={(e) => {
                                                e.preventDefault();
                                                e.stopPropagation();
                                                loadExplanationScoreDetail(
                                                  explanation as ExplanationWithPartialRelations,
                                                  score as ExplanationScoreWithPartialRelations,
                                                );
                                              }}
                                              className="h-auto whitespace-pre rounded-md bg-emerald-500 p-1.5 px-2 text-[12px] font-bold text-white hover:bg-emerald-600"
                                              size="sm"
                                            >
                                              {((score.value || 0) * 100).toFixed(0)}
                                            </Button>
                                            <div className="flex flex-1 flex-row items-center justify-start gap-x-0.5 whitespace-pre text-left text-[11px]">
                                              {score.explanationScoreTypeName} · {score.explanationScoreModelName}
                                              <button
                                                type="button"
                                                onClick={(e) => {
                                                  e.preventDefault();
                                                  e.stopPropagation();
                                                  window.open(
                                                    `/score-type/${score.explanationScoreTypeName}`,
                                                    '_blank',
                                                    'noopener,noreferrer',
                                                  );
                                                }}
                                                className="rounded p-1 hover:bg-sky-300 hover:text-sky-700"
                                              >
                                                <Info className="h-3 w-3 text-slate-400 hover:text-sky-700" />
                                              </button>
                                            </div>
                                            <div className="flex flex-row items-center justify-center gap-x-3">
                                              <Button
                                                onClick={(e) => {
                                                  e.preventDefault();
                                                  e.stopPropagation();
                                                  loadExplanationScoreDetail(
                                                    explanation as ExplanationWithPartialRelations,
                                                    score as ExplanationScoreWithPartialRelations,
                                                  );
                                                }}
                                                variant="outline"
                                                className="h-7 font-sans text-[11px] text-slate-500"
                                                size="sm"
                                              >
                                                Details
                                              </Button>
                                            </div>

                                            {score.initiatedByUserId === user?.id && user && (
                                              <button
                                                type="button"
                                                onClick={async (e) => {
                                                  e.preventDefault();
                                                  if (window.confirm('Delete this score?')) {
                                                    const result = await fetch(
                                                      `/api/explanation/score/${score.id}/delete`,
                                                      {
                                                        method: 'POST',
                                                        headers: {
                                                          'Content-Type': 'application/json',
                                                        },
                                                      },
                                                    );
                                                    if (result.status !== 200) {
                                                      showToastServerError();
                                                    } else {
                                                      const newExplanationScores = explanation.scores?.filter(
                                                        (curExpScore) => curExpScore.id !== score.id,
                                                      );
                                                      const newExplanations = currentNeuron?.explanations?.map(
                                                        (curExp) => {
                                                          if (curExp.id === explanation.id) {
                                                            return {
                                                              ...curExp,
                                                              scores: newExplanationScores,
                                                            };
                                                          }
                                                          return curExp;
                                                        },
                                                      );
                                                      const newCurrentNeuron = {
                                                        ...currentNeuron,
                                                      };
                                                      newCurrentNeuron.explanations = newExplanations;
                                                      setCurrentNeuron(newCurrentNeuron);
                                                    }
                                                  }
                                                }}
                                                className="rounded bg-red-400 p-0.5 hover:bg-red-600"
                                              >
                                                <X className="h-2.5 w-2.5 text-[5px] font-bold uppercase text-white" />
                                              </button>
                                            )}
                                          </div>
                                        ))}
                                    </div>
                                  </div>
                                )}

                                <div className="flex w-full min-w-[500px] flex-col items-center justify-center gap-x-1 border-t border-slate-200 px-3 pb-5 pt-3 text-slate-500">
                                  <div className="mb-1.5 text-[10px] font-normal uppercase text-slate-400">
                                    Generate New Score
                                  </div>
                                  <div className="flex w-full flex-row gap-x-2">
                                    <div className="flex min-w-[calc(50%-35px)] max-w-[calc(50%-35px)] flex-1 flex-col items-center">
                                      <DropdownMenu.Root>
                                        <DropdownMenu.Trigger asChild>
                                          <button
                                            type="button"
                                            className="flex w-full flex-1 flex-row items-center justify-center gap-x-1 whitespace-pre rounded border border-slate-300 bg-white px-2 text-[11px] font-medium text-slate-500 hover:bg-slate-50 focus:outline-none sm:pl-3 sm:pr-2"
                                          >
                                            <div className="flex flex-col items-center justify-center gap-y-1 py-2 leading-none">
                                              <div className="flex flex-row items-center justify-center gap-x-0.5">
                                                {selectedExplanationScoreType?.displayName || 'Scorer Type'}
                                              </div>
                                            </div>
                                            <ChevronDownIcon className="-mr-1 ml-0 w-4 leading-none text-slate-400" />
                                          </button>
                                        </DropdownMenu.Trigger>
                                        <DropdownMenu.Portal>
                                          <DropdownMenu.Content
                                            className="z-30 max-h-[900px] cursor-pointer overflow-hidden rounded-md border border-slate-300 bg-white text-xs font-medium text-slate-500 shadow"
                                            sideOffset={2}
                                            align="center"
                                          >
                                            <div className="bg-slate-200 py-1.5 text-center text-[10px] uppercase text-slate-500">
                                              Scorer Types
                                            </div>
                                            {explanationScoreTypes
                                              .filter((type) => type.featured)
                                              .map((s) => (
                                                <DropdownMenu.Item
                                                  key={s.name}
                                                  className={`overflow-hidden ${
                                                    selectedExplanationScoreType?.name === s.name
                                                      ? 'bg-sky-100 text-slate-600'
                                                      : 'text-slate-600'
                                                  } flex flex-col items-start gap-y-0.5 border-b px-3 py-2.5 font-mono text-xs hover:bg-slate-100 focus:outline-none`}
                                                  onSelect={() => setSelectedExplanationScoreType(s)}
                                                >
                                                  <div className="flex w-full flex-row items-center justify-between gap-x-3">
                                                    <span className="">{s.displayName}</span>
                                                    <button
                                                      type="button"
                                                      onClick={(e) => {
                                                        e.preventDefault();
                                                        e.stopPropagation();
                                                        window.open(
                                                          `/score-type/${s.name}`,
                                                          '_blank',
                                                          'noopener,noreferrer',
                                                        );
                                                      }}
                                                      className="rounded p-1 transition-all hover:bg-sky-300 hover:text-sky-700"
                                                    >
                                                      <Info className="h-3.5 w-3.5" />
                                                    </button>
                                                  </div>
                                                </DropdownMenu.Item>
                                              ))}
                                          </DropdownMenu.Content>
                                        </DropdownMenu.Portal>
                                      </DropdownMenu.Root>
                                    </div>
                                    <div
                                      className={`${
                                        selectedExplanationScoreType?.name === 'eleuther_embedding' ? 'hidden' : 'flex'
                                      } min-w-[calc(50%-35px)] max-w-[calc(50%-35px)] flex-1 flex-row items-center justify-start gap-x-2 overflow-hidden text-ellipsis whitespace-pre`}
                                    >
                                      <DropdownMenu.Root>
                                        <DropdownMenu.Trigger asChild>
                                          <button
                                            type="button"
                                            className="relative flex h-7 w-full flex-row items-center justify-center gap-x-1 overflow-hidden text-ellipsis whitespace-pre rounded border border-slate-300 bg-white px-2 text-[11px] font-medium text-slate-500 hover:bg-slate-50 focus:outline-none sm:pl-3 sm:pr-2"
                                          >
                                            {selectedExplanationScoreModelType?.displayName || 'Scorer Model'}
                                            <ChevronDownIcon className="absolute right-0 bg-white px-1 leading-none text-slate-400" />
                                          </button>
                                        </DropdownMenu.Trigger>
                                        <DropdownMenu.Portal>
                                          <DropdownMenu.Content
                                            className="z-30 max-h-[420px] cursor-pointer overflow-scroll rounded-md border border-slate-300 bg-white text-xs font-medium text-slate-600 shadow"
                                            sideOffset={2}
                                            align="center"
                                          >
                                            <div className="bg-slate-200 py-1.5 text-center text-[10px] uppercase text-slate-500">
                                              Scorer Models
                                            </div>
                                            {explanationScoreModelTypes
                                              .filter((type) => type.featured)
                                              .map((s) => (
                                                <DropdownMenu.Item
                                                  key={s.name}
                                                  className={`overflow-hidden ${
                                                    selectedExplanationScoreModelType?.name === s.name
                                                      ? 'bg-sky-100 text-slate-600'
                                                      : 'text-slate-600'
                                                  } flex flex-col items-start gap-y-0.5 border-b px-3 py-2.5 font-mono text-xs hover:bg-slate-100 focus:outline-none`}
                                                  onSelect={() => setSelectedExplanationScoreModelType(s)}
                                                >
                                                  <div className="flex w-full flex-row items-center justify-between gap-x-3">
                                                    <span className="">{s.displayName}</span>
                                                  </div>
                                                </DropdownMenu.Item>
                                              ))}
                                          </DropdownMenu.Content>
                                        </DropdownMenu.Portal>
                                      </DropdownMenu.Root>
                                    </div>
                                    <button
                                      type="button"
                                      onClick={async (e) => {
                                        e.preventDefault();
                                        const existingScore = currentNeuron?.explanations?.find((exp) =>
                                          exp.scores?.find(
                                            (s) =>
                                              s.explanationId === explanation.id &&
                                              s.explanationScoreTypeName === selectedExplanationScoreType?.name &&
                                              s.explanationScoreModelName === selectedExplanationScoreModelType?.name,
                                          ),
                                        );
                                        if (existingScore) {
                                          alert(
                                            'An auto-interp score with this explanation type and model already exists.',
                                          );
                                          return;
                                        }
                                        if (!selectedExplanationScoreType) {
                                          alert('Please select a scoring method.');
                                          return;
                                        }
                                        if (
                                          !selectedExplanationScoreModelType &&
                                          selectedExplanationScoreType?.name !== 'eleuther_embedding'
                                        ) {
                                          alert('Please select a scorer model.');
                                          return;
                                        }
                                        if (!user) {
                                          alert('Please log in to generate auto-interp explanation scores.');
                                          setSignInModalOpen(true);
                                          return;
                                        }

                                        setIsScoringId(explanation.id);
                                        const result = await fetch(`/api/explanation/score`, {
                                          method: 'POST',
                                          headers: {
                                            'Content-Type': 'application/json',
                                          },
                                          body: JSON.stringify({
                                            explanationId: explanation.id,
                                            scorerType: selectedExplanationScoreType?.name,
                                            scorerModel: selectedExplanationScoreModelType?.name,
                                          }),
                                        });
                                        if (result.status === 429 || result.status === 405) {
                                          alert("Sorry, you've been rate limited. Try again later.");
                                          setIsScoringId(undefined);
                                        } else if (result.status !== 200) {
                                          try {
                                            const res = await result.json();
                                            if (res.message === ERROR_NO_AUTOINTERP_KEY) {
                                              showToastMessage(
                                                <div className="flex flex-col items-center justify-center">
                                                  <div>
                                                    Missing auto-interp API key for {selectedExplanationModel?.name}.
                                                  </div>
                                                  <div>Set your key under Settings {`>`} Auto-Interp Keys</div>
                                                  <Link
                                                    href="/account"
                                                    className="mt-2 rounded-md bg-sky-600 px-2.5 py-1.5 text-xs font-semibold uppercase text-sky-100"
                                                  >
                                                    OPEN SETTINGS
                                                  </Link>
                                                </div>,
                                              );
                                              setIsScoringId(undefined);
                                            } else if (res.message === ERROR_REQUIRES_OPENROUTER) {
                                              showToastMessage(
                                                <div className="flex flex-col items-center justify-center">
                                                  <div>This auto-interp type requires an OpenRouter key.</div>
                                                  <div>Set your key under Settings {`>`} Auto-Interp Keys.</div>
                                                  <Link
                                                    href="/account"
                                                    className="mt-2 rounded-md bg-sky-600 px-2.5 py-1.5 text-xs font-semibold uppercase text-sky-100"
                                                  >
                                                    OPEN SETTINGS
                                                  </Link>
                                                </div>,
                                              );
                                              setIsScoringId(undefined);
                                            } else if (res.message === ERROR_RECALL_ALT_FAILED) {
                                              showToastMessage(
                                                <div className="flex flex-col items-center justify-center text-xs">
                                                  <div>Error: API Key</div>
                                                  <div>
                                                    Your API key was valid, but we received errors when using it.
                                                    <br />
                                                    Check that your OpenRouter/other account has sufficient funds and
                                                    that it has not been revoked.
                                                  </div>
                                                  <Link
                                                    href="/account"
                                                    className="mt-2 rounded-md bg-sky-600 px-2.5 py-1.5 text-xs font-semibold uppercase text-sky-100"
                                                  >
                                                    OPEN SETTINGS
                                                  </Link>
                                                </div>,
                                              );
                                              setIsScoringId(undefined);
                                            } else {
                                              showToastServerError();
                                              setIsScoringId(undefined);
                                            }
                                          } catch (err) {
                                            showToastServerError();
                                            setIsScoringId(undefined);
                                          }
                                        } else {
                                          const res = await result.json();
                                          const exp = res.score as ExplanationScore;

                                          // update the explanation with the ExplanationScore
                                          if (currentNeuron) {
                                            const newCurrentNeuron = {
                                              ...currentNeuron,
                                            };
                                            const newExplanations = currentNeuron.explanations?.map((currentExp) => {
                                              if (currentExp.id === exp.explanationId) {
                                                return {
                                                  ...currentExp,
                                                  scores: currentExp.scores ? [...currentExp.scores, exp] : [exp],
                                                };
                                              }
                                              return currentExp;
                                            });
                                            newCurrentNeuron.explanations = newExplanations;
                                            setCurrentNeuron(newCurrentNeuron);
                                          }

                                          // show it in the popup
                                          loadExplanationScoreDetail(
                                            explanation as ExplanationWithPartialRelations,
                                            res.score as ExplanationScoreWithPartialRelations,
                                          );

                                          setSelectedExplanationScoreModelType(undefined);
                                          setSelectedExplanationScoreType(undefined);
                                          setIsScoringId(undefined);
                                        }
                                      }}
                                      disabled={isScoringId !== undefined}
                                      className="flex max-w-[70px] flex-1 items-center justify-center rounded bg-slate-200 px-0 py-1.5 text-[10px] font-medium text-slate-500 transition-all hover:bg-slate-300 disabled:bg-slate-300"
                                    >
                                      {isScoringId !== undefined ? (
                                        <LoadingSpinner size={16} className="text-sky-700" />
                                      ) : (
                                        'Generate'
                                      )}
                                    </button>
                                  </div>
                                </div>
                              </div>
                            </DropdownMenu.Content>
                          </DropdownMenu.Portal>
                        </DropdownMenu.Root>
                      )}
                    </div>
                  </div>
                  {explanation.triggeredByUser?.id === user?.id && user && (
                    <button
                      type="button"
                      onClick={async (e) => {
                        e.preventDefault();
                        if (window.confirm('Delete this explanation?')) {
                          const result = await fetch(`/api/explanation/${explanation.id}/delete`, {
                            method: 'POST',
                            headers: {
                              'Content-Type': 'application/json',
                            },
                          });
                          if (result.status !== 200) {
                            showToastServerError();
                          } else {
                            const newExplanations = currentNeuron?.explanations?.filter(
                              (curExp) => curExp.id !== explanation.id,
                            );
                            if (currentNeuron) {
                              const newCurrentNeuron = {
                                ...currentNeuron,
                              };
                              newCurrentNeuron.explanations = newExplanations;
                              setCurrentNeuron(newCurrentNeuron);
                            }
                          }
                        }
                      }}
                      className="absolute right-2 top-0 rounded bg-red-400 p-0.5 hover:bg-red-600"
                    >
                      <X className="h-2.5 w-2.5 text-[5px] font-bold uppercase text-white" />
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
      <div className="flex w-full flex-col rounded-b-md border-t border-slate-200 px-3 py-3 pt-3">
        <div className="mb-1.5 w-full text-center text-[10px] font-normal uppercase leading-none text-slate-400">
          New Auto-Interp
        </div>
        <div className="flex w-full min-w-0 flex-row items-center justify-center gap-x-1 text-slate-500">
          <div className="flex min-w-[calc(50%-35px)] max-w-[calc(50%-35px)] flex-1 flex-col items-center">
            <DropdownMenu.Root>
              <DropdownMenu.Trigger asChild>
                <button
                  type="button"
                  className="flex w-full flex-1 flex-row items-center justify-center gap-x-1 whitespace-pre rounded border border-slate-300 bg-white px-2 text-[11px] font-medium text-slate-500 hover:bg-slate-50 focus:outline-none sm:pl-3 sm:pr-2"
                >
                  <div className="flex flex-col items-center justify-center gap-y-1 py-2 leading-none">
                    <div className="flex flex-row items-center justify-center gap-x-0.5">
                      {selectedExplanationType?.displayName || 'AutoInterp Type'}
                    </div>
                  </div>
                  <ChevronDownIcon className="-mr-1 ml-0 w-4 leading-none text-slate-400" />
                </button>
              </DropdownMenu.Trigger>
              <DropdownMenu.Portal>
                <DropdownMenu.Content
                  className="z-30 max-h-[900px] cursor-pointer overflow-hidden rounded-md border border-slate-300 bg-white text-xs font-medium text-slate-500 shadow"
                  sideOffset={2}
                  align="center"
                >
                  <div className="bg-slate-200 py-1.5 text-center text-[10px] uppercase text-slate-500">
                    Auto-Interp Methods
                  </div>
                  {explanationTypesFiltered.map((s) => (
                    <DropdownMenu.Item
                      key={s.name}
                      className={`overflow-hidden ${
                        selectedExplanationType?.name === s.name ? 'bg-sky-100 text-slate-600' : 'text-slate-600'
                      } flex flex-col items-start gap-y-0.5 border-b px-3 py-2.5 font-mono text-xs hover:bg-slate-100 focus:outline-none`}
                      onSelect={() => setSelectedExplanationType(s)}
                    >
                      <div className="flex w-full flex-row items-center justify-between gap-x-3">
                        <span className="">{s.displayName}</span>
                        <button
                          type="button"
                          onClick={(e) => {
                            e.preventDefault();
                            e.stopPropagation();
                            window.open(`/explanation-type/${s.name}`, '_blank', 'noopener,noreferrer');
                          }}
                          className="rounded p-1 transition-all hover:bg-sky-300 hover:text-sky-700"
                        >
                          <Info className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    </DropdownMenu.Item>
                  ))}
                </DropdownMenu.Content>
              </DropdownMenu.Portal>
            </DropdownMenu.Root>
          </div>
          <div className="flex min-w-[calc(50%-35px)] max-w-[calc(50%-35px)] flex-1 flex-row items-center justify-start gap-x-2 overflow-hidden text-ellipsis whitespace-pre">
            <DropdownMenu.Root>
              <DropdownMenu.Trigger asChild>
                <button
                  type="button"
                  className="relative flex h-7 w-full flex-row items-center justify-center gap-x-1 overflow-hidden text-ellipsis whitespace-pre rounded border border-slate-300 bg-white px-2 text-[11px] font-medium text-slate-500 hover:bg-slate-50 focus:outline-none sm:pl-3 sm:pr-2"
                >
                  {selectedExplanationModel?.displayName || 'AutoInterp Model'}
                  <ChevronDownIcon className="absolute right-0 bg-white px-1 leading-none text-slate-400" />
                </button>
              </DropdownMenu.Trigger>
              <DropdownMenu.Portal>
                <DropdownMenu.Content
                  className="z-30 max-h-[420px] cursor-pointer overflow-scroll rounded-md border border-slate-300 bg-white text-xs font-medium text-slate-600 shadow"
                  sideOffset={2}
                  align="center"
                >
                  <div className="bg-slate-200 py-1.5 text-center text-[10px] uppercase text-slate-500">
                    Auto-Interp Models
                  </div>
                  {explanationModels.map((s) => (
                    <DropdownMenu.Item
                      key={s.name}
                      className={`overflow-hidden ${
                        selectedExplanationModel?.name === s.name ? 'bg-sky-100 text-slate-600' : 'text-slate-600'
                      } flex flex-col items-start gap-y-0.5 border-b px-3 py-2.5 font-mono text-xs hover:bg-slate-100 focus:outline-none`}
                      onSelect={() => setSelectedExplanationModel(s)}
                    >
                      <div className="flex w-full flex-row items-center justify-between gap-x-3">
                        <span className="">{s.displayName}</span>
                        {/* <button
                                    onClick={(e) => {
                                      e.preventDefault();
                                      e.stopPropagation();
                                      window.open(
                                        `/explanation-model-type/${s.name}`,
                                        "_blank",
                                        "noopener,noreferrer",
                                      );
                                    }}
                                    className="rounded p-1 transition-all hover:bg-sky-300 hover:text-sky-700"
                                  >
                                    <Info className="h-3.5 w-3.5" />
                                  </button> */}
                      </div>
                    </DropdownMenu.Item>
                  ))}
                </DropdownMenu.Content>
              </DropdownMenu.Portal>
            </DropdownMenu.Root>
          </div>
          <button
            type="button"
            onClick={async (e) => {
              e.preventDefault();

              // if the explanationType and explanationModelName already exist, don't generate a new one
              const existingExplanation = currentNeuron?.explanations?.find(
                (curExp) =>
                  curExp.typeName === selectedExplanationType?.name &&
                  curExp.explanationModelName === selectedExplanationModel?.name,
              );
              if (existingExplanation) {
                alert('An auto-interp with this explanation type and model already exists.');
                return;
              }
              if (!user) {
                alert('Please log in to generate auto-interp explanations.');
                setSignInModalOpen(true);
                return;
              }
              setIsAutoInterping(true);
              const result = await fetch(`/api/explanation/generate`, {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                  modelId: currentNeuron?.modelId,
                  layer: currentNeuron?.layer,
                  index: currentNeuron?.index,
                  explanationType: selectedExplanationType?.name || '',
                  explanationModelName: selectedExplanationModel?.name || '',
                }),
              });
              if (result.status === 429 || result.status === 405) {
                alert("Sorry, you've been rate limited. Try again later.");
                setIsAutoInterping(false);
              } else if (result.status !== 200) {
                try {
                  const res = await result.json();
                  if (res.message === ERROR_NO_AUTOINTERP_KEY) {
                    showToastMessage(
                      <div className="flex flex-col items-center justify-center">
                        <div>Missing auto-interp API key for {selectedExplanationModel?.displayName}.</div>
                        <div>Set your key under Settings {`>`} Auto-Interp Keys.</div>
                        <Link
                          href="/account"
                          className="mt-2 rounded-md bg-sky-600 px-2.5 py-1.5 text-xs font-semibold uppercase text-sky-100"
                        >
                          OPEN SETTINGS
                        </Link>
                      </div>,
                    );
                  } else if (res.message === ERROR_REQUIRES_OPENROUTER) {
                    showToastMessage(
                      <div className="flex flex-col items-center justify-center">
                        <div>This auto-interp type requires an OpenRouter key.</div>
                        <div>Set your key under Settings {`>`} Auto-Interp Keys.</div>
                        <Link
                          href="/account"
                          className="mt-2 rounded-md bg-sky-600 px-2.5 py-1.5 text-xs font-semibold uppercase text-sky-100"
                        >
                          OPEN SETTINGS
                        </Link>
                      </div>,
                    );
                  } else {
                    showToastServerError();
                  }
                } catch (_) {
                  showToastServerError();
                }
                setIsAutoInterping(false);
              } else {
                const res = await result.json();
                const exp = res.explanation as Explanation;
                // append the explanation to the currentNeuron.explanations
                if (currentNeuron && currentNeuron.explanations) {
                  const newCurrentNeuron = { ...currentNeuron };
                  newCurrentNeuron.explanations = [...currentNeuron.explanations, exp];
                  setCurrentNeuron(newCurrentNeuron);
                }
                setIsAutoInterping(false);
              }
            }}
            disabled={isAutoInterping}
            className="flex max-w-[70px] flex-1 items-center justify-center rounded bg-slate-200 px-0 py-1.5 text-[10px] font-medium text-slate-500 transition-all hover:bg-slate-300 disabled:bg-slate-300"
          >
            {isAutoInterping ? <LoadingSpinner size={16} className="text-sky-700" /> : 'Generate'}
          </button>
        </div>
      </div>
      {/* <div className="rounded-b-md px-2 py-2 pt-0">
        </div>
        <div className="relative mb-0 flex flex-col rounded-lg border border-slate-200 bg-white text-xs shadow transition-all sm:mt-0 ">             
        <Formik
            initialValues={{
            modelId: currentNeuron?.modelId,
            layer: currentNeuron?.layer,
            index: currentNeuron?.index,
            explanation: "",
            save: true,
            peeked: true
            }}
            onSubmit={(values, { setFieldValue }) => {
            if (!session.data?.user) {
                setSignInModalOpen(true);
                return;
            }
            if (!currentNeuron) {
                return;
            }
            if (values.explanation.trim().length === 0) {
                alert("Please enter an explanation.");
                return;
            }
            setIsExplaining(true);

            values.modelId = currentNeuron?.modelId;
            values.layer = currentNeuron?.layer;
            values.index = currentNeuron?.index;

            fetch(`/api/explanation/score`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(values),
            })
                .then((response) => {
                return response.json();
                })
                .then((result: ScoreResult) => {
                if (
                    result.explanation &&
                    currentNeuron &&
                    currentNeuron.explanations
                ) {
                    var newCurrentNeuron = { ...currentNeuron };
                    newCurrentNeuron.explanations = [
                    ...currentNeuron?.explanations,
                    result.explanation,
                    ];
                    setCurrentNeuron(newCurrentNeuron);
                } else if (result.error) {
                    alert(result.error);
                }
                setFieldValue("explanation", "");
                setIsExplaining(false);
                })
                .catch((error) => {
                alert("Error\n" + error);
                setIsExplaining(false);
                });
            }}
        >
            {({ submitForm, values, setFieldValue }) => (
            <Form
                className={`flex gap-x-1.5 gap-y-1.5 sm:flex-row`}
            >
                <div
                className={`mt-0 flex flex-1 flex-row gap-0 overflow-hidden rounded-md ${
                    isExplaining
                    ? "border-slate-300"
                    : "border-sky-700"
                } bg-white`}
                >
                <Field
                    name="explanation"
                    disabled={isExplaining}
                    type="text"
                    required
                    onKeyDown={(event: React.KeyboardEvent<any>) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                        event.preventDefault();
                        submitForm();
                    }
                    }}
                    placeholder={`Add Explanation`}
                    className={`mt-0 min-w-[10px] flex-1 rounded-md border-0 bg-slate-200 pl-2 pr-0 text-left text-xs font-normal text-slate-800 placeholder-slate-700/60 focus:outline-none focus:ring-0 disabled:bg-slate-200`}
                />
                </div>
            </Form>
            )}
        </Formik>
        {isExplaining && (
            <div className="mt-1 text-center text-[10px] font-medium text-slate-400">
            Please wait - scoring your explanation...
            </div>
        )}
        </div> */}
    </div>
  );
}
