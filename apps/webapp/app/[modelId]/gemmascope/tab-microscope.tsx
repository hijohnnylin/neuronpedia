'use client';

import DeepDiveTooltip from '@/app/[modelId]/gemmascope/tooltip-deepdive';
import WarningTooltip from '@/app/[modelId]/gemmascope/tooltip-warning';
import CustomTooltip from '@/components/custom-tooltip';
import { useGlobalContext } from '@/components/provider/global-provider';
import { LoadingSquare } from '@/components/svg/loading-square';
import { SearchTopKResult } from '@/lib/utils/inference';
import { INFERENCE_EXAMPLE_TEXTS, INFERENCE_EXAMPLE_TEXTS_CATEGORIZED } from '@/lib/utils/inference-example-texts';
import { Form, Formik, FormikProps } from 'formik';
import { BrainCircuit, ExternalLink, Search, X } from 'lucide-react';
import { NeuronWithPartialRelations } from 'prisma/generated/zod';
import { useEffect, useRef, useState } from 'react';
import FeatureTooltip from './feature-tooltip';

const SEARCHER_TOPK_MAX_CHAR_LENGTH = 250;
const DOGS_INDEX = 14838;

export default function TabMicroscope({
  modelId,
  layer,
  tabUpdater,
  completedTabsAdd,
}: {
  modelId: string;
  layer: string;
  tabUpdater: (tab: string) => void;
  completedTabsAdd: (tab: string) => void;
}) {
  const [topkResult, setTopkResult] = useState<SearchTopKResult | undefined>();
  type TopKFeature = {
    index: number;
    feature: NeuronWithPartialRelations;
    activation_value: number;
    frequency: number;
  };
  const [topkFeatures, setTopkFeatures] = useState<TopKFeature[] | undefined>(undefined);
  const [hoveredTokenPosition, setHoveredTokenPosition] = useState<number>(-1);
  const [lockedTokenPosition, setLockedTokenPosition] = useState<number>(-1);
  const [hoveredNeuronIndex, setHoveredNeuronIndex] = useState<number>(-1);
  const formRef = useRef<
    FormikProps<{
      searchQuery: string;
    }>
  >(null);
  const [isSearching, setIsSearching] = useState(false);
  const [forceExtraCredit, setForceExtraCredit] = useState(false);
  const [forceWhatsNext, setForceWhatsNext] = useState(false);
  const { showToastServerError, setFeatureModalFeature, setFeatureModalOpen } = useGlobalContext();
  const [searchQuery, setSearchQuery] = useState<string>('');
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [sortNeurons, setSortNeurons] = useState<'frequency' | 'strength'>('frequency');

  async function searchClicked(overrideText?: string) {
    setIsSearching(true);
    setLockedTokenPosition(-1);
    setHoveredTokenPosition(-1);
    setSearchQuery(overrideText || formRef.current?.values.searchQuery || '');
    const result = await fetch(`/api/search-topk-by-token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        modelId,
        text: overrideText || formRef.current?.values.searchQuery,
        source: layer,
      }),
    });
    if (result.status === 429 || result.status === 405) {
      alert('Sorry, we are limiting each user to 100 search requests per hour. Please try again later.');
    } else if (result.status !== 200) {
      showToastServerError();
    } else {
      const resultData = (await result.json()) as SearchTopKResult;

      // filter out boring indexes and ones with act desnsity > 1%
      resultData.results = resultData.results.map((resultDataResult) => {
        // eslint-disable-next-line
        resultDataResult.topFeatures = resultDataResult.topFeatures.filter(
          (feature) => feature.feature?.frac_nonzero !== undefined,
        );
        return resultDataResult;
      });

      setTopkResult(resultData);
    }
    setIsSearching(false);
  }

  function getSortedFeatures(feats: TopKFeature[]) {
    if (sortNeurons === 'strength') {
      return feats.toSorted((a, b) => {
        if (a.index === DOGS_INDEX) {
          return 1;
        }
        if (b.index === DOGS_INDEX) {
          return -1;
        }
        return b.activation_value - a.activation_value;
      });
    }
    if (sortNeurons === 'frequency') {
      return feats.toSorted((a, b) => {
        if (a.index === DOGS_INDEX) {
          return -1;
        }
        if (b.index === DOGS_INDEX) {
          return 1;
        }
        return b.frequency - a.frequency;
      });
    }
    return feats;
  }

  useEffect(() => {
    if (topkResult) {
      let toSet: TopKFeature[] = [];
      topkResult.results.forEach((result) => {
        result.topFeatures.forEach((feature) => {
          if (result.token === '<bos>' || result.token === '<eos>') {
            return;
          }
          toSet.push({
            feature: feature.feature as NeuronWithPartialRelations,
            activation_value: feature.activationValue,
            frequency: 1,
            index: feature.featureIndex,
          });
        });
      });

      // set the frequency, which is the number of times that index appears
      toSet = toSet.map((f) => {
        // eslint-disable-next-line
        f.frequency = toSet.filter((f2) => f2.index === f.index).length;
        return f;
      });

      // deduplicate by toSet.index
      toSet = toSet.filter((v, i, a) => a.findIndex((t) => t.index === v.index) === i);

      // check if index has the SF_INDEX
      toSet.forEach((f) => {
        if (f.index === DOGS_INDEX) {
          completedTabsAdd('microscope');
        }
      });

      setTopkFeatures(getSortedFeatures(toSet));
    }
  }, [topkResult]);

  useEffect(() => {
    if (topkFeatures) {
      setTopkFeatures(getSortedFeatures(topkFeatures));
    }
  }, [sortNeurons]);

  const hoverGbgColors = [
    'enabled:hover:bg-[#4285f4]',
    'enabled:hover:bg-[#34a853]',
    'enabled:hover:bg-[#fbbc05]',
    'enabled:hover:bg-[#ea4335]',
  ];
  const gTextColors = ['text-[#4285f4]', 'text-[#34a853]', 'text-[#fbbc05]', 'text-[#ea4335]'];
  const gBorderColors = [
    'enabled:border-[#4285f4]',
    'enabled:border-[#34a853]',
    'enabled:border-[#fbbc05]',
    'enabled:border-[#ea4335]',
  ];

  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (ref.current) {
      ref.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, []);

  return (
    <div className="mt-0 flex w-full max-w-screen-xl flex-col items-center justify-center pb-24 pt-1">
      <div ref={ref} className="pt-20 sm:pt-0" />
      <div className="mb-10 mt-5 flex w-full flex-row items-center justify-start px-2 sm:mb-3 sm:px-5 sm:pt-0">
        <div className="flex w-full flex-col items-start justify-start gap-x-4 gap-y-1.5 rounded px-2 py-1 sm:flex-row">
          <span className="w-[105px] min-w-[105px] max-w-[105px] whitespace-nowrap rounded-full bg-slate-100 px-3 py-1 text-center text-[10px] font-bold uppercase text-slate-600">
            💡 GOAL
          </span>
          <div className="text-sm font-medium leading-normal text-slate-500">
            <span>
              To understand what AI is thinking, Gemma Scope breaks down {`Gemma's`} brain into millions of parts called{' '}
              <strong>features</strong>.
            </span>
          </div>
        </div>
      </div>
      <div className="mb-10 flex w-full flex-row items-center justify-start px-2 sm:mb-4 sm:px-5">
        <div className="flex w-full flex-col items-start justify-start gap-x-4 gap-y-1.5 rounded px-2 py-1 sm:flex-row">
          <span className="w-[105px] min-w-[105px] max-w-[105px] whitespace-nowrap rounded-full bg-slate-100 px-3 py-1 text-center text-[10px] font-bold uppercase text-slate-600">
            ❗️ Key Term
          </span>
          <div className="text-sm font-medium leading-normal text-slate-500">
            <span>
              A <FeatureTooltip s={false} /> is something that activates in the {`AI's brain`} when it see a specific
              concept or idea, like
              {` "words about cats"`}.<br />
              For example, when you send an AI the message {`"I like cats"`}, it activates the {`"words about cats"`}{' '}
              feature in the {`AI's brain`}.
            </span>
          </div>
        </div>
      </div>

      <div className="mb-10 flex w-full flex-row items-center justify-start px-2 sm:mb-5 sm:px-5">
        <div className="flex w-full flex-col items-start justify-start gap-x-4 gap-y-1.5 rounded px-2 py-1 sm:flex-row">
          <div className="mb-2 flex w-full flex-row items-center justify-between gap-x-2 sm:mb-0 sm:w-auto sm:flex-col sm:justify-start">
            <span className="w-[105px] min-w-[105px] max-w-[105px] whitespace-nowrap rounded-full bg-slate-100 px-0 py-1 text-center text-[10px] font-bold uppercase text-slate-600">
              🎨 Demo
            </span>
            <div className="flex flex-row gap-x-1.5 sm:flex-auto sm:flex-col">
              <WarningTooltip>
                <div className="font-medium text-slate-600">
                  Labels for features {`e.g, "words about cats"`} are generated by other AIs that analyze the neurons of
                  Gemma 2. These labels are sometimes misleading, inaccurate, or just plain wrong. More details about
                  this in the next {`"Analyze Features"`} section.
                </div>
              </WarningTooltip>
              <DeepDiveTooltip>
                <div className="font-medium text-slate-600">
                  <div className="text-center">Technical Deep Dive</div>
                  <br />
                  We find the top five features that activated for each token. The default list combines all the top
                  five features from all tokens. Hovering over a token filters the list to just the features that fired
                  on that token.
                  <br />
                  <br />
                  {`We've`} manually updated a few labels/explanations for the purposes of this demo to be more concise.
                  <br />
                  <br />
                  All messages with Gemma 2 start with a special token {`<bos>`}. For the purposes of this demo,{' '}
                  {`we've`} hidden this token and its associated features. It is fine to ignore the fact that this token
                  is hidden.
                  <br />
                  <br />
                  For the purposes of this demo, we have also hidden all features that have a density of activations
                  greater than 1%.
                </div>
              </DeepDiveTooltip>
            </div>
          </div>
          <div className="flex w-full flex-col">
            <div className="text-sm font-medium leading-normal text-slate-500">
              <span>
                What{`'`}s Gemma thinking? Click a button below to send Gemma a sentence and see what features it
                activates. 👇
              </span>
            </div>
            <div className="mt-2 grid w-full grid-cols-3 gap-x-1.5 gap-y-1.5 sm:grid-cols-3 sm:gap-x-2 sm:gap-y-2">
              <button
                type="button"
                disabled={isSearching}
                onClick={async (e) => {
                  e.preventDefault();
                  const randomSentence =
                    INFERENCE_EXAMPLE_TEXTS[Math.floor(Math.random() * INFERENCE_EXAMPLE_TEXTS.length)];
                  await searchClicked(randomSentence);
                }}
                className="flex flex-row items-center justify-center gap-x-1.5 rounded-full border bg-white px-0 py-2 text-[10px] font-medium text-slate-600 shadow transition-all enabled:border-[#ea4335] enabled:hover:bg-[#ea4335] enabled:hover:text-white disabled:opacity-50 sm:text-[14px]"
              >
                {`🎲 I'm Feeling Lucky`}
              </button>
              {INFERENCE_EXAMPLE_TEXTS_CATEGORIZED.slice(0, 5).map((category, i) => (
                <button
                  type="button"
                  key={category.type}
                  disabled={isSearching}
                  onClick={async (e) => {
                    e.preventDefault();
                    const sentences = INFERENCE_EXAMPLE_TEXTS_CATEGORIZED.find(
                      (c) => c.type === category.type,
                    )?.examples;
                    if (sentences) {
                      const randomSentence = sentences[Math.floor(Math.random() * sentences.length)];
                      await searchClicked(randomSentence);
                    }
                  }}
                  className={`flex flex-row items-center justify-center gap-x-2 rounded-full bg-white px-0 py-2 text-[10px] font-medium text-slate-600 shadow transition-all enabled:hover:text-white sm:text-[14px] ${
                    hoverGbgColors[i % hoverGbgColors.length]
                  } border ${gBorderColors[i % gBorderColors.length]} text-[10px] disabled:opacity-50`}
                >
                  {category.type}
                </button>
              ))}
            </div>
            <div
              className={`mt-4 flex w-full flex-col gap-x-3 gap-y-4 sm:flex-row ${
                !topkResult && !isSearching ? 'opacity-40' : ''
              }`}
            >
              <div className="flex flex-1 basis-1/2 flex-col">
                <div className="flex w-full flex-col justify-center">
                  <div className="mb-0 w-full text-center text-sm font-bold text-slate-600">Text Sent to Gemma</div>
                  {!isSearching && !topkResult && (
                    <div className="pt-3 text-center text-sm font-medium text-slate-400">
                      <span>Click a preset above to send a text to Gemma.</span>
                    </div>
                  )}
                  {(isSearching || (topkResult && !isSearching)) && (
                    <div className="flex flex-col gap-y-1 pt-2">
                      <div className="text-xs">{searchQuery}</div>
                      <div className="flex flex-row items-center justify-start">
                        <span className="mb-0 mt-0.5 flex flex-row items-center justify-start gap-x-1.5 rounded border-slate-200 px-3 py-1">
                          <div className="text-[10px]">⬇️</div>
                          <div className="max-w-[300px] text-center text-[9px] font-bold uppercase leading-snug text-slate-400">
                            <span className="py-0.5">Message is turned into </span>
                            <CustomTooltip
                              trigger={
                                <span
                                  className={`${gTextColors[1]} flex cursor-pointer flex-row gap-x-0.5 font-mono font-bold underline decoration-dashed`}
                                >
                                  <span className="flex flex-row rounded bg-slate-200 px-1 py-0.5">tokens</span>
                                </span>
                              }
                            >
                              <div>
                                Tokens are bits of text that AI models are trained to understand. Sometimes tokens
                                correspond to whole words, but tokens can be anything: parts of words, symbols,
                                punctuation, etc.
                              </div>
                              <div className="mt-2">Gemma 2 has a fixed vocabulary of ~256,000 tokens.</div>
                            </CustomTooltip>
                            , then sent to Gemma.
                          </div>
                        </span>
                      </div>
                    </div>
                  )}
                  {isSearching && (
                    <div className="mt-5 flex w-full flex-row items-center justify-center">
                      <div className="mb-1 flex items-center justify-center px-0 font-bold text-slate-300">
                        <LoadingSquare size={32} className="text-sky-700" />
                      </div>
                    </div>
                  )}
                </div>
                {topkResult && !isSearching && (
                  <div className="flex w-full flex-col">
                    <div className="mb-2.5 mt-1 flex w-full flex-row items-center justify-center gap-x-1 py-1 text-center text-[11px] text-gBlue">
                      <span className="mr-1 w-[60px] min-w-[60px] max-w-[60px] whitespace-nowrap rounded bg-gBlue px-3 py-1 text-center text-[8px] font-bold uppercase leading-none text-white">
                        TRY THIS{' '}
                      </span>
                      <div className="flex flex-1 flex-col items-start justify-start gap-y-0.5 pr-1 text-left leading-snug">
                        <div className="">
                          <strong>
                            <span className="hidden sm:inline-block">Hover over or click</span>{' '}
                            <span className="inline-block sm:hidden">Tap</span> a token
                          </strong>{' '}
                          to see which features were activated{' '}
                          <span className="hidden sm:inline-block">to the right</span>.
                        </div>
                      </div>
                      <div className="hidden text-base sm:block">➡️</div>
                    </div>
                    <div className="flex flex-row flex-wrap items-start justify-start gap-x-1">
                      {topkResult.results &&
                        topkResult.results.map((result, i) => {
                          if (result.token === '<bos>' || result.token === '<eos>') {
                            return <div key={i} className="hidden" />;
                          }
                          return (
                            <button
                              type="button"
                              onMouseEnter={() => {
                                setHoveredTokenPosition(result.position);
                              }}
                              onMouseLeave={() => {
                                setHoveredTokenPosition(-1);
                              }}
                              onClick={() => {
                                if (lockedTokenPosition === result.position) {
                                  setLockedTokenPosition(-1);
                                } else {
                                  setLockedTokenPosition(i);
                                }
                              }}
                              key={result.position}
                              className={`mb-1 inline-block cursor-pointer select-none rounded font-mono text-xs transition-all sm:text-sm ${
                                lockedTokenPosition === result.position
                                  ? 'bg-emerald-600 text-white'
                                  : result.topFeatures.filter((f) => f.featureIndex === hoveredNeuronIndex).length > 0
                                    ? 'bg-emerald-200'
                                    : 'bg-slate-100 hover:bg-emerald-200 hover:text-emerald-700'
                              } ${result.token.endsWith(' ') && result.token.length > 1 ? 'pr-2' : ''} ${result.token.startsWith(' ') && result.token.length > 1 ? 'pl-2' : ''}`}
                            >
                              {result.token.trim().length === 0 ? ' ' : result.token}
                            </button>
                          );
                        })}
                    </div>
                  </div>
                )}
              </div>
              <div className="flex flex-1 basis-1/2 flex-col">
                <div className="mb-0 w-full text-center text-sm font-bold text-slate-600">Top Features Activated</div>
                {(!topkResult || isSearching) && (
                  <div className="max-h-[260px] pt-3 text-center text-sm font-medium text-slate-400 sm:h-[260px] sm:min-h-[260px]">
                    <span>Features that fire will appear here.</span>
                  </div>
                )}
                {topkResult && !isSearching && (
                  // eslint-disable-next-line
                  <>
                    {lockedTokenPosition === -1 && hoveredTokenPosition === -1 ? (
                      <div className="flex flex-1 flex-col">
                        <div className="mb-0 flex flex-row items-center justify-between">
                          <div className="flex flex-1 flex-col justify-center">
                            <div className="mb-2 hidden w-full flex-row items-center justify-center gap-x-1 py-1 text-center text-[11px] text-gBlue sm:flex">
                              <span className="mr-1 w-[60px] min-w-[60px] max-w-[60px] whitespace-nowrap rounded bg-gBlue px-3 py-1 text-center text-[8px] font-bold uppercase leading-none text-white">
                                TRY THIS{' '}
                              </span>
                              <div className="flex flex-col items-start justify-start gap-y-0.5 text-left leading-none">
                                <div className="block text-left">
                                  <strong>Hover over a feature</strong> to see which tokens it activated on.
                                </div>
                                {/* <div>
                                  <strong>Click a neuron</strong> for its
                                  technical details.
                                </div> */}
                              </div>
                            </div>
                          </div>
                        </div>
                        <div className="flex h-[300px] max-h-[300px] w-full flex-col gap-y-1 overflow-y-hidden sm:h-[280px] sm:max-h-[280px] sm:overflow-y-scroll">
                          {topkFeatures?.map((f, i) => (
                            <div
                              key={i}
                              onMouseEnter={() => {
                                setHoveredNeuronIndex(f.index);
                              }}
                              onMouseLeave={() => {
                                setHoveredNeuronIndex(-1);
                              }}
                              className="group relative flex w-full flex-row items-center justify-center gap-x-3 rounded-xl bg-emerald-100 py-1 pl-5 pr-3 text-emerald-800 transition-all hover:bg-emerald-200"
                            >
                              <div className="h-2 w-2 rounded-full bg-emerald-700" />
                              {f.index === DOGS_INDEX && (
                                <div className="flex cursor-default flex-col items-center justify-center py-1.5">
                                  <div className="mb-1 animate-bounce text-[18px]">🐶</div>
                                  <div className="text-center font-mono text-[7px] uppercase leading-none">
                                    Challenge
                                    <br />
                                    Complete!
                                  </div>
                                </div>
                              )}
                              <div className="flex flex-1 flex-col py-2">
                                <div className="cursor-default text-xs transition-all group-hover:text-emerald-800">
                                  {f.feature &&
                                    f.feature.explanations &&
                                    f.feature.explanations.length > 0 &&
                                    f.feature.explanations[0].description}
                                </div>
                              </div>
                              <div className="flex w-6 flex-col items-center justify-center gap-y-0 rounded-lg font-mono text-xs font-bold leading-none text-emerald-700">
                                {f.frequency}
                                <div className="pt-[1px] text-[8px] font-bold leading-none">
                                  TOKEN
                                  {f.frequency > 1 ? 'S' : ''}
                                </div>
                              </div>
                              <div className="ml-1 flex max-w-7 flex-col items-center justify-center gap-y-1 overflow-visible pb-1 pt-2.5">
                                <ExternalLink
                                  className="h-7 w-7 cursor-pointer rounded bg-emerald-600/30 p-1.5 hover:bg-emerald-600/50"
                                  onClick={() => {
                                    setFeatureModalFeature(f.feature as NeuronWithPartialRelations);
                                    setFeatureModalOpen(true);
                                  }}
                                />
                                <div className="whitespace-nowrap font-mono text-[6px] uppercase leading-none text-emerald-500 group-hover:text-emerald-600">
                                  {/* Layer {20} */}
                                  ID {f.index}
                                </div>
                              </div>
                              {/* <div className="absolute top-1 ml-auto flex w-full flex-row items-center justify-end pr-3">
                                  <div className="font-mono text-[6px] uppercase leading-none text-emerald-500 transition-all group-hover:text-emerald-600">
                                    Strength {f.activation_value.toFixed(0)}
                                  </div>
                                </div> */}
                              {/* <div className="absolute bottom-1 ml-auto flex w-full flex-row items-center justify-end pr-3">
                                  <div className="font-mono text-[6px] uppercase leading-none text-emerald-500 transition-all group-hover:text-emerald-600">
                                    Layer {20}
                                    ID {f.index}
                                  </div>
                                </div> */}
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <div className="flex flex-1 flex-col">
                        <div className="mb-2 flex w-full flex-col justify-center">
                          <div className="mb-0 mt-2 flex w-full flex-row items-center justify-center">
                            <span className="rounded bg-emerald-200 px-1 py-0.5 font-mono text-[12px] leading-none text-emerald-700">
                              {`${
                                topkResult.results[
                                  lockedTokenPosition > -1 ? lockedTokenPosition : hoveredTokenPosition
                                ].token.trim().length === 0
                                  ? ' '
                                  : topkResult.results[
                                      lockedTokenPosition > -1 ? lockedTokenPosition : hoveredTokenPosition
                                    ].token
                              }`}
                            </span>
                            {lockedTokenPosition > -1 && (
                              <button
                                type="button"
                                onClick={() => {
                                  setLockedTokenPosition(-1);
                                }}
                                className="flex cursor-pointer flex-row items-center justify-center gap-x-0.5 text-[8px] font-bold uppercase leading-none text-red-400 hover:text-red-600"
                              >
                                <X className="ml-1 h-3 w-3 cursor-pointer" />
                                Show All
                              </button>
                            )}
                          </div>
                        </div>
                        <div className="flex h-[280px] max-h-[280px] w-full flex-col gap-y-1 overflow-y-scroll">
                          {topkResult.results[
                            lockedTokenPosition > -1 ? lockedTokenPosition : hoveredTokenPosition
                          ].topFeatures.map((f, i) => (
                            <div
                              key={i}
                              className="group relative flex w-full cursor-default flex-row items-center justify-center gap-x-3 rounded-xl bg-emerald-100 py-1 pl-5 pr-3 text-emerald-800 transition-all hover:bg-emerald-200"
                            >
                              <div className="h-2 w-2 rounded-full bg-emerald-700" />
                              {f.featureIndex === DOGS_INDEX && (
                                <div className="flex cursor-default flex-col items-center justify-center py-1.5">
                                  <div className="mb-1 animate-bounce text-[18px]">🐶</div>
                                  <div className="text-center font-mono text-[7px] uppercase leading-none">
                                    Challenge
                                    <br />
                                    Complete!
                                  </div>
                                </div>
                              )}
                              <div className="flex flex-1 flex-col">
                                <div className="cursor-default pb-3 pt-3 text-xs transition-all group-hover:text-emerald-800">
                                  {f.feature &&
                                    f.feature.explanations &&
                                    f.feature.explanations.length > 0 &&
                                    f.feature.explanations[0].description}
                                </div>
                              </div>
                              <div className="ml-1 flex max-w-7 flex-col items-center justify-center gap-y-1 overflow-visible pb-1 pt-2.5">
                                <ExternalLink
                                  className="h-7 w-7 cursor-pointer rounded bg-emerald-600/30 p-1.5 hover:bg-emerald-600/50"
                                  onClick={() => {
                                    setFeatureModalFeature(f.feature as NeuronWithPartialRelations);
                                    setFeatureModalOpen(true);
                                  }}
                                />
                                <div className="whitespace-nowrap font-mono text-[6px] uppercase leading-none text-emerald-500 group-hover:text-emerald-600">
                                  {/* Layer {20} */}
                                  ID {f.feature?.index}
                                </div>
                              </div>
                              {/* <div className="absolute bottom-1 mx-auto flex w-full flex-row items-center justify-end pr-3">
                                  <div
                                    onClick={() => {
                                      setFeatureModalFeature(
                                        f.feature as NeuronWithPartialRelations,
                                      );
                                      setFeatureModalOpen(true);
                                    }}
                                    className="font-mono text-[6px] uppercase leading-none text-emerald-500 transition-all group-hover:text-emerald-600"
                                  >
                                    Layer{" "}
                                    {getLayerNumFromLayer(
                                      f.feature?.layer || "0",
                                    )}{" "}
                                    ID {f.feature?.index}
                                    Strength{" "}
                                    {f.activation_value.toFixed(0)} ↗
                                  </div>
                                </div> */}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div
        className={`mb-10 flex w-full flex-row items-center justify-start px-2 sm:mb-7 sm:px-5 ${
          !(topkResult && !isSearching) && !forceExtraCredit ? 'opacity-40' : ''
        }`}
      >
        <div className="flex w-full flex-col items-start justify-start gap-x-4 gap-y-1.5 rounded px-2 py-1 sm:flex-row">
          <span className="w-[105px] min-w-[105px] max-w-[105px] whitespace-nowrap rounded-full bg-slate-100 px-3 py-1 text-center text-[10px] font-bold uppercase text-slate-600">
            ⭐️ Challenge
          </span>
          {!(topkResult && !isSearching) && !forceExtraCredit ? (
            <div className="flex flex-row items-center justify-center text-sm font-medium text-slate-400">
              Unlocked after doing the demo, or{' '}
              <button
                type="button"
                onClick={() => setForceExtraCredit(true)}
                className="ml-1 px-0 py-0 font-medium leading-none text-slate-400 underline hover:bg-slate-300"
              >
                skip demo
              </button>
              .
            </div>
          ) : (
            <div className="flex w-full flex-col items-start justify-start text-sm font-medium text-slate-500">
              <div className="mb-2 text-sm leading-normal">
                Great! Now, can you find a <strong className="text-gBlue">feature related to dogs</strong>? Use this
                field to send Gemma a custom text to think about.
              </div>
              <Formik
                innerRef={formRef}
                initialValues={{ searchQuery: '' }}
                onSubmit={async () => {
                  await searchClicked();
                }}
              >
                {({ values, setFieldValue }) => (
                  <Form className="relative mb-0 flex w-full max-w-screen-sm flex-col gap-x-1.5 gap-y-1.5 sm:flex-row">
                    <input
                      name="searchQuery"
                      disabled={isSearching}
                      value={values.searchQuery}
                      maxLength={SEARCHER_TOPK_MAX_CHAR_LENGTH}
                      onChange={(e) => {
                        setFieldValue('searchQuery', e.target.value);
                      }}
                      required
                      placeholder="Hint: Try a short sentence instead of a word or phrase."
                      className="mt-0 min-w-[10px] flex-1 resize-none rounded-xl border border-slate-300 px-3 py-3 pl-10 text-left font-mono text-xs font-medium text-slate-800 placeholder-slate-400 transition-all hover:shadow-lg focus:border-slate-300 focus:shadow-lg focus:outline-none focus:ring-0 disabled:bg-slate-200 sm:text-[13px]"
                    />
                    <Search className="absolute left-3 my-auto h-full w-4 text-slate-400" />
                  </Form>
                )}
              </Formik>
            </div>
          )}
        </div>
      </div>

      <div
        className={`mb-5 flex w-full flex-row items-center justify-start px-2 sm:px-5 ${
          !(topkResult && !isSearching) && !forceWhatsNext ? 'opacity-40' : ''
        }`}
      >
        <div className="flex w-full flex-col items-start justify-start gap-x-4 gap-y-1.5 rounded px-2 py-1 sm:flex-row">
          <span className="w-[105px] min-w-[105px] max-w-[105px] whitespace-nowrap rounded-full bg-slate-100 px-3 py-1 text-center text-[10px] font-bold uppercase text-slate-600">
            🎁 Next
          </span>
          {!(topkResult && !isSearching) && !forceWhatsNext ? (
            <div className="flex flex-row items-center justify-center text-sm font-medium text-slate-400">
              Unlocked after doing the demo, or{' '}
              <button
                type="button"
                onClick={() => setForceWhatsNext(true)}
                className="ml-1 px-0 py-0 font-medium leading-none text-slate-400 underline hover:bg-slate-300"
              >
                skip demo
              </button>
              .
            </div>
          ) : (
            <div className="flex w-full flex-col items-start justify-start text-sm font-medium text-slate-500">
              <div className="mb-3 text-sm">
                Now we know how to find features that are activated in texts.{' '}
                <strong className="">
                  But how do features get their labels/names? And how do we know the labels are correct?
                </strong>
              </div>
              <button
                type="button"
                onClick={() => {
                  tabUpdater('analyze');
                }}
                className="flex min-w-[160px] cursor-pointer flex-row gap-x-2 rounded-full border border-gYellow bg-white px-5 py-2.5 text-sm font-medium text-gYellow shadow transition-all hover:scale-105 hover:bg-gYellow/10"
              >
                <BrainCircuit className="h-5 w-5" /> Next - Analyze Features
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
