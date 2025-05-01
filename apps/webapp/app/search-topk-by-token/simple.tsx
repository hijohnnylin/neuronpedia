'use client';

// This is a one-time use for Gemma Scope
// TODO: duplicated code with normal topk by token search - refactor

import LayerSelector from '@/components/feature-selector/layer-selector';
import ModelSelector from '@/components/feature-selector/model-selector';
import SourceSetSelector from '@/components/feature-selector/sourceset-selector';
import { useGlobalContext } from '@/components/provider/global-provider';
import { LoadingSquare } from '@/components/svg/loading-square';
import { useIsMount } from '@/lib/hooks/use-is-mount';
import useWindowSize from '@/lib/hooks/use-window-size';
import { SearchTopKResult } from '@/lib/utils/inference';
import { INFERENCE_EXAMPLE_TEXTS } from '@/lib/utils/inference-example-texts';
import { getFirstSourceSetForModel, getSourceSetNameFromSource } from '@/lib/utils/source';
import { Visibility } from '@prisma/client';
import { Form, Formik, FormikProps } from 'formik';
import { Dices, ExternalLink, PlayIcon, Search, X } from 'lucide-react';
import { NeuronWithPartialRelations } from 'prisma/generated/zod';
import { useEffect, useRef, useState } from 'react';

const SEARCHER_TOPK_MAX_CHAR_LENGTH = 250;

type TopKFeature = {
  index: number;
  feature: NeuronWithPartialRelations;
  activation_value: number;
  frequency: number;
};

export default function SearchTopkByTokenSimple({
  initialModelId,
  initialLayer,
  filterModelsToRelease,
}: {
  initialModelId: string;
  initialLayer: string;
  filterModelsToRelease?: string;
}) {
  const { setFeatureModalFeature, setFeatureModalOpen } = useGlobalContext();
  const [topkResult, setTopkResult] = useState<SearchTopKResult | undefined>();
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
  const [modelId, setModelId] = useState(initialModelId);
  const [layer, setLayer] = useState(initialLayer);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [sortNeurons, setSortNeurons] = useState<'frequency' | 'strength'>('frequency');
  const { getFirstSourceForSourceSet, globalModels, showToastServerError } = useGlobalContext();
  const [maxAct, setMaxAct] = useState<number>(0);

  async function searchClicked(overrideText?: string) {
    setIsSearching(true);
    setLockedTokenPosition(-1);
    setHoveredTokenPosition(-1);
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
      const res = await result.json();
      const resultData = res as SearchTopKResult;

      // filter out boring indexes and ones with act desnsity > 1%
      resultData.results = resultData.results.map((resultDataItem) => {
        // eslint-disable-next-line
        resultDataItem.topFeatures = resultDataItem.topFeatures.filter(
          (feature) => feature.feature?.frac_nonzero !== undefined && feature.feature?.frac_nonzero <= 0.01,
        );
        return resultDataItem;
      });
      setTopkResult(resultData);
    }
    setIsSearching(false);
  }

  function getSortedFeatures(feats: TopKFeature[]) {
    if (sortNeurons === 'strength') {
      return feats.toSorted((a, b) => b.activation_value - a.activation_value);
    }
    if (sortNeurons === 'frequency') {
      return feats.toSorted((a, b) => b.frequency - a.frequency);
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

      // get the max activating value
      setMaxAct(Math.max(...toSet.map((f) => f.activation_value)));

      setTopkFeatures(getSortedFeatures(toSet));
    }
  }, [topkResult]);

  useEffect(() => {
    if (topkFeatures) {
      setTopkFeatures(getSortedFeatures(topkFeatures));
    }
  }, [sortNeurons]);

  const scrollRef = useRef<HTMLDivElement | null>(null);

  const { windowSize } = useWindowSize();

  const isMount = useIsMount();

  useEffect(() => {
    if (scrollRef.current && windowSize.width && windowSize.width < 640) {
      scrollRef.current.scrollTo(0, 0);
    }
  }, [topkFeatures, lockedTokenPosition, hoveredTokenPosition]);

  const [sourceSet, setSourceSet] = useState<string>(getSourceSetNameFromSource(initialLayer));

  const sourceSetChangedCallback = (newSourceSet: string) => {
    setSourceSet(newSourceSet);
    const sourceId = getFirstSourceForSourceSet(modelId, newSourceSet);
    if (sourceId && !isMount) {
      setLayer(sourceId);
    }
  };

  useEffect(() => {
    if (sourceSet) {
      sourceSetChangedCallback(sourceSet);
    }
  }, [sourceSet]);

  const modelIdChangedCallback = (newModelId: string) => {
    if (globalModels[newModelId]) {
      setModelId(newModelId);
      const sourceSetFirst = getFirstSourceSetForModel(globalModels[newModelId], Visibility.PUBLIC, true, false);
      setSourceSet(sourceSetFirst?.name || '');
    }
  };

  useEffect(() => {
    if (modelId) {
      modelIdChangedCallback(modelId);
    }
  }, [modelId]);

  return (
    <div className="mt-0 flex h-full w-full max-w-screen-xl flex-col items-center justify-center overflow-y-scroll pt-0 sm:pb-3">
      <div className="flex flex-row items-center justify-start gap-x-4">
        <ModelSelector
          filterToRelease={filterModelsToRelease}
          modelId={modelId}
          filterToInferenceEnabled
          modelIdChangedCallback={modelIdChangedCallback}
        />
        <SourceSetSelector
          modelId={modelId}
          sourceSet={sourceSet}
          filterToInferenceEnabled
          sourceSetChangedCallback={sourceSetChangedCallback}
        />
        <LayerSelector
          modelId={modelId}
          layer={layer}
          sourceSet={sourceSet}
          filterToInferenceEnabled
          layerChangedCallback={setLayer}
        />
      </div>
      <div
        className={`flex w-full flex-row items-center justify-start px-2 sm:px-5 ${
          !topkResult && !isSearching ? '' : ''
        }`}
      >
        <div className="flex w-full flex-col items-start justify-start gap-x-4 gap-y-1.5 rounded px-2 py-0 sm:mt-0 sm:flex-row">
          <div className="flex w-full flex-col items-center justify-start text-sm font-medium text-slate-500">
            <div id="searchfield" className="order-3 my-3 mt-4 flex w-full flex-row items-center justify-center">
              <Formik
                innerRef={formRef}
                initialValues={{ searchQuery: '' }}
                onSubmit={async () => {
                  await searchClicked();
                }}
              >
                {({ values, setFieldValue }) => (
                  <Form className="relative flex w-full max-w-[480px] flex-col gap-x-1.5 gap-y-1.5 sm:flex-row">
                    <input
                      name="searchQuery"
                      disabled={isSearching}
                      value={values.searchQuery}
                      maxLength={SEARCHER_TOPK_MAX_CHAR_LENGTH}
                      onChange={(e) => {
                        setFieldValue('searchQuery', e.target.value);
                      }}
                      required
                      placeholder={`Make ${modelId} think about...`}
                      className="mt-0 min-w-[10px] flex-1 resize-none rounded-xl border border-slate-300 px-3 py-3 pl-10 pr-8 text-left text-sm font-medium text-gBlue placeholder-slate-500 transition-all hover:shadow-lg focus:border-gBlue focus:shadow-lg focus:outline-none focus:ring-0 disabled:bg-slate-200 sm:text-[16px]"
                    />
                    <Search className="absolute left-3 my-auto h-full w-4 text-slate-400" />
                    {values.searchQuery.length > 0 && (
                      <div className="absolute right-2 my-auto flex h-full items-center justify-center">
                        <X
                          className="h-5 w-5 rounded-full bg-slate-200 p-1 hover:bg-slate-300"
                          onClick={() => {
                            setFieldValue('searchQuery', '');
                          }}
                        />
                      </div>
                    )}
                  </Form>
                )}
              </Formik>
            </div>
            <div
              id="imfeelinglucky"
              className="order-4 mb-0 mt-0 flex w-full max-w-[480px] flex-row items-center justify-between"
            >
              <button
                type="button"
                disabled={isSearching}
                onClick={async (e) => {
                  e.preventDefault();
                  const randomSentence =
                    INFERENCE_EXAMPLE_TEXTS[Math.floor(Math.random() * INFERENCE_EXAMPLE_TEXTS.length)];
                  formRef.current?.setFieldValue('searchQuery', randomSentence);
                  await searchClicked(randomSentence);
                }}
                className="flex min-w-[140px] cursor-pointer flex-row justify-center gap-x-2 rounded-full border border-gBlue bg-white px-5 py-2 text-[12px] font-medium text-gBlue shadow transition-all hover:scale-105 hover:bg-gBlue/20 disabled:opacity-50"
              >
                <Dices className="h-5 w-5 text-gBlue" />
                {`I'm Feeling Lucky`}
              </button>

              <button
                type="button"
                onClick={() => {
                  searchClicked();
                }}
                disabled={isSearching}
                className="flex min-w-[140px] flex-row items-center justify-center gap-x-1.5 rounded-full bg-gBlue px-5 py-2 text-[12px] font-medium text-white shadow transition-all enabled:hover:bg-gBlue/80 enabled:hover:text-white disabled:opacity-50"
              >
                <PlayIcon className="h-5 w-5 text-white" /> Go
              </button>
            </div>
            <div
              id="tokens"
              className="order-2 flex h-[28vh] max-h-[28vh] min-h-[28vh] max-w-[480px] flex-1 flex-col overflow-y-scroll pt-3"
            >
              <div className={`flex w-full flex-col justify-center ${isSearching ? 'flex-1' : ''}`}>
                {isSearching && (
                  <div className="mt-5 flex w-full flex-1 flex-row items-center justify-center">
                    <div className="mb-1 flex items-center justify-center px-0 font-bold text-slate-300">
                      <LoadingSquare size={32} className="text-sky-700" />
                    </div>
                  </div>
                )}
              </div>
              {topkResult && !isSearching && (
                <div className="flex w-full flex-col">
                  <div className="mt-0 flex flex-row flex-wrap items-center justify-start gap-x-1 sm:justify-center">
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
                            style={{
                              backgroundColor:
                                lockedTokenPosition === result.position
                                  ? 'rgba(66,133,244,1)'
                                  : hoveredTokenPosition === result.position
                                    ? 'rgba(66,133,244, 0.5)'
                                    : result.topFeatures.filter((f) => f.featureIndex === hoveredNeuronIndex).length > 0
                                      ? 'rgba(66,133,244, 0.5)'
                                      : lockedTokenPosition === -1 &&
                                          hoveredTokenPosition === -1 &&
                                          hoveredNeuronIndex === -1
                                        ? `rgba(66,133,244,${
                                            result.topFeatures && result.topFeatures.length > 0
                                              ? Math.min((result.topFeatures[0].activationValue / maxAct) ** 2, 0.8)
                                              : '0'
                                          })`
                                        : 'rgba(0,0,0,0)',
                            }}
                            className={`mb-0.5 inline-block cursor-pointer select-none rounded px-[5px] py-[5px] text-sm font-normal text-slate-800 transition-all sm:mb-2 sm:text-[20px] ${
                              lockedTokenPosition === result.position
                                ? 'text-white'
                                : result.topFeatures.filter((f) => f.featureIndex === hoveredNeuronIndex).length > 0
                                  ? ''
                                  : 'hover:text-gBlue'
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
            <div
              ref={scrollRef}
              className={`order-1 mt-3 flex h-[35vh] max-h-[35vh] min-h-[35vh] w-full flex-col overflow-y-hidden ${
                !topkResult && !isSearching && 'items-center justify-end'
              }`}
            >
              {!topkResult && !isSearching && (
                <div className="flex flex-col items-center justify-center">
                  <div className="mb-3 text-5xl">🛝</div>
                  <div className="px-2 text-center">{`Enter something below to see what features activate, or try I'm Feeling Lucky.`}</div>
                </div>
              )}
              {topkResult && !isSearching && (
                // eslint-disable-next-line
                <>
                  {lockedTokenPosition === -1 && hoveredTokenPosition === -1 ? (
                    <div className="flex h-full flex-1 flex-col items-center justify-center">
                      <div className="mt-0 flex w-full max-w-[480px] flex-col gap-y-2 overflow-y-hidden sm:h-full sm:max-h-full sm:overflow-y-scroll">
                        {topkFeatures?.map((f, i) => (
                          <div
                            key={i}
                            onMouseEnter={() => {
                              setHoveredNeuronIndex(f.index);
                            }}
                            onMouseLeave={() => {
                              setHoveredNeuronIndex(-1);
                            }}
                            className="group relative flex w-full flex-row items-center justify-center gap-x-3 rounded-xl bg-gBlue/5 py-0 pl-5 pr-5 text-gBlue transition-all hover:bg-gBlue/20 sm:py-2.5"
                          >
                            <div className="flex flex-1 flex-col py-2">
                              <div className="w-full cursor-default text-[13px] transition-all group-hover:text-gBlue sm:text-[15px]">
                                {f.feature &&
                                  f.feature.explanations &&
                                  f.feature.explanations.length > 0 &&
                                  f.feature.explanations[0].description}
                              </div>
                            </div>
                            <div className="ml-1 flex max-w-7 flex-col items-center justify-center gap-y-1 overflow-visible py-2">
                              <ExternalLink
                                className="h-7 w-7 cursor-pointer rounded bg-white/50 p-1.5 hover:bg-white/90"
                                onClick={() => {
                                  setFeatureModalFeature(f.feature as NeuronWithPartialRelations);
                                  setFeatureModalOpen(true);
                                }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div className="flex w-full flex-1 flex-col items-center justify-start">
                      <div className="flex w-full max-w-[480px] flex-col gap-y-2 overflow-y-scroll sm:h-[500px] sm:max-h-[500px]">
                        {topkResult.results[
                          lockedTokenPosition > -1 ? lockedTokenPosition : hoveredTokenPosition
                        ].topFeatures.map((f, i) => (
                          <div
                            key={i}
                            className="group relative flex w-full cursor-default flex-row items-center justify-center gap-x-3 rounded-xl bg-gBlue/5 py-0 pl-5 pr-5 text-gBlue transition-all hover:bg-gBlue/20 sm:py-2.5"
                          >
                            <div className="flex flex-1 flex-col">
                              <div className="w-full cursor-default py-2 text-[13px] transition-all group-hover:text-gBlue sm:text-[15px]">
                                {f.feature &&
                                  f.feature.explanations &&
                                  f.feature.explanations.length > 0 &&
                                  f.feature.explanations[0].description}
                              </div>
                            </div>
                            <div className="ml-1 flex max-w-7 flex-col items-center justify-center gap-y-1 overflow-visible py-2">
                              <ExternalLink
                                className="h-7 w-7 cursor-pointer rounded bg-white/50 p-1.5 hover:bg-white/90"
                                onClick={() => {
                                  setFeatureModalFeature(f.feature as NeuronWithPartialRelations);
                                  setFeatureModalOpen(true);
                                }}
                              />
                            </div>
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
  );
}
