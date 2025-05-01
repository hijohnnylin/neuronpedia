'use client';

import LayerSelector from '@/components/feature-selector/layer-selector';
import ModelSelector from '@/components/feature-selector/model-selector';
import SourceSetSelector from '@/components/feature-selector/sourceset-selector';
import { useGlobalContext } from '@/components/provider/global-provider';
import { LoadingSquare } from '@/components/svg/loading-square';
import { DEFAULT_MODELID, DEFAULT_SOURCE, DEFAULT_SOURCESET, NEXT_PUBLIC_SEARCH_TOPK_MAX_CHAR_LENGTH } from '@/lib/env';
import useWindowSize from '@/lib/hooks/use-window-size';
import { SearchTopKResult } from '@/lib/utils/inference';
import { INFERENCE_EXAMPLE_TEXTS } from '@/lib/utils/inference-example-texts';
import { getFirstSourceSetForModel, getSourceSetNameFromSource } from '@/lib/utils/source';
import { Visibility } from '@prisma/client';
import * as Slider from '@radix-ui/react-slider';
import * as ToggleGroup from '@radix-ui/react-toggle-group';
import { Form, Formik, FormikProps } from 'formik';
import { Dices, Search, SearchIcon, X } from 'lucide-react';
import { NeuronWithPartialRelations } from 'prisma/generated/zod';
import { useEffect, useRef, useState } from 'react';
import ReactTextareaAutosize from 'react-textarea-autosize';

const DEFAULT_DENSITY_THRESHOLD_CLIENT = 0.0075;
const DENSITY_SLIDER_MAX = 100;
const DENSITY_SLIDER_POWER = 3;
const valueToPosition = (value: number) => DENSITY_SLIDER_MAX * value ** (1 / DENSITY_SLIDER_POWER);
const positionToValue = (position: number) => (position / DENSITY_SLIDER_MAX) ** DENSITY_SLIDER_POWER;

type TopKFeature = {
  index: number;
  feature: NeuronWithPartialRelations;
  activation_value: number;
  frequency: number;
};

export default function SearchTopkByToken({
  initialModelId,
  initialSource,
  initialText,
  initialDensityThreshold,
  initialIgnoreBos,
  initialSortBy,
  filterModelsToRelease,
  hideSettings = false,
  showResultsInNewPage = false,
}: {
  initialModelId?: string;
  initialSource?: string;
  initialText?: string;
  initialDensityThreshold?: number;
  initialIgnoreBos?: boolean;
  initialSortBy?: 'frequency' | 'strength' | 'density';
  filterModelsToRelease?: string;
  hideSettings?: boolean;
  showResultsInNewPage?: boolean;
}) {
  // Use a ref to track initialization state - won't trigger re-renders
  const isInitializedRef = useRef(false);

  const [modelId, setModelId] = useState(initialModelId || DEFAULT_MODELID);
  const [sourceSet, setSourceSet] = useState<string>(
    initialSource ? getSourceSetNameFromSource(initialSource) : DEFAULT_SOURCESET,
  );
  const [source, setSource] = useState<string>(initialSource || DEFAULT_SOURCE);
  const [searchQuery, setSearchQuery] = useState<string>(initialText || '');
  const [topkResult, setTopkResult] = useState<SearchTopKResult | undefined>();
  const [topkFeatures, setTopkFeatures] = useState<TopKFeature[] | undefined>(undefined);
  const [hoveredTokenPosition, setHoveredTokenPosition] = useState<number>(-1);
  const [lockedTokenPosition, setLockedTokenPosition] = useState<number>(-1);
  const [hoveredNeuronIndex, setHoveredNeuronIndex] = useState<number>(-1);
  const [isSearching, setIsSearching] = useState(false);
  const [needsReloadSearch, setNeedsReloadSearch] = useState(false);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const { windowSize } = useWindowSize();
  const [sortBy, setSortBy] = useState<'frequency' | 'strength' | 'density'>(initialSortBy ?? 'frequency');
  const {
    getFirstSourceForSourceSet,
    globalModels,
    showToastServerError,
    setFeatureModalFeature,
    setFeatureModalOpen,
  } = useGlobalContext();
  const [maxAct, setMaxAct] = useState<number>(0);
  const [densityThreshold, setDensityThreshold] = useState<number>(
    initialDensityThreshold ?? DEFAULT_DENSITY_THRESHOLD_CLIENT,
  );
  const [ignoreBos, setIgnoreBos] = useState<boolean>(initialIgnoreBos ?? true);
  const formRef = useRef<
    FormikProps<{
      searchQuery: string;
    }>
  >(null);

  async function searchClicked(overrideModelId?: string, overrideSource?: string, overrideText?: string) {
    const modelIdToUse = overrideModelId || modelId;
    const sourceToUse = overrideSource || source;
    const textToUse = overrideText || formRef.current?.values.searchQuery || '';
    if (showResultsInNewPage && textToUse.length > 0) {
      window.location.href = `/search-topk-by-token?modelId=${modelIdToUse}&source=${sourceToUse}&text=${textToUse}`;
      return;
    }
    setIsSearching(true);
    setNeedsReloadSearch(false);
    setLockedTokenPosition(-1);
    setHoveredTokenPosition(-1);
    setSearchQuery(textToUse);
    if (textToUse.length > NEXT_PUBLIC_SEARCH_TOPK_MAX_CHAR_LENGTH) {
      alert(
        `We currently support a maximum of ${NEXT_PUBLIC_SEARCH_TOPK_MAX_CHAR_LENGTH} characters in a search query. Your query is ${textToUse.length} characters long.`,
      );
      setIsSearching(false);
      return;
    }
    const result = await fetch(`/api/search-topk-by-token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        modelId: modelIdToUse,
        text: textToUse,
        source: sourceToUse,
        ignoreBos,
        densityThreshold,
      }),
    });
    setNeedsReloadSearch(false);
    if (result.status === 429 || result.status === 405) {
      alert('Sorry, we are limiting each user to 100 search requests per hour. Please try again later.');
    } else if (result.status !== 200) {
      showToastServerError();
    } else {
      const res = await result.json();
      const resultData = res as SearchTopKResult;
      // update the url params with the current modelId, source, and searchQuery
      const url = new URL(window.location.href);
      url.searchParams.set('modelId', modelIdToUse);
      url.searchParams.set('source', sourceToUse);
      url.searchParams.set('text', textToUse);
      url.searchParams.set('densityThreshold', densityThreshold.toString());
      url.searchParams.set('ignoreBos', ignoreBos.toString());
      url.searchParams.set('sortBy', sortBy);
      window.history.replaceState({}, '', url.toString());
      setTopkResult(resultData);
    }
    setIsSearching(false);
  }

  function getSortedFeatures(feats: TopKFeature[]) {
    if (sortBy === 'strength') {
      return feats.toSorted((a, b) => b.activation_value - a.activation_value);
    }
    if (sortBy === 'frequency') {
      return feats.toSorted((a, b) => b.frequency - a.frequency);
    }
    if (sortBy === 'density') {
      return feats.toSorted((a, b) => b.feature.frac_nonzero - a.feature.frac_nonzero);
    }
    return feats;
  }

  useEffect(() => {
    if (topkResult) {
      let toSet: TopKFeature[] = [];
      topkResult.results.forEach((result) => {
        result.topFeatures.forEach((feature) => {
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
  }, [sortBy]);

  useEffect(() => {
    if (scrollRef.current && windowSize.width && windowSize.width < 640) {
      scrollRef.current.scrollTo(0, 0);
    }
  }, [topkFeatures, lockedTokenPosition, hoveredTokenPosition]);

  const sourceSetChangedCallback = (newSourceSet: string) => {
    setSourceSet(newSourceSet);
    // Only run the side effect if we're past initialization
    if (isInitializedRef.current) {
      const sourceId = getFirstSourceForSourceSet(modelId, newSourceSet);
      if (sourceId) {
        setSource(sourceId);
      }
    }
  };

  useEffect(() => {
    // Skip the first render - only run on subsequent changes
    if (isInitializedRef.current) {
      sourceSetChangedCallback(sourceSet);
    }
  }, [sourceSet]);

  useEffect(() => {
    if (searchQuery.length > 0) {
      setNeedsReloadSearch(true);
    }
  }, [ignoreBos, densityThreshold, source, modelId, sourceSet]);

  const modelIdChangedCallback = (newModelId: string) => {
    if (globalModels[newModelId]) {
      setModelId(newModelId);
      // Only run the side effect if we're past initialization
      if (isInitializedRef.current) {
        const sourceSetFirst = getFirstSourceSetForModel(
          globalModels[newModelId],
          Visibility.PUBLIC,
          true,
          false,
          true,
        );
        setSourceSet(sourceSetFirst?.name || '');
      }
    } else {
      alert('not initialized');
    }
  };

  useEffect(() => {
    // Skip the first render - only run on subsequent changes
    if (isInitializedRef.current) {
      modelIdChangedCallback(modelId);
    }
  }, [modelId]);

  // Initialization effect - runs once on mount
  useEffect(() => {
    // Handle initial URL parameters if provided
    if (
      initialModelId !== undefined &&
      initialSource !== undefined &&
      initialText !== undefined &&
      initialText.length > 0
    ) {
      setModelId(initialModelId);
      setSourceSet(getSourceSetNameFromSource(initialSource));
      setSource(initialSource);
      setSearchQuery(initialText);
      if (initialDensityThreshold !== undefined) {
        setDensityThreshold(initialDensityThreshold);
      }
      if (initialIgnoreBos !== undefined) {
        setIgnoreBos(initialIgnoreBos);
      }
      if (initialSortBy !== undefined) {
        setSortBy(initialSortBy);
      }
      searchClicked(initialModelId, initialSource, initialText);
    }

    // Mark initialization as complete - for all future renders
    isInitializedRef.current = true;
  }, []);

  return (
    <div className="mt-0 flex h-full w-full flex-col items-center justify-center gap-y-0 pt-0 sm:flex-row">
      <div className="flex w-full max-w-screen-md flex-col sm:h-full sm:flex-1">
        <div className="flex w-full flex-row items-center justify-start gap-x-2">
          <ModelSelector
            filterToRelease={filterModelsToRelease}
            modelId={modelId}
            filterToInferenceEnabled
            modelIdChangedCallback={modelIdChangedCallback}
          />
          <SourceSetSelector
            modelId={modelId}
            sourceSet={sourceSet}
            filterToOnlyVisible
            filterToInferenceEnabled
            sourceSetChangedCallback={sourceSetChangedCallback}
            filterToAllowInferenceSearch
          />
          <LayerSelector
            modelId={modelId}
            layer={source}
            sourceSet={sourceSet}
            filterToInferenceEnabled
            layerChangedCallback={setSource}
          />
        </div>
        <div id="searchfield" className="mb-2 mt-2 flex w-full flex-row items-center justify-center">
          <Formik
            innerRef={formRef}
            initialValues={{ searchQuery: initialText || '' }}
            onSubmit={async () => {
              await searchClicked();
            }}
          >
            {({ values, setFieldValue }) => (
              <Form className="relative flex w-full flex-col gap-x-1.5 gap-y-1.5 sm:flex-row">
                <ReactTextareaAutosize
                  name="searchQuery"
                  disabled={isSearching}
                  value={values.searchQuery}
                  minRows={2}
                  onChange={(e) => {
                    if (topkResult) {
                      setNeedsReloadSearch(true);
                    }
                    setFieldValue('searchQuery', e.target.value);
                    setSearchQuery(e.target.value);
                  }}
                  required
                  placeholder={`Make ${modelId} think about...`}
                  className="mt-0 min-w-[10px] flex-1 resize-none rounded-xl border border-slate-300 px-3 py-3 pl-9 pr-9 text-left text-sm font-medium text-slate-700 placeholder-slate-400 shadow-sm transition-all hover:shadow-lg focus:border-sky-700 focus:shadow-lg focus:outline-none focus:ring-0 disabled:bg-slate-200 sm:text-[13px]"
                />
                <Search className="absolute left-3 my-auto h-full w-4 text-slate-400" />
                {values.searchQuery.length > 0 && (
                  <div className="absolute right-3 my-auto flex h-full items-center justify-center">
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

        <div id="imfeelinglucky" className="mb-2 mt-0 flex w-full flex-row items-center justify-between">
          <button
            type="button"
            disabled={isSearching}
            onClick={async (e) => {
              e.preventDefault();
              const randomSentence =
                INFERENCE_EXAMPLE_TEXTS[Math.floor(Math.random() * INFERENCE_EXAMPLE_TEXTS.length)];
              formRef.current?.setFieldValue('searchQuery', randomSentence);
              await searchClicked(undefined, undefined, randomSentence);
            }}
            className="flex min-w-[140px] cursor-pointer flex-row items-center justify-center gap-x-2 rounded-full border border-sky-700 bg-white px-5 py-2 text-[10px] font-medium text-sky-700 shadow transition-all hover:scale-105 hover:bg-sky-700/20 disabled:opacity-50"
          >
            <Dices className="h-5 w-5 text-sky-700" />
            RANDOM
          </button>
          <button
            type="button"
            onClick={() => {
              searchClicked();
            }}
            disabled={
              isSearching ||
              searchQuery.length === 0 ||
              (searchQuery.length > 0 && !needsReloadSearch && topkResult && topkResult.results.length > 0)
            }
            className="flex min-w-[140px] flex-row items-center justify-center gap-x-1.5 rounded-full bg-sky-700 px-5 py-2 text-[10px] font-medium text-white shadow transition-all enabled:hover:bg-sky-700/80 enabled:hover:text-white disabled:opacity-50"
          >
            <SearchIcon className="h-5 w-5 text-white" /> SEARCH
          </button>
        </div>

        <div className={`mt-4 w-full flex-col gap-y-2 ${hideSettings ? 'hidden' : 'flex'}`}>
          <div className="flex flex-col items-center justify-center gap-y-1 rounded-md border border-slate-300 bg-white px-5 py-5 pt-3">
            <div className="flex w-full flex-row items-center justify-between">
              <label htmlFor="density-slider" className="text-[11px] font-medium text-slate-500">
                Density Threshold: <span className="font-mono">{(densityThreshold * 100).toFixed(2)}%</span>
              </label>
              <button
                type="button"
                onClick={() => setDensityThreshold(DEFAULT_DENSITY_THRESHOLD_CLIENT)}
                className="text-[10px] font-medium text-sky-700 hover:text-sky-800"
              >
                Reset
              </button>
            </div>
            <div className="relative w-full pt-0">
              <div className="absolute mt-1.5 h-12 w-full">
                {[0, 0.01, 0.1, 1].map((value) => {
                  const position = valueToPosition(value);
                  const percentage = value * 100;
                  return (
                    <div
                      key={value}
                      className="absolute top-0 flex h-full -translate-x-1/2 flex-col items-center"
                      style={{ left: `${position}%` }}
                    >
                      <div className="h-5 w-px bg-slate-300" />
                      <span className="mt-1 h-4 select-none text-[9px] text-slate-400">{`${percentage}%`}</span>
                    </div>
                  );
                })}
              </div>

              <Slider.Root
                id="density-slider"
                className="relative flex h-8 w-full touch-none select-none items-center"
                defaultValue={[valueToPosition(densityThreshold)]} // Use helper
                value={[valueToPosition(densityThreshold)]} // Use helper
                onValueChange={(position) => {
                  let value = Number(positionToValue(position[0]).toFixed(4));
                  if (value === 0) {
                    value = 0.0001;
                  } else if (value > 0.9999) {
                    value = 0.9999;
                  }
                  setDensityThreshold(Number(value.toFixed(4))); // Use helper, rounded to 4 decimal places
                }}
                min={0}
                max={DENSITY_SLIDER_MAX}
                step={0.1} // Finer step for position
                aria-label="Density Threshold"
              >
                <Slider.Track className="relative h-1 flex-grow rounded-full bg-slate-200">
                  <Slider.Range className="absolute h-full rounded-full bg-sky-700" />
                </Slider.Track>
                <Slider.Thumb className="block h-4 w-4 cursor-pointer rounded-full border-0 bg-sky-700 shadow-md ring-0 focus:outline-none focus:ring-0 focus:ring-sky-500" />
              </Slider.Root>
            </div>
          </div>

          <ToggleGroup.Root
            className="inline-flex flex-1 overflow-hidden rounded-md border border-slate-300 bg-slate-300 px-0 py-0"
            type="single"
            defaultValue={ignoreBos ? 'ignore' : 'show'}
            value={ignoreBos ? 'ignore' : 'show'}
            onValueChange={(value) => {
              setIgnoreBos(value === 'ignore');
            }}
            aria-label="Show results where the max token was the BOS token"
          >
            <ToggleGroup.Item
              key="show"
              className="flex-1 rounded-r-md px-1 py-2.5 text-[10px] font-medium leading-none text-slate-500 transition-all hover:bg-slate-100 data-[state=on]:bg-white data-[state=on]:text-slate-600 sm:px-4 sm:text-[11px]"
              value="show"
              aria-label="show"
            >
              Show BOS
            </ToggleGroup.Item>
            <ToggleGroup.Item
              key="ignore"
              className="flex-1 rounded-l-md px-1 py-2.5 text-[10px] font-medium leading-none text-slate-500 transition-all hover:bg-slate-100 data-[state=on]:bg-white data-[state=on]:text-slate-600 sm:px-4 sm:text-[11px]"
              value="ignore"
              aria-label="ignore"
            >
              Hide BOS
            </ToggleGroup.Item>
          </ToggleGroup.Root>

          <ToggleGroup.Root
            className="inline-flex flex-1 overflow-hidden rounded-md border border-slate-300 bg-slate-300 px-0 py-0"
            type="single"
            defaultValue={sortBy}
            value={sortBy}
            onValueChange={(value) => {
              setSortBy(value as 'frequency' | 'strength' | 'density');
            }}
            aria-label="Sort neurons by frequency or max activation"
          >
            <ToggleGroup.Item
              key="frequency"
              className="flex-1 rounded-r-md px-1 py-2.5 text-[10px] font-medium leading-none text-slate-500 transition-all hover:bg-slate-100 data-[state=on]:bg-white data-[state=on]:text-slate-600 sm:px-4 sm:text-[11px]"
              value="frequency"
              aria-label="frequency"
            >
              Sort by Frequency
            </ToggleGroup.Item>
            <ToggleGroup.Item
              key="strength"
              className="flex-1 rounded-md px-1 py-2.5 text-[10px] font-medium leading-none text-slate-500 transition-all hover:bg-slate-100 data-[state=on]:bg-white data-[state=on]:text-slate-600 sm:px-4 sm:text-[11px]"
              value="strength"
              aria-label="strength"
            >
              Sort by Max Act
            </ToggleGroup.Item>
            <ToggleGroup.Item
              key="density"
              className="flex-1 rounded-l-md px-1 py-2.5 text-[10px] font-medium leading-none text-slate-500 transition-all hover:bg-slate-100 data-[state=on]:bg-white data-[state=on]:text-slate-600 sm:px-4 sm:text-[11px]"
              value="density"
              aria-label="density"
            >
              Sort by Density
            </ToggleGroup.Item>
          </ToggleGroup.Root>
        </div>

        {needsReloadSearch && (
          <div className="mt-4 flex w-full flex-row items-center justify-center gap-x-3 rounded-md bg-slate-100 py-2 text-[13px] text-slate-600">
            Search parameters changed.{' '}
            <button
              type="button"
              className="rounded-md bg-sky-700 px-3 py-2 text-[10px] font-medium uppercase leading-none text-white transition-all hover:bg-sky-900"
              onClick={() => {
                searchClicked();
              }}
            >
              Reload Search
            </button>
          </div>
        )}
      </div>
      <div
        className={`flex h-full flex-col pt-5 transition-all sm:min-h-full sm:pt-0 ${
          !topkResult && !isSearching ? 'w-[0px] max-w-[0px] overflow-hidden' : 'flex-1 sm:pl-5'
        }`}
      >
        <div id="tokens" className="flex flex-col">
          <div className={`flex w-full flex-col justify-center ${isSearching ? 'flex-1' : ''}`}>
            {isSearching && (
              <div className="mt-5 flex w-full flex-1 flex-row items-center justify-center">
                <div className="mb-1 flex items-center justify-center px-0 font-bold text-slate-300">
                  <LoadingSquare size={36} className="text-sky-700" />
                </div>
              </div>
            )}
          </div>
          {topkResult && !isSearching && (
            <div className="mt-0 flex w-full flex-row flex-wrap items-center justify-start gap-x-1 sm:justify-center">
              {topkResult.results &&
                topkResult.results.map((result, i) => (
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
                          ? 'rgba(2,132,199,1)'
                          : hoveredTokenPosition === result.position
                            ? 'rgba(2,132,199, 0.5)'
                            : result.topFeatures.filter((f) => f.featureIndex === hoveredNeuronIndex).length > 0
                              ? 'rgba(2,132,199, 0.5)'
                              : lockedTokenPosition === -1 && hoveredTokenPosition === -1 && hoveredNeuronIndex === -1
                                ? `rgba(2,132,199,${
                                    result.topFeatures && result.topFeatures.length > 0
                                      ? Math.min((result.topFeatures[0].activationValue / maxAct) ** 2, 0.8)
                                      : '0'
                                  })`
                                : 'rgba(0,0,0,0)',
                      borderColor:
                        lockedTokenPosition === result.position
                          ? 'rgba(2,132,199,1)'
                          : hoveredTokenPosition === result.position
                            ? 'rgba(2,132,199, 0.5)'
                            : 'rgba(2,132,199, 0.25',
                    }}
                    className={`mb-[2px] inline-block cursor-pointer select-none rounded border px-[5px] py-[1px] text-sm font-normal text-slate-800 transition-all sm:mb-1 ${
                      lockedTokenPosition === result.position
                        ? 'text-white'
                        : result.topFeatures.filter((f) => f.featureIndex === hoveredNeuronIndex).length > 0
                          ? ''
                          : 'hover:text-sky-700'
                    } ${result.token.endsWith(' ') && result.token.length > 1 ? 'pr-2' : ''} ${result.token.startsWith(' ') && result.token.length > 1 ? 'pl-2' : ''}`}
                  >
                    {result.token.trim().length === 0 ? ' ' : result.token}
                  </button>
                ))}
            </div>
          )}
        </div>
        <div ref={scrollRef} className="mt-3 flex w-full flex-1 flex-col overflow-y-hidden transition-all">
          {topkResult && !isSearching && (
            <div className="flex h-full max-h-full flex-1 flex-col items-center justify-center">
              <div className="mt-0 flex w-full flex-col gap-y-2 pb-24 sm:h-full sm:max-h-full sm:overflow-y-scroll">
                {lockedTokenPosition === -1 && hoveredTokenPosition === -1
                  ? topkFeatures?.map((f, i) => (
                      <button
                        key={i}
                        type="button"
                        onMouseEnter={() => {
                          setHoveredNeuronIndex(f.index);
                        }}
                        onMouseLeave={() => {
                          setHoveredNeuronIndex(-1);
                        }}
                        onClick={() => {
                          setFeatureModalFeature(f.feature as NeuronWithPartialRelations);
                          setFeatureModalOpen(true);
                        }}
                        className="group relative flex w-full flex-row items-center justify-center gap-x-3 rounded-xl bg-sky-700/5 py-0 pl-5 pr-5 text-sky-800 transition-all hover:bg-sky-700/20 sm:py-2.5"
                      >
                        <div className="flex flex-1 flex-col items-start justify-start py-1">
                          <div className="w-full text-left text-[13px] font-semibold transition-all group-hover:text-sky-700 sm:text-[13px]">
                            {f.feature &&
                              f.feature.explanations &&
                              f.feature.explanations.length > 0 &&
                              f.feature.explanations[0].description}
                          </div>
                          <span className="mb-0 rounded-md pt-1 text-left text-[10px] font-bold uppercase text-slate-400">
                            {f.feature.layer}:{f.feature.index}
                          </span>
                        </div>
                        <div className="ml-0.5 flex flex-col items-center justify-center gap-y-0 overflow-visible py-2 font-mono text-xs font-bold text-slate-600">
                          {f.frequency}
                          <div className="font-sans text-[8px] text-slate-400">TOKENS</div>
                        </div>
                        <div className="ml-0.5 flex w-11 flex-col items-center justify-center gap-y-0 overflow-visible py-2 font-mono text-xs font-bold text-slate-600">
                          {f.activation_value.toFixed(2)}
                          <div className="font-sans text-[8px] text-slate-400">MAX ACT</div>
                        </div>
                        <div className="ml-0.5 flex flex-col items-center justify-center gap-y-0 overflow-visible py-2 font-mono text-xs font-bold text-slate-600">
                          {(f.feature.frac_nonzero * 100).toFixed(2)}%
                          <div className="font-sans text-[8px] text-slate-400">DENSITY</div>
                        </div>
                      </button>
                    ))
                  : topkResult.results[
                      lockedTokenPosition > -1 ? lockedTokenPosition : hoveredTokenPosition
                    ].topFeatures.map((f, i) => (
                      <button
                        key={i}
                        type="button"
                        onClick={() => {
                          setFeatureModalFeature(f.feature as NeuronWithPartialRelations);
                          setFeatureModalOpen(true);
                        }}
                        className="group relative flex w-full flex-row items-center justify-center gap-x-3 rounded-xl bg-sky-700/5 py-0 pl-5 pr-5 text-sky-800 transition-all hover:bg-sky-700/20 sm:py-2.5"
                      >
                        <div className="flex flex-1 flex-col items-start justify-start py-1">
                          <div className="w-full text-left text-[13px] font-semibold transition-all group-hover:text-sky-700 sm:text-[13px]">
                            {f.feature &&
                              f.feature.explanations &&
                              f.feature.explanations.length > 0 &&
                              f.feature.explanations[0].description}
                          </div>
                          <span className="mb-0 rounded-md pt-1 text-left text-[10px] font-bold uppercase text-slate-400">
                            {f.feature?.layer}:{f.feature?.index}
                          </span>
                        </div>
                        <div className="ml-1 flex w-11 flex-col items-center justify-center gap-y-0 overflow-visible py-2 font-mono text-xs font-bold text-slate-600">
                          {f.activationValue.toFixed(2)}
                          <div className="font-sans text-[8px] text-slate-400">MAX ACT</div>
                        </div>
                      </button>
                    ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
