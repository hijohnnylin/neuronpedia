'use client';

import ActivationItem from '@/components/activation-item';
import ModelSelector from '@/components/feature-selector/model-selector';
import ReleaseSelector from '@/components/feature-selector/release-selector';
import SourceSetSelector from '@/components/feature-selector/sourceset-selector';
import FeatureStats from '@/components/feature-stats';
import PanelLoader from '@/components/panel-loader';
import { useGlobalContext } from '@/components/provider/global-provider';
import { Button } from '@/components/shadcn/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/shadcn/tabs';
import { env } from '@/lib/env';
import { replaceHtmlAnomalies } from '@/lib/utils/activations';
import { EXPLANATIONTYPE_HUMAN } from '@/lib/utils/autointerp';
import { SearchExplanationsResponse, SearchExplanationsType } from '@/lib/utils/general';
import { getAdditionalInfoFromSource, getLayerNumAsStringFromSource } from '@/lib/utils/source';
import { replaceSteerModelIdIfNeeded } from '@/lib/utils/steer';
import { ExplanationWithPartialRelations, NeuronWithPartialRelations } from '@/prisma/generated/zod';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import * as Select from '@radix-ui/react-select';
import * as ToggleGroup from '@radix-ui/react-toggle-group';
import { Form, Formik, FormikProps } from 'formik';
import { Check, ChevronDownIcon, Search } from 'lucide-react';
import { useRouter, useSearchParams } from 'next/navigation';
import { useEffect, useRef, useState } from 'react';
import InfiniteScroll from 'react-infinite-scroll-component';

export const MAX_SEARCH_EXP_QUERY_LENGTH_CHARS = 50;

enum ExplanationSearchState {
  DEFAULT,
  SEARCHING,
  LOADED,
}

export default function ExplanationsSearcher({
  initialSearchQuery,
  initialSelectedLayers = [],
  initialReleaseName,
  initialModelId,
  initialSourceSetName,
  initialDensityThreshold,
  filterToRelease,
  showTabs = false,
  defaultTab = SearchExplanationsType.BY_ALL,
  fromNav = false,
  showModelSelector = true,
  allowSourceSetChange = true,
  hideResultDetails = false,
  isSteerSearch = false,
  allowSteerSearchFullHeight = false,
  onClickResultCallback,
  filterToInferenceEnabled = false,
  neverChangePageOnSearch = false,
}: {
  initialSearchQuery?: string;
  initialSelectedLayers?: string[];
  initialReleaseName?: string;
  initialModelId?: string;
  initialSourceSetName?: string;
  initialDensityThreshold?: number;
  filterToRelease?: string | undefined;
  showTabs?: boolean;
  defaultTab?: SearchExplanationsType;
  fromNav?: boolean;
  showModelSelector?: boolean;
  allowSourceSetChange?: boolean;
  hideResultDetails?: boolean;
  isSteerSearch?: boolean;
  allowSteerSearchFullHeight?: boolean;
  onClickResultCallback?: (result: ExplanationWithPartialRelations) => void;
  filterToInferenceEnabled?: boolean;
  neverChangePageOnSearch?: boolean;
}) {
  const searchParams = useSearchParams();
  const router = useRouter();
  const loadResultsInNewPage = initialSearchQuery === undefined;
  const isEmbed = searchParams.get('embed') === 'true';
  const [searchState, setSearchState] = useState<ExplanationSearchState>(ExplanationSearchState.DEFAULT);
  const {
    getSourceSetsForModelId,
    getSourceSet,
    getSourcesForSourceSet,
    getInferenceEnabledSourcesForModel,
    getDefaultModel,
    showToastServerError,
    setFeatureModalFeature,
    setFeatureModalOpen,
  } = useGlobalContext();
  const [modelId, setModelId] = useState(
    initialModelId || env.NEXT_PUBLIC_DEFAULT_MODELID || getDefaultModel()?.id || env.NEXT_PUBLIC_DEFAULT_MODELID,
  );
  const [sourceSet, setSourceSet] = useState(
    initialSourceSetName ||
      (modelId
        ? getSourceSetsForModelId(modelId).length > 0
          ? getSourceSetsForModelId(modelId)[0].name
          : env.NEXT_PUBLIC_DEFAULT_SOURCESET
        : env.NEXT_PUBLIC_DEFAULT_SOURCESET),
  );
  const [selectedLayers, setSelectedLayers] = useState<string[] | undefined>(initialSelectedLayers);
  const [needsReloadSearch, setNeedsReloadSearch] = useState(false);
  const [showDashboards, setShowDashboards] = useState(true);
  const [selectedTab, setSelectedTab] = useState<SearchExplanationsType>(defaultTab);
  const [selectedRelease, setSelectedRelease] = useState<string>(
    initialReleaseName || filterToRelease || env.NEXT_PUBLIC_DEFAULT_RELEASE_NAME,
  );
  const [loadedResults, setLoadedResults] = useState<ExplanationWithPartialRelations[]>([]);
  const [hasMore, setHasMore] = useState(true);
  const formRef = useRef<
    FormikProps<{
      searchQuery: string;
    }>
  >(null);
  // TODO: density threshold filtering
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [densityThreshold, setDensityThreshold] = useState<number>(initialDensityThreshold || 0.01);

  function makeUrl(query: string) {
    if (selectedTab === SearchExplanationsType.BY_RELEASE) {
      return `/search-explanations/?releaseName=${selectedRelease}&q=${encodeURIComponent(query)}${
        isEmbed ? '&embed=true' : ''
      }`;
    }
    if (selectedTab === SearchExplanationsType.BY_MODEL) {
      return `/search-explanations/?modelId=${modelId}&q=${encodeURIComponent(query)}${isEmbed ? '&embed=true' : ''}`;
    }
    if (selectedTab === SearchExplanationsType.BY_SOURCE) {
      let urlSourcesToSet = selectedLayers;
      if (!selectedLayers || selectedLayers.length === 0) {
        urlSourcesToSet = getSourcesForSourceSet(modelId, sourceSet, true, false, false);
      }
      return `/search-explanations/?modelId=${modelId}&sources=${JSON.stringify(urlSourcesToSet)}&q=${encodeURIComponent(
        query,
      )}${isEmbed ? '&embed=true' : ''}`;
    }
    if (selectedTab === SearchExplanationsType.BY_ALL) {
      return `/search-explanations/?q=${encodeURIComponent(query)}${isEmbed ? '&embed=true' : ''}`;
    }
    return '';
  }

  const sourceSetChanged = (newSourceSet: string) => {
    setSourceSet(newSourceSet);
    setSelectedLayers([]);
  };

  const modelIdChanged = (newModelId: string) => {
    setModelId(newModelId);
    const newSourceSet = getSourceSetsForModelId(newModelId).sort((a, b) => a.name.localeCompare(b.name))[0].name;
    setSourceSet(newSourceSet);
    setSelectedLayers([]);
  };

  function callSearchApi(uri: string, body: any, isFetchMore = false) {
    if (!isFetchMore) {
      // only clear results if we're not fetching more
      setLoadedResults([]);
    }
    setSearchState(ExplanationSearchState.SEARCHING);
    fetch(uri, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
      .then((response) => response.json())
      .then((resp: SearchExplanationsResponse) => {
        setSearchState(ExplanationSearchState.LOADED);
        if (resp.results && resp.results.length > 0) {
          if (isFetchMore) {
            setLoadedResults((prevItems) => [...prevItems, ...resp.results]);
          } else {
            setLoadedResults(resp.results);
          }
          setHasMore(true);
        } else {
          setHasMore(false);
        }
      })
      .catch((error) => {
        showToastServerError();
        setSearchState(ExplanationSearchState.DEFAULT);
        console.error(error);
      });
  }

  function submitSearch(searchQuery: string, isFetchMore = false) {
    if (searchQuery.length <= 2) {
      alert('Search query must be at least 3 characters long');
      return;
    }
    if (searchQuery.length > MAX_SEARCH_EXP_QUERY_LENGTH_CHARS) {
      alert(`Must be shorter than ${MAX_SEARCH_EXP_QUERY_LENGTH_CHARS} characters`);
      return;
    }
    if (selectedTab === SearchExplanationsType.BY_SOURCE) {
      if (!selectedLayers) {
        alert('Please select at least one layer to search.');
        return;
      }
    }
    if (searchQuery.trim().length === 0) {
      alert('Please enter a search query.');
      return;
    }
    if (fromNav) {
      setSearchState(ExplanationSearchState.SEARCHING);
      window.location.href = makeUrl(searchQuery) || '';
      return;
    }
    if (loadResultsInNewPage && !neverChangePageOnSearch) {
      router.push(makeUrl(searchQuery) || '');
      return;
    }
    if (selectedTab === SearchExplanationsType.BY_SOURCE) {
      let layersToSend = selectedLayers;
      if (!layersToSend || layersToSend.length === 0) {
        layersToSend = getSourcesForSourceSet(modelId, sourceSet, true, false, false);
      }
      callSearchApi(
        `/api/explanation/search`,
        {
          query: searchQuery,
          modelId,
          layers: layersToSend,
          offset: loadedResults.length,
        },
        isFetchMore,
      );
    } else if (selectedTab === SearchExplanationsType.BY_RELEASE) {
      callSearchApi(
        `/api/explanation/search-release`,
        {
          query: searchQuery,
          releaseName: selectedRelease,
          offset: loadedResults.length,
        },
        isFetchMore,
      );
    } else if (selectedTab === SearchExplanationsType.BY_MODEL) {
      if (filterToInferenceEnabled) {
        let layersToSend = [];
        layersToSend = getInferenceEnabledSourcesForModel(modelId);
        callSearchApi(
          `/api/explanation/search`,
          {
            query: searchQuery,
            modelId: replaceSteerModelIdIfNeeded(modelId),
            layers: layersToSend,
            offset: loadedResults.length,
          },
          isFetchMore,
        );
      } else {
        callSearchApi(
          `/api/explanation/search-model`,
          {
            query: searchQuery,
            modelId,
            offset: loadedResults.length,
          },
          isFetchMore,
        );
      }
    } else if (selectedTab === SearchExplanationsType.BY_ALL) {
      callSearchApi(
        `/api/explanation/search-all`,
        {
          query: searchQuery,
          offset: loadedResults.length,
        },
        isFetchMore,
      );
    }
  }

  // update the URL with query param for the search
  useEffect(() => {
    if (!isSteerSearch) {
      if (searchState === ExplanationSearchState.LOADED && formRef.current) {
        window.history.pushState({}, '', makeUrl(formRef.current.values.searchQuery));
      } else {
        // remove query params
        window.history.pushState({}, '', document.location.toString().split(/[?#]/)[0]);
      }
    }
    if (searchState === ExplanationSearchState.SEARCHING) {
      setNeedsReloadSearch(false);
    }
  }, [searchState]);

  useEffect(() => {
    if (searchState === ExplanationSearchState.LOADED) {
      setNeedsReloadSearch(true);
    }
  }, [selectedLayers]);

  useEffect(() => {
    setNeedsReloadSearch(true);
  }, [selectedTab, selectedRelease, modelId]);

  // load query if we have one
  useEffect(() => {
    if (initialSearchQuery !== undefined) {
      submitSearch(initialSearchQuery);
      formRef.current?.setFieldValue('searchQuery', initialSearchQuery);
    }
  }, [initialSearchQuery]);

  return (
    <div className="flex w-full flex-col items-center justify-center gap-x-1.5">
      <div
        className={`z-10 flex w-full flex-col items-start justify-center transition-all sm:max-w-screen-lg ${!isEmbed && 'sticky'} ${isSteerSearch ? 'pt-0' : 'top-12 pt-2'}`}
      >
        <div className="flex w-full flex-col items-center justify-center">
          <Tabs value={selectedTab} className="min-w-[400px] max-w-[400px]">
            <TabsList className={`mb-0 w-full ${!showTabs && 'hidden'}`}>
              <TabsTrigger
                value={SearchExplanationsType.BY_ALL}
                className="flex-1"
                onClick={() => setSelectedTab(SearchExplanationsType.BY_ALL)}
              >
                All
              </TabsTrigger>
              <TabsTrigger
                value={SearchExplanationsType.BY_RELEASE}
                className="flex-1"
                onClick={() => setSelectedTab(SearchExplanationsType.BY_RELEASE)}
              >
                By Release
              </TabsTrigger>
              <TabsTrigger
                value={SearchExplanationsType.BY_MODEL}
                className="flex-1"
                onClick={() => setSelectedTab(SearchExplanationsType.BY_MODEL)}
              >
                By Model
              </TabsTrigger>
              <TabsTrigger
                value={SearchExplanationsType.BY_SOURCE}
                className="flex-1"
                onClick={() => setSelectedTab(SearchExplanationsType.BY_SOURCE)}
              >
                By Sources
              </TabsTrigger>
            </TabsList>
            <TabsContent value={SearchExplanationsType.BY_ALL} className="mb-4">
              <div className="flex flex-col items-center justify-center">
                <p className="mt-3 text-center text-sm font-medium text-slate-400">
                  {!env.IS_ACTUALLY_NEURONPEDIA_ORG
                    ? 'Search all features across all loaded models'
                    : 'Search 50,000,000+ features across 7 models'}
                </p>
              </div>
            </TabsContent>
            <TabsContent value={SearchExplanationsType.BY_RELEASE} className="mb-2">
              <ReleaseSelector
                defaultReleaseName={selectedRelease}
                onReleaseChange={(releaseName: string) => {
                  setSelectedRelease(releaseName);
                }}
              />
            </TabsContent>
            <TabsContent value={SearchExplanationsType.BY_MODEL} className="mb-2 flex flex-row justify-center">
              {showModelSelector && (
                <div className="mx-auto flex min-w-[300px] max-w-[300px] flex-col">
                  <ModelSelector
                    modelId={modelId}
                    modelIdChangedCallback={modelIdChanged}
                    filterToRelease={filterToRelease}
                  />
                </div>
              )}
            </TabsContent>
            <TabsContent value={SearchExplanationsType.BY_SOURCE} className="mb-2">
              <div className="flex w-full flex-row items-center justify-center gap-x-2 text-center font-sans text-xs font-medium uppercase leading-none text-slate-500">
                {showModelSelector && (
                  <div className="flex flex-col font-mono">
                    <ModelSelector
                      modelId={modelId}
                      modelIdChangedCallback={modelIdChanged}
                      filterToRelease={filterToRelease}
                    />
                  </div>
                )}
                {allowSourceSetChange && (
                  <SourceSetSelector
                    modelId={modelId}
                    sourceSet={sourceSet}
                    filterToInferenceEnabled={filterToInferenceEnabled}
                    sourceSetChangedCallback={sourceSetChanged}
                    filterToRelease={filterToRelease}
                    filterToOnlyVisible
                    filterToOnlyHasDashboards
                  />
                )}
                <div className="flex flex-col">
                  <DropdownMenu.Root>
                    <DropdownMenu.Trigger className="flex h-10 max-h-[40px] min-h-[40px] w-full flex-1 flex-row items-center justify-center gap-x-1 whitespace-pre rounded border border-slate-300 bg-white px-3 font-mono text-xs font-medium leading-tight text-sky-700 hover:bg-slate-50 focus:outline-none sm:pl-4 sm:pr-2">
                      {selectedLayers
                        ? selectedLayers?.length === 0 ||
                          selectedLayers?.length ===
                            getSourcesForSourceSet(modelId, sourceSet, true, false, false).length
                          ? 'All Layers'
                          : `Layer${selectedLayers.length > 1 ? 's ' : ' '}${selectedLayers
                              .map(
                                (l) =>
                                  getLayerNumAsStringFromSource(l) +
                                  (getAdditionalInfoFromSource(l).length > 0
                                    ? ` ${getAdditionalInfoFromSource(l)}`
                                    : ''),
                              )
                              .join(', ')}`
                        : 'No Layers Selected'}
                      <Select.Icon>
                        <ChevronDownIcon className="-mr-1 ml-0 w-4 leading-none" />
                      </Select.Icon>
                    </DropdownMenu.Trigger>
                    <DropdownMenu.Portal>
                      <DropdownMenu.Content
                        className="z-[100] max-h-[305px] cursor-pointer overflow-scroll rounded-md border border-slate-300 bg-white text-[10px] font-medium text-sky-700 shadow"
                        sideOffset={5}
                      >
                        <DropdownMenu.CheckboxItem
                          className="flex h-8 min-w-[100px] flex-row items-center overflow-hidden border-b px-3 font-mono text-xs hover:bg-slate-100 focus:outline-none"
                          checked={selectedLayers?.length === 0}
                          onSelect={(e) => {
                            e.preventDefault();
                            setSelectedLayers([]);
                          }}
                        >
                          <div className="w-7">
                            <DropdownMenu.ItemIndicator>
                              <Check className="h-4 w-4" />
                            </DropdownMenu.ItemIndicator>
                          </div>
                          <div className="flex items-center justify-center leading-none">All Layers</div>
                        </DropdownMenu.CheckboxItem>
                        {getSourcesForSourceSet(modelId, sourceSet, true, false, false).map((layer) => (
                          <DropdownMenu.CheckboxItem
                            key={layer}
                            className="flex h-8 flex-row items-center overflow-hidden border-b px-3 font-mono text-xs hover:bg-slate-100 focus:outline-none"
                            checked={selectedLayers?.indexOf(layer) !== -1}
                            onSelect={(e) => {
                              e.preventDefault();
                              if (selectedLayers?.indexOf(layer) === -1) {
                                setSelectedLayers([...selectedLayers, layer].toSorted());
                              } else {
                                setSelectedLayers(selectedLayers?.filter((layerToCheck) => layerToCheck !== layer));
                              }
                            }}
                          >
                            <div className="w-7">
                              <DropdownMenu.ItemIndicator>
                                <Check className="h-4 w-4" />
                              </DropdownMenu.ItemIndicator>
                            </div>
                            <div className="flex items-center justify-center uppercase leading-none">
                              {getLayerNumAsStringFromSource(layer)}
                              {getAdditionalInfoFromSource(layer).length > 0 &&
                                ` ${getAdditionalInfoFromSource(layer)}`}
                            </div>
                          </DropdownMenu.CheckboxItem>
                        ))}
                      </DropdownMenu.Content>
                    </DropdownMenu.Portal>
                  </DropdownMenu.Root>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </div>
        <div className="mb-2 flex w-full flex-col items-center justify-center px-0">
          <Formik
            innerRef={formRef}
            initialValues={{ searchQuery: '' }}
            onSubmit={(values) => {
              submitSearch(values.searchQuery);
            }}
          >
            {({ submitForm, values, setFieldValue }) => (
              <Form
                className={`flex w-full gap-x-1.5 gap-y-1.5 sm:max-w-screen-lg sm:flex-row ${searchState === ExplanationSearchState.LOADED ? 'flex-row' : 'flex-col'}`}
              >
                <div className="mt-0 flex flex-1 flex-row justify-center gap-x-2">
                  <input
                    name="searchQuery"
                    disabled={searchState === ExplanationSearchState.SEARCHING}
                    value={values.searchQuery}
                    onChange={(e) => {
                      setNeedsReloadSearch(true);
                      setFieldValue('searchQuery', e.target.value);
                    }}
                    required
                    placeholder={`Search for anything (eg 'cats', 'blue', 'joyful')`}
                    className="mt-0 max-w-sm flex-1 resize-none rounded-md border border-slate-300 px-3 py-2.5 text-center text-xs font-medium text-slate-800 placeholder-slate-400 shadow-sm transition-all focus:border-sky-700 focus:outline-none focus:ring-0 disabled:bg-slate-200 sm:text-[14px]"
                  />
                  <Button
                    variant="default"
                    type="submit"
                    disabled={
                      values.searchQuery.length === 0 ||
                      searchState === ExplanationSearchState.SEARCHING ||
                      (searchState === ExplanationSearchState.LOADED && !needsReloadSearch)
                    }
                    onClick={(e) => {
                      e.preventDefault();
                      submitForm();
                    }}
                    className="group flex h-full flex-col items-center justify-center gap-y-1 overflow-hidden px-3 font-bold uppercase transition-all sm:text-xs"
                  >
                    <Search className="h-5 w-5" />
                  </Button>
                </div>
              </Form>
            )}
          </Formik>
        </div>
      </div>

      <div className="mb-3 mt-0 hidden w-full flex-col items-start justify-center overflow-x-hidden border-b px-0 pb-3 pt-0">
        <div className="mb-2 flex w-full max-w-sm flex-row gap-x-0.5 px-0 leading-tight sm:max-w-screen-lg sm:gap-x-2 sm:leading-normal">
          <ToggleGroup.Root
            className="inline-flex flex-1 overflow-hidden rounded border border-slate-300 bg-slate-300 px-0 py-0 sm:rounded"
            type="single"
            defaultValue={showDashboards ? 'showDashboards' : 'hideDashboards'}
            value={showDashboards ? 'showDashboards' : 'hideDashboards'}
            onValueChange={(value) => {
              setShowDashboards(value === 'showDashboards');
            }}
            aria-label="show dashboards or not"
          >
            <ToggleGroup.Item
              key="showDashboards"
              className="flex-auto items-center rounded-l px-1 py-1 text-[10px] font-medium text-slate-500 transition-all hover:bg-slate-100 data-[state=on]:bg-white data-[state=on]:text-slate-600 sm:px-4 sm:text-[11px]"
              value="showDashboards"
              aria-label="showDashboards"
            >
              Show Dashboards
            </ToggleGroup.Item>
            <ToggleGroup.Item
              key="hideDashboards"
              className="flex-auto items-center rounded-r px-1 py-1 text-[10px] font-medium text-slate-500 transition-all hover:bg-slate-100 data-[state=on]:bg-white data-[state=on]:text-slate-600 sm:px-4 sm:text-[11px]"
              value="hideDashboards"
              aria-label="hideDashboards"
            >
              Hide Dashboards
            </ToggleGroup.Item>
          </ToggleGroup.Root>
          {/* <ToggleGroup.Root
            className="inline-flex flex-1 overflow-hidden rounded border border-slate-300 bg-slate-300 px-0 py-0 sm:rounded"
            type="single"
            defaultValue={hideDense ? "hide" : "show"}
            value={hideDense ? "hide" : "show"}
            onValueChange={(value) => {
              setHideDense(value === "hide");
            }}
            aria-label="Show results where activation density is > 1%"
          >
            <ToggleGroup.Item
              key={"show"}
              className="flex-auto items-center rounded-l px-1 py-1 text-[10px] font-medium text-slate-500  transition-all hover:bg-slate-100 data-[state=on]:bg-white data-[state=on]:text-slate-600 sm:px-4  sm:text-[11px]"
              value={"show"}
              aria-label={"show"}
            >
              Show &gt;1% Density
            </ToggleGroup.Item>
            <ToggleGroup.Item
              key={"hide"}
              className="flex-auto items-center rounded-r px-1 py-1 text-[10px] font-medium text-slate-500 transition-all hover:bg-slate-100 data-[state=on]:bg-white data-[state=on]:text-slate-600 sm:px-4 sm:text-[11px]"
              value={"hide"}
              aria-label={"hide"}
            >
              Hide &gt;1% Density
            </ToggleGroup.Item>
          </ToggleGroup.Root> */}
        </div>
      </div>

      {searchState === ExplanationSearchState.SEARCHING && loadedResults.length === 0 && !fromNav && (
        <div className="">
          <PanelLoader showBackground={false} />
        </div>
      )}
      {(searchState === ExplanationSearchState.LOADED || loadedResults.length > 0) && (
        <div className="flex w-full max-w-screen-lg flex-col overscroll-contain bg-white">
          <div className="mx-auto flex w-full flex-col items-center overscroll-contain bg-white">
            {searchState === ExplanationSearchState.LOADED && needsReloadSearch && (
              <div className="flex w-full flex-row items-center justify-center gap-x-3 rounded bg-slate-100 py-2 text-[13px] text-slate-600">
                Search parameters changed.{' '}
                <button
                  type="button"
                  className="rounded-md bg-sky-700 px-3 py-2 text-[10px] font-medium uppercase leading-none text-white transition-all hover:bg-sky-900"
                  onClick={() => {
                    submitSearch(formRef.current?.values.searchQuery || '');
                  }}
                >
                  Reload Search
                </button>
              </div>
            )}
            <InfiniteScroll
              dataLength={loadedResults.length}
              next={() => {
                submitSearch(formRef.current?.values.searchQuery || '', true);
              }}
              scrollableTarget=""
              loader={
                <div
                  className="loader flex items-center justify-center py-5 text-center text-xs font-bold uppercase text-slate-500"
                  key={0}
                >
                  <PanelLoader showBackground={false} />
                </div>
              }
              hasMore={hasMore}
              className={`forceShowScrollBar relative flex flex-1 flex-col ${
                isSteerSearch
                  ? `mt-1 h-full ${allowSteerSearchFullHeight ? 'max-h-[480px]' : 'max-h-[250px]'} overflow-y-scroll px-1 py-1`
                  : 'bg-white'
              }`}
            >
              {loadedResults.map((result) => (
                <a
                  onClick={(e: React.MouseEvent<HTMLAnchorElement>) => {
                    if (isSteerSearch && onClickResultCallback) {
                      e.preventDefault();
                    }
                  }}
                  href={`/${result.modelId}/${result.layer}/${result.index}`}
                  target="_blank"
                  rel="noreferrer"
                  key={result.index}
                  className={`flex w-full flex-col items-center justify-center rounded-md border bg-white px-3 sm:px-3 ${
                    isSteerSearch
                      ? 'mb-2 cursor-default border-slate-200 py-2'
                      : 'group cursor-pointer border-transparent py-5 hover:border-sky-600'
                  }`}
                >
                  <div className="flex w-full max-w-screen-lg flex-col items-center justify-center gap-x-2 sm:flex-row sm:gap-x-5">
                    <div
                      className={`flex flex-col items-center justify-center font-mono text-[11px] font-medium text-slate-500 ${
                        hideResultDetails ? 'w-full' : 'basis-4/12 sm:basis-3/12'
                      }`}
                    >
                      <span
                        className={`${
                          isSteerSearch ? 'mb-0 mt-0' : 'mb-1'
                        } flex flex-col text-center font-sans text-[12px] leading-tight text-slate-700 group-hover:text-sky-700 sm:text-[13px]`}
                      >
                        {result.description}
                      </span>
                      <div>
                        {result.typeName && !hideResultDetails && result.typeName !== EXPLANATIONTYPE_HUMAN && (
                          <span className="font-sans text-[11px] font-medium text-slate-400">
                            {result.typeName} · {result.explanationModelName}
                          </span>
                        )}
                      </div>
                      {isSteerSearch && (
                        <span className="mt-2 text-center text-[10px] font-medium text-slate-500">
                          Activation Density:{' '}
                          {result.neuron?.frac_nonzero
                            ? `${((result.neuron?.frac_nonzero || 0) * 100).toFixed(2)}%`
                            : 'N/A'}
                        </span>
                      )}
                      {result.neuron &&
                        result.neuron.activations &&
                        result.neuron.activations.length > 0 &&
                        result.neuron.activations[0] &&
                        result.neuron.activations[0].tokens && (
                          <div className="mt-2 flex w-full flex-row justify-between gap-x-2">
                            {isSteerSearch && (
                              <div className="flex flex-col items-center justify-center gap-x-2">
                                <Button
                                  variant="default"
                                  size="sm"
                                  className="font-sans"
                                  onClick={(e) => {
                                    if (onClickResultCallback) {
                                      e.preventDefault();
                                      onClickResultCallback(result);
                                    }
                                  }}
                                >
                                  + Steer
                                </Button>
                              </div>
                            )}
                            <button
                              type="button"
                              onClick={(e) => {
                                if (isSteerSearch && result.neuron) {
                                  e.stopPropagation();
                                  e.preventDefault();
                                  const toFeedCopy = {
                                    ...result.neuron,
                                  } as NeuronWithPartialRelations;
                                  toFeedCopy.activations = undefined;
                                  setFeatureModalFeature(toFeedCopy);
                                  setFeatureModalOpen(true);
                                }
                              }}
                              className={`flex shrink-0 flex-row items-center gap-x-1 whitespace-nowrap rounded bg-slate-100 px-[6px] py-[6px] text-[8.5px] font-medium leading-none text-slate-500 group-hover:bg-sky-100 group-hover:text-slate-600 sm:px-[8px] sm:py-[6px] sm:text-[9px] ${
                                isSteerSearch
                                  ? 'cursor-pointer hover:bg-sky-100 hover:text-slate-600'
                                  : 'group-hover:bg-sky-100 group-hover:text-slate-600'
                              }`}
                            >
                              <div className="flex flex-col gap-y-[3px]">
                                <div className="">{result.neuron?.modelId?.toUpperCase()}</div>
                                <div className="">{result.neuron?.layer?.toUpperCase()}</div>
                                <div className="">INDEX {result.neuron?.index?.toUpperCase()}</div>
                              </div>
                            </button>
                            {!isSteerSearch && (
                              <div className="flex flex-col items-center justify-center gap-x-2">
                                <div className="whitespace-pre rounded bg-slate-100 text-slate-600">
                                  {replaceHtmlAnomalies(
                                    result.neuron?.activations[0].tokens[
                                      result.neuron?.activations[0].maxValueTokenIndex || 0
                                    ],
                                  )}
                                </div>
                                <span className="text-emerald-700">{result.neuron?.maxActApprox?.toFixed(2)}</span>
                              </div>
                            )}
                          </div>
                        )}
                    </div>
                    <div className={`flex-1 ${hideResultDetails ? 'hidden' : 'flex'}`}>
                      {showDashboards && result.neuron && (
                        <div className="flex flex-1 flex-row items-center gap-x-3 px-0">
                          <div className="flex flex-1 flex-col justify-start gap-y-1">
                            {result.neuron &&
                              result.neuron.activations &&
                              result.neuron.activations.map((activation) => (
                                <div className="flex flex-col" key={activation.id}>
                                  <div className="pointer-events-none mt-0 flex flex-row gap-x-2">
                                    <ActivationItem
                                      showLineBreaks={false}
                                      activation={activation}
                                      enableExpanding={false}
                                      tokensToDisplayAroundMaxActToken={10}
                                      overrideLeading="leading-none"
                                      overrideTextSize="text-[11px]"
                                      dfa={
                                        getSourceSet(result.neuron?.modelId || '', result.neuron?.sourceSetName || '')
                                          ?.showDfa
                                      }
                                    />
                                  </div>
                                </div>
                              ))}
                          </div>
                          <div className="pointer-events-none mb-0 min-w-[300px] max-w-[300px]">
                            <FeatureStats
                              embed
                              embedPlots={false}
                              currentNeuron={result.neuron as NeuronWithPartialRelations}
                            />
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </a>
              ))}
              {!hasMore && (
                <div className="bg-white px-5 py-5 text-center font-bold text-slate-400">End of Results</div>
              )}
            </InfiniteScroll>
          </div>
        </div>
      )}
    </div>
  );
}
