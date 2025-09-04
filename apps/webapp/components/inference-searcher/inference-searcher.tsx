'use client';

import ModelSelector from '@/components/feature-selector/model-selector';
import SourceSetSelector from '@/components/feature-selector/sourceset-selector';
import PanelLoader from '@/components/panel-loader';
import { useGlobalContext } from '@/components/provider/global-provider';
import {
  InferenceActivationAllState,
  useInferenceActivationAllContext,
} from '@/components/provider/inference-activation-all-provider';
import { Button } from '@/components/shadcn/button';
import { env } from '@/lib/env';
import { BOS_TOKENS, replaceHtmlAnomalies } from '@/lib/utils/activations';
import {
  getAdditionalInfoFromSource,
  getFirstSourceSetForModel,
  getLayerNumAsStringFromSource,
} from '@/lib/utils/source';
import { SourceSetWithPartialRelations } from '@/prisma/generated/zod';
import { Visibility } from '@prisma/client';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import * as Select from '@radix-ui/react-select';
import * as ToggleGroup from '@radix-ui/react-toggle-group';
import { Form, Formik, FormikProps } from 'formik';
import { Check, ChevronDownIcon, HelpCircle, Search } from 'lucide-react';
import { useRouter } from 'next-nprogress-bar';
import { useEffect, useRef, useState } from 'react';
import ReactTextareaAutosize from 'react-textarea-autosize';
import ExamplesButtons from './examples-buttons';
import ResultItem from './result-item';

const MAX_SEARCH_QUERY_LENGTH_CHARS = 800;

export default function InferenceSearcher({
  q,
  initialSelectedLayers = [],
  initialSortIndexes,
  initialModelId,
  initialSourceSet,
  initialIgnoreBos,
  showSourceSets = true,
  showModels = true,
  showExamples = true,
  filterToRelease,
}: {
  q?: string;
  initialSelectedLayers?: string[];
  initialSortIndexes?: number[];
  initialModelId?: string;
  initialSourceSet?: string;
  initialIgnoreBos?: boolean;
  showSourceSets?: boolean;
  showModels?: boolean;
  showExamples?: boolean;
  filterToRelease?: string | undefined;
}) {
  const {
    exploreState,
    submitSearchAll,
    setTokens,
    tokens,
    searchSortIndexes,
    setSearchResults,
    searchResults,
    setSearchCounts,
  } = useInferenceActivationAllContext();
  const { getSourcesForSourceSet, getDefaultModel, globalModels } = useGlobalContext();
  const [sortIndexes, setSortIndexes] = useState<number[]>(initialSortIndexes || []);
  const [modelId, setModelId] = useState(
    initialModelId || env.NEXT_PUBLIC_DEFAULT_MODELID || getDefaultModel()?.id || env.NEXT_PUBLIC_DEFAULT_MODELID,
  );
  const [sourceSet, setSourceSet] = useState(initialSourceSet || env.NEXT_PUBLIC_DEFAULT_SOURCESET);
  const [selectedLayers, setSelectedLayers] = useState<string[] | undefined>(initialSelectedLayers);
  const [ignoreBos, setIgnoreBos] = useState(initialIgnoreBos !== undefined ? initialIgnoreBos : true);
  const [hideDense, setHideDense] = useState(true);
  const [availableLayers, setAvailableLayers] = useState<string[]>([]);
  const [showDashboards, setShowDashboards] = useState(true);
  const [needsReloadSearch, setNeedsReloadSearch] = useState(false);
  const formRef = useRef<
    FormikProps<{
      searchQuery: string;
    }>
  >(null);
  const router = useRouter();
  const loadResultsInNewPage = q === undefined;
  // const [showTable, setShowTable] = useState(false);

  function makeUrl(query: string) {
    return `/${modelId}/?sourceSet=${sourceSet}&selectedLayers=${JSON.stringify(
      selectedLayers,
    )}&sortIndexes=${JSON.stringify(sortIndexes)}&ignoreBos=${ignoreBos}&q=${encodeURIComponent(query)}`;
  }

  const sourceSetChanged = (newSourceSet: string) => {
    setSourceSet(newSourceSet);
    setAvailableLayers(getSourcesForSourceSet(modelId, newSourceSet, false, true, false));
    setSelectedLayers([]);
    setSearchCounts([]);
  };

  const modelIdChanged = (newModelId: string) => {
    setModelId(newModelId);
    const newSourceSet = getFirstSourceSetForModel(
      globalModels[newModelId],
      Visibility.PUBLIC,
      true,
      false,
    ) as SourceSetWithPartialRelations;
    if (newSourceSet) {
      setSourceSet(newSourceSet.name);
      setAvailableLayers(getSourcesForSourceSet(newModelId, newSourceSet.name, false, true, false));
    } else {
      setSourceSet(env.NEXT_PUBLIC_DEFAULT_SOURCESET);
      setAvailableLayers([]);
    }
    setSelectedLayers([]);
    setSearchCounts([]);
  };

  useEffect(() => {
    if (exploreState === InferenceActivationAllState.LOADED) {
      setNeedsReloadSearch(true);
    }
  }, [selectedLayers, sortIndexes, ignoreBos]);

  function searchClicked() {
    const values = formRef.current?.values;
    if (values === undefined) {
      return;
    }
    if (values.searchQuery.length > MAX_SEARCH_QUERY_LENGTH_CHARS) {
      alert(`Must be shorter than ${MAX_SEARCH_QUERY_LENGTH_CHARS} characters`);
      return;
    }
    if (!selectedLayers) {
      alert('Please select at least one layer to search.');
      return;
    }
    if (values.searchQuery.trim().length === 0) {
      alert('Please enter a search query.');
      return;
    }
    if (loadResultsInNewPage) {
      router.push(makeUrl(values.searchQuery));
      return;
    }
    setSearchResults([]);
    setTokens([]);
    submitSearchAll(modelId, values.searchQuery, selectedLayers, sourceSet, ignoreBos, sortIndexes);
  }

  // load query if we have one
  useEffect(() => {
    if (q !== undefined) {
      submitSearchAll(modelId, q, selectedLayers, sourceSet, ignoreBos, sortIndexes);
      formRef.current?.setFieldValue('searchQuery', q);
    }
  }, [q]);

  // update the URL with query param for the search
  useEffect(() => {
    if (exploreState === InferenceActivationAllState.LOADED && formRef.current) {
      window.history.pushState({}, '', makeUrl(formRef.current.values.searchQuery));
      setAvailableLayers(getSourcesForSourceSet(modelId, sourceSet, false, true, false));
    } else {
      // remove query params
      window.history.pushState({}, '', document.location.toString().split(/[?#]/)[0]);
    }
    if (exploreState === InferenceActivationAllState.RUNNING) {
      setNeedsReloadSearch(false);
    }
  }, [exploreState]);

  // load initial
  useEffect(() => {
    setAvailableLayers(getSourcesForSourceSet(modelId, sourceSet, false, true, false));
  }, []);

  return (
    <div className="flex w-full flex-col items-center justify-center gap-x-1.5">
      <div
        className={`${
          exploreState === InferenceActivationAllState.LOADED ? 'pt-0' : 'mt-0'
        } flex w-full flex-col items-start justify-center transition-all sm:max-w-screen-lg`}
      >
        <div className="flex flex-col items-center justify-center">
          <div className="mb-2 flex w-auto flex-row items-center justify-center gap-x-2 text-center font-sans text-xs font-medium uppercase leading-none text-slate-500">
            {showModels && (
              <div className="flex flex-col font-mono">
                <ModelSelector
                  modelId={modelId}
                  modelIdChangedCallback={modelIdChanged}
                  filterToInferenceEnabled
                  filterToRelease={filterToRelease}
                />
              </div>
            )}
            {showSourceSets && (
              <SourceSetSelector
                modelId={modelId}
                sourceSet={sourceSet}
                filterToOnlyVisible
                sourceSetChangedCallback={sourceSetChanged}
                filterToRelease={filterToRelease}
                filterToInferenceEnabled
                filterToAllowInferenceSearch
              />
            )}
            <div className="flex flex-col">
              <DropdownMenu.Root>
                <DropdownMenu.Trigger className="flex h-10 max-h-[40px] min-h-[40px] w-full flex-1 flex-row items-center justify-center gap-x-1 whitespace-pre rounded border border-slate-300 bg-white px-2 font-mono text-[10px] font-medium leading-tight text-sky-700 hover:bg-slate-50 sm:pl-4 sm:pr-2 sm:text-xs">
                  {selectedLayers
                    ? selectedLayers?.length === 0
                      ? 'All Layers'
                      : `Layer${selectedLayers.length > 1 ? 's ' : ' '}${selectedLayers
                          .map(
                            (l) =>
                              getLayerNumAsStringFromSource(l) +
                              (getAdditionalInfoFromSource(l).length > 0 ? ` ${getAdditionalInfoFromSource(l)}` : ''),
                          )
                          .join(', ')}`
                    : 'No Layers Selected'}
                  <Select.Icon>
                    <ChevronDownIcon className="-mr-1 ml-0 w-4 leading-none" />
                  </Select.Icon>
                </DropdownMenu.Trigger>
                <DropdownMenu.Portal>
                  <DropdownMenu.Content
                    className="z-30 max-h-[305px] cursor-pointer overflow-scroll rounded-md border border-slate-300 bg-white text-[10px] font-medium text-sky-700 shadow"
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
                    {availableLayers.map((layer) => (
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
                          {getAdditionalInfoFromSource(layer).length > 0 && ` ${getAdditionalInfoFromSource(layer)}`}
                        </div>
                      </DropdownMenu.CheckboxItem>
                    ))}
                  </DropdownMenu.Content>
                </DropdownMenu.Portal>
              </DropdownMenu.Root>
            </div>
          </div>
        </div>
        <div className="mb-2 flex w-full flex-col items-center justify-center px-0">
          <Formik
            innerRef={formRef}
            initialValues={{ searchQuery: '' }}
            onSubmit={() => {
              searchClicked();
            }}
          >
            {({ submitForm, values, setFieldValue }) => (
              <Form
                className={`flex w-full gap-x-1.5 gap-y-1.5 sm:max-w-screen-lg sm:flex-row ${exploreState === InferenceActivationAllState.LOADED ? 'flex-row' : 'flex-col'}`}
              >
                <div className="mt-0 flex flex-1 flex-row gap-x-2">
                  <ReactTextareaAutosize
                    name="searchQuery"
                    disabled={exploreState === InferenceActivationAllState.RUNNING}
                    value={values.searchQuery}
                    minRows={2}
                    onChange={(e) => {
                      setNeedsReloadSearch(true);
                      if (sortIndexes.length > 0) {
                        setSortIndexes([]);
                      }
                      setFieldValue('searchQuery', e.target.value);
                    }}
                    required
                    placeholder={`Enter any text or sentence to search (e.g, 'I like cats!')`}
                    className="mt-0 min-w-[10px] flex-1 resize-none rounded-md border border-slate-200 px-3 py-2 text-left font-mono text-xs font-medium text-slate-800 placeholder-slate-400 shadow-sm transition-all focus:border-sky-700 focus:outline-none focus:ring-0 disabled:bg-slate-200 sm:text-[13px]"
                  />
                  <Button
                    variant="default"
                    type="submit"
                    disabled={
                      values.searchQuery.length === 0 ||
                      exploreState === InferenceActivationAllState.RUNNING ||
                      (exploreState === InferenceActivationAllState.LOADED && !needsReloadSearch)
                    }
                    onClick={(e) => {
                      e.preventDefault();
                      submitForm();
                    }}
                    className="group flex h-full min-w-[48px] flex-col items-center justify-center gap-y-1 overflow-hidden font-bold uppercase transition-all sm:min-w-[54px] sm:max-w-[70px] sm:text-xs"
                  >
                    <Search className="h-5 w-5" />
                    <div className="block text-[9px] font-medium uppercase leading-none sm:w-24 sm:px-2">SEARCH</div>
                  </Button>
                </div>
              </Form>
            )}
          </Formik>
        </div>
      </div>

      {exploreState === InferenceActivationAllState.DEFAULT && (
        <div className="mt-4 flex w-full max-w-screen-lg flex-col px-0 sm:mt-4">
          {showExamples && (
            <ExamplesButtons
              // eslint-disable-next-line
              makeUrl={makeUrl}
            />
          )}
        </div>
      )}
      {exploreState === InferenceActivationAllState.RUNNING && (
        <div>
          <PanelLoader showBackground={false} />
        </div>
      )}
      {exploreState === InferenceActivationAllState.LOADED && (
        <div className="flex w-full flex-col overscroll-contain bg-white">
          <div className="mb-3 mt-0 flex w-full flex-col items-start justify-center overflow-x-hidden border-b px-0 pb-3 pt-0">
            <div className="mb-2 flex w-full max-w-sm flex-row gap-x-0.5 px-0 leading-tight sm:max-w-screen-lg sm:gap-x-2 sm:leading-normal">
              <button
                type="button"
                onClick={() => {
                  alert('Click a token to sort results by highest activations for that token.');
                }}
                className="flex w-16 flex-row items-center justify-center rounded border border-slate-300 bg-slate-300 px-2 py-1 text-center text-[11px] text-slate-600 hover:bg-white sm:w-36"
              >
                How To <div className="ml-0.5 hidden sm:block"> Use</div>
                <HelpCircle className="ml-1 hidden h-3 w-3 sm:block" />
              </button>
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
              <ToggleGroup.Root
                className="inline-flex flex-1 overflow-hidden rounded border border-slate-300 bg-slate-300 px-0 py-0 sm:rounded"
                type="single"
                defaultValue={hideDense ? 'hide' : 'show'}
                value={hideDense ? 'hide' : 'show'}
                onValueChange={(value) => {
                  setHideDense(value === 'hide');
                }}
                aria-label="Show results where activation density is > 1%"
              >
                <ToggleGroup.Item
                  key="show"
                  className="flex-auto items-center rounded-l px-1 py-1 text-[10px] font-medium text-slate-500 transition-all hover:bg-slate-100 data-[state=on]:bg-white data-[state=on]:text-slate-600 sm:px-4 sm:text-[11px]"
                  value="show"
                  aria-label="show"
                >
                  Show &gt;1% Density
                </ToggleGroup.Item>
                <ToggleGroup.Item
                  key="hide"
                  className="flex-auto items-center rounded-r px-1 py-1 text-[10px] font-medium text-slate-500 transition-all hover:bg-slate-100 data-[state=on]:bg-white data-[state=on]:text-slate-600 sm:px-4 sm:text-[11px]"
                  value="hide"
                  aria-label="hide"
                >
                  Hide &gt;1% Density
                </ToggleGroup.Item>
              </ToggleGroup.Root>
              <ToggleGroup.Root
                className="inline-flex flex-1 overflow-hidden rounded border border-slate-300 bg-slate-300 px-0 py-0 sm:rounded"
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
                  className="flex-auto items-center rounded-l px-1 py-1 text-[10px] font-medium text-slate-500 transition-all hover:bg-slate-100 data-[state=on]:bg-white data-[state=on]:text-slate-600 sm:px-4 sm:text-[11px]"
                  value="show"
                  aria-label="show"
                >
                  Show BOS Results
                </ToggleGroup.Item>
                <ToggleGroup.Item
                  key="ignore"
                  className="flex-auto items-center rounded-r px-1 py-1 text-[10px] font-medium text-slate-500 transition-all hover:bg-slate-100 data-[state=on]:bg-white data-[state=on]:text-slate-600 sm:px-4 sm:text-[11px]"
                  value="ignore"
                  aria-label="ignore"
                >
                  Ignore BOS Results
                </ToggleGroup.Item>
              </ToggleGroup.Root>
            </div>
            <div className="sticky left-0.5 top-0 mb-1 mt-2 hidden w-full text-left font-sans text-[11px] font-medium text-slate-400 sm:block">
              Select tokens below to sort results by highest activations for those tokens.
            </div>
            <div className="forceShowScrollBarHorizontal my-0 flex w-full max-w-sm flex-col overflow-x-scroll pb-1 font-mono sm:max-w-[calc(1024px-50px)]">
              <div className="flex flex-row items-center gap-x-0.5">
                <button
                  type="button"
                  className={`flex w-20 min-w-20 cursor-pointer items-center justify-center whitespace-nowrap rounded ${
                    sortIndexes.length === tokens.length
                      ? 'bg-emerald-700 text-white hover:bg-emerald-600'
                      : 'bg-slate-100 text-slate-500 hover:bg-emerald-100'
                  } select-none py-1.5 text-[9px] font-bold uppercase leading-none shadow outline-none`}
                  onClick={() => {
                    if (sortIndexes.length < tokens.length) {
                      setSortIndexes(Array.from(Array(tokens.length).keys()));
                    } else {
                      setSortIndexes([]);
                    }
                  }}
                >
                  ALL TOKENS
                </button>

                {tokens.map((token, index) => (
                  <div key={index} className="px-[1px] text-center leading-none">
                    <button
                      type="button"
                      onClick={() => {
                        if (sortIndexes.indexOf(index) !== -1) {
                          // has it, remove it
                          setSortIndexes(sortIndexes.filter((sortIndex) => sortIndex !== index).toSorted());
                        } else {
                          // doesnt have it, add it
                          setSortIndexes([...sortIndexes, index].toSorted());
                        }
                      }}
                      className={`flex-1 cursor-pointer whitespace-pre rounded border px-[2px] py-0.5 font-mono text-[12px] font-medium leading-none hover:border-emerald-600 ${
                        sortIndexes.indexOf(index) !== -1
                          ? 'border-emerald-600 bg-emerald-600 text-white'
                          : 'border-slate-300 text-slate-600'
                      }`}
                    >
                      {replaceHtmlAnomalies(token)}
                    </button>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="mx-auto flex w-full flex-col items-center overscroll-contain bg-white">
            {exploreState === InferenceActivationAllState.LOADED && needsReloadSearch && (
              <div className="flex w-full flex-row items-center justify-center gap-x-3 bg-slate-200 py-2 text-[13px] text-slate-600">
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
            {searchResults
              .filter((result) => {
                if (hideDense && result.neuron && result.neuron?.frac_nonzero > 0.01) {
                  return false;
                }
                if (
                  ignoreBos &&
                  result.neuron &&
                  result.neuron.pos_str.length > 0 &&
                  BOS_TOKENS.indexOf(result.neuron.pos_str[0]) !== -1
                ) {
                  return false;
                }
                return true;
              })
              .map((result) => {
                const topExplanation = result.neuron?.explanations ? result.neuron?.explanations[0] : undefined;
                return (
                  <ResultItem
                    key={result.index}
                    result={result}
                    tokens={tokens}
                    topExplanation={topExplanation}
                    searchSortIndexes={searchSortIndexes}
                    showDashboards={showDashboards}
                  />
                );
              })}
          </div>
        </div>
      )}
    </div>
  );
}

// TODO: Old Table counts code that shows how many activations per token per layer

// TODO: this is for table counts, which we may bring back later
// function makeUrlWithReplacedSelectedLayerAndIndex(query: string, layer: number, index: number) {
//   const layerString = sourceSet === NEURONS_SOURCESET ? layer.toString() : `${layer.toString()}-${sourceSet}`;
//   return `/${modelId}/?sourceSet=${sourceSet}&selectedLayers=${JSON.stringify([
//     layerString,
//   ])}&sortIndexes=${JSON.stringify([index])}&ignoreBos=${ignoreBos}&q=${query}`;
// }

/*
<ToggleGroup.Root
  className="inline-flex flex-1 overflow-hidden rounded border border-slate-300 bg-slate-300 px-0 py-0 sm:rounded"
  type="single"
  defaultValue={showTable ? "show" : "hide"}
  value={showTable ? "show" : "hide"}
  onValueChange={(value) => {
    setShowTable(value === "show");
  }}
  aria-label="Show the table with numbers of how many features were activated"
>
  <ToggleGroup.Item
    key={"show"}
    className="flex-auto items-center rounded-l px-1 py-1 text-[10px] font-medium text-slate-500  transition-all hover:bg-slate-100 data-[state=on]:bg-white data-[state=on]:text-slate-600 sm:px-4  sm:text-[11px]"
    value={"show"}
    aria-label={"show"}
  >
    Show Layers
  </ToggleGroup.Item>
  <ToggleGroup.Item
    key={"hide"}
    className="flex-auto items-center rounded-r px-1 py-1 text-[10px] font-medium text-slate-500 transition-all hover:bg-slate-100 data-[state=on]:bg-white data-[state=on]:text-slate-600 sm:px-4 sm:text-[11px]"
    value={"hide"}
    aria-label={"hide"}
  >
    Hide Layers
  </ToggleGroup.Item>
</ToggleGroup.Root>
*/

/* {showTable && (
  <>
    {getSourcesForSourceSet(modelId, sourceSet, false, true, false).map((layerName, i) => (
      <tr key={i} className="text-center text-xs">
        <td className="flex w-[90px] cursor-pointer flex-row items-center gap-x-1 px-0 pr-1 font-sans text-[11px] font-bold leading-none text-slate-500">
          <Checkbox.Root
            className="flex h-3.5 w-3.5 appearance-none items-center justify-center rounded bg-white shadow outline-none hover:bg-emerald-200"
            checked={
              selectedLayers
                ? selectedLayers.length === 0
                  ? true
                  : selectedLayers?.indexOf(layerName) !== -1
                : false
            }
            onCheckedChange={(e) => {
              if (e) {
                if (selectedLayers) {
                  // if we end up selecting all layers, then set it to []
                  const newSelectedLayers = [...selectedLayers, layerName];
                  if (newSelectedLayers.length === availableLayers.length) {
                    setSelectedLayers([]);
                  } else {
                    setSelectedLayers([...selectedLayers, layerName]);
                  }
                } else {
                  setSelectedLayers([layerName]);
                }
              } else if (selectedLayers && selectedLayers.length === 0) {
                // all selected, so filter out clicked
                setSelectedLayers(availableLayers.filter((l) => l !== layerName));
              } else {
                // if unchecking it leads to all unselected, then set selectedlayers to undefined
                const newSelectedLayers = selectedLayers?.filter((l) => l !== layerName);
                if (newSelectedLayers?.length === 0) {
                  setSelectedLayers(undefined);
                } else {
                  setSelectedLayers(newSelectedLayers);
                }
              }
            }}
            id={`layer${i}`}
          >
            <Checkbox.Indicator className="text-emerald-700">
              <Check className="h-2.5 w-2.5" />
            </Checkbox.Indicator>
          </Checkbox.Root>
          <label
            className="cursor-pointer select-none whitespace-nowrap text-[9px] font-medium uppercase leading-none"
            htmlFor={`layer${i}`}
          >
            Layer {getLayerNumAsStringFromSource(layerName)}
          </label>
        </td>
        {searchCounts &&
          searchCounts.length > i &&
          // searchCounts[i].length > tokens.length &&
          tokens.map((token, tokenIndex) => {
            let maxResult = 0;
            searchCounts.forEach((c) => {
              c.forEach((cc) => {
                if (cc > maxResult) {
                  maxResult = cc;
                }
              });
            });
            const resultsForCell = searchCounts[i][tokenIndex];
            return (
              <td key={tokenIndex} className="px-[1px]">
                <Link
                  // target="_blank"
                  rel="noreferrer"
                  href={makeUrlWithReplacedSelectedLayerAndIndex(
                    encodeURIComponent(formRef.current?.values.searchQuery || ''),
                    i,
                    tokenIndex,
                  )}
                  className={`rounded ${
                    resultsForCell === 0
                      ? 'bg-slate-100 text-slate-300'
                      : resultsForCell < 3
                      ? 'bg-emerald-100 text-slate-600 hover:bg-emerald-200'
                      : 'bg-emerald-300 text-slate-700 hover:bg-emerald-400'
                  } px-[3px] font-bold hover:scale-125`}
                  style={{
                    backgroundColor: `rgba(52,211,153, ${resultsForCell / maxResult})`,
                  }}
                >
                  {resultsForCell}
                </Link>
              </td>
            );
          })}
      </tr>
    ))}
    <tr>
      <td className=" flex cursor-pointer flex-row items-center gap-x-1 px-0 pr-1 font-sans text-[11px] font-bold leading-none text-slate-500">
        <Checkbox.Root
          className="flex h-3.5 w-3.5 appearance-none items-center justify-center rounded bg-white shadow outline-none hover:bg-emerald-200"
          checked={selectedLayers?.length === 0 || selectedLayers?.length === availableLayers.length}
          onCheckedChange={(e) => {
            if (e) {
              setSelectedLayers([]);
            } else {
              setSelectedLayers(undefined);
            }
          }}
          id="layerAll"
        >
          <Checkbox.Indicator className="text-emerald-700">
            <Check className="h-2.5 w-2.5" />
          </Checkbox.Indicator>
        </Checkbox.Root>
        <label
          className="cursor-pointer select-none whitespace-nowrap text-[9px] font-medium uppercase leading-none"
          htmlFor="layerAll"
        >
          ALL LAYERS
        </label>
      </td>
    </tr>
  </>
)} */
