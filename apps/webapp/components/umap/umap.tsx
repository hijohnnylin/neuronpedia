/* eslint no-nested-ternary: 0 */

/* TODO: Refactor this. It ended up being a messy mini-app that uh, miraculously works somehow. */

import PanelLoader from '@/components/panel-loader';
import { useGlobalContext } from '@/components/provider/global-provider';
import {
  SPARSITY_COLOR_MAX,
  SPARSITY_COLOR_MIN,
  SPARSITY_COLORS,
  UMAP_HEIGHT,
  UMAP_INITIAL_COLORS,
  UmapListItem,
  useUmapContext,
  ZOOM_PADDING_X,
  ZOOM_PADDING_Y,
} from '@/components/provider/umap-provider';
import { ListNeuronToAdd } from '@/lib/utils/list';
import { getExplanationNeuronIdentifier, NeuronIdentifier } from '@/lib/utils/neuron-identifier';
import * as Dialog from '@radix-ui/react-dialog';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import * as Popover from '@radix-ui/react-popover';
import * as Select from '@radix-ui/react-select';
import * as ToggleGroup from '@radix-ui/react-toggle-group';
import { Field, Form, Formik } from 'formik';
import _ from 'lodash';
import { HelpCircle, Maximize, X, ZoomIn, ZoomOut } from 'lucide-react';
import { useSession } from 'next-auth/react';
import {
  ExplanationWithPartialRelations,
  ListWithPartialRelations,
  NeuronWithPartialRelations,
} from 'prisma/generated/zod';
import { useEffect, useRef, useState } from 'react';
import { Virtuoso, VirtuosoHandle } from 'react-virtuoso';
import UmapListRow from './umap-list-row';
import UmapPlotActive, { SEARCH_NOT_MATCHED_COLOR } from './umap-plot-active';
import UmapPlotInactive from './umap-plot-inactive';
import UmapSearch from './umap-search';
import UmapSelectedItem from './umap-selected-item';

const ZOOM_IN_OUT_FACTOR = 0.32; // 32% zoom each click

const usePrevious = <T,>(value: T): T | undefined => {
  const ref = useRef<T>();
  useEffect(() => {
    ref.current = value;
  });
  return ref.current;
};

export default function Umap({ modelId, sourceSet, layers }: { modelId: string; sourceSet: string; layers: string[] }) {
  const session = useSession();
  const { setSignInModalOpen, showToastMessage, getSourceSet, user, refreshUser } = useGlobalContext();
  const {
    searchText,
    setSearchText,
    setIsSearching,
    loadingFeature,
    selectedFeatures,
    setSelectedFeatures,
    graphRanges,
    setGraphRanges,
    openedList,
    setOpenedList,
    highlightedUmapExplanations,
    setHighlightedUmapExplanations,
    plotActiveRef,
    loadFeature,
    visibleUmapExplanations,
    setVisibleUmapExplanations,
    showLogSparsity,
    setShowLogSparsity,
    layerToInitialColor,
    setLayerToInitialColor,
  } = useUmapContext();
  const [umapExplanations, setUmapExplanations] = useState<ExplanationWithPartialRelations[]>([]);
  // faster comparison when searching
  const [umapExplanationsLowercased, setUmapExplanationsLowercased] = useState<string[]>([]);
  const [isAddingToExistingList, setIsAddingToExistingList] = useState<boolean>(false);
  const [isAddingToNewList, setIsAddingToNewList] = useState<boolean>(false);
  const listRef = useRef<VirtuosoHandle>(null);
  const [newListDialogOpen, setNewListDialogOpen] = useState(false);
  const [defaultGraphRange, setDefaultGraphRange] = useState<{
    minX: number;
    minY: number;
    maxX: number;
    maxY: number;
  }>({ minX: 0, minY: 0, maxX: 0, maxY: 0 });

  const sourceHasUmapClusters = (sourceId: string) => {
    const ss = getSourceSet(modelId, sourceSet);
    if (!ss) {
      return false;
    }
    let hasUmapClusters = false;
    ss.sources?.forEach((s) => {
      if (s.id === sourceId) {
        hasUmapClusters = s.hasUmapClusters === true;
      }
    });
    return hasUmapClusters;
  };

  const sourceHasUmapLogSparsity = (sourceId: string) => {
    const ss = getSourceSet(modelId, sourceSet);
    if (!ss) {
      return false;
    }
    let hasUmapLogSparsity = false;
    ss.sources?.forEach((s) => {
      if (s.id === sourceId) {
        hasUmapLogSparsity = s.hasUmapLogSparsity === true;
      }
    });
    return hasUmapLogSparsity;
  };

  const hasClusters = sourceHasUmapClusters(sourceSet);
  const hasLogSparsity = sourceHasUmapLogSparsity(sourceSet);

  const previousSelectedFeatures = usePrevious({ selectedFeatures });

  const handleSearch = () => {
    const searchTextLowercased = searchText.toLowerCase();
    let newEindexes: number[] = [];
    newEindexes = umapExplanationsLowercased.reduce((a, e, i) => {
      if (e.indexOf(searchTextLowercased) !== -1) {
        a.push(i);
      }
      return a;
    }, newEindexes);
    let newE = _.at(umapExplanations, newEindexes);
    // only set highlighted things that are in the current grid
    newE = newE.filter((e) => {
      if (e.umap_x !== null && e.umap_y !== null) {
        if (
          e.umap_x >= graphRanges.minX &&
          e.umap_x <= graphRanges.maxX &&
          e.umap_y >= graphRanges.minY &&
          e.umap_y <= graphRanges.maxY
        ) {
          return true;
        }
        return false;
      }
      return false;
    });
    setHighlightedUmapExplanations(newE);
  };

  function resetScroll() {
    listRef.current?.scrollToIndex({
      index: 0,
      align: 'center',
      behavior: 'auto',
    });
  }

  async function addFeatureToOpenedList(feature: NeuronIdentifier) {
    if (openedList) {
      await fetch(`/api/list/add-features`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          listId: openedList.id,
          featuresToAdd: [
            {
              modelId: feature.modelId,
              layer: feature.layer,
              index: feature.index,
              description: '',
            },
          ],
        }),
      });
      showToastMessage(`Added feature ${feature.toString()} to list.`);
    }
  }

  // https://github.com/radix-ui/primitives/issues/1241
  // https://github.com/radix-ui/primitives/discussions/1234
  useEffect(() => {
    if (!newListDialogOpen) {
      setTimeout(() => {
        document.body.style.pointerEvents = '';
      }, 500);
      setIsAddingToNewList(false);
    }
  }, [newListDialogOpen]);

  useEffect(() => {
    // find and load the new one
    if (selectedFeatures.size > 0 && previousSelectedFeatures) {
      const prevFeaturesArray = [...previousSelectedFeatures.selectedFeatures.keys()];
      if (prevFeaturesArray)
        // find the selectedFeature that is new (and has null neuron)
        selectedFeatures.forEach((umapListItem, selectedFeature) => {
          // ensure this entry is a null neuron (not yet loaded)
          if (
            umapListItem.neuron === null &&
            prevFeaturesArray.find((prevFeature) =>
              // this new item was not in previous
              prevFeature.equals(selectedFeature),
            ) === undefined
          ) {
            loadFeature(selectedFeature);
            // if we have an opened list, add it to that list
            addFeatureToOpenedList(selectedFeature);
          }
        });
    }
  }, [previousSelectedFeatures, selectedFeatures]);

  useEffect(() => {
    if (openedList === undefined) {
      setSelectedFeatures(new Map());
    } else if (openedList === null) {
      // loading, ignore
    } else {
      // got a list, set it
      setSelectedFeatures(() => {
        const newMap = new Map<NeuronIdentifier, UmapListItem>();
        openedList.neurons?.forEach((neuron) => {
          newMap.set(new NeuronIdentifier(neuron.neuron?.modelId, neuron.neuron?.layer, neuron.neuron?.index), {
            isEditing: false,
            description: neuron.description || '',
            neuron: neuron.neuron as NeuronWithPartialRelations,
          });
        });
        return newMap;
      });
    }
  }, [openedList, setSelectedFeatures]);

  useEffect(() => {
    if (umapExplanations.length > 0) {
      // get the max, min range
      let minX = 0;
      let minY = 0;
      let maxX = 0;
      let maxY = 0;
      umapExplanations.forEach((exp) => {
        if (exp.umap_x !== null && exp.umap_y !== null) {
          if (exp.umap_x < minX) {
            minX = exp.umap_x;
          }
          if (exp.umap_x > maxX) {
            maxX = exp.umap_x;
          }
          if (exp.umap_y < minY) {
            minY = exp.umap_y;
          }
          if (exp.umap_y > maxY) {
            maxY = exp.umap_y;
          }
        }
      });
      minX -= (maxX - minX) * ZOOM_PADDING_X;
      maxX += (maxX - minX) * ZOOM_PADDING_X;
      minY -= (maxY - minY) * ZOOM_PADDING_Y;
      maxY += (maxY - minY) * ZOOM_PADDING_Y;

      setDefaultGraphRange({ minX, minY, maxX, maxY });
      setGraphRanges({ minX, minY, maxX, maxY });
    }
  }, [umapExplanations, setGraphRanges]);

  function isZoomDefault() {
    if (
      defaultGraphRange.minX === graphRanges.minX &&
      defaultGraphRange.minY === graphRanges.minY &&
      defaultGraphRange.maxX === graphRanges.maxX &&
      defaultGraphRange.maxY === graphRanges.maxY
    ) {
      return true;
    }
    return false;
  }

  function setZoomDefault() {
    if (!isZoomDefault()) {
      setGraphRanges(defaultGraphRange);
    }
  }

  async function openListWithId(id: string) {
    setOpenedList(null);
    const response = await fetch(`/api/list/get`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        listId: id,
      }),
    });
    const list = await response.json();
    setOpenedList(list);
  }

  useEffect(() => {
    if (searchText.length > 0 && showLogSparsity === true) {
      setSearchText('');
    }
  }, [showLogSparsity]);

  useEffect(() => {
    if (visibleUmapExplanations.length > 1 && searchText.length > 1) {
      handleSearch();
    }
  }, [visibleUmapExplanations]);

  function setNewZoomedRange(zoomFactor: number) {
    // get center
    const centerX = (graphRanges.minX + graphRanges.maxX) / 2;
    const centerY = (graphRanges.minY + graphRanges.maxY) / 2;
    const width = graphRanges.maxX - graphRanges.minX;
    const height = graphRanges.maxY - graphRanges.minY;
    const newWidth = width * zoomFactor;
    const newHeight = height * zoomFactor;
    const newMinX = centerX - newWidth / 2;
    const newMaxX = centerX + newWidth / 2;
    const newMinY = centerY - newHeight / 2;
    const newMaxY = centerY + newHeight / 2;
    setGraphRanges({
      minX: newMinX,
      minY: newMinY,
      maxX: newMaxX,
      maxY: newMaxY,
    });
  }

  useEffect(() => {
    if (plotActiveRef.current && (plotActiveRef.current as any)?.el) {
      const newE = umapExplanations.filter((e) => {
        if (e.umap_x !== null && e.umap_y !== null) {
          if (
            e.umap_x >= graphRanges.minX &&
            e.umap_x <= graphRanges.maxX &&
            e.umap_y >= graphRanges.minY &&
            e.umap_y <= graphRanges.maxY
          ) {
            return true;
          }
          return false;
        }
        return false;
      });
      newE.sort((a, b) => {
        const aValue = a.umap_cluster === -1 ? Number.MAX_SAFE_INTEGER : a.umap_cluster || 0;
        const bValue = b.umap_cluster === -1 ? Number.MAX_SAFE_INTEGER : b.umap_cluster || 0;
        if (aValue === bValue) {
          return parseInt(a.index, 10) - parseInt(b.index, 10);
        }
        return aValue - bValue;
      });
      setVisibleUmapExplanations(newE);
    }
  }, [graphRanges, setVisibleUmapExplanations]);

  useEffect(() => {
    resetScroll();
    setZoomDefault();
    if (searchText.length > 0) {
      setShowLogSparsity(false);
      handleSearch();
      setIsSearching(false);
    } else {
      setShowLogSparsity(!!hasLogSparsity);
      setHighlightedUmapExplanations([]);
    }
  }, [searchText]);

  function findAndScrollToFeature(feature: NeuronIdentifier) {
    let targetIndex = 0;
    const listToUse = searchText.length > 0 ? highlightedUmapExplanations : visibleUmapExplanations;

    for (const [indexInList, e] of listToUse.entries()) {
      if (feature.equals(getExplanationNeuronIdentifier(e))) {
        targetIndex = indexInList;
        break;
      }
    }
    listRef.current?.scrollToIndex({
      index: targetIndex,
      align: 'center',
      behavior: 'smooth',
    });
  }

  useEffect(() => {
    if (loadingFeature !== null) {
      findAndScrollToFeature(loadingFeature);
    }
  }, [loadingFeature]);

  useEffect(() => {
    fetch(`/api/umap`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        modelId,
        layers,
      }),
    })
      .then((response) => response.json())
      .then((response: { [layer: string]: ExplanationWithPartialRelations[] }) => {
        let allResponseExplanations: ExplanationWithPartialRelations[] = [];
        const layerToInitialColorToSet: {
          [layer: string]: string;
        } = {};
        // iterate through layers and set the initial colors for each layer
        layers.forEach((layer, layerIndex) => {
          if (!response[layer]) {
            console.error(`no response for layer: ${layer}`);
          } else {
            allResponseExplanations = allResponseExplanations.concat(response[layer]);
            layerToInitialColorToSet[layer] = UMAP_INITIAL_COLORS[layerIndex];
          }
          // sort by cluster then neuron index
          allResponseExplanations.sort((a, b) => {
            const aValue = a.umap_cluster === -1 ? Number.MAX_SAFE_INTEGER : a.umap_cluster || 0;
            const bValue = b.umap_cluster === -1 ? Number.MAX_SAFE_INTEGER : b.umap_cluster || 0;
            if (aValue === bValue) {
              return parseInt(a.index, 10) - parseInt(b.index, 10);
            }
            return aValue - bValue;
          });
          // add modelId to responseExplanations (server doesn't send it for less data)
          allResponseExplanations.forEach((e) => {
            e.modelId = modelId;
          });
          setUmapExplanationsLowercased(allResponseExplanations.map((e) => e.description.toLowerCase()));
          setUmapExplanations(allResponseExplanations);
        });
        setLayerToInitialColor(layerToInitialColorToSet);
      })
      .catch((error) => {
        console.error(`error getting umap explanations: ${error}`);
      });
  }, [modelId, layers, setLayerToInitialColor]);

  function useDoubleClick(handleDoubleClick: () => void, delay = 250) {
    const [click, setClick] = useState(0);

    useEffect(() => {
      const timer = setTimeout(() => {
        setClick(0);
      }, delay);

      if (click === 2) {
        handleDoubleClick();
      }

      return () => clearTimeout(timer);
    }, [click, handleDoubleClick, delay]);

    return () => setClick((prev) => prev + 1);
  }

  async function addSelectedFeaturesToList(id: string) {
    const neuronsToAdd: ListNeuronToAdd[] = Array.from(selectedFeatures).map(([feature, umapListItem]) => ({
      modelId: umapListItem.neuron?.modelId || '',
      layer: umapListItem.neuron?.layer || '',
      index: feature.index,
      description: umapListItem.description,
    }));
    await fetch(`/api/list/add-features`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        listId: id,
        featuresToAdd: neuronsToAdd,
      }),
    });

    showToastMessage(`Added ${neuronsToAdd.length} features to list.`);
  }

  return (
    <div className="flex w-full flex-col items-center">
      <Dialog.Root open={newListDialogOpen}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed left-0 top-0 z-20 h-full w-full bg-white/50">
            <Dialog.Content className="fixed left-[50%] top-[50%] max-h-[85vh] w-[90vw] max-w-[450px] translate-x-[-50%] translate-y-[-50%] rounded-md border bg-white p-[25px] text-slate-500 shadow-md focus:outline-none">
              <Dialog.Title className="mb-2 text-center text-sm font-medium text-slate-500">
                New List
                {selectedFeatures.size > 0 ? ` With ${selectedFeatures.size} Features` : ''}
              </Dialog.Title>
              <Formik
                initialValues={{
                  listName: '',
                  listDescription: '',
                }}
                onSubmit={async (values, { resetForm }) => {
                  if (values.listName.trim().length === 0) {
                    alert('List name cannot be blank.');
                    return;
                  }

                  const response = await fetch(`/api/list/new`, {
                    method: 'POST',
                    headers: {
                      'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                      name: values.listName,
                      description: values.listDescription,
                    }),
                  });
                  const responseJson = (await response.json()) as ListWithPartialRelations;
                  // now load the new list
                  if (selectedFeatures.size > 0) {
                    await addSelectedFeaturesToList(responseJson.id || '');
                  }
                  await openListWithId(responseJson.id || '');

                  resetForm();
                  setNewListDialogOpen(false);
                  refreshUser();
                }}
              >
                {({ submitForm, isSubmitting }) => (
                  <Form
                    onClick={(e) => {
                      e.stopPropagation();
                    }}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        submitForm();
                      }
                    }}
                    className="flex max-w-sm flex-1 flex-col gap-y-1.5"
                  >
                    <Field
                      type="text"
                      name="listName"
                      required
                      placeholder="List Name"
                      className="mt-0 flex-1 rounded border border-slate-200 px-3 py-2 text-left text-xs font-medium text-slate-800 placeholder-slate-400 transition-all focus:outline-none focus:ring-0 disabled:bg-slate-200 sm:text-[13px]"
                    />
                    <Field
                      type="text"
                      name="listDescription"
                      placeholder="List Description (optional)"
                      className="mt-0 flex-1 rounded border border-slate-200 px-3 py-2 text-left text-xs font-medium text-slate-800 placeholder-slate-400 transition-all focus:outline-none focus:ring-0 disabled:bg-slate-200 sm:text-[13px]"
                    />

                    <div className="mt-3 flex flex-row gap-x-1">
                      <button
                        type="button"
                        disabled={isSubmitting}
                        onClick={(e) => {
                          e.preventDefault();
                          setNewListDialogOpen(false);
                        }}
                        className="mb-0 mt-0 flex w-full flex-1 items-center justify-center overflow-hidden rounded bg-slate-700 py-1.5 font-sans text-xs font-medium text-white transition-all hover:bg-slate-400 hover:text-slate-700 focus:ring-0 disabled:bg-slate-300 disabled:text-slate-400"
                      >
                        Cancel
                      </button>

                      <button
                        type="submit"
                        disabled={isSubmitting}
                        onClick={(e) => {
                          e.preventDefault();
                          e.stopPropagation();
                          submitForm();
                        }}
                        className="mb-0 mt-0 flex w-full flex-1 items-center justify-center overflow-hidden rounded bg-sky-700 py-1.5 font-sans text-xs font-medium text-white transition-all hover:bg-slate-400 hover:text-slate-700 focus:ring-0 disabled:bg-slate-300 disabled:text-slate-400"
                      >
                        Add
                      </button>
                    </div>
                  </Form>
                )}
              </Formik>
            </Dialog.Content>
          </Dialog.Overlay>
        </Dialog.Portal>
      </Dialog.Root>
      <div className="flex w-full max-w-screen-xl flex-col rounded border border-slate-200 bg-slate-100">
        <div className="flex w-full flex-row items-center gap-x-1 px-2.5 py-1 pt-2 text-xs uppercase">
          <div className="flex flex-1 flex-row items-center px-0.5 text-slate-400">
            {hasClusters && <div className="w-14 text-[11px]">Cluster</div>}
            <div className="flex w-20 flex-row gap-x-1 pl-1 text-[11px]">
              <div className="h-3 w-3" />
              <div>Feature</div>
            </div>
            <div className="relative z-10 flex-1">
              <UmapSearch />
            </div>
          </div>
          <div className="flex flex-1 flex-row justify-between">
            <div className="z-10 flex flex-row">
              {searchText.length > 0 && (
                <button
                  type="button"
                  onClick={() => {
                    setSearchText('');
                  }}
                  className="flex h-7 flex-row items-center justify-center gap-x-1.5 rounded bg-amber-400 px-2.5 py-1.5 text-xs hover:bg-amber-500"
                >
                  <X className="h-4 w-4" />
                  Clear Search
                </button>
              )}
            </div>
            <div className="flex flex-row gap-x-1.5">
              {hasLogSparsity && (
                <div className="flex flex-row items-center justify-center gap-x-2">
                  <div className="flex w-16 flex-col gap-y-[3px]">
                    <div className="w-16 text-center text-[7px] font-medium leading-[1.1] text-slate-700">
                      Log Feature Sparsity
                    </div>
                    <div className="flex flex-row items-center justify-center gap-x-0.5 font-mono text-[7px] font-bold">
                      <div
                        className="leading-none"
                        style={{
                          color: showLogSparsity
                            ? SPARSITY_COLORS[0]
                            : searchText.length > 0
                              ? SEARCH_NOT_MATCHED_COLOR
                              : UMAP_INITIAL_COLORS[0],
                        }}
                      >
                        {Math.floor(SPARSITY_COLOR_MIN)}
                      </div>
                      <div
                        className="h-2 flex-1 rounded-full"
                        style={{
                          backgroundImage: `linear-gradient(to right, ${
                            showLogSparsity
                              ? SPARSITY_COLORS[0]
                              : searchText.length > 0
                                ? SEARCH_NOT_MATCHED_COLOR
                                : UMAP_INITIAL_COLORS[0]
                          }, ${
                            showLogSparsity
                              ? SPARSITY_COLORS[SPARSITY_COLORS.length - 1]
                              : searchText.length > 0
                                ? SEARCH_NOT_MATCHED_COLOR
                                : UMAP_INITIAL_COLORS[0]
                          })`,
                        }}
                      />
                      <div
                        className="leading-none"
                        style={{
                          color: showLogSparsity
                            ? SPARSITY_COLORS[SPARSITY_COLORS.length - 1]
                            : searchText.length > 0
                              ? SEARCH_NOT_MATCHED_COLOR
                              : UMAP_INITIAL_COLORS[0],
                        }}
                      >
                        {Math.ceil(SPARSITY_COLOR_MAX)}
                      </div>
                    </div>
                  </div>
                  <ToggleGroup.Root
                    className="inline-flex h-7 flex-1 overflow-hidden rounded border border-slate-300 bg-slate-300 px-0 py-0 sm:rounded"
                    type="single"
                    defaultValue={showLogSparsity ? 'showLogSparsity' : 'hideLogSparsity'}
                    value={showLogSparsity ? 'showLogSparsity' : 'hideLogSparsity'}
                    onValueChange={(value) => {
                      setShowLogSparsity(value === 'showLogSparsity');
                    }}
                    aria-label="Range of text to display before and after the highest token"
                  >
                    <ToggleGroup.Item
                      key="hideLogSparsity"
                      className="flex-auto items-center rounded-r px-1 py-1 text-[10px] font-medium text-slate-500 transition-all hover:bg-slate-100 data-[state=on]:bg-white data-[state=on]:text-slate-600 sm:px-4 sm:text-[11px]"
                      value="hideLogSparsity"
                      aria-label="hideLogSparsity"
                    >
                      Off
                    </ToggleGroup.Item>
                    <ToggleGroup.Item
                      key="showLogSparsity"
                      className="flex-auto items-center rounded-l px-1 py-1 text-[10px] font-medium text-slate-500 transition-all hover:bg-slate-100 data-[state=on]:bg-white data-[state=on]:text-slate-600 sm:px-4 sm:text-[11px]"
                      value="showLogSparsity"
                      aria-label="showLogSparsity"
                    >
                      On
                    </ToggleGroup.Item>
                  </ToggleGroup.Root>
                </div>
              )}
              <button
                type="button"
                onClick={() => {
                  setNewZoomedRange(1 - ZOOM_IN_OUT_FACTOR);
                }}
                className="flex h-7 w-7 flex-row items-center justify-center gap-x-1.5 rounded bg-slate-300 px-1 py-1 text-xs text-slate-700 transition-all hover:bg-slate-400"
              >
                <ZoomIn className="h-4 w-4" />
              </button>
              <button
                type="button"
                onClick={() => {
                  setNewZoomedRange(1 + ZOOM_IN_OUT_FACTOR);
                }}
                className="flex h-7 w-7 flex-row items-center justify-center gap-x-1.5 rounded bg-slate-300 px-1 py-1 text-xs text-slate-700 transition-all hover:bg-slate-400"
              >
                <ZoomOut className="h-4 w-4" />
              </button>
              <button
                type="button"
                disabled={isZoomDefault()}
                onClick={() => {
                  setZoomDefault();
                  resetScroll();
                }}
                className={`flex h-7 flex-row gap-x-1 rounded transition-all ${
                  isZoomDefault()
                    ? 'disabled bg-slate-200 text-slate-400'
                    : 'bg-slate-300 text-slate-700 hover:bg-slate-400'
                } flex items-center justify-center px-2 py-1 text-xs`}
              >
                <Maximize className="h-4 w-4" />
                Reset
              </button>
              <Popover.Root>
                <Popover.Trigger asChild>
                  <button
                    type="button"
                    className="flex h-7 flex-row items-center justify-center gap-x-1 rounded bg-slate-300 px-2 py-1 text-xs text-slate-700 transition-all hover:bg-slate-400"
                  >
                    <HelpCircle className="h-4 w-4" />
                  </button>
                </Popover.Trigger>
                <Popover.Portal>
                  <Popover.Content
                    className="rounded bg-white px-3 py-3 text-xs text-slate-400 shadow-md"
                    sideOffset={5}
                  >
                    <div className="mx-auto flex max-w-md flex-col gap-y-1 text-left text-xs font-medium text-slate-400">
                      <div className="text-center text-slate-500">Using the UMAP</div>
                      <div className="">&bull; Click a feature to select and view details.</div>
                      <div className="">&bull; Click and drag the map to zoom into a cluster.</div>
                      <div className="">&bull; Filter by searching explanations.</div>
                      <div className="">&bull; Select features to create, update, and load lists.</div>
                      <div className="">&bull; Use the dropdown to choose a different SAE.</div>
                      {/* <div className="">
                        &bull; Learn how this{" "}
                        <a
                          href="https://transformer-circuits.pub/2023/monosemantic-features#appendix-umap"
                          className="text-slate-500 hover:underline"
                          target="_blank"
                          rel="noreferrer"
                        >
                          UMAP is generated†
                        </a>
                        .
                      </div>
                      <div className="text-[9px] leading-tight">
                        †UMAP: n_neighbors=5, min_dist=0.005, metric=cosine, random_state=1.
                        Clustering: dbscan_eps=0.05, dbscan_min_samples=2. Sparsity metric
                        calculated in window at end of training.
                      </div> */}
                    </div>
                    <Popover.Arrow className="fill-white" />
                  </Popover.Content>
                </Popover.Portal>
              </Popover.Root>
            </div>
          </div>
        </div>
        <div className="flex w-full flex-row gap-x-2 px-2.5">
          <div className={`z-10 flex h-[${UMAP_HEIGHT}px] w-full flex-1 flex-col border-t border-t-slate-300`}>
            {umapExplanations.length === 0 && (
              <div
                className={`flex h-[${UMAP_HEIGHT}px] max-h-[${UMAP_HEIGHT}px] min-h-[${UMAP_HEIGHT}px] w-full flex-1 flex-row items-center justify-center pt-0`}
              >
                <PanelLoader showBackground={false} />
              </div>
            )}
            <Virtuoso
              ref={listRef}
              style={{ height: `${UMAP_HEIGHT}px` }}
              className="forceShowScrollBar overscroll-contain"
              data={searchText.length > 0 ? highlightedUmapExplanations : visibleUmapExplanations}
              itemContent={(i, exp) => (
                <div
                  key={exp.id}
                  className={`flex w-full cursor-pointer flex-row ${
                    i > 0
                      ? searchText.length > 0
                        ? highlightedUmapExplanations[i - 1].umap_cluster !== exp.umap_cluster
                          ? 'border-t border-t-slate-200'
                          : ''
                        : visibleUmapExplanations[i - 1].umap_cluster !== exp.umap_cluster
                          ? 'border-t border-t-slate-200'
                          : ''
                      : ''
                  }`}
                >
                  {hasClusters && (
                    <div className="w-14 py-[3px] pl-1 text-xs uppercase text-slate-500">
                      {i > 0 ? (
                        searchText.length > 0 ? (
                          highlightedUmapExplanations[i - 1].umap_cluster !== exp.umap_cluster && (
                            <>#{exp.umap_cluster}</>
                          )
                        ) : (
                          visibleUmapExplanations[i - 1].umap_cluster !== exp.umap_cluster && <>#{exp.umap_cluster}</>
                        )
                      ) : (
                        <>#{exp.umap_cluster}</>
                      )}
                    </div>
                  )}
                  {layers.length > 1 && (
                    <div className="flex w-6 flex-row justify-center gap-x-1 pt-[4px] font-mono text-[10px] uppercase text-slate-500">
                      {searchText.length > 0 ? (
                        <div
                          className="mt-[3.5px] h-2 w-2 rounded-full"
                          style={{
                            backgroundColor: layerToInitialColor[highlightedUmapExplanations[i].layer],
                          }}
                        />
                      ) : (
                        <div
                          className={`mt-[3.5px] h-2 w-2 rounded-full ${
                            layerToInitialColor[visibleUmapExplanations[i].layer]
                          }`}
                          style={{
                            backgroundColor: layerToInitialColor[visibleUmapExplanations[i].layer],
                          }}
                        />
                      )}
                    </div>
                  )}
                  <UmapListRow exp={exp} modelId={modelId} />
                </div>
              )}
            />
          </div>
          <button
            type="button"
            className="z-0 grid flex-1 grid-cols-1"
            // https://github.com/plotly/react-plotly.js/issues/257
            onClick={useDoubleClick(() => {
              setZoomDefault();
            })}
          >
            <UmapPlotActive modelId={modelId} showLists />
            <UmapPlotInactive showLists />
          </button>
        </div>

        <div className="z-10 flex h-10 w-full flex-row justify-between rounded-b border-t border-slate-200 bg-slate-200 px-2.5">
          <div className="flex items-center justify-center">
            {selectedFeatures.size > 0 && openedList === undefined ? (
              <div className="flex flex-row gap-x-2">
                <DropdownMenu.Root
                  onOpenChange={() => {
                    if (!session.data?.user) {
                      setSignInModalOpen(true);
                      return false;
                    }
                    return true;
                  }}
                >
                  <DropdownMenu.Trigger className="flex min-w-[200px] flex-1 flex-row items-center justify-center overflow-hidden rounded bg-white px-5 py-1.5 text-xs font-medium leading-tight text-sky-700 shadow outline-none hover:bg-slate-100">
                    {isAddingToExistingList ? 'Adding...' : `Add ${selectedFeatures.size} to Existing List`}
                    <Select.Icon className="ml-1 text-[10px]" />
                  </DropdownMenu.Trigger>
                  <DropdownMenu.Portal>
                    <DropdownMenu.Content
                      className="z-10 max-h-[340px] cursor-pointer overflow-scroll rounded-md border-slate-300 bg-white text-xs font-medium text-sky-700 shadow"
                      sideOffset={-5}
                      side="top"
                      align="start"
                    >
                      {user &&
                        user.lists &&
                        user?.lists?.map((list) => (
                          <DropdownMenu.Item
                            key={list.id}
                            className="flex h-8 min-w-[200px] flex-row items-center overflow-hidden border-b px-3 font-sans text-xs hover:bg-slate-100 focus:outline-none"
                            onSelect={async () => {
                              setIsAddingToExistingList(true);
                              await addSelectedFeaturesToList(list.id || '');
                              setIsAddingToExistingList(false);
                              // now load the new list
                              await openListWithId(list.id || '');
                            }}
                          >
                            <div className="flex items-center justify-center leading-none">{list.name}</div>
                          </DropdownMenu.Item>
                        ))}
                    </DropdownMenu.Content>
                  </DropdownMenu.Portal>
                </DropdownMenu.Root>
                {openedList === undefined && (
                  <>
                    <button
                      type="submit"
                      onClick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        if (!session.data?.user) {
                          setSignInModalOpen(true);
                          return false;
                        }
                        setIsAddingToNewList(true);
                        setNewListDialogOpen(true);
                        return true;
                      }}
                      className="flex min-w-[160px] flex-1 flex-row items-center justify-center overflow-hidden whitespace-pre rounded bg-white px-3 py-1.5 text-xs font-medium leading-tight text-slate-600 shadow outline-none hover:bg-slate-100"
                    >
                      {isAddingToNewList ? 'Adding...' : `Add ${selectedFeatures.size} to New List`}
                    </button>
                    <button
                      type="submit"
                      onClick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        setSelectedFeatures(new Map());
                      }}
                      className="flex min-w-[100px] flex-1 flex-row items-center justify-center overflow-hidden whitespace-pre rounded bg-white px-3 py-1.5 text-xs font-medium leading-tight text-slate-600 shadow outline-none hover:bg-slate-100"
                    >
                      Deselect All
                    </button>
                  </>
                )}
              </div>
            ) : (
              <div className="flex flex-row gap-x-2">
                <DropdownMenu.Root
                  onOpenChange={() => {
                    if (!session.data?.user) {
                      setSignInModalOpen(true);
                      return false;
                    }
                    return true;
                  }}
                >
                  <DropdownMenu.Trigger className="flex min-w-[200px] flex-1 flex-row items-center justify-center overflow-hidden rounded bg-white px-3 py-1.5 text-xs font-medium leading-tight text-sky-700 shadow outline-none hover:bg-slate-100">
                    <span className="max-w-[180px] overflow-hidden text-ellipsis whitespace-nowrap">
                      {openedList === undefined ? 'Open List' : openedList === null ? 'Loading...' : openedList.name}
                    </span>
                    <Select.Icon className="ml-1 text-[10px]" />
                  </DropdownMenu.Trigger>
                  <DropdownMenu.Portal>
                    <DropdownMenu.Content
                      className="z-10 max-h-[340px] cursor-pointer overflow-scroll rounded-md border-slate-300 bg-white text-xs font-medium text-sky-700 shadow"
                      sideOffset={3}
                      side="top"
                      align="start"
                    >
                      {user &&
                        user.lists &&
                        user?.lists?.map((list) => (
                          <DropdownMenu.Item
                            key={list.id}
                            className="flex h-8 min-w-[200px] flex-row items-center overflow-hidden border-b px-3 font-sans text-xs hover:bg-slate-100 focus:outline-none"
                            onSelect={async () => {
                              openListWithId(list.id || '');
                            }}
                          >
                            <div className="flex items-center justify-center leading-none">{list.name}</div>
                          </DropdownMenu.Item>
                        ))}
                    </DropdownMenu.Content>
                  </DropdownMenu.Portal>
                </DropdownMenu.Root>
                {openedList === undefined ? (
                  <button
                    type="button"
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      if (!session.data?.user) {
                        setSignInModalOpen(true);
                        return false;
                      }
                      setIsAddingToNewList(true);
                      setNewListDialogOpen(true);
                      return true;
                    }}
                    className="flex min-w-[120px] flex-1 flex-row items-center justify-center overflow-hidden whitespace-pre rounded bg-white px-3 py-1.5 text-xs font-medium leading-tight text-slate-600 shadow outline-none hover:bg-slate-100"
                  >
                    {isAddingToNewList ? 'Creating...' : '+ New List'}
                  </button>
                ) : (
                  openedList !== null && (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        setOpenedList(undefined);
                      }}
                      className="flex flex-1 flex-row items-center justify-center overflow-hidden whitespace-pre rounded bg-white px-3 py-1.5 text-xs font-medium leading-tight text-slate-600 shadow outline-none hover:bg-slate-100"
                    >
                      Close List
                    </button>
                  )
                )}
              </div>
            )}
          </div>

          <div className="flex flex-col items-end justify-center gap-x-2 gap-y-1">
            {openedList && (
              <a
                href={`/list/${openedList.id}`}
                target="_blank"
                rel="noreferrer"
                className="text-[11px] font-medium leading-none text-sky-800 hover:underline"
              >
                {openedList.name} ↗
              </a>
            )}
            <div className="text-[11px] font-medium leading-none text-slate-400">{selectedFeatures.size} Features</div>
          </div>
        </div>
        {/* )} */}
      </div>
      {selectedFeatures.size > 0 && (
        <div className="mt-2 flex w-full max-w-screen-xl flex-1 flex-col overflow-y-scroll">
          <div className="flex flex-row gap-x-2 overflow-x-scroll">
            {Array.from(selectedFeatures)
              .toReversed()
              .map(([feature, umapListItem]) => (
                <UmapSelectedItem
                  feature={feature}
                  listItem={umapListItem}
                  key={`${feature.modelId}-${feature.layer}-${feature.index}`}
                  listToUse={openedList}
                  umap
                />
              ))}
          </div>
        </div>
      )}
    </div>
  );
}
