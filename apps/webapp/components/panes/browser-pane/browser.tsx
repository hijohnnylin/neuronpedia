'use client';

/* eslint no-nested-ternary: 0 */

import ActivationItem from '@/components/activation-item';
import ActivationsList from '@/components/activations-list';
import LayerSelector from '@/components/feature-selector/layer-selector';
import ModelSelector from '@/components/feature-selector/model-selector';
import SourceSetSelector from '@/components/feature-selector/sourceset-selector';
import FeatureStats from '@/components/feature-stats';
import PanelLoader from '@/components/panel-loader';
import { useFeatureContext } from '@/components/provider/feature-provider';
import { useGlobalContext } from '@/components/provider/global-provider';
import { env } from '@/lib/env';
import { getFirstSourceSetForModel } from '@/lib/utils/source';
import { Visibility } from '@prisma/client';
import Link from 'next/link';
import { NeuronWithPartialRelations, SourceSetWithPartialRelations } from 'prisma/generated/zod';
import { useEffect, useRef, useState } from 'react';
import InfiniteScroll from 'react-infinite-scroll-component';

export const FEATURE_BROWSER_HEIGHT = 520;

export default function Browser({
  defaultModelId,
  defaultSourceSet,
  defaultLayer,
  index,
  showModel = true,
  filterToRelease,
  newWindowOnSaeChange = false,
}: {
  defaultModelId: string;
  defaultSourceSet: string;
  defaultLayer: string;
  index?: string;
  showModel?: boolean;
  filterToRelease?: string;
  newWindowOnSaeChange?: boolean;
}) {
  const { globalModels, getFirstSourceForSourceSet } = useGlobalContext();
  const [modelId, setModelId] = useState(defaultModelId);
  const [sourceSet, setSourceSet] = useState(defaultSourceSet);
  const [layer, setLayer] = useState(defaultLayer);
  const [loadedNeurons, setLoadedNeurons] = useState<NeuronWithPartialRelations[]>([]);
  const scrollParentRef = useRef<HTMLDivElement>(null);
  const [hasMore, setHasMore] = useState(true);
  const { currentFeature: currentNeuron, setCurrentFeature: setCurrentNeuron } = useFeatureContext();
  const [locked, setLocked] = useState(false);

  if (index !== undefined) {
    setLocked(true);
  }

  // load index if we have one
  useEffect(() => {
    if (index !== undefined) {
      setLocked(true);
      // load the neuron remotely
      fetch(`/api/feature/${modelId}/${layer}/${index}`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      })
        .then((response) => response.json())
        .then((n: NeuronWithPartialRelations) => {
          setCurrentNeuron(n);
          setLocked(true);
        })
        .catch((error) => {
          console.error(`error submitting getting rest of neuron: ${error}`);
        });
    }
  }, [index]);

  const modelIdChanged = (newModelId: string) => {
    const newSourceSet = getFirstSourceSetForModel(
      globalModels[newModelId],
      Visibility.PUBLIC,
      true,
      false,
    ) as SourceSetWithPartialRelations;
    if (newSourceSet) {
      setSourceSet(newSourceSet.name);
      if (newWindowOnSaeChange) {
        window.open(`/${newModelId}/${getFirstSourceForSourceSet(newModelId, newSourceSet.name)}`, '_blank');
      } else {
        setModelId(newModelId);
        setSourceSet(newSourceSet.name);
        setLayer(getFirstSourceForSourceSet(newModelId, newSourceSet.name));
      }
    } else {
      setSourceSet(env.NEXT_PUBLIC_DEFAULT_SOURCESET);
    }
  };

  const sourceSetChanged = (newSourceSet: string) => {
    if (newWindowOnSaeChange) {
      window.open(`/${modelId}/${getFirstSourceForSourceSet(modelId, newSourceSet)}`, '_blank');
    } else {
      setSourceSet(newSourceSet);
      setLayer(getFirstSourceForSourceSet(modelId, newSourceSet));
    }
  };

  const layerChanged = (newLayer: string) => {
    setLayer(newLayer);
  };

  // TODO: fix this mess (3x duplicated code)

  const fetchMoreData = () => {
    fetch(`/api/features-offset`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        modelId,
        layer,
        offset: loadedNeurons.length,
      }),
    })
      .then((response) => response.json())
      .then((responseJson: NeuronWithPartialRelations[]) => {
        responseJson.sort((a, b) => {
          const aInt = parseInt(a.index, 10);
          const bInt = parseInt(b.index, 10);
          if (aInt > bInt) {
            return 1;
          }
          if (aInt === bInt) {
            return 0;
          }
          return -1;
        });
        setLoadedNeurons((prevItems) => [...prevItems, ...responseJson]);
        if (responseJson.length > 0) {
          setHasMore(true);
        } else {
          setHasMore(false);
        }
      })
      .catch((err) => {
        console.log(err);
      });
  };

  useEffect(() => {
    fetch(`/api/features-offset`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        modelId,
        layer,
        offset: 0,
      }),
    })
      .then((response) => response.json())
      .then((responseJson: NeuronWithPartialRelations[]) => {
        console.log('got response length', responseJson.length);
        responseJson.sort((a, b) => {
          const aInt = parseInt(a.index, 10);
          const bInt = parseInt(b.index, 10);
          if (aInt > bInt) {
            return 1;
          }
          if (aInt === bInt) {
            return 0;
          }
          return -1;
        });
        setLoadedNeurons(responseJson);
        if (responseJson.length > 0) {
          setHasMore(true);
        } else {
          setHasMore(false);
        }
      })
      .catch((err) => {
        console.log(err);
      });
  }, []);

  useEffect(() => {
    setCurrentNeuron(undefined);
    if (loadedNeurons.length > 0) {
      setLoadedNeurons([]);
    }
    setHasMore(true);
    fetch(`/api/features-offset`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        modelId,
        layer,
        offset: 0,
      }),
    })
      .then((response) => response.json())
      .then((responseJson: NeuronWithPartialRelations[]) => {
        responseJson.sort((a, b) => {
          const aInt = parseInt(a.index, 10);
          const bInt = parseInt(b.index, 10);
          if (aInt > bInt) {
            return 1;
          }
          if (aInt === bInt) {
            return 0;
          }
          return -1;
        });
        setLoadedNeurons(responseJson);
        if (responseJson.length > 0) {
          setHasMore(true);
        } else {
          setHasMore(false);
        }
      })
      .catch((err) => {
        console.log(err);
      });
  }, [modelId, layer]);

  return (
    <div
      className="flex w-full flex-col gap-x-3 sm:pt-0"
      style={{
        height: FEATURE_BROWSER_HEIGHT,
        minHeight: FEATURE_BROWSER_HEIGHT,
        maxHeight: FEATURE_BROWSER_HEIGHT,
      }}
    >
      {/* <div className="absolute right-0 top-3 sm:mr-3">
        {getVisibilityBadge(getSourceSet(modelId, sourceSet)?.visibility || Visibility.PUBLIC)}
      </div> */}
      <div className="relative mx-auto mb-2 flex w-full flex-row items-center justify-center gap-y-2 sm:mb-0 sm:gap-y-0">
        <div className="flex flex-1 select-none flex-row justify-center gap-x-1.5 overflow-hidden pb-2 text-center font-mono text-sm font-bold uppercase leading-none text-slate-700 sm:w-auto sm:gap-x-1.5 sm:text-sm">
          {showModel && (
            <ModelSelector
              modelId={modelId}
              modelIdChangedCallback={modelIdChanged}
              filterToRelease={filterToRelease}
            />
          )}
          <SourceSetSelector
            modelId={modelId}
            sourceSet={sourceSet}
            sourceSetChangedCallback={sourceSetChanged}
            filterToRelease={filterToRelease}
          />
          <LayerSelector modelId={modelId} sourceSet={sourceSet} layer={layer} layerChangedCallback={layerChanged} />
        </div>
      </div>

      <div className="flex h-[calc(100%-48px)] flex-col gap-y-2 sm:flex-row sm:gap-x-3 sm:gap-y-0">
        <div className="flex h-full flex-col sm:basis-1/3">
          <div className="flex flex-row items-center justify-center rounded-t-lg bg-slate-200 px-3 text-center text-[10px] font-medium uppercase tracking-wide text-slate-600 transition-all">
            <div className="py-1.5 text-[10px]">
              {'Features '}
              in{' '}
              <Link href={`/${modelId}`} className="flex-1 text-center text-sky-700">
                {Object.keys(globalModels).includes(modelId) && globalModels[modelId].displayNameShort}
              </Link>
              @
              <Link href={`/${modelId}/${layer}`} className="flex-1 text-center text-sky-700">
                {layer}
              </Link>
            </div>
          </div>
          <div
            className="relative flex flex-col overflow-auto rounded-b-lg border border-t-0 border-slate-200 shadow-md sm:flex-1"
            ref={scrollParentRef}
            id="parentScroll"
          >
            {hasMore && (
              <div
                className="loader absolute flex h-[200px] min-h-[200px] w-full items-center justify-center py-5 text-center text-xs font-bold uppercase text-slate-500"
                key={0}
              >
                <PanelLoader showBackground={false} />
              </div>
            )}
            <InfiniteScroll
              dataLength={loadedNeurons.length}
              next={fetchMoreData}
              scrollableTarget="parentScroll"
              loader={
                <div
                  className="loader flex items-center justify-center py-5 text-center text-xs font-bold uppercase text-slate-500"
                  key={0}
                >
                  <PanelLoader showBackground={false} />
                </div>
              }
              hasMore={hasMore}
              className="forceShowScrollBar relative flex flex-1 flex-col bg-white"
            >
              {hasMore === false && (
                <div className="bg-slate-50 py-5 text-center font-bold text-slate-300">No Features Found</div>
              )}
              {loadedNeurons.map((neuron) => (
                <button
                  type="button"
                  key={`${neuron?.layer}-${neuron?.index}`}
                  className={`relative flex cursor-pointer flex-col items-start justify-start border-b px-3 py-2.5 pb-4 text-left text-xs transition-all last:border-b-0 ${
                    locked &&
                    currentNeuron?.modelId === neuron.modelId &&
                    currentNeuron?.layer === neuron.layer &&
                    currentNeuron?.index === neuron.index
                      ? 'bg-sky-100 shadow-[inset_0_0px_0px_1.5px_rgba(3,105,161,.5)]'
                      : 'bg-white hover:bg-slate-100'
                  }`}
                  onMouseEnter={() => {
                    if (!locked) {
                      setCurrentNeuron(neuron);
                    }
                  }}
                  onMouseLeave={() => {
                    if (!locked) {
                      setCurrentNeuron(undefined);
                    }
                  }}
                  onClick={() => {
                    if (
                      currentNeuron?.modelId === neuron.modelId &&
                      currentNeuron?.layer === neuron.layer &&
                      currentNeuron?.index === neuron.index &&
                      locked
                    ) {
                      setLocked(false);
                      setCurrentNeuron(undefined);
                    } else {
                      setCurrentNeuron(neuron);
                      setLocked(true);
                    }
                  }}
                >
                  {neuron.explanations && neuron.explanations.length > 0 ? (
                    <div className="mb-1 text-[13px] font-medium text-slate-700">
                      {neuron.explanations[0].description}
                    </div>
                  ) : (
                    <div className="mb-1 font-bold text-slate-300">Unexplained Feature</div>
                  )}
                  {neuron.activations && neuron.activations.length > 0 ? (
                    <div className="pointer-events-none leading-none">
                      <ActivationItem
                        enableExpanding={false}
                        activation={neuron.activations[0]}
                        tokensToDisplayAroundMaxActToken={10}
                        overrideTextSize="text-[11px]"
                        overrideLeading="leading-none"
                      />
                    </div>
                  ) : (
                    <div className="pl-4 font-medium text-slate-300">No Activations Found</div>
                  )}
                  <div className="absolute bottom-1 right-1 font-mono text-[8px] font-bold leading-none text-slate-400">
                    {neuron.index}
                  </div>
                </button>
              ))}
            </InfiniteScroll>
          </div>
        </div>
        <div className="flex flex-col sm:basis-2/3 sm:overflow-x-hidden">
          {currentNeuron ? (
            <div
              className="relative flex h-full w-full flex-1 cursor-default select-none flex-col overflow-hidden rounded-xl bg-white px-0 pt-0 shadow-md"
              id="activationScrollDiv"
            >
              <div className="relative flex flex-row items-center justify-center rounded-t-xl bg-slate-200 px-3 text-center text-[11.5px] font-bold uppercase text-slate-600 transition-all">
                <div className="flex flex-row items-center justify-center gap-x-1 pb-1.5 pt-1.5 text-center text-[10px] font-medium uppercase text-slate-600">
                  {currentNeuron.modelId}@{currentNeuron.layer}:{currentNeuron.index}
                </div>
                <a
                  href={`/${currentNeuron?.modelId}/${currentNeuron?.layer}/${currentNeuron?.index}`}
                  target="_blank"
                  rel="noreferrer"
                  className="absolute right-3 top-1.5 text-[10px] font-bold capitalize text-sky-700"
                >
                  ↗ Popup
                </a>
              </div>
              <div className="overflow-y-scroll overscroll-contain rounded-b-xl border border-slate-200">
                {currentNeuron?.pos_str && currentNeuron?.pos_str?.length > 0 && (
                  <div className="border-b px-4 pb-2 pt-3">
                    <FeatureStats currentNeuron={currentNeuron} />
                  </div>
                )}
                <ActivationsList
                  feature={currentNeuron}
                  defaultRange={currentNeuron.sourceSet?.defaultRange}
                  defaultShowLineBreaks={currentNeuron.sourceSet?.defaultShowBreaks}
                  showTest={false}
                  activations={currentNeuron?.activations}
                />
              </div>
            </div>
          ) : (
            <div className="z-0 hidden flex-1 flex-col gap-2 rounded-xl border bg-white px-4 py-3 shadow-md sm:flex">
              <div className="flex h-full items-center justify-center text-center text-lg font-medium leading-relaxed text-slate-400">
                <ol>
                  <li className="pb-3">Hover over a feature on the left to preview its details.</li>
                  <li className="pb-20">Click a feature to lock it and interact with it.</li>
                </ol>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
