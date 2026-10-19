'use client';

import {
  getAdditionalInfoFromSource,
  getLayerNumFromSource,
  getNeuronLayerLabel,
  getSourceSetNameFromSource,
  isCanonicalSource,
  isNeuronLayerSource,
  NEURONS_SOURCESET,
} from '@/lib/utils/source';
import { SourceReleaseWithPartialRelations, SourceSetWithPartialRelations } from '@/prisma/generated/zod';
import { useRouter } from '@bprogress/next';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import { ChevronDownIcon, ChevronRightIcon } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useIsMount } from '../../lib/hooks/use-is-mount';
import { useGlobalContext } from '../provider/global-provider';

// Sentinel name for the manually-injected "Attention Heads" pseudo-release that is always
// pinned to the top of the release list when `includeHeads` is enabled.
const ATTENTION_HEADS_RELEASE = '__attention_heads__';

// Sentinel name for the "MLP Neurons" pseudo-release. The neurons source set has no release.
const NEURONS_RELEASE = '__mlp_neurons__';

// Models that have no attention head data, so the "Attention Heads" release/source is hidden for them.
const MODELS_WITHOUT_HEAD_DATA = ['circuitgpt-python', 'deepseek-r1-llama-8b'];

export default function SourceSelector({
  modelId,
  defaultSource,
  defaultSourceSet,
  sourceChangedCallback,
  filterToRelease,
  filterToInferenceEnabled = false, // should we sourceSets that don't support activation testing
  filterToFeaturedReleases = true, // should we only show featured sourceSets
  filterToOnlyHasDashboards = true,
  filterToPublic = false,
  filterToLayerNumber = undefined,
  includeHeads = false,
  numHeadLayers = 0,
  headLayers = undefined,
  selectedHeadLayer = undefined,
  headLayerChangedCallback,
  compact = false,
}: {
  modelId: string;
  defaultSource?: string;
  defaultSourceSet?: string;
  sourceChangedCallback?: (sourceId: string) => void;
  filterToRelease?: string | undefined;
  filterToInferenceEnabled?: boolean | undefined;
  filterToFeaturedReleases?: boolean | undefined;
  filterToOnlyHasDashboards?: boolean;
  filterToLayerNumber?: number | undefined;
  filterToPublic?: boolean;
  // When true, injects an "Attention Heads" pseudo-release at the top that lets users pick a layer.
  includeHeads?: boolean;
  numHeadLayers?: number;
  // Explicit (possibly non-contiguous) list of layers with head data. Falls back
  // to a contiguous 0..numHeadLayers-1 range when omitted.
  headLayers?: number[] | undefined;
  selectedHeadLayer?: number | undefined;
  headLayerChangedCallback?: (layer: number) => void;
  // Shorter trigger, to leave room for a row below it in the same height
  compact?: boolean;
}) {
  const isMount = useIsMount();
  const router = useRouter();
  // Layers offered in the "Attention At Layer" submenu. Prefer an explicit list
  // (non-contiguous for hybrid models); otherwise fall back to 0..numHeadLayers-1.
  const headLayerList = headLayers ?? Array.from({ length: numHeadLayers }, (_, i) => i);
  const { getSourceSetsForModelId, releases, getSourceSet, getReleaseForSourceSet, globalModels } = useGlobalContext();

  const modelDefaultSourceSet = globalModels[modelId]?.defaultSourceSetName;
  // By name, with the model's defaultSourceSetName first
  const byModelDefaultThenName = (a: SourceSetWithPartialRelations, b: SourceSetWithPartialRelations) =>
    Number(b.name === modelDefaultSourceSet) - Number(a.name === modelDefaultSourceSet) ||
    (a.name || '').localeCompare(b.name || '');

  function getFirstSourceSetForModelId() {
    let sourceSets = getSourceSetsForModelId(modelId);
    sourceSets = sourceSets
      .filter((ss) => ss.modelId === modelId && ss.name !== NEURONS_SOURCESET)
      .filter((ss) => (filterToRelease ? ss.releaseName === filterToRelease : true))
      .filter((ss) =>
        // filter to sourcesets that have at least one source with inference enabled
        filterToInferenceEnabled ? ss.sources?.some((s) => s.inferenceEnabled) : true,
      )
      .filter((ss) => (filterToOnlyHasDashboards ? ss.hasDashboards : true))
      .sort(byModelDefaultThenName);
    if (sourceSets.length > 0) {
      return sourceSets[0].name;
    }
    return '';
  }

  const [release, setRelease] = useState<SourceReleaseWithPartialRelations | undefined>(
    defaultSource
      ? getReleaseForSourceSet(modelId, getSourceSetNameFromSource(defaultSource))
      : getReleaseForSourceSet(modelId, defaultSourceSet || getFirstSourceSetForModelId()),
  );

  // Neuron layers to offer. Not for inference or release-filtered pickers: those need SAE sources.
  // Like the release list below, this ignores filterToPublic.
  const neuronsSourceSet = getSourceSet(modelId, NEURONS_SOURCESET);
  const neuronLayers: string[] =
    filterToInferenceEnabled || filterToRelease || (filterToOnlyHasDashboards && !neuronsSourceSet?.hasDashboards)
      ? []
      : (neuronsSourceSet?.sources || [])
          .map((s) => s.id || '')
          .filter((id) => isNeuronLayerSource(id))
          .filter((id) =>
            filterToLayerNumber !== undefined ? getLayerNumFromSource(id) === filterToLayerNumber : true,
          )
          .toSorted((a, b) => getLayerNumFromSource(a) - getLayerNumFromSource(b));

  function getOpenReleaseName() {
    if (selectedHeadLayer !== undefined) {
      return ATTENTION_HEADS_RELEASE;
    }
    if (defaultSource && isNeuronLayerSource(defaultSource)) {
      return NEURONS_RELEASE;
    }
    return release?.name;
  }

  const [releaseOpen, setReleaseOpen] = useState<string | undefined>();
  const [sourceSetOpen, setSourceSetOpen] = useState<string>();
  const [dropdownOpen, setDropdownOpen] = useState(false);

  const modelIdSourceToRef = useRef<{
    [key: string]: HTMLDivElement | null;
  }>({});

  const scrollToModelIdSource = (source: string) => {
    if (modelIdSourceToRef.current[`${modelId}-${source}`]) {
      modelIdSourceToRef.current[`${modelId}-${source}`]?.scrollIntoView({
        behavior: 'smooth',
        block: 'center',
      });
    }
  };

  useEffect(() => {
    if (modelId && !isMount) {
      // In attention-head mode, switching models keeps us on the head page (the parent pins us to
      // layer 0), so we must not auto-select an SAE source — that would drop us out of head mode.
      const inHeadMode = includeHeads && selectedHeadLayer !== undefined;
      // update the sourceSet, release, and layer
      let sourceSets = getSourceSetsForModelId(modelId);
      sourceSets = sourceSets
        .filter((ss) => ss.modelId === modelId && ss.name !== NEURONS_SOURCESET)
        .filter((ss) => (filterToRelease ? ss.releaseName === filterToRelease : true))
        .filter((ss) => (filterToPublic ? ss.visibility === 'PUBLIC' : true))
        .filter((ss) => (filterToInferenceEnabled ? ss.sources?.some((s) => s.inferenceEnabled) : true))
        .filter((ss) => (filterToOnlyHasDashboards ? ss.hasDashboards : true))
        .sort(byModelDefaultThenName);
      // Start on neurons when the model has them, unless its default source set is offered
      if (neuronLayers.length > 0 && !sourceSets.some((ss) => ss.name === modelDefaultSourceSet)) {
        setRelease(undefined);
        if (!inHeadMode) {
          sourceChangedCallback?.(neuronLayers[0]);
        }
        return;
      }
      if (sourceSets.length > 0) {
        const newSourceSet = sourceSets[0].name;
        setSourceSetOpen(newSourceSet);
        setRelease(getReleaseForSourceSet(modelId, newSourceSet));
        // filter to public sources if specified
        if (filterToPublic) {
          sourceSets = sourceSets.map((ss) => ({
            ...ss,
            sources: ss.sources?.filter((s) => s.visibility === 'PUBLIC'),
          }));
        }
        if (!inHeadMode && sourceSets[0].sources && sourceSets[0].sources.length > 0 && sourceSets[0].sources[0].id) {
          sourceChangedCallback?.(
            sourceSets[0].sources.sort((a, b) => {
              if (a.id && b.id) {
                return getLayerNumFromSource(a.id) - getLayerNumFromSource(b.id);
              }
              return 0;
            })[0].id || '',
          );
        }
      } else if (!inHeadMode) {
        sourceChangedCallback?.('');
      }
    }
  }, [modelId]);

  return (
    <div className="flex flex-col">
      <DropdownMenu.Root open={dropdownOpen}>
        <DropdownMenu.Trigger asChild>
          <button
            type="button"
            onClick={() => {
              setDropdownOpen(!dropdownOpen);
              setReleaseOpen(undefined);
              setSourceSetOpen(undefined);
              setTimeout(() => {
                setReleaseOpen(getOpenReleaseName());
                setSourceSetOpen(defaultSource ? getSourceSetNameFromSource(defaultSource) : '');
              }, 100);
              setTimeout(() => {
                scrollToModelIdSource(defaultSource || '');
              }, 200);
            }}
            className={`flex w-full flex-1 flex-row items-center justify-center gap-x-1 whitespace-pre rounded border border-slate-300 bg-white px-2 text-[10px] font-medium uppercase text-sky-700 hover:bg-slate-50 focus:outline-none sm:pl-5 sm:pr-2 sm:text-xs ${
              compact ? 'h-[26px] max-h-[26px] min-h-[26px]' : 'h-10 max-h-[40px] min-h-[40px]'
            }`}
          >
            {includeHeads && selectedHeadLayer !== undefined ? (
              <div className="flex flex-col items-center justify-center gap-y-0.5 leading-none">
                <div className="flex flex-row items-center justify-center gap-x-0.5 font-mono">
                  Attention Layer {selectedHeadLayer}
                </div>
                <div className="mt-0.5 text-center font-mono text-[8px] font-medium leading-none text-slate-400">
                  Attention Head
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center gap-y-0.5 leading-none">
                <div className="flex flex-row items-center justify-center gap-x-0.5 font-mono">
                  {defaultSource && isNeuronLayerSource(defaultSource)
                    ? getNeuronLayerLabel(defaultSource)
                    : defaultSource}
                </div>
                {release && !compact && (
                  <div className="mt-0.5 text-center font-mono text-[8px] font-medium leading-none text-slate-400">
                    Source/SAE
                  </div>
                )}
              </div>
            )}
            <ChevronDownIcon className="-mr-1 ml-0 w-2 leading-none sm:w-4" />
          </button>
        </DropdownMenu.Trigger>
        <DropdownMenu.Portal>
          <DropdownMenu.Content
            onPointerDownOutside={() => {
              setDropdownOpen(false);
              setReleaseOpen(getOpenReleaseName());
              setSourceSetOpen(defaultSource ? getSourceSetNameFromSource(defaultSource) : '');
            }}
            sideOffset={3}
            className="z-30 cursor-pointer overflow-hidden rounded bg-white text-xs font-medium text-sky-700 shadow-[0px_10px_38px_-10px_rgba(22,_23,_24,_0.35),_0px_10px_20px_-15px_rgba(22,_23,_24,_0.2)]"
          >
            <DropdownMenu.Label className="sticky top-0 cursor-default border-b border-slate-100 bg-white py-1.5 text-center text-[10px] uppercase text-slate-400">
              Release
            </DropdownMenu.Label>
            {includeHeads &&
              !MODELS_WITHOUT_HEAD_DATA.includes(modelId) &&
              (headLayerList.length === 0 ? (
                // No per-layer metrics available yet: make the row a direct link into the head finder.
                <button
                  type="button"
                  onMouseEnter={() => {
                    // Collapse any previously hovered release submenu when hovering this item.
                    setReleaseOpen(ATTENTION_HEADS_RELEASE);
                  }}
                  onClick={() => {
                    setDropdownOpen(false);
                    router.push(`/${modelId}/head/0/0?headFinder=true`);
                  }}
                  className="group flex w-full max-w-[340px] flex-1 cursor-pointer flex-row items-center justify-between gap-x-1 border-b border-b-slate-100 bg-white px-3 py-3 text-xs font-medium hover:bg-amber-100 hover:text-amber-600 focus:outline-none"
                >
                  <div className="flex w-full flex-col items-start justify-center gap-y-1.5 leading-tight">
                    <div className="flex w-full flex-row items-center justify-between gap-x-5 text-[10.5px] leading-none text-amber-700/70 group-hover:text-amber-700">
                      <div className="whitespace-pre font-mono font-bold uppercase">Attention Heads</div>
                      <div className="font-medium text-slate-400">Luger, Kamath, et al.</div>
                    </div>
                    <div className="font-sans text-[12px] font-semibold capitalize text-slate-500 group-hover:text-slate-600">
                      HeadVis - Browse & Investigate Attention Heads
                    </div>
                  </div>
                </button>
              ) : (
                <DropdownMenu.Sub open={releaseOpen === ATTENTION_HEADS_RELEASE}>
                  <DropdownMenu.SubTrigger
                    onMouseEnter={() => {
                      setReleaseOpen(ATTENTION_HEADS_RELEASE);
                    }}
                    className="group flex w-full max-w-[340px] flex-1 cursor-pointer flex-row items-center justify-between gap-x-1 border-b border-b-slate-100 bg-white px-3 py-3 text-xs font-medium hover:bg-sky-100 hover:text-slate-600 focus:outline-none data-[state=open]:bg-sky-100 data-[state=open]:text-slate-600"
                  >
                    <div className="flex w-full flex-col items-start justify-center gap-y-1.5 leading-tight">
                      <div className="flex w-full flex-row items-center justify-between gap-x-5 text-[10.5px] leading-none text-sky-700/70 group-hover:text-sky-700 group-data-[state=open]:text-sky-700">
                        <div className="whitespace-pre font-mono font-bold uppercase">Attention Heads</div>
                        <div className="font-medium text-slate-400">Luger, Kamath, et al.</div>
                      </div>
                      <div className="font-sans text-[12px] font-semibold capitalize text-slate-500 group-hover:text-slate-600 group-data-[state=open]:text-slate-600">
                        HeadVis - Browse & Investigate Attention Heads
                      </div>
                    </div>
                    <ChevronRightIcon className="-mr-2 ml-0 w-3 leading-none text-slate-400 group-hover:text-slate-600 group-data-[state=open]:text-slate-600" />
                  </DropdownMenu.SubTrigger>
                  <DropdownMenu.Portal>
                    <DropdownMenu.SubContent
                      sideOffset={1}
                      className="forceShowScrollBar z-40 max-h-[340px] w-full min-w-[160px] cursor-pointer divide-y divide-slate-100 overflow-y-scroll rounded bg-white text-xs font-medium text-sky-700 shadow-[0px_10px_38px_-10px_rgba(22,_23,_24,_0.35),_0px_10px_20px_-15px_rgba(22,_23,_24,_0.2)] sm:max-h-[500px]"
                    >
                      <DropdownMenu.Label className="sticky top-0 cursor-default border-b border-slate-100 bg-white py-1.5 text-center text-[10px] uppercase text-slate-400">
                        Attention At Layer
                      </DropdownMenu.Label>
                      {headLayerList.map((layer) => (
                        <button
                          key={layer}
                          type="button"
                          onClick={() => {
                            setReleaseOpen(ATTENTION_HEADS_RELEASE);
                            headLayerChangedCallback?.(layer);
                            setDropdownOpen(false);
                          }}
                          className={`${
                            selectedHeadLayer === layer
                              ? 'bg-sky-200 text-sky-700'
                              : 'bg-white text-sky-700/70 hover:bg-sky-100'
                          } flex w-full flex-1 cursor-pointer items-center justify-center px-3 py-2.5 font-mono text-[12px] font-bold uppercase hover:text-sky-700 focus:outline-none`}
                        >
                          Layer {layer}
                        </button>
                      ))}
                    </DropdownMenu.SubContent>
                  </DropdownMenu.Portal>
                </DropdownMenu.Sub>
              ))}
            {neuronLayers.length > 0 && (
              <DropdownMenu.Sub open={releaseOpen === NEURONS_RELEASE}>
                <DropdownMenu.SubTrigger
                  onMouseEnter={() => {
                    setReleaseOpen(NEURONS_RELEASE);
                  }}
                  className="group flex w-full max-w-[340px] flex-1 cursor-pointer flex-row items-center justify-between gap-x-1 border-b border-b-slate-100 bg-white px-3 py-3 text-xs font-medium hover:bg-sky-100 hover:text-slate-600 focus:outline-none data-[state=open]:bg-sky-100 data-[state=open]:text-slate-600"
                >
                  <div className="flex w-full flex-col items-start justify-center gap-y-1.5 leading-tight">
                    <div className="flex w-full flex-row items-center justify-between gap-x-5 text-[10.5px] leading-none text-sky-700/70 group-hover:text-sky-700 group-data-[state=open]:text-sky-700">
                      <div className="whitespace-pre font-mono font-bold uppercase">MLP Neurons</div>
                      <div className="font-medium text-slate-400">{neuronsSourceSet?.creatorName}</div>
                    </div>
                    <div className="font-sans text-[12px] font-semibold capitalize text-slate-500 group-hover:text-slate-600 group-data-[state=open]:text-slate-600">
                      {neuronsSourceSet?.description || 'MLP Neurons'}
                    </div>
                  </div>
                  <ChevronRightIcon className="-mr-2 ml-0 w-3 leading-none text-slate-400 group-hover:text-slate-600 group-data-[state=open]:text-slate-600" />
                </DropdownMenu.SubTrigger>
                <DropdownMenu.Portal>
                  <DropdownMenu.SubContent
                    sideOffset={1}
                    className="forceShowScrollBar z-40 max-h-[340px] w-full min-w-[160px] cursor-pointer divide-y divide-slate-100 overflow-y-scroll rounded bg-white text-xs font-medium text-sky-700 shadow-[0px_10px_38px_-10px_rgba(22,_23,_24,_0.35),_0px_10px_20px_-15px_rgba(22,_23,_24,_0.2)] sm:max-h-[500px]"
                  >
                    <DropdownMenu.Label className="sticky top-0 cursor-default border-b border-slate-100 bg-white py-1.5 text-center text-[10px] uppercase text-slate-400">
                      Layer
                    </DropdownMenu.Label>
                    {neuronLayers.map((layer) => (
                      <div
                        key={layer}
                        ref={(el) => {
                          modelIdSourceToRef.current[`${modelId}-${layer}`] = el;
                        }}
                      >
                        <button
                          type="button"
                          onClick={() => {
                            setReleaseOpen(NEURONS_RELEASE);
                            sourceChangedCallback?.(layer);
                            setDropdownOpen(false);
                          }}
                          className={`${
                            defaultSource === layer
                              ? 'bg-sky-200 text-sky-700'
                              : 'bg-white text-sky-700/70 hover:bg-sky-100'
                          } flex w-full flex-1 cursor-pointer items-center justify-center px-3 py-2.5 font-mono text-[12px] font-bold uppercase hover:text-sky-700 focus:outline-none`}
                        >
                          {getNeuronLayerLabel(layer)}
                        </button>
                      </div>
                    ))}
                  </DropdownMenu.SubContent>
                </DropdownMenu.Portal>
              </DropdownMenu.Sub>
            )}
            {releases
              .filter((r) => r.sourceSets?.some((ss) => ss.modelId === modelId))
              .filter((r) => (filterToRelease ? r.name === filterToRelease : true))
              .filter((r) => (filterToFeaturedReleases ? r.featured : true))
              .sort((a, b) => {
                if (a.description && b.description) {
                  return a.description.localeCompare(b.description);
                }
                return 0;
              })
              .map((releaseFiltered) => (
                <DropdownMenu.Sub key={releaseFiltered.name} open={releaseOpen === releaseFiltered.name}>
                  <DropdownMenu.SubTrigger
                    onMouseEnter={() => {
                      setReleaseOpen(releaseFiltered.name);
                    }}
                    className="group flex w-full max-w-[340px] flex-1 cursor-pointer flex-row items-center justify-between gap-x-1 border-b-slate-100 bg-white px-3 py-3 text-xs font-medium hover:bg-sky-100 hover:text-slate-600 focus:outline-none data-[state=open]:bg-sky-100 data-[state=open]:text-slate-600 [&:not(:last-child)]:border-b"
                  >
                    <div className="flex w-full flex-col items-start justify-center gap-y-1.5 leading-tight">
                      <div className="flex w-full flex-row items-center justify-between gap-x-5 text-[10.5px] leading-none text-sky-700/70 group-hover:text-sky-700 group-data-[state=open]:text-sky-700">
                        <div className="whitespace-pre font-mono font-bold uppercase">{releaseFiltered?.name}</div>
                        <div className="font-medium text-slate-400">{releaseFiltered.creatorNameShort}</div>
                      </div>
                      {releaseFiltered && (
                        <div className="font-sans text-[12px] font-semibold capitalize text-slate-500 group-hover:text-slate-600 group-data-[state=open]:text-slate-600">
                          {releaseFiltered.description}
                        </div>
                      )}
                    </div>
                    <ChevronRightIcon className="-mr-2 ml-0 w-3 leading-none text-slate-400 group-hover:text-slate-600 group-data-[state=open]:text-slate-600" />
                  </DropdownMenu.SubTrigger>
                  <DropdownMenu.Portal>
                    <DropdownMenu.SubContent
                      sideOffset={1}
                      className="z-40 w-full cursor-pointer divide-y divide-slate-100 overflow-hidden rounded border-slate-300 bg-white text-xs font-medium text-sky-700 shadow-[0px_10px_38px_-10px_rgba(22,_23,_24,_0.35),_0px_10px_20px_-15px_rgba(22,_23,_24,_0.2)]"
                    >
                      <DropdownMenu.Label className="sticky top-0 cursor-default border-b border-slate-100 bg-white py-1.5 text-center text-[10px] uppercase text-slate-400">
                        Source/SAE Set
                      </DropdownMenu.Label>
                      {releaseFiltered.sourceSets
                        ?.filter((ss) => ss.modelId === modelId && ss.name !== NEURONS_SOURCESET)
                        .filter((ss) => (filterToRelease ? ss.releaseName === filterToRelease : true))
                        .filter((ss) => (filterToInferenceEnabled ? ss.sources?.some((s) => s.inferenceEnabled) : true))
                        .filter((ss) => (filterToOnlyHasDashboards ? ss.hasDashboards : true))
                        .sort((a, b) => {
                          if (a.name && b.name) {
                            return a.name.localeCompare(b.name);
                          }
                          return 0;
                        })
                        .map((ss) => {
                          const sourceSet = ss.name || '';
                          return (
                            <DropdownMenu.Sub key={sourceSet} open={sourceSetOpen === sourceSet}>
                              <DropdownMenu.SubTrigger
                                onMouseEnter={() => {
                                  setSourceSetOpen(sourceSet);
                                }}
                                asChild
                              >
                                <div className="group flex w-full flex-1 cursor-pointer flex-row items-center justify-between gap-x-1 border-b-slate-100 bg-white px-3 py-2.5 text-xs font-medium hover:bg-sky-100 hover:text-slate-600 focus:outline-none data-[state=open]:bg-sky-100 data-[state=open]:text-slate-600 [&:not(:last-child)]:border-b">
                                  <div className="flex w-full flex-col items-start justify-center gap-y-1 leading-snug">
                                    <div className="flex w-full flex-row items-center justify-between gap-x-5 whitespace-pre font-mono text-[12px] font-semibold uppercase text-sky-700/70 group-hover:text-sky-700 group-data-[state=open]:text-sky-700">
                                      {sourceSet}
                                    </div>
                                    <div className="w-full text-left font-sans text-[10.5px] font-semibold text-slate-500 group-hover:text-slate-600 group-data-[state=open]:text-slate-600">
                                      {getSourceSet(modelId, sourceSet)?.type}
                                    </div>
                                  </div>
                                  <ChevronRightIcon className="-mr-2 ml-0 w-3 leading-none text-slate-400 group-hover:text-slate-600 group-data-[state=open]:text-slate-600" />
                                </div>
                              </DropdownMenu.SubTrigger>

                              <DropdownMenu.Portal>
                                <DropdownMenu.SubContent
                                  sideOffset={1}
                                  className="forceShowScrollBar z-50 max-h-[340px] w-full cursor-pointer divide-y divide-slate-100 overflow-hidden overflow-y-scroll rounded bg-white text-xs font-medium text-sky-700 shadow-[0px_10px_38px_-10px_rgba(22,_23,_24,_0.35),_0px_10px_20px_-15px_rgba(22,_23,_24,_0.2)] sm:max-h-[500px]"
                                >
                                  <DropdownMenu.Label className="sticky top-0 cursor-default border-b border-slate-100 bg-white py-1.5 text-center text-[10px] uppercase text-slate-400">
                                    Source/SAE
                                  </DropdownMenu.Label>
                                  {ss.sources
                                    ?.filter((s) => (filterToInferenceEnabled ? s.inferenceEnabled : true))
                                    .filter((s) => (filterToOnlyHasDashboards ? s.hasDashboards : true))
                                    .filter((s) =>
                                      filterToLayerNumber
                                        ? getLayerNumFromSource(s.id || '') === filterToLayerNumber
                                        : true,
                                    )
                                    .sort((a, b) => {
                                      if (a.id && b.id) {
                                        return a.id.localeCompare(b.id, undefined, {
                                          numeric: true,
                                          sensitivity: 'base',
                                        });
                                      }
                                      return 0;
                                    })
                                    .map((source) => (
                                      <button
                                        key={source.id}
                                        type="button"
                                        ref={(el) => {
                                          modelIdSourceToRef.current[`${modelId}-${source.id}`] =
                                            el as unknown as HTMLDivElement;
                                        }}
                                        onClick={() => {
                                          setReleaseOpen(releaseFiltered.name);
                                          setSourceSetOpen(sourceSet);
                                          sourceChangedCallback?.(source.id || '');
                                          setDropdownOpen(false);
                                        }}
                                        className={`${
                                          defaultSource === source.id ? 'bg-sky-200' : 'bg-white hover:bg-sky-100'
                                        } group flex w-full flex-1 cursor-pointer flex-col items-start justify-start gap-x-1 border-b-slate-100 px-3 py-2.5 text-xs font-medium focus:outline-none`}
                                      >
                                        <div className="flex w-full flex-col items-start justify-center gap-y-1 leading-snug">
                                          <div
                                            className={`flex w-full flex-row items-center justify-between gap-x-5 font-mono text-[12px] font-bold uppercase ${
                                              defaultSource === source.id ? 'text-sky-700' : 'text-sky-700/70'
                                            } group-hover:text-sky-700 group-data-[state=open]:text-sky-700`}
                                          >
                                            {source.id}
                                          </div>
                                          <div
                                            className={`font-sans text-[10.5px] font-bold capitalize ${
                                              defaultSource === source.id ? 'text-slate-600' : 'text-slate-500'
                                            } group-hover:text-slate-600 group-data-[state=open]:text-slate-600`}
                                          >
                                            Layer {getLayerNumFromSource(source.id || '')}{' '}
                                            {isCanonicalSource(source.id || '')
                                              ? '· Canonical'
                                              : `· ${getAdditionalInfoFromSource(source.id || '')}`}
                                          </div>
                                        </div>
                                      </button>
                                    ))}
                                </DropdownMenu.SubContent>
                              </DropdownMenu.Portal>
                            </DropdownMenu.Sub>
                          );
                        })}
                    </DropdownMenu.SubContent>
                  </DropdownMenu.Portal>
                </DropdownMenu.Sub>
              ))}

            <DropdownMenu.Arrow className="fill-white" />
          </DropdownMenu.Content>
        </DropdownMenu.Portal>
      </DropdownMenu.Root>
    </div>
  );
}
