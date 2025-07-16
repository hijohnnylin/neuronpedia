'use client';

import {
  getAdditionalInfoFromSource,
  getLayerNumFromSource,
  getSourceSetNameFromSource,
  isCanonicalSource,
  NEURONS_SOURCESET,
} from '@/lib/utils/source';
import { SourceReleaseWithPartialRelations } from '@/prisma/generated/zod';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import { ChevronDownIcon, ChevronRightIcon } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useIsMount } from '../../lib/hooks/use-is-mount';
import { useGlobalContext } from '../provider/global-provider';

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
}) {
  const isMount = useIsMount();
  const { getSourceSetsForModelId, releases, getSourceSet, getReleaseForSourceSet } = useGlobalContext();

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
      .sort((a, b) => {
        if (a.name && b.name) {
          return a.name.localeCompare(b.name);
        }
        return 0;
      });
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
      // update the sourceSet, release, and layer
      let sourceSets = getSourceSetsForModelId(modelId);
      sourceSets = sourceSets
        .filter((ss) => ss.modelId === modelId && ss.name !== NEURONS_SOURCESET)
        .filter((ss) => (filterToRelease ? ss.releaseName === filterToRelease : true))
        .filter((ss) => (filterToPublic ? ss.visibility === 'PUBLIC' : true))
        .filter((ss) => (filterToInferenceEnabled ? ss.sources?.some((s) => s.inferenceEnabled) : true))
        .filter((ss) => (filterToOnlyHasDashboards ? ss.hasDashboards : true))
        .sort((a, b) => {
          if (a.name && b.name) {
            return a.name.localeCompare(b.name);
          }
          return 0;
        });
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
        if (sourceSets[0].sources && sourceSets[0].sources.length > 0 && sourceSets[0].sources[0].id) {
          sourceChangedCallback?.(
            sourceSets[0].sources.sort((a, b) => {
              if (a.id && b.id) {
                return getLayerNumFromSource(a.id) - getLayerNumFromSource(b.id);
              }
              return 0;
            })[0].id || '',
          );
        }
      } else {
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
                setReleaseOpen(release?.name);
                setSourceSetOpen(defaultSource ? getSourceSetNameFromSource(defaultSource) : '');
              }, 100);
              setTimeout(() => {
                scrollToModelIdSource(defaultSource || '');
              }, 200);
            }}
            className="flex h-10 max-h-[40px] min-h-[40px] w-full flex-1 flex-row items-center justify-center gap-x-1 whitespace-pre rounded border border-slate-300 bg-white px-2 text-[10px] font-medium uppercase text-sky-700 hover:bg-slate-50 focus:outline-none sm:pl-5 sm:pr-2 sm:text-xs"
          >
            <div className="flex flex-col items-center justify-center gap-y-0.5 leading-none">
              <div className="flex flex-row items-center justify-center gap-x-0.5 font-mono">{defaultSource}</div>
              {release && (
                <div className="mt-0.5 text-center font-mono text-[8px] font-medium leading-none text-slate-400">
                  Source/SAE
                </div>
              )}
              {/* <div className="max-w font-sans text-[9px] capitalize text-slate-500">
                {release.creatorName}
              </div> */}
            </div>
            <ChevronDownIcon className="-mr-1 ml-0 w-2 leading-none sm:w-4" />
          </button>
        </DropdownMenu.Trigger>
        <DropdownMenu.Portal>
          <DropdownMenu.Content
            onPointerDownOutside={() => {
              setDropdownOpen(false);
              setReleaseOpen(release?.name);
              setSourceSetOpen(defaultSource ? getSourceSetNameFromSource(defaultSource) : '');
            }}
            sideOffset={3}
            className="z-30 cursor-pointer overflow-hidden rounded bg-white text-xs font-medium text-sky-700 shadow-[0px_10px_38px_-10px_rgba(22,_23,_24,_0.35),_0px_10px_20px_-15px_rgba(22,_23,_24,_0.2)]"
          >
            <DropdownMenu.Label className="sticky top-0 cursor-default border-b border-slate-100 bg-white py-1.5 text-center text-[10px] uppercase text-slate-400">
              Release
            </DropdownMenu.Label>
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
