'use client';

import { DEFAULT_MODELID, DEFAULT_SOURCESET } from '@/lib/env';
import { useRouter } from '@bprogress/next';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import { ChevronDownIcon, ChevronLeft, ChevronRight, Search } from 'lucide-react';
import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';
import { useGlobalContext } from '../provider/global-provider';
import ModelSelector from './model-selector';
import SourceSelector from './source-selector';

// Styled dropdown for picking an attention head index. Navigates to the head page on select.
function HeadIndexDropdown({
  modelId,
  headLayer,
  numHeadIndexes,
  selectedIndex,
  openInNewTab,
  keepHeadFinderOpen = false,
}: {
  modelId: string;
  headLayer: number;
  numHeadIndexes: number;
  selectedIndex?: string;
  openInNewTab?: boolean;
  keepHeadFinderOpen?: boolean;
}) {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  return (
    <DropdownMenu.Root open={open} onOpenChange={setOpen}>
      <DropdownMenu.Trigger asChild>
        <button
          type="button"
          className="flex h-10 max-h-[40px] min-h-[40px] flex-row items-center justify-center gap-x-1 rounded border border-slate-300 bg-white px-3 py-0 pr-1 text-center font-mono text-xs font-medium text-sky-700 hover:bg-slate-50 focus:outline-none"
        >
          <div className="flex flex-1 flex-col items-center justify-center">
            <div className="flex flex-row items-center justify-center gap-x-0.5 text-[11px] leading-none text-sky-700 sm:text-xs">
              HEAD {selectedIndex && selectedIndex.length > 0 ? selectedIndex : '0'}
            </div>
            {/* <div className="mt-0.5 text-center font-mono text-[8px] font-medium leading-none text-slate-400">HEAD</div> */}
          </div>
          <ChevronDownIcon className="w-4" />
        </button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content
          sideOffset={3}
          className="forceShowScrollBar z-50 max-h-[340px] min-w-[72px] cursor-pointer overflow-y-scroll rounded bg-white text-xs font-medium text-sky-700 shadow-[0px_10px_38px_-10px_rgba(22,_23,_24,_0.35),_0px_10px_20px_-15px_rgba(22,_23,_24,_0.2)] sm:max-h-[500px]"
        >
          <DropdownMenu.Label className="sticky top-0 cursor-default border-b border-slate-100 bg-white py-1.5 text-center text-[10px] uppercase text-slate-400">
            Head
          </DropdownMenu.Label>
          {Array.from({ length: numHeadIndexes }, (_, i) => i).map((idx) => (
            <DropdownMenu.Item
              key={idx}
              onSelect={() => {
                const url = `/${modelId}/head/${headLayer}/${idx}${keepHeadFinderOpen ? '?headFinder=true' : ''}`;
                if (openInNewTab) {
                  window.open(url, '_blank');
                } else {
                  router.push(url);
                }
              }}
              className={`${
                String(idx) === selectedIndex ? 'bg-sky-200 text-sky-700' : 'bg-white text-sky-700/70 hover:bg-sky-100'
              } flex w-full cursor-pointer items-center justify-center px-5 py-2 font-mono text-[12px] font-bold focus:outline-none`}
            >
              {idx}
            </DropdownMenu.Item>
          ))}
          <DropdownMenu.Arrow className="fill-white" />
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}

export default function FeatureSelector({
  defaultModelId,
  defaultSourceSet,
  defaultSource,
  defaultIndex = '0',
  filterToRelease,
  filterToInferenceEnabled = false,
  filterToFeaturedReleases = true,
  filterToPublic = false,
  showModel = true,
  showNextPrev = false,
  openInNewTab = true,
  modelOnSeparateRow = false,
  autoFocus = true,
  callback,
  exclusiveCallback = false,
  includeHeads = false,
  numHeadLayers = 0,
  numHeadIndexes = 0,
  defaultHeadLayer,
  showHeadFinderToggle = false,
  headFinderActive = false,
  onHeadFinderToggle,
}: {
  defaultModelId?: string;
  defaultSourceSet?: string;
  defaultSource?: string;
  defaultIndex?: string | undefined;
  filterToRelease?: string | undefined;
  filterToFeaturedReleases?: boolean;
  filterToInferenceEnabled?: boolean;
  filterToPublic?: boolean;
  showModel?: boolean;
  showNextPrev?: boolean;
  openInNewTab?: boolean;
  modelOnSeparateRow?: boolean;
  autoFocus?: boolean;
  callback?: (feature: { modelId: string; layer: string; index: string }) => void;
  exclusiveCallback?: boolean;
  // When true, the source dropdown gains an "Attention Heads" entry and the index field
  // becomes a head-index dropdown that navigates to /[modelId]/head/[layer]/[index].
  includeHeads?: boolean;
  numHeadLayers?: number;
  numHeadIndexes?: number;
  defaultHeadLayer?: number;
  // Shows a "Head Finder" toggle button (only in head mode) to reveal an external head finder UI.
  showHeadFinderToggle?: boolean;
  headFinderActive?: boolean;
  onHeadFinderToggle?: () => void;
}) {
  const { getSourceSetsForModelId, getFirstSourceForSourceSet, globalModels, getDefaultModel } = useGlobalContext();
  const [modelId, setModelId] = useState(defaultModelId || getDefaultModel()?.id || DEFAULT_MODELID);
  const [sourceSet, setSourceSet] = useState(defaultSourceSet || DEFAULT_SOURCESET);
  const [source, setSource] = useState(defaultSource);
  const [index, setIndex] = useState<string | undefined>(defaultIndex);
  // When defined, the selector is in "attention head" mode and `index` refers to a head index.
  const [headLayer, setHeadLayer] = useState<number | undefined>(defaultHeadLayer);
  const router = useRouter();

  const indexInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (globalModels && !defaultSource) {
      setSource(getFirstSourceForSourceSet(modelId, sourceSet));
    }
  }, []);

  const modelIdChanged = (newModelId: string) => {
    setModelId(newModelId);
    setHeadLayer(undefined);
    const newSourceSet = getSourceSetsForModelId(newModelId, filterToPublic)?.[0].name;
    setSourceSet(newSourceSet);
    setSource(getFirstSourceForSourceSet(newModelId, newSourceSet));
  };

  const layerChanged = (newLayer: string) => {
    setHeadLayer(undefined);
    setSource(newLayer);
  };

  const headLayerChanged = (newHeadLayer: number) => {
    setHeadLayer(newHeadLayer);
  };

  useEffect(() => {
    if (source !== defaultSource) {
      setIndex('0');
      if (autoFocus) {
        setTimeout(() => {
          indexInputRef.current?.focus();
        }, 200);
      }
    }
  }, [source]);

  return (
    <div className="flex flex-col items-start justify-start">
      {showModel && modelOnSeparateRow && (
        <div className="mb-1.5 flex flex-col">
          <ModelSelector modelId={modelId} modelIdChangedCallback={modelIdChanged} filterToRelease={filterToRelease} />
        </div>
      )}
      <div className="flex items-start justify-center gap-x-1 sm:gap-x-2">
        {showNextPrev && (
          <div className="flex flex-row divide-x divide-slate-300 overflow-hidden rounded bg-slate-200">
            <Link
              className={`group hidden h-10 min-h-[40px] select-none flex-col items-center justify-center px-1.5 text-[11px] font-medium uppercase text-slate-500 hover:bg-sky-700 hover:text-white sm:flex ${
                parseInt(defaultIndex, 10) > 0 ? '' : 'pointer-events-none opacity-50'
              }`}
              href={`/${modelId}/${source}/${parseInt(defaultIndex, 10) - 1}`}
            >
              <ChevronLeft className="h-4 w-4" />
              <div className="mt-0.5 text-center text-[8px] font-medium uppercase leading-none text-slate-400 group-hover:text-white">
                Prev
              </div>
            </Link>

            <Link
              className={`group hidden h-10 min-h-[40px] select-none flex-col items-center justify-center px-1.5 text-[11px] font-medium uppercase text-slate-500 hover:bg-sky-700 hover:text-white sm:flex ${
                parseInt(defaultIndex, 10) > 0 ? '' : ''
              }`}
              href={`/${modelId}/${source}/${parseInt(defaultIndex, 10) + 1}`}
            >
              <ChevronRight className="h-4 w-4" />
              <div className="mt-0.5 text-center text-[8px] font-medium uppercase leading-none text-slate-400 group-hover:text-white">
                Next
              </div>
            </Link>
          </div>
        )}
        {showModel && !modelOnSeparateRow && (
          <ModelSelector modelId={modelId} modelIdChangedCallback={modelIdChanged} filterToRelease={filterToRelease} />
        )}
        <SourceSelector
          modelId={modelId}
          defaultSource={source}
          sourceChangedCallback={layerChanged}
          defaultSourceSet={defaultSourceSet}
          filterToRelease={filterToRelease}
          filterToInferenceEnabled={filterToInferenceEnabled}
          filterToFeaturedReleases={filterToFeaturedReleases}
          filterToPublic={filterToPublic}
          includeHeads={includeHeads}
          numHeadLayers={numHeadLayers}
          selectedHeadLayer={headLayer}
          headLayerChangedCallback={headLayerChanged}
        />

        {includeHeads && headLayer !== undefined ? (
          <>
            <HeadIndexDropdown
              modelId={modelId}
              headLayer={headLayer}
              numHeadIndexes={numHeadIndexes}
              selectedIndex={index}
              openInNewTab={openInNewTab}
              keepHeadFinderOpen={headFinderActive}
            />
            {showHeadFinderToggle && includeHeads && headLayer !== undefined && (
              <button
                type="button"
                onClick={onHeadFinderToggle}
                aria-pressed={headFinderActive}
                className={`flex h-10 max-h-[40px] min-h-[40px] select-none flex-col items-center justify-center rounded border px-2 focus:outline-none ${
                  headFinderActive
                    ? 'border-sky-600 bg-sky-600 text-white hover:bg-sky-700'
                    : 'bg-slate-200 text-slate-500 hover:bg-sky-200 hover:text-sky-700'
                }`}
              >
                <Search className="h-3.5 w-3.5" />
                <div className="mt-0.5 text-center text-[8px] font-medium uppercase leading-none">Finder</div>
              </button>
            )}
          </>
        ) : (
          <>
            <div className="flex flex-col">
              <div className="flex h-10 max-h-[40px] min-h-[40px] flex-col items-center justify-center rounded border border-slate-300 bg-white px-0.5 py-0 text-center font-mono text-xs font-medium text-sky-700 placeholder-slate-400 focus:border-sky-700 focus:ring-0 sm:w-[60px] sm:px-1">
                <input
                  type="text"
                  ref={indexInputRef}
                  value={index}
                  onChange={(e) => {
                    const newValue = parseInt(e.target.value, 10);
                    if (!Number.isNaN(newValue)) {
                      setIndex(newValue.toString());
                    } else {
                      setIndex('');
                    }
                  }}
                  onKeyDown={(event: React.KeyboardEvent<any>) => {
                    if (event.key === 'Enter') {
                      event.preventDefault();
                      router.push(`/${modelId}/${source}/${index}`);
                    }
                  }}
                  placeholder="0"
                  className="w-[50px] border-none bg-transparent px-0 py-0 text-center text-[11px] leading-none text-sky-700 placeholder-slate-400 focus:border-none focus:ring-0 sm:w-[60px] sm:text-xs"
                />
                <div className="mt-0.5 text-center font-mono text-[8px] font-medium leading-none text-slate-400">
                  INDEX
                </div>
              </div>
            </div>
            <button
              type="button"
              onClick={(e) => {
                if (index === undefined || index.trim().length === 0) {
                  alert('Must enter a valid index.');
                  e.preventDefault();
                } else if (/^\+?\d+$/.test(index) === false) {
                  alert('Index must be a positive integer.');
                  e.preventDefault();
                } else {
                  // Get defaulttesttext from query params if present
                  const urlParams = new URLSearchParams(window.location.search);
                  const defaultTestText = urlParams.get('defaulttesttext');
                  const url = `/${modelId}/${source}/${index}${defaultTestText ? `?defaulttesttext=${defaultTestText}` : ''}`;
                  if (!exclusiveCallback) {
                    if (openInNewTab || e.metaKey || e.ctrlKey) {
                      window.open(url, '_blank');
                    } else {
                      router.push(url);
                    }
                  }
                  if (callback) {
                    callback({
                      modelId,
                      layer: source || '',
                      index,
                    });
                  }
                }
              }}
              className="flex h-10 max-h-[40px] min-h-[40px] select-none items-center justify-center rounded bg-slate-200 px-3 text-[11px] font-medium uppercase text-slate-500 hover:bg-sky-700 hover:text-white"
            >
              Go
            </button>
          </>
        )}
      </div>
    </div>
  );
}
