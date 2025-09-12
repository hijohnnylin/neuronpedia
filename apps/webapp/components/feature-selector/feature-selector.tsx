'use client';

import { env } from '@/lib/env';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { useRouter } from 'next-nprogress-bar';
import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';
import { useGlobalContext } from '../provider/global-provider';
import ModelSelector from './model-selector';
import SourceSelector from './source-selector';

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
}) {
  const { getSourceSetsForModelId, getFirstSourceForSourceSet, globalModels, getDefaultModel } = useGlobalContext();
  const [modelId, setModelId] = useState(defaultModelId || getDefaultModel()?.id || env.NEXT_PUBLIC_DEFAULT_MODELID);
  const [sourceSet, setSourceSet] = useState(defaultSourceSet || env.NEXT_PUBLIC_DEFAULT_SOURCESET);
  const [source, setSource] = useState(defaultSource);
  const [index, setIndex] = useState<string | undefined>(defaultIndex);
  const router = useRouter();

  const indexInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (globalModels && !defaultSource) {
      setSource(getFirstSourceForSourceSet(modelId, sourceSet));
    }
  }, []);

  const modelIdChanged = (newModelId: string) => {
    setModelId(newModelId);
    const newSourceSet = getSourceSetsForModelId(newModelId, filterToPublic)?.[0].name;
    setSourceSet(newSourceSet);
    setSource(getFirstSourceForSourceSet(newModelId, newSourceSet));
  };

  const layerChanged = (newLayer: string) => {
    setSource(newLayer);
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
        />
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
            <div className="mt-0.5 text-center font-mono text-[8px] font-medium leading-none text-slate-400">INDEX</div>
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
              if (!exclusiveCallback) {
                if (openInNewTab || e.metaKey || e.ctrlKey) {
                  window.open(`/${modelId}/${source}/${index}`, '_blank');
                } else {
                  router.push(`/${modelId}/${source}/${index}`);
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
      </div>
    </div>
  );
}
