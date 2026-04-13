'use client';

import ModelSelector from '@/components/feature-selector/model-selector';
import SourceSelector from '@/components/feature-selector/source-selector';
import { useRouter } from '@bprogress/next';
import React from 'react';

export default function JumpToSAE({
  modelId,
  layer,
  filterToRelease,
  filterToFeaturedReleases = true,
  callback = () => {},
  modelOnSeparateRow = false,
  showModel = true,
}: {
  modelId?: string;
  layer?: string;
  filterToRelease?: string;
  filterToFeaturedReleases?: boolean;
  callback?: (modelId: string, layer: string) => void;
  showModel?: boolean;
  modelOnSeparateRow?: boolean;
}) {
  const router = useRouter();
  const [jumpToSAEModelId, setJumpToSAEModelId] = React.useState(modelId || '');
  const [jumpToSAELayer, setJumpToSAELayer] = React.useState(layer || '');
  return (
    <div
      onSelect={(e) => {
        e.preventDefault();
      }}
      className="flex cursor-pointer flex-col items-start text-sm font-medium text-sky-700 outline-none"
    >
      <div className="text-[10px] uppercase text-slate-500">Jump to Source/SAE</div>
      {modelOnSeparateRow && showModel && (
        <div className="mb-1.5 flex flex-col">
          <ModelSelector
            modelIdChangedCallback={(newModelId) => {
              setJumpToSAEModelId(newModelId);
            }}
            modelId={jumpToSAEModelId}
            filterToRelease={filterToRelease}
          />
        </div>
      )}
      <div className="flex flex-row gap-x-2">
        {showModel && !modelOnSeparateRow && (
          <ModelSelector
            modelIdChangedCallback={(newModelId) => {
              setJumpToSAEModelId(newModelId);
            }}
            modelId={jumpToSAEModelId}
            filterToRelease={filterToRelease}
          />
        )}
        <SourceSelector
          modelId={jumpToSAEModelId}
          defaultSource={jumpToSAELayer}
          sourceChangedCallback={setJumpToSAELayer}
          filterToRelease={filterToRelease}
          filterToFeaturedReleases={filterToFeaturedReleases}
          filterToPublic
        />
        <button
          type="button"
          onClick={(e) => {
            if (e.ctrlKey || e.metaKey) {
              window.open(`/${jumpToSAEModelId}/${jumpToSAELayer}`, '_blank');
            } else {
              router.push(`/${jumpToSAEModelId}/${jumpToSAELayer}`);
            }
            callback(jumpToSAEModelId, jumpToSAELayer);
          }}
          className="flex h-10 max-h-[40px] min-h-[40px] select-none items-center justify-center rounded bg-slate-200 px-3 text-[11px] font-medium uppercase text-slate-500 hover:bg-sky-700 hover:text-white"
        >
          Go
        </button>
      </div>
    </div>
  );
}
