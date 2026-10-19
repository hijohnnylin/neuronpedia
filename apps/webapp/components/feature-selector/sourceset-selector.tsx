'use client';

import { NEURONS_SOURCESET } from '@/lib/utils/source';
import * as Select from '@radix-ui/react-select';
import { ChevronDownIcon, ChevronUpIcon } from 'lucide-react';
import { useGlobalContext } from '../provider/global-provider';
import { getVisibilityBadge } from '../visibility-badge';

export default function SourceSetSelector({
  modelId,
  sourceSet = NEURONS_SOURCESET,
  sourceSetChangedCallback,
  filterToOnlyVisible = false,
  filterToRelease,
  filterToInferenceEnabled = false, // should filter out sourceSets that don't support activation testing
  filterToAllowInferenceSearch = false, // some sourceSets support activation single feature testing and steeering but not inference search across the whole set - reason is because the vectors are in the db, not in saelens/inference server
  filterToOnlyHasDashboards = true,
}: {
  modelId: string;
  sourceSet?: string;
  sourceSetChangedCallback?: (sourceSet: string) => void;
  filterToOnlyVisible?: boolean;
  filterToRelease?: string | undefined;
  filterToInferenceEnabled?: boolean | undefined;
  filterToOnlyHasDashboards?: boolean | undefined;
  filterToAllowInferenceSearch?: boolean | undefined;
}) {
  const { getSourceSetsForModelId, getSourceSet } = useGlobalContext();

  const sourceSets = getSourceSetsForModelId(modelId)
    .filter((ss) => (filterToRelease ? ss.releaseName === filterToRelease : true))
    .filter((ss) => (filterToOnlyVisible ? ss.visibility === 'PUBLIC' : true))
    // check that the sourceSet has at least one source that supports activation testing
    .filter((ss) => (filterToInferenceEnabled ? ss.sources?.some((s) => s.inferenceEnabled) : true))
    .filter((ss) => (filterToAllowInferenceSearch ? ss.allowInferenceSearch : true))
    .filter((ss) => (filterToOnlyHasDashboards ? ss.hasDashboards : true))
    .sort((a, b) => {
      if (a.name === NEURONS_SOURCESET) {
        return 1;
      }
      if (b.name === NEURONS_SOURCESET) {
        return -1;
      }
      return a.name.localeCompare(b.name);
    });

  if (sourceSets.length === 0) {
    return (
      <div className="flex flex-col">
        <button
          type="button"
          disabled
          className="flex h-10 max-h-[40px] min-h-[40px] w-full flex-1 cursor-default flex-row items-center justify-center whitespace-pre rounded border border-slate-200 bg-slate-50 px-2 font-mono text-[10px] font-medium uppercase text-slate-400 sm:px-3 sm:text-xs"
        >
          No SAEs Available
        </button>
      </div>
    );
  }

  return (
    <div className="flex flex-col">
      <Select.Root
        defaultValue={sourceSet}
        value={sourceSet}
        onValueChange={(newVal) => {
          sourceSetChangedCallback?.(newVal);
        }}
      >
        <Select.Trigger
          className={`flex ${
            sourceSet === NEURONS_SOURCESET ? 'capitalize' : 'uppercase'
          } h-10 max-h-[40px] min-h-[40px] w-full flex-1 flex-row items-center justify-center gap-x-1 whitespace-pre rounded border border-slate-300 bg-white px-2 text-[10px] font-medium text-sky-700 hover:bg-slate-50 focus:outline-none sm:pl-5 sm:pr-2 sm:text-xs`}
        >
          <div className="flex flex-col items-center justify-center gap-y-1 leading-none">
            <div className="flex flex-row items-center justify-center gap-x-0.5 font-mono">
              <Select.Value />
            </div>
            {sourceSet !== NEURONS_SOURCESET && (
              <div className="font-sans text-[9px] capitalize text-slate-500">
                {getSourceSet(modelId, sourceSet)?.type}
              </div>
            )}
          </div>
          <Select.Icon>
            <ChevronDownIcon className="-mr-1 ml-0 w-4 leading-none" />
          </Select.Icon>
        </Select.Trigger>
        <Select.Portal>
          <Select.Content
            position="popper"
            align="center"
            sideOffset={2}
            className="z-[100] max-h-[420px] cursor-pointer overflow-hidden rounded-md border border-slate-300 bg-white text-xs font-medium text-sky-700 shadow"
          >
            <Select.ScrollUpButton className="flex justify-center border-b bg-slate-100 text-slate-300">
              <ChevronUpIcon className="h-4 w-4" />
            </Select.ScrollUpButton>
            <Select.Viewport className="text-center">
              {sourceSets.map((s) => (
                <Select.Item
                  key={s.name}
                  value={s.name || ''}
                  className={`overflow-hidden ${
                    sourceSet === s.name ? 'bg-sky-100 text-sky-700' : 'text-slate-600'
                  } flex flex-col items-start gap-y-0.5 border-b px-3 py-2.5 font-mono text-xs hover:bg-slate-100 focus:outline-none`}
                >
                  <div className="flex w-full flex-row items-center justify-between gap-x-3">
                    <Select.ItemText>
                      <span className={s.name === NEURONS_SOURCESET ? 'capitalize' : 'uppercase'}>{s.name}</span>
                    </Select.ItemText>
                    {getVisibilityBadge(s.visibility, true)}
                  </div>
                  <Select.ItemText className="text-left" />
                  {s.name !== NEURONS_SOURCESET && (
                    <div className="mt-0 flex w-full flex-row justify-between gap-x-3 font-sans text-[10px] font-normal text-slate-500">
                      <div>{s.type}</div>
                      <div>{s.creatorName}</div>
                    </div>
                  )}
                </Select.Item>
              ))}
            </Select.Viewport>
            <Select.ScrollDownButton className="flex justify-center border-b bg-slate-100 text-slate-300">
              <ChevronDownIcon className="h-4 w-4" />
            </Select.ScrollDownButton>
          </Select.Content>
        </Select.Portal>
      </Select.Root>
    </div>
  );
}
