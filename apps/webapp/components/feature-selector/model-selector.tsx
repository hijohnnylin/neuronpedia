'use client';

import { DEFAULT_MODELID } from '@/lib/env';
import * as Select from '@radix-ui/react-select';
import { ChevronDownIcon, ChevronUpIcon } from 'lucide-react';
import { useGlobalContext } from '../provider/global-provider';

export default function ModelSelector({
  modelId = DEFAULT_MODELID,
  modelIdChangedCallback,
  filterToInferenceEnabled = false,
  filterToRelease,
  showUnlisted = false,
  overrideModels,
  id,
}: {
  modelId: string;
  modelIdChangedCallback: (modelId: string) => void;
  filterToInferenceEnabled?: boolean;
  filterToRelease?: string | undefined;
  showUnlisted?: boolean;
  overrideModels?: string[];
  id?: string;
}) {
  const { globalModels, getSourceSetsForModelId, getInferenceEnabledForModel } = useGlobalContext();

  const modelIdsToUse = overrideModels || Object.keys(globalModels);

  return (
    <div className="flex select-none flex-col">
      <Select.Root
        defaultValue={modelId}
        value={modelId}
        onValueChange={(newVal) => {
          modelIdChangedCallback(newVal);
        }}
      >
        <Select.Trigger
          id={id}
          className="flex h-10 max-h-[40px] min-h-[40px] w-full flex-1 flex-row items-center justify-center gap-x-1 whitespace-pre rounded border border-slate-300 bg-white px-2 font-mono text-[10px] font-medium text-sky-700 hover:bg-slate-50 focus:outline-none sm:pl-4 sm:pr-2 sm:text-xs"
        >
          <div className="flex flex-col gap-y-0">
            <Select.Value />
            <div className="mt-0.5 text-center text-[8px] font-medium leading-none text-slate-400">MODEL</div>
          </div>
          <Select.Icon>
            <ChevronDownIcon className="-mr-1 ml-0 w-4 leading-none sm:w-4" />
          </Select.Icon>
        </Select.Trigger>
        <Select.Portal>
          <Select.Content
            position="popper"
            align="center"
            sideOffset={3}
            className="z-30 max-h-[400px] cursor-pointer overflow-hidden rounded-md border-slate-300 bg-white text-xs font-medium text-sky-700 shadow-[0px_10px_38px_-10px_rgba(22,_23,_24,_0.35),_0px_10px_20px_-15px_rgba(22,_23,_24,_0.2)]"
          >
            <Select.ScrollUpButton className="flex justify-center border-b text-slate-600">
              <ChevronUpIcon />
            </Select.ScrollUpButton>
            <Select.Viewport className="text-left">
              {modelIdsToUse
                .filter((m) => {
                  if (filterToInferenceEnabled) {
                    return getInferenceEnabledForModel(m);
                  }
                  return true;
                })
                .filter((mId) => {
                  if (showUnlisted || overrideModels) {
                    return true;
                  }
                  return globalModels[mId].visibility !== 'UNLISTED';
                })
                .filter((mId) => {
                  if (filterToRelease) {
                    const ssets = getSourceSetsForModelId(mId).filter((ss) => {
                      if (filterToRelease) {
                        return ss.releaseName === filterToRelease;
                      }
                      return true;
                    });
                    return ssets.length > 0;
                  }
                  return true;
                })
                .sort((a, b) => {
                  if (a && b) {
                    return a.localeCompare(b);
                  }
                  return 0;
                })
                .map((mId) => (
                  <Select.Item
                    key={mId}
                    value={mId}
                    className={`flex flex-col items-start gap-y-0.5 overflow-hidden border-b border-b-slate-100 px-3 py-2.5 text-xs ${mId === modelId ? 'bg-sky-100 text-sky-700' : 'text-slate-600'
                      } hover:bg-slate-100 focus:outline-none`}
                  >
                    <div className="flex w-full flex-row items-center justify-between gap-x-3">
                      <Select.ItemText>
                        <span className="font-mono">{globalModels[mId].displayNameShort}</span>
                      </Select.ItemText>
                      {/* {getVisibilityBadge(models[mId].visibility, true)} */}
                    </div>
                    <div className="mt-0 flex w-full flex-row justify-between gap-x-5 text-[10px] font-normal text-slate-500">
                      <div>{globalModels[mId].displayName}</div>
                      <div>{globalModels[mId].owner}</div>
                    </div>
                  </Select.Item>
                ))}
            </Select.Viewport>
            <Select.ScrollDownButton className="flex justify-center border-t text-slate-600">
              <ChevronDownIcon />
            </Select.ScrollDownButton>

            <Select.Arrow className="fill-white" />
          </Select.Content>
        </Select.Portal>
      </Select.Root>
    </div>
  );
}
