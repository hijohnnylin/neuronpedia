'use client';

import { useGlobalContext } from '@/components/provider/global-provider';
import { env } from '@/lib/env';
import * as Select from '@radix-ui/react-select';
import { ChevronDown, ChevronUp, Plus } from 'lucide-react';
import { useRouter } from 'next-nprogress-bar';
import Link from 'next/link';

export default function ModelsDropdown({ isInBreadcrumb = false }: { isInBreadcrumb?: boolean }) {
  const router = useRouter();
  const { globalModels } = useGlobalContext();

  return (
    <Select.Root
      defaultValue="model"
      value="model"
      onValueChange={(newVal) => {
        router.push(`/${newVal}`);
      }}
    >
      <Select.Trigger
        className={`flex cursor-pointer items-center whitespace-nowrap rounded-full px-2.5 py-1 ${
          isInBreadcrumb ? '-mx-1.5' : 'text-[13px]'
        } transition-all hover:bg-sky-100 hover:text-sky-700 focus:outline-none data-[state=open]:bg-sky-700 data-[state=open]:text-white`}
      >
        Models
        <Select.Icon>
          <ChevronDown className="-mr-0.5 ml-0.5 h-3.5 w-3.5" />
        </Select.Icon>
      </Select.Trigger>
      <Select.Portal>
        <Select.Content
          position="popper"
          align="center"
          sideOffset={-3}
          className="z-30 cursor-pointer overflow-hidden rounded-md bg-white text-xs font-medium text-sky-700 shadow-[0px_10px_38px_-10px_rgba(22,_23,_24,_0.35),_0px_10px_20px_-15px_rgba(22,_23,_24,_0.2)]"
        >
          <Select.ScrollUpButton className="flex justify-center border-b text-slate-600">
            <ChevronUp />
          </Select.ScrollUpButton>
          <Select.Viewport className="text-left">
            <Select.Item
              key="model"
              value="model"
              className="hidden flex-col overflow-hidden border-b px-6 py-2.5 font-sans text-xs text-slate-400 hover:bg-slate-100 focus:outline-none"
            >
              <Select.ItemText>Models</Select.ItemText>
            </Select.Item>
            {Object.keys(globalModels)
              .sort((a, b) => a.localeCompare(b))
              .filter((mId) => globalModels[mId].visibility === 'PUBLIC')
              .map((mId) => (
                <Select.Item
                  key={mId}
                  value={mId}
                  className="flex flex-col overflow-hidden border-b px-6 py-3 font-sans text-xs text-sky-700 hover:bg-slate-100 focus:outline-none"
                >
                  <Select.ItemText className="font-mono">
                    <span className="font-mono">{globalModels[mId].displayNameShort}</span>
                  </Select.ItemText>
                  <div className="mt-0 flex flex-row justify-between gap-x-5 text-[10px] font-normal text-slate-500">
                    <div>{globalModels[mId].displayName}</div>
                    <div>{globalModels[mId].owner}</div>
                  </div>
                </Select.Item>
              ))}
            {!isInBreadcrumb && (
              <Link
                href={env.IS_LOCALHOST ? '/model/new' : 'https://forms.gle/Yg51TYFutJysiyDP7'}
                target={env.IS_LOCALHOST ? undefined : '_blank'}
                rel="noreferrer"
                className="col-span-2 flex h-9 flex-row items-center justify-center gap-x-0.5 rounded bg-white px-3 py-1 text-[11px] text-slate-600 hover:bg-slate-100"
              >
                <Plus className="h-3.5 w-3.5" />
                Add New Model
              </Link>
            )}
          </Select.Viewport>
          <Select.ScrollDownButton className="flex justify-center border-t text-slate-600">
            <ChevronDown />
          </Select.ScrollDownButton>

          <Select.Arrow className="fill-white" />
        </Select.Content>
      </Select.Portal>
    </Select.Root>
  );
}
