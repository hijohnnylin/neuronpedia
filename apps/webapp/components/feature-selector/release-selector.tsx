'use client';

import { env } from '@/lib/env';
import { SourceReleaseWithPartialRelations } from '@/prisma/generated/zod';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import { ChevronDownIcon } from 'lucide-react';
import { useState } from 'react';
import { useGlobalContext } from '../provider/global-provider';

export default function ReleaseSelector({
  defaultReleaseName,
  onReleaseChange,
}: {
  defaultReleaseName: string;
  onReleaseChange: (releaseName: string) => void;
}) {
  const { releases } = useGlobalContext();
  const getReleaseForName = (releaseName: string) =>
    releases.find((r) => r.name === releaseName) as SourceReleaseWithPartialRelations;
  const [release, setRelease] = useState<SourceReleaseWithPartialRelations | undefined>(
    defaultReleaseName
      ? getReleaseForName(defaultReleaseName)
      : getReleaseForName(env.NEXT_PUBLIC_DEFAULT_RELEASE_NAME),
  );

  return (
    <div className="flex select-none flex-col">
      <DropdownMenu.Root>
        <DropdownMenu.Trigger asChild>
          <button
            type="button"
            className="flex h-10 max-h-10 min-h-10 w-full flex-1 flex-row items-center justify-center gap-x-1 rounded border border-slate-300 bg-white px-2 py-2 text-[10px] font-medium text-sky-700 hover:bg-slate-50 focus:outline-none sm:pl-5 sm:pr-2 sm:text-xs"
          >
            <div className="flex flex-1 flex-col items-center justify-center gap-y-0 overflow-x-hidden">
              <div className="flex w-full flex-row items-center justify-center gap-x-0.5 overflow-x-hidden whitespace-pre text-center">
                {release?.descriptionShort}
              </div>
              <div className="py-0.5 font-sans text-[9px] capitalize leading-none text-slate-500">
                {release?.creatorName}
              </div>
            </div>
            <ChevronDownIcon className="-mr-1 ml-0 w-2 min-w-4 leading-none sm:w-4" />
          </button>
        </DropdownMenu.Trigger>
        <DropdownMenu.Portal>
          <DropdownMenu.Content
            sideOffset={-3}
            className="z-30 cursor-pointer overflow-hidden rounded bg-white text-xs font-medium text-sky-700 shadow-[0px_10px_38px_-10px_rgba(22,_23,_24,_0.35),_0px_10px_20px_-15px_rgba(22,_23,_24,_0.2)]"
          >
            {releases
              .filter((r) => r.featured)
              .sort((a, b) => {
                if (a.descriptionShort && b.descriptionShort) {
                  return a.descriptionShort.localeCompare(b.descriptionShort);
                }
                return 0;
              })
              .map((releaseItem) => (
                <DropdownMenu.Item
                  onSelect={() => {
                    setRelease(releaseItem as SourceReleaseWithPartialRelations);
                    onReleaseChange(releaseItem.name ?? '');
                  }}
                  key={releaseItem.name}
                  className={`${
                    releaseItem.name === release?.name ? 'bg-sky-200 text-slate-600' : 'bg-white'
                  } group flex w-full max-w-[380px] flex-1 cursor-pointer flex-row items-center justify-between gap-x-1 border-b-slate-100 px-3 py-2.5 text-xs font-medium hover:bg-sky-100 hover:text-slate-600 focus:outline-none data-[state=open]:bg-sky-100 data-[state=open]:text-slate-600 [&:not(:last-child)]:border-b`}
                >
                  <div className="flex w-full flex-col items-start justify-center gap-y-1.5 leading-tight">
                    {releaseItem && (
                      <div
                        className={`font-sans text-[12px] font-semibold capitalize ${
                          releaseItem.name === release?.name ? 'text-slate-600' : 'text-slate-500'
                        } group-hover:text-slate-600 group-data-[state=open]:text-slate-600`}
                      >
                        {releaseItem.descriptionShort}
                      </div>
                    )}
                    <div className="flex w-full flex-row items-center justify-between gap-x-5 text-[10.5px] text-sky-700/70 group-hover:text-sky-700 group-data-[state=open]:text-sky-700">
                      <div className="font-medium text-slate-500">{releaseItem.creatorNameShort}</div>
                      <div className="whitespace-pre font-mono font-bold uppercase">{releaseItem?.name}</div>
                    </div>
                  </div>
                </DropdownMenu.Item>
              ))}
            <DropdownMenu.Arrow className="fill-white" />
          </DropdownMenu.Content>
        </DropdownMenu.Portal>
      </DropdownMenu.Root>
    </div>
  );
}
