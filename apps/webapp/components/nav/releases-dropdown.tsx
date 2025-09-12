'use client';

import { useGlobalContext } from '@/components/provider/global-provider';
import { env } from '@/lib/env';
import { UNNAMED_AUTHOR_NAME } from '@/lib/utils/general';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import { StarFilledIcon } from '@radix-ui/react-icons';
import { ChevronDown, Plus } from 'lucide-react';
import { useRouter } from 'next-nprogress-bar';
import Link from 'next/link';

export default function ReleasesDropdown({ breadcrumb = false }: { breadcrumb?: boolean }) {
  const router = useRouter();
  const { releases } = useGlobalContext();
  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger
        className={`flex cursor-pointer items-center whitespace-nowrap rounded-full px-2.5 py-1 ${
          breadcrumb ? '-mx-1.5' : 'text-[13px]'
        } transition-all hover:bg-sky-100 hover:text-sky-700 focus:outline-none data-[state=open]:bg-sky-700 data-[state=open]:text-white`}
      >
        <span>Releases</span>
        <ChevronDown className="-mr-0.5 ml-0.5 h-3.5 w-3.5" />
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content
          align="center"
          sideOffset={-3}
          className="forceShowScrollBar z-50 max-h-[550px] w-full cursor-pointer overflow-hidden overflow-y-scroll rounded-md bg-white text-xs font-medium text-sky-700 shadow-[0px_10px_38px_-10px_rgba(22,_23,_24,_0.35),_0px_10px_20px_-15px_rgba(22,_23,_24,_0.2)]"
        >
          {releases
            .sort((a, b) => (b.createdAt?.getTime() || 0) - (a.createdAt?.getTime() || 0))
            .filter((release) => release.featured)
            .map((release) => (
              <DropdownMenu.Item
                key={release.name}
                className="flex max-w-[380px] flex-row items-center justify-between gap-x-1.5 border-b border-slate-100 hover:bg-slate-100 focus:outline-none"
              >
                <button
                  type="button"
                  onClick={() => {
                    router.push(`/${release.name}`);
                  }}
                  className="flex w-full flex-col gap-y-0 overflow-hidden px-6 py-2.5 text-left font-sans text-xs text-sky-700 focus:outline-none"
                >
                  <div className="mt-0 flex flex-row justify-between gap-x-3 text-[12px] font-medium">
                    <div>{release.descriptionShort}</div>
                    {release.name === 'gemma-scope' && (
                      <div className="flex flex-row items-center gap-x-0.5 text-[8.5px] font-bold uppercase text-emerald-600">
                        <StarFilledIcon className="h-2 w-2" /> Featured
                      </div>
                    )}
                    {/* <div className="whitespace-pre text-[10px]  font-medium text-slate-400">
                        {release?.createdAt?.toLocaleDateString("en-US", {
                          year: "numeric",
                          month: "long",
                        })}
                      </div> */}
                  </div>
                  <div className="flex flex-row justify-between gap-x-3 text-[9px] font-normal text-slate-500">
                    {release.creatorName !== UNNAMED_AUTHOR_NAME
                      ? `${release.creatorName}`
                      : 'Under Anonymous Peer Review'}
                    <div className="font-mono text-slate-400">{release.name?.toUpperCase()}</div>
                  </div>
                </button>
              </DropdownMenu.Item>
            ))}
          {!breadcrumb && (
            <Link
              href={env.IS_LOCALHOST ? '/sae/new' : 'https://forms.gle/Yg51TYFutJysiyDP7'}
              prefetch={false}
              target={env.IS_LOCALHOST ? undefined : '_blank'}
              rel="noreferrer"
              className="col-span-2 flex h-9 flex-row items-center justify-center gap-x-0.5 rounded bg-white px-3 py-1 text-[11px] text-slate-600 hover:bg-slate-100"
            >
              <Plus className="h-3.5 w-3.5" />
              Submit New Release
            </Link>
          )}
          <DropdownMenu.Arrow className="fill-white" />
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}
