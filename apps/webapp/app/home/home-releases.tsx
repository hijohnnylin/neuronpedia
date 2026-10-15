'use client';

import {
  betaReleases,
  communityReleases,
  featuredStarReleases,
  sortReleases,
} from '@/components/nav/releases-dropdown';
import { useGlobalContext } from '@/components/provider/global-provider';
import { UNNAMED_AUTHOR_NAME } from '@/lib/utils/general';
import { StarFilledIcon } from '@radix-ui/react-icons';
import { CircleAlertIcon, University } from 'lucide-react';
import Link from 'next/link';

export default function HomeReleases() {
  const { releases } = useGlobalContext();

  return (
    <div className="forceShowScrollBar flex max-h-[420px] flex-1 flex-col divide-y divide-slate-100 overflow-y-scroll">
      {sortReleases(releases.filter((release) => release.featured)).map((release) => (
        <Link
          key={release.name}
          href={`/${release.name}`}
          prefetch={false}
          className="relative flex w-full flex-col items-start justify-center gap-x-2 gap-y-0.5 rounded px-3 py-2 pr-5 text-xs font-medium hover:bg-sky-100 hover:text-sky-800"
        >
          {featuredStarReleases.includes(release.name ?? '') && (
            <div className="mb-0 inline-flex flex-row items-center gap-x-0.5 text-[8.5px] font-bold uppercase text-emerald-600">
              <StarFilledIcon className="h-2 w-2" /> Featured
            </div>
          )}
          {communityReleases.includes(release.name ?? '') && (
            <div className="mb-0 inline-flex flex-row items-center gap-x-0.5 text-[8.5px] font-bold uppercase text-sky-600">
              <University className="h-2.5 w-2.5" /> Community
            </div>
          )}
          {betaReleases.includes(release.name ?? '') && (
            <div className="mb-0 inline-flex flex-row items-center gap-x-0.5 text-[8.5px] font-bold uppercase text-yellow-600">
              <CircleAlertIcon className="h-2 w-2" /> Beta
            </div>
          )}
          <div className="mb-0.5 text-left font-sans text-[12px] font-bold leading-tight text-sky-700">
            <span>{release.description}</span>
          </div>

          <div className="text-left text-[11px] font-normal leading-tight text-slate-600">
            {release.creatorName !== UNNAMED_AUTHOR_NAME ? `${release.creatorName}` : 'Anonymous Peer Review'}
          </div>
        </Link>
      ))}
    </div>
  );
}
