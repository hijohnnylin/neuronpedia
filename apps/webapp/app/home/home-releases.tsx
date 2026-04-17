'use client';

import { featuredStarReleases } from '@/components/nav/releases-dropdown';
import { useGlobalContext } from '@/components/provider/global-provider';
import { UNNAMED_AUTHOR_NAME } from '@/lib/utils/general';
import { StarFilledIcon } from '@radix-ui/react-icons';
import Link from 'next/link';

export default function HomeReleases() {
  const { releases } = useGlobalContext();

  return (
    <div className="forceShowScrollBar flex max-h-[420px] flex-1 flex-col divide-y divide-slate-100 overflow-y-scroll">
      {releases
        .sort((a, b) => {
          const aIsStar = featuredStarReleases.includes(a.name ?? '');
          const bIsStar = featuredStarReleases.includes(b.name ?? '');
          if (aIsStar && !bIsStar) return -1;
          if (!aIsStar && bIsStar) return 1;
          // If both are (or aren't) starred, order by date
          return (b.createdAt?.getTime() || 0) - (a.createdAt?.getTime() || 0);
        })
        .filter((release) => release.featured)
        .map((release) => (
          <Link
            key={release.name}
            href={`/${release.name}`}
            prefetch={false}
            className="relative flex w-full flex-col items-start justify-center gap-x-2 gap-y-0.5 rounded px-3 py-3 pr-5 text-xs font-medium hover:bg-sky-100 hover:text-sky-800"
          >
            {featuredStarReleases.includes(release.name ?? '') && (
              <div className="mb-0 inline-flex flex-row items-center gap-x-0.5 text-[8.5px] font-bold uppercase text-emerald-600">
                <StarFilledIcon className="h-2 w-2" /> Featured
              </div>
            )}
            <div className="mb-1 text-left font-sans text-[12px] font-bold leading-tight text-sky-700">
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
