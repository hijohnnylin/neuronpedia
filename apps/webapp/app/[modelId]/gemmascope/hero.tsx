'use client';

import { Button } from '@/components/shadcn/button';
import { getVisibilityBadge } from '@/components/visibility-badge';
import { SourceReleaseWithPartialRelations } from '@/prisma/generated/zod';
import { Visibility } from '@prisma/client';
import React from 'react';

export default function Hero({ release }: { release: SourceReleaseWithPartialRelations }) {
  return (
    <div className="flex w-full flex-row items-center justify-center border-b border-slate-200 py-6">
      <div className="flex w-full max-w-screen-lg flex-col items-center justify-between gap-x-5 px-3 sm:flex-row sm:px-0">
        <div className="flex flex-1 flex-col items-start">
          {release.visibility !== Visibility.PUBLIC && (
            <div className="pb-1">{getVisibilityBadge(release.visibility)}</div>
          )}
          <div className="text-[17px] font-bold text-slate-900">{release.description}</div>
          <div className="mt-1 flex w-full flex-1 flex-col items-start justify-start gap-x-2 text-sm font-normal text-slate-500 sm:flex-row sm:items-center sm:justify-start">
            {release.creatorName}
            <div className="flex flex-col items-start justify-center gap-x-1.5 sm:flex-row sm:items-center">
              {release?.urls.map((url, i) => (
                <React.Fragment key={i}>
                  <span className="hidden sm:block">·</span>
                  <a
                    key={i}
                    href={url}
                    target="_blank"
                    rel="noreferrer"
                    className="text-[11px] text-slate-400 hover:text-sky-700 hover:underline"
                  >
                    {new URL(url).hostname.replace('www.', '')} ↗
                  </a>
                </React.Fragment>
              ))}
            </div>
          </div>
          {release?.creatorEmail && (
            <div className="mt-2 flex flex-col items-end">
              <Button
                onClick={() => {
                  window.open(
                    `mailto:${
                      release?.creatorEmail
                    }?subject=SAE%20Detail%20Request%3A%20${release.name.toUpperCase()}&body=I'd%20like%20to%20contact%20the%20researcher%20who%20created%20the%20${release.name.toUpperCase()}%20SAEs.%20Please%20put%20me%20in%20touch%20-%20thanks!`,
                    '_blank',
                  );
                }}
                variant="outline"
                size="sm"
              >
                Contact Admin
              </Button>
            </div>
          )}
        </div>
        <div className="hidden flex-col text-right text-sm text-slate-500 sm:flex">
          {/* <div>
            {release.createdAt.toLocaleDateString('en-US', {
              year: 'numeric',
              month: 'long',
            })}
          </div> */}
          <div className="mt-1 whitespace-pre font-mono text-xs font-medium uppercase text-sky-700">{release.name}</div>
        </div>
      </div>
    </div>
  );
}
