'use client';

import { useGlobalContext } from '@/components/provider/global-provider';
import Link from 'next/link';

export default function HomeModels() {
  const { globalModels } = useGlobalContext();

  const models = Object.values(globalModels);

  return (
    <div className="forceShowScrollBar flex max-h-[420px] w-full flex-col divide-y divide-slate-100 overflow-y-scroll">
      {models
        .sort((a, b) => (b.createdAt?.getTime() || 0) - (a.createdAt?.getTime() || 0))
        .filter((model) => model.visibility === 'PUBLIC')
        .map((model) => (
          <Link
            key={model.id}
            href={`/${model.id}`}
            prefetch={false}
            className="relative flex w-full flex-col items-start justify-center gap-x-2 gap-y-0.5 rounded px-3 py-2 pr-5 text-xs font-medium hover:bg-sky-100 hover:text-sky-800"
          >
            <div className="mb-1 text-left font-mono text-[12px] font-bold leading-tight text-sky-700">
              {model.displayNameShort}
            </div>
            <div className="flex w-full flex-row justify-between text-[11px] font-normal leading-tight text-slate-600">
              <span className="">{model.displayName}</span>
              <span className="">{model.owner}</span>
            </div>
          </Link>
        ))}
    </div>
  );
}
