'use client';

import { getFullPageHref, isInteractiveEmbedPath } from '@/lib/utils/embed';
import { ExternalLinkIcon } from '@radix-ui/react-icons';
import Image from 'next/image';
import { usePathname, useSearchParams } from 'next/navigation';

// CSS hides this bar until the root layout's script marks the page as framed (html.in-iframe).
export default function EmbedOpenFullPageBar() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  if (!isInteractiveEmbedPath(pathname)) {
    return null;
  }

  return (
    <a
      href={getFullPageHref(pathname, new URLSearchParams(searchParams.toString()))}
      target="_blank"
      rel="noopener noreferrer"
      className="hidden h-[var(--embed-bar-height)] w-full shrink-0 items-center justify-center gap-x-1.5 border-b border-slate-200 bg-slate-100 text-[11px] font-medium text-sky-700 transition-colors hover:bg-slate-200 [.in-iframe_&]:flex"
    >
      <Image src="/logo.png" alt="" width="14" height="14" />
      Open in Neuronpedia
      <ExternalLinkIcon className="h-3 w-3" />
    </a>
  );
}
