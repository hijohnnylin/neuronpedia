'use client';

import { Youtube } from 'lucide-react';
import { useState } from 'react';
import NlaIntroVideoModal from '../[modelId]/nla/nla-intro-video-modal';

export default function HomeWatchIntroButton() {
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="mt-1 h-12 min-h-12 transition-all hover:scale-105 sm:mt-0 sm:w-auto"
      >
        <div className="flex h-12 min-h-12 flex-row items-center justify-center gap-x-1.5 rounded-xl bg-[#c46e7c] px-5 py-2 text-white shadow-sm shadow-[#5f333b]/30 sm:px-3">
          <Youtube className="h-5 w-5" />
          <span className="text-[12px] font-semibold leading-tight sm:hidden">Watch Intro</span>
        </div>
      </button>

      <NlaIntroVideoModal open={open} onOpenChange={setOpen} />
    </>
  );
}
