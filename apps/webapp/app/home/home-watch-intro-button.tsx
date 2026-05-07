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
        className="h-12 min-h-12 transition-all hover:scale-105 sm:w-auto"
      >
        <div className="flex h-12 min-h-12 flex-row items-center justify-center gap-x-1.5 rounded-xl bg-[#D4A274] px-5 py-2 text-[#262625] shadow-sm shadow-[#666663]/60 sm:px-3">
          <Youtube className="h-5 w-5" />
          <div className="text-[12px] font-semibold leading-tight">Watch Intro</div>
        </div>
      </button>

      <NlaIntroVideoModal open={open} onOpenChange={setOpen} />
    </>
  );
}
