'use client';

import { Youtube } from 'lucide-react';
import { useState } from 'react';
import JlensIntroVideoModal from '../[modelId]/jlens/jlens-intro-video-modal';

export default function HomeJlensWatchIntroButton() {
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="mt-1 h-11 max-h-11 min-h-11 w-[136px] min-w-[136px] transition-all hover:scale-105 sm:mt-0 sm:w-auto sm:min-w-0"
      >
        <div className="flex h-11 max-h-11 min-h-11 flex-row items-center justify-center gap-x-1.5 rounded-xl bg-[#D4A274] px-2 py-2 text-[#262625] shadow-sm shadow-[#666663]/60 sm:px-5">
          <Youtube className="h-5 w-5" />
          <div className="text-[12px] font-semibold leading-tight">Watch</div>
        </div>
      </button>

      <JlensIntroVideoModal open={open} onOpenChange={setOpen} />
    </>
  );
}
