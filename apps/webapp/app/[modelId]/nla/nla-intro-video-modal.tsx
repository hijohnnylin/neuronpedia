'use client';

import { Button } from '@/components/shadcn/button';
import { Dialog, DialogClose, DialogContent, DialogTitle } from '@/components/shadcn/dialog';

// https://youtu.be/j2knrqAzYVY -> j2knrqAzYVY
const NLA_YOUTUBE_VIDEO_ID = 'j2knrqAzYVY';

export default function NlaIntroVideoModal({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        hideDefaultClose
        className="flex w-[95vw] max-w-4xl flex-col gap-3 border-none bg-black p-3 sm:rounded-2xl sm:p-4"
      >
        <DialogTitle className="sr-only">Natural Language Autoencoders Intro Video</DialogTitle>
        <div className="relative w-full overflow-hidden rounded-lg" style={{ paddingBottom: '56.25%' }}>
          {open && (
            <iframe
              title="Natural Language Autoencoders Intro Video"
              src={`https://www.youtube.com/embed/${NLA_YOUTUBE_VIDEO_ID}?autoplay=1&rel=0`}
              className="absolute left-0 top-0 h-full w-full border-0"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
              allowFullScreen
            />
          )}
        </div>
        <div className="flex w-full flex-row items-center justify-center pb-1 pt-1">
          <DialogClose asChild>
            <Button
              type="button"
              variant="secondary"
              className="min-w-[120px] bg-white text-slate-800 hover:bg-slate-200"
            >
              Close
            </Button>
          </DialogClose>
        </div>
      </DialogContent>
    </Dialog>
  );
}
