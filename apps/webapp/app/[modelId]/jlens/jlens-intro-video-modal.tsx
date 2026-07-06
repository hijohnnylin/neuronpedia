'use client';

import { Button } from '@/components/shadcn/button';
import { Dialog, DialogClose, DialogContent, DialogTitle } from '@/components/shadcn/dialog';
import { JLENS_YOUTUBE_URL } from './jlens-urls';

// Extracts the YouTube video ID from a full URL, e.g.
// https://youtu.be/j2knrqAzYVY -> j2knrqAzYVY
// https://www.youtube.com/watch?v=j2knrqAzYVY -> j2knrqAzYVY
function getYoutubeVideoId(url: string): string {
  if (!url) {
    return '';
  }
  try {
    const parsed = new URL(url);
    if (parsed.hostname.includes('youtu.be')) {
      return parsed.pathname.replace('/', '');
    }
    const idFromQuery = parsed.searchParams.get('v');
    if (idFromQuery) {
      return idFromQuery;
    }
    if (parsed.pathname.includes('/embed/')) {
      return parsed.pathname.split('/embed/')[1] ?? '';
    }
  } catch {
    return '';
  }
  return '';
}

const JLENS_YOUTUBE_VIDEO_ID = getYoutubeVideoId(JLENS_YOUTUBE_URL);

export default function JlensIntroVideoModal({
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
        <DialogTitle className="sr-only">Jacobian Lens Intro Video</DialogTitle>
        <div className="relative w-full overflow-hidden rounded-lg" style={{ paddingBottom: '56.25%' }}>
          {open && (
            <iframe
              title="Jacobian Lens Intro Video"
              src={`https://www.youtube.com/embed/${JLENS_YOUTUBE_VIDEO_ID}?autoplay=1&rel=0`}
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
