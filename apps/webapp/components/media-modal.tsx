'use client';

import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/shadcn/dialog';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';

export type MediaItem = {
  type: 'image' | 'video';
  src: string;
  alt?: string;
  title?: string;
  subtitle?: string;
};

export function MediaModal({
  open,
  onOpenChange,
  title,
  description,
  items,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description?: string;
  items: MediaItem[];
}) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);
  const hasMultiple = items.length > 1;

  useEffect(() => {
    if (open) setCurrentIndex(0);
  }, [open]);

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.load();
    }
  }, [currentIndex]);

  const goTo = useCallback(
    (index: number) => {
      if (index >= 0 && index < items.length) setCurrentIndex(index);
    },
    [items.length],
  );

  const item = items[currentIndex];
  if (!item) return null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex max-h-[90vh] w-[95vw] max-w-2xl flex-col gap-0 overflow-hidden border-0 bg-white p-0 text-slate-700">
        <DialogHeader className="shrink-0 space-y-1 px-5 pb-3 pt-5 sm:px-6 sm:pt-6">
          <DialogTitle className="text-lg font-bold leading-snug tracking-tight text-slate-800">{title}</DialogTitle>
          {description && <DialogDescription className="text-[13px] text-slate-500">{description}</DialogDescription>}
        </DialogHeader>

        {hasMultiple && (
          <>
            {/* Desktop tabs */}
            <div className="hidden shrink-0 sm:flex">
              {items.map((it, i) => (
                <button
                  type="button"
                  key={i}
                  onClick={() => goTo(i)}
                  className={`flex flex-1 flex-col items-center justify-center gap-y-0.5 px-3 py-2.5 text-xs font-medium transition-colors focus:outline-none ${
                    i === currentIndex
                      ? 'bg-slate-300 text-slate-700'
                      : 'bg-slate-200 text-slate-500 hover:bg-slate-300/80'
                  }`}
                >
                  {it.title && (
                    <span className={`text-[12px] leading-none ${i === currentIndex ? 'font-semibold' : ''}`}>
                      {it.title}
                    </span>
                  )}
                  {it.subtitle && (
                    <span
                      className={`text-[9px] uppercase leading-none ${i === currentIndex ? 'text-slate-500' : 'text-slate-400'}`}
                    >
                      {it.subtitle}
                    </span>
                  )}
                  {!it.title && !it.subtitle && <span className="text-[12px] leading-none">{i + 1}</span>}
                </button>
              ))}
            </div>

            {/* Mobile: title + subtitle above media */}
            {(item.title || item.subtitle) && (
              <div className="flex flex-col items-center gap-0.5 px-5 pb-1 pt-2 sm:hidden">
                {item.title && <span className="text-sm font-semibold text-slate-700">{item.title}</span>}
                {item.subtitle && <span className="text-[11px] uppercase text-slate-400">{item.subtitle}</span>}
              </div>
            )}
          </>
        )}

        <div className="relative flex min-h-0 flex-1 flex-col items-center justify-center px-5 py-4 sm:px-6">
          {item.type === 'image' ? (
            <img
              src={item.src}
              alt={item.alt || item.title || ''}
              className="max-h-[55vh] w-full rounded-lg object-contain"
            />
          ) : (
            <video
              ref={videoRef}
              src={item.src}
              controls
              muted
              loop
              className="max-h-[55vh] w-full rounded-lg object-contain"
            />
          )}

          {hasMultiple && (
            <>
              <button
                type="button"
                onClick={() => goTo(currentIndex - 1)}
                disabled={currentIndex === 0}
                className="absolute left-1 top-1/2 -translate-y-1/2 rounded-full bg-white/80 p-1.5 text-slate-600 shadow transition-opacity hover:bg-white disabled:opacity-0 sm:left-2 sm:p-2"
              >
                <ChevronLeft className="h-4 w-4 sm:h-5 sm:w-5" />
              </button>
              <button
                type="button"
                onClick={() => goTo(currentIndex + 1)}
                disabled={currentIndex === items.length - 1}
                className="absolute right-1 top-1/2 -translate-y-1/2 rounded-full bg-white/80 p-1.5 text-slate-600 shadow transition-opacity hover:bg-white disabled:opacity-0 sm:left-auto sm:right-2 sm:p-2"
              >
                <ChevronRight className="h-4 w-4 sm:h-5 sm:w-5" />
              </button>
            </>
          )}
        </div>

        {/* Mobile dots */}
        {hasMultiple && (
          <div className="flex shrink-0 items-center justify-center gap-1.5 pb-4 sm:hidden">
            {items.map((_, i) => (
              <button
                type="button"
                key={i}
                onClick={() => goTo(i)}
                className={`h-2 w-2 rounded-full transition-colors ${
                  i === currentIndex ? 'bg-slate-700' : 'bg-slate-300'
                }`}
              />
            ))}
          </div>
        )}

        {/* Desktop prev/next footer */}
        {/* {hasMultiple && (
          <div className="hidden shrink-0 items-center justify-between border-t border-slate-200 px-6 py-3 sm:flex">
            <button
              type="button"
              onClick={() => goTo(currentIndex - 1)}
              disabled={currentIndex === 0}
              className="flex items-center gap-1 text-sm font-medium text-slate-600 transition-colors hover:text-slate-800 disabled:opacity-30"
            >
              <ChevronLeft className="h-4 w-4" /> Prev
            </button>
            <span className="text-xs text-slate-400">
              {currentIndex + 1} / {items.length}
            </span>
            <button
              type="button"
              onClick={() => goTo(currentIndex + 1)}
              disabled={currentIndex === items.length - 1}
              className="flex items-center gap-1 text-sm font-medium text-slate-600 transition-colors hover:text-slate-800 disabled:opacity-30"
            >
              Next <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        )} */}
      </DialogContent>
    </Dialog>
  );
}
