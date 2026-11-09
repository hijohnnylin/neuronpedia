'use client';

import { ReactNode, useCallback, useLayoutEffect, useRef, useState } from 'react';

/** A tooltip that follows the pointer, for the SVG charts whose hit targets are not DOM-sized elements. */
export interface TipState {
  x: number;
  y: number;
  content: ReactNode;
}

export function useHoverTip() {
  const [tip, setTip] = useState<TipState | null>(null);
  const show = useCallback((e: { clientX: number; clientY: number }, content: ReactNode | null) => {
    setTip(content == null ? null : { x: e.clientX, y: e.clientY, content });
  }, []);
  const hide = useCallback(() => setTip(null), []);
  return { tip, show, hide };
}

export function HoverTip({ tip }: { tip: TipState | null }) {
  const ref = useRef<HTMLDivElement>(null);
  // Positioned after layout so the box can be kept inside the window once its size is known.
  useLayoutEffect(() => {
    const el = ref.current;
    if (!tip || !el) return;
    el.style.left = `${Math.max(8, Math.min(tip.x + 12, window.innerWidth - el.offsetWidth - 8))}px`;
    el.style.top = `${Math.max(8, Math.min(tip.y + 14, window.innerHeight - el.offsetHeight - 8))}px`;
  }, [tip]);
  if (!tip) return null;
  return (
    <div
      ref={ref}
      className="pointer-events-none fixed left-0 top-0 z-50 max-w-[320px] rounded-md border border-slate-200 bg-white px-2.5 py-1.5 text-xs leading-snug text-slate-700 shadow-md"
    >
      {tip.content}
    </div>
  );
}
