'use client';

import * as OldTooltip from '@radix-ui/react-tooltip';
import Tooltip from './shadcn/tooltip';

export default function CustomTooltip({
  trigger,
  wide = false,
  children,
  side,
  delayDuration,
  minMargin = false,
  className,
  align,
  alignOffset,
  collisionPadding,
}: {
  trigger: React.ReactNode;
  children: React.ReactNode;
  wide?: boolean;
  side?: 'top' | 'bottom' | 'left' | 'right';
  /** Hover open delay in ms. Defaults to the inner Tooltip default (250ms). */
  delayDuration?: number;
  /** If true, applies minimal margin for compact tooltips */
  minMargin?: boolean;
  className?: string;
  align?: 'start' | 'center' | 'end';
  alignOffset?: number;
  /** Minimum gap in px between the tooltip and the window edge. */
  collisionPadding?: number;
}) {
  return (
    <OldTooltip.Provider delayDuration={0} skipDelayDuration={0}>
      <Tooltip alwaysOpen={false} delayDuration={delayDuration} className={className}>
        <OldTooltip.Trigger asChild>{trigger}</OldTooltip.Trigger>
        <OldTooltip.Portal>
          <OldTooltip.Content
            className={`z-50 rounded-lg border border-slate-200 bg-white ${
              minMargin ? 'px-2.5 py-2' : 'px-5 py-3.5'
            } text-xs text-slate-600 shadow ${wide ? 'max-w-[640px]' : 'max-w-[320px]'}`}
            sideOffset={3}
            side={side}
            align={align}
            alignOffset={alignOffset}
            collisionPadding={collisionPadding}
          >
            {children}
          </OldTooltip.Content>
        </OldTooltip.Portal>
      </Tooltip>
    </OldTooltip.Provider>
  );
}
