'use client';

import * as OldTooltip from '@radix-ui/react-tooltip';
import { ReactNode, useState } from 'react';

export default function Tooltip({
  alwaysOpen,
  children,
  delayDuration = 250,
  className,
}: {
  alwaysOpen: boolean;
  children: ReactNode;
  delayDuration?: number;
  className?: string;
}) {
  const [open, setOpen] = useState(false);

  return (
    <OldTooltip.Root open={alwaysOpen || open} delayDuration={delayDuration} onOpenChange={setOpen}>
      <span onClick={() => setOpen(true)} className={`inline-block ${className}`}>
        {children}
      </span>
    </OldTooltip.Root>
  );
}
