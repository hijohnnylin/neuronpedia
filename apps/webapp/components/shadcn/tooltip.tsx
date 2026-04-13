'use client';

import * as OldTooltip from '@radix-ui/react-tooltip';
import { ReactNode, useState } from 'react';

export default function Tooltip({ alwaysOpen, children }: { alwaysOpen: boolean; children: ReactNode }) {
  const [open, setOpen] = useState(false);

  return (
    <OldTooltip.Root open={alwaysOpen || open} delayDuration={250} onOpenChange={setOpen}>
      <span onClick={() => setOpen(true)} className="inline-block">
        {children}
      </span>
    </OldTooltip.Root>
  );
}
