'use client';

import useWindowSize from '@/lib/hooks/use-window-size';
import * as PopoverPrimitive from '@radix-ui/react-popover';
import { Dispatch, ReactNode, SetStateAction } from 'react';
import Leaflet from './leaflet';

export default function Popover({
  children,
  content,
  align = 'center',
  openPopover,
  setOpenPopover,
}: {
  children: ReactNode;
  content: ReactNode | string;
  align?: 'center' | 'start' | 'end';
  openPopover: boolean;
  setOpenPopover: Dispatch<SetStateAction<boolean>>;
}) {
  const { isMobile, isDesktop } = useWindowSize();

  if (!isMobile && !isDesktop) return <>{children}</>;
  return (
    <>
      {isMobile && children}
      {openPopover && isMobile && <Leaflet setShow={setOpenPopover}>{content}</Leaflet>}
      {isDesktop && (
        <PopoverPrimitive.Root open={openPopover} onOpenChange={(isOpen) => setOpenPopover(isOpen)}>
          <PopoverPrimitive.Trigger className="inline-flex" asChild>
            {children}
          </PopoverPrimitive.Trigger>
          <PopoverPrimitive.Portal>
            <PopoverPrimitive.Content
              sideOffset={4}
              align={align}
              className="z-20 animate-slide-up-fade items-center rounded-md border border-slate-200 bg-white drop-shadow-lg"
            >
              {content}
            </PopoverPrimitive.Content>
          </PopoverPrimitive.Portal>
        </PopoverPrimitive.Root>
      )}
    </>
  );
}
