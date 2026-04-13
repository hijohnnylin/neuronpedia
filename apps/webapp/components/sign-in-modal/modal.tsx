'use client';

import useWindowSize from '@/lib/hooks/use-window-size';
import FocusTrap from 'focus-trap-react';
import { Dispatch, SetStateAction, useCallback, useEffect, useRef } from 'react';
import Leaflet from './leaflet';

export default function Modal({
  children,
  showModal,
  setShowModal,
}: {
  children: React.ReactNode;
  showModal: boolean;
  setShowModal: Dispatch<SetStateAction<boolean>>;
}) {
  const desktopModalRef = useRef(null);

  const onKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setShowModal(false);
      }
    },
    [setShowModal],
  );

  useEffect(() => {
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [onKeyDown]);

  const { isMobile, isDesktop } = useWindowSize();

  if (!showModal) return null;

  return (
    <>
      {isMobile && <Leaflet setShow={setShowModal}>{children}</Leaflet>}
      {isDesktop && (
        <>
          <FocusTrap focusTrapOptions={{ initialFocus: false }}>
            <div
              ref={desktopModalRef}
              className="fixed inset-0 z-40 hidden min-h-screen items-center justify-center md:flex"
              onMouseDown={(e) => {
                if (desktopModalRef.current === e.target) {
                  setShowModal(false);
                }
              }}
            >
              {children}
            </div>
          </FocusTrap>
          <div
            className="fixed inset-0 z-30 bg-slate-100 bg-opacity-10 backdrop-blur"
            onClick={() => setShowModal(false)}
          />
        </>
      )}
    </>
  );
}
