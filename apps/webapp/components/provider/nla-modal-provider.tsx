'use client';

import { createContext, ReactNode, useContext, useMemo, useState } from 'react';

type NLAModalContextType = {
  isWelcomeModalOpen: boolean;
  setIsWelcomeModalOpen: (isOpen: boolean) => void;
};

const NLAModalContext = createContext<NLAModalContextType | undefined>(undefined);

export function NLAModalProvider({ children }: { children: ReactNode }) {
  const [isWelcomeModalOpen, setIsWelcomeModalOpen] = useState<boolean>(false);

  const contextValue = useMemo(
    () => ({
      isWelcomeModalOpen,
      setIsWelcomeModalOpen,
    }),
    [isWelcomeModalOpen],
  );

  return <NLAModalContext.Provider value={contextValue}>{children}</NLAModalContext.Provider>;
}

export function useNLAModalContext() {
  const context = useContext(NLAModalContext);
  if (context === undefined) {
    throw new Error('useNLAModalContext must be used within an NLAModalProvider');
  }
  return context;
}
