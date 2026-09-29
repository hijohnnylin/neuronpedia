'use client';

import { createContext, ReactNode, useContext } from 'react';

// Empty when Turnstile is off.
const TurnstileSiteKeyContext = createContext('');

export function TurnstileProvider({ siteKey, children }: { siteKey: string; children?: ReactNode }) {
  return <TurnstileSiteKeyContext.Provider value={siteKey}>{children}</TurnstileSiteKeyContext.Provider>;
}

export function useTurnstileSiteKey() {
  return useContext(TurnstileSiteKeyContext);
}
