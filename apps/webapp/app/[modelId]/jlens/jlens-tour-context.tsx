'use client';

// Shares the tour's currently-highlighted `DriveStep` with the rest of the
// JLens tree so components can tweak their UI for specific tour steps (e.g.
// hiding the steer/swap buttons while the spider readout is spotlighted).
//
// Kept separate from `jlens-tour.tsx` because that module imports
// `driver.js/dist/driver.css`; here we only need the `DriveStep` *type*
// (erased at build time), so consumers don't pull in the tour's CSS.
import { createContext, useContext } from 'react';
import type { DriveStep } from 'driver.js';

// The step driver.js is currently spotlighting, or `null` when no tour is
// running.
export const JlensTourStepContext = createContext<DriveStep | null>(null);

export function useJlensTourStep(): DriveStep | null {
  return useContext(JlensTourStepContext);
}
