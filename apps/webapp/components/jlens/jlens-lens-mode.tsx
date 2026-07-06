'use client';

// Global lens-display mode, shared by the page toggle, the sidebar, and the
// token popup. "DIFF" shows Jacobian + Logit side by side (ranked by each
// lens's advantage over the other); the single modes show just that lens type.

import { LensMode, LensType } from '@/lib/utils/lens';
import { QuestionMarkCircledIcon } from '@radix-ui/react-icons';
import { createContext } from 'react';
import CustomTooltip from '../custom-tooltip';

export { LensMode } from '@/lib/utils/lens';

export const LensModeContext = createContext<LensMode>(LensMode.JACOBIAN_LENS);

// Setter for the lens mode, provided alongside `LensModeContext` so the toggle
// can be rendered wherever it's convenient (e.g. inside the chat sidebar) while
// the state still lives at the page level.
export const LensModeSetContext = createContext<(m: LensMode) => void>(() => {});

// The lens types to display for a given mode, in render order. DIFF is a
// two-column mode over the pair of lens types.
export function lensTypesForMode(mode: LensMode): LensType[] {
  if (mode === LensMode.DIFF) {
    return [LensType.JACOBIAN_LENS, LensType.LOGIT_LENS];
  }
  return [mode];
}

const MODE_OPTIONS: { value: LensMode; label: string }[] = [
  { value: LensMode.JACOBIAN_LENS, label: 'Jacobian' },
  { value: LensMode.LOGIT_LENS, label: 'Logit' },
  { value: LensMode.DIFF, label: 'Diff' },
];

export function LensModeToggle({ mode, setMode }: { mode: LensMode; setMode: (m: LensMode) => void }) {
  return (
    <div className="flex flex-1 flex-col gap-y-1 pl-0 pr-3 sm:pr-1">
      <div className="mb-0 text-start text-[10px] font-medium uppercase tracking-wide text-slate-400">Lens Mode</div>
      <div className="mx-0 flex flex-row items-center justify-center overflow-hidden rounded border border-sky-600">
        {MODE_OPTIONS.map((o) => (
          <button
            key={o.value}
            type="button"
            onClick={() => setMode(o.value)}
            className={`flex h-[23px] max-h-[23px] flex-1 items-center justify-center gap-x-1 whitespace-nowrap px-0 py-1.5 text-[9.5px] font-semibold leading-none tracking-wide transition-colors ${mode === o.value ? 'bg-sky-600 text-white' : 'bg-white text-sky-600 hover:bg-sky-500 hover:text-white'} ${o.value !== MODE_OPTIONS[MODE_OPTIONS.length - 1].value ? 'border-r border-sky-600' : ''} `}
          >
            {o.label.toUpperCase()}
            {o.value === LensMode.DIFF && (
              <CustomTooltip
                minMargin
                trigger={
                  <span
                    role="button"
                    tabIndex={0}
                    aria-label="How Diff columns are sorted"
                    className="inline-flex items-center opacity-100"
                  >
                    <QuestionMarkCircledIcon className="h-3.5 w-3.5" />
                  </span>
                }
              >
                In Diff mode each lens gets its own column. A column only lists tokens that lens predicts more often
                than the other lens, ranked by how disproportionately it favors them — a smoothed ratio of{' '}
                <span className="whitespace-nowrap">(this count + α) / (other count + α)</span> (α = 2). The smoothing
                keeps low-support cases (like 1 vs 0) near neutral so genuinely lopsided, well-supported tokens rise to
                the top.
              </CustomTooltip>
            )}
          </button>
        ))}
      </div>
    </div>
  );
}
