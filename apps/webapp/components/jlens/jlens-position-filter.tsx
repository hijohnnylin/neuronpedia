'use client';

// The shared "position filter" readout + controls shown once one or more chat
// positions are selected (via click-and-drag on the transcript). It renders the
// selected-position summary, the SHOW/HIDE positions popup trigger (+ the popup
// itself), and the clear button. Used both in the analysis sidebar's status row
// and in the action bar's "Filter Positions" section so the two stay identical.

import { LensTokenMessage } from '@/lib/utils/lens';
import { X } from 'lucide-react';
import { createPortal } from 'react-dom';
import { displayToken } from './jlens-token-popup';
import { JlensAnalysis } from './use-jlens-analysis';

export function JlensPositionFilter({
  analysis,
  tokens,
  stacked = false,
}: {
  analysis: JlensAnalysis;
  tokens: LensTokenMessage[];
  // When stacked, the SHOW/clear controls sit centered below the position
  // label (used in the action bar) instead of inline beside it (sidebar).
  stacked?: boolean;
}) {
  const {
    selectedPositions,
    clearSelectedPositions,
    removeSelectedPosition,
    positionsPopupOpen,
    setPositionsPopupOpen,
    positionsPopupAnchor,
    setPositionsPopupAnchor,
    setHighlightedPosition,
    hoveredChatToken,
  } = analysis;

  return (
    <div
      className={`flex flex-1 items-center justify-center px-3 text-center leading-none text-slate-500 ${
        stacked ? 'flex-row gap-x-2 text-[9px] sm:text-[10px]' : 'flex-row gap-x-2 text-[11px] sm:text-[12px]'
      }`}
    >
      <div className="flex flex-row items-center justify-center gap-x-0">
        {selectedPositions.size > 0 ? (
          selectedPositions.size === 1 ? (
            <>
              Selected Pos{' '}
              <span className="ml-1 mr-0 border-slate-300 font-semibold text-slate-500">
                {[...selectedPositions][0]}
              </span>
              {/* <span className="rounded bg-slate-200 px-2 py-1 font-mono text-[10px] text-slate-600">
                {displayToken(tokens.find((t) => t.position === [...selectedPositions][0])?.token ?? '')}
              </span> */}
            </>
          ) : (
            <>
              <span className="mr-1 font-semibold text-slate-500">{selectedPositions.size}</span> Positions Selected
            </>
          )
        ) : hoveredChatToken ? (
          <div className="flex flex-row items-center justify-center gap-x-2">
            <div>
              Pos <span className="font-semibold text-slate-500">{hoveredChatToken.position}</span>
            </div>

            <div className="rounded bg-slate-200 px-1.5 py-0.5 font-mono text-[10px] text-slate-600">
              {displayToken(hoveredChatToken.token)}
            </div>
          </div>
        ) : (
          'All Positions'
        )}
      </div>
      {selectedPositions.size > 0 && (
        <div className="flex flex-row items-center gap-x-1">
          {selectedPositions.size > 1 && (
            <button
              type="button"
              onClick={(e) => {
                const r = e.currentTarget.getBoundingClientRect();
                setPositionsPopupAnchor({ x: r.left, y: r.bottom });
                setPositionsPopupOpen((o) => !o);
              }}
              className={`flex w-11 min-w-11 max-w-11 shrink-0 flex-row items-center justify-center gap-x-0.5 rounded border py-1 text-center text-[9px] font-medium transition-colors ${
                positionsPopupOpen
                  ? 'border-slate-500 bg-slate-200 text-slate-800'
                  : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-100 hover:text-slate-800'
              }`}
            >
              {positionsPopupOpen ? 'HIDE' : 'SHOW'}
            </button>
          )}
          <button
            type="button"
            onClick={() => {
              // eslint-disable-next-line no-alert
              if (window.confirm('Clear selected positions?')) {
                clearSelectedPositions();
                setHighlightedPosition(null);
                setPositionsPopupOpen(false);
              }
            }}
            className="flex h-4 w-4 min-w-4 max-w-4 shrink-0 flex-row items-center justify-center rounded border border-rose-300 bg-rose-100 py-0.5 text-center text-[9px] font-medium text-rose-500 transition-colors hover:bg-rose-200 hover:text-rose-800"
          >
            <X className="h-2.5 w-2.5" />
          </button>
        </div>
      )}
      {positionsPopupOpen &&
        selectedPositions.size > 0 &&
        positionsPopupAnchor &&
        typeof document !== 'undefined' &&
        createPortal(
          <>
            <div className="fixed inset-0 z-[69]" onClick={() => setPositionsPopupOpen(false)} />
            <div
              className="fixed z-[70] max-h-[300px] w-[160px] overflow-y-auto rounded-md border border-slate-200 bg-white text-left text-[11px] shadow-xl"
              style={{ left: positionsPopupAnchor.x, top: positionsPopupAnchor.y + 4 }}
              onMouseLeave={() => setHighlightedPosition(null)}
            >
              <div className="sticky top-0 flex flex-row items-center gap-x-2 border-b border-slate-200 bg-white px-2 py-1 text-[9px] font-semibold uppercase tracking-wide text-slate-400">
                <span className="w-6 shrink-0 text-right">Pos</span>
                <span className="min-w-0 flex-1">Token</span>
                <span className="w-4 shrink-0" />
              </div>
              {[...selectedPositions]
                .sort((a, b) => a - b)
                .map((pos) => {
                  const tok = tokens.find((t) => t.position === pos);
                  return (
                    <div
                      key={pos}
                      onMouseEnter={() => setHighlightedPosition(pos)}
                      className="flex cursor-default flex-row items-center gap-x-2 px-2 py-1 hover:bg-slate-100"
                    >
                      <span className="w-6 shrink-0 text-right tabular-nums text-slate-400">{pos}</span>
                      <span className="min-w-0 flex-1 truncate font-mono text-slate-700">
                        {tok ? displayToken(tok.token) : '—'}
                      </span>
                      <button
                        type="button"
                        onClick={() => {
                          // Clear the hover highlight too: removing the last
                          // position unmounts this popup, so its onMouseLeave
                          // never fires and the highlighted token's box shadow
                          // would otherwise stay stuck on the transcript.
                          setHighlightedPosition(null);
                          // Removing the last position closes the popup, so a
                          // later multi-position selection doesn't reopen it
                          // immediately from the lingering open state.
                          if (selectedPositions.size <= 1) {
                            setPositionsPopupOpen(false);
                          }
                          removeSelectedPosition(pos);
                        }}
                        aria-label={`Remove position ${pos}`}
                        className="flex h-4 w-4 shrink-0 items-center justify-center rounded text-slate-400 transition-colors hover:bg-rose-100 hover:text-rose-600"
                      >
                        <X className="h-3 w-3" />
                      </button>
                    </div>
                  );
                })}
            </div>
          </>,
          document.body,
        )}
    </div>
  );
}
