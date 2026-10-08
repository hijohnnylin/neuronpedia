'use client';

// The lens columns shown by the sidebar and the token popup: any non-empty set
// of Jacobian, J++ and Logit, in that order. Each column keeps its own ranking,
// so two or three columns compare the lenses side by side. The oracle is not a
// column here: the token popup shows it for a model that has it.

import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/shadcn/hover-card';
import { isLensTypeColumn, LENS_COLUMN_LABELS, LENS_COLUMN_ORDER, LensColumn, LensType } from '@/lib/utils/lens';
import { QuestionMarkCircledIcon } from '@radix-ui/react-icons';
import { createContext } from 'react';

export const LensColumnsContext = createContext<LensColumn[]>([LensColumn.JACOBIAN_LENS]);

// Setter for the columns, provided next to `LensColumnsContext` so the toggle
// can render anywhere while the state stays at the page level.
export const LensColumnsSetContext = createContext<(c: LensColumn[]) => void>(() => {});

// `columns` in display order, without repeats.
export function orderLensColumns(columns: LensColumn[]): LensColumn[] {
  return LENS_COLUMN_ORDER.filter((c) => columns.includes(c));
}

// Whether the run has J++ Lens read-outs (only servers with the J++ Lens send them).
export function hasJppLens(layersByType: Record<string, number[]> | null | undefined): boolean {
  return (layersByType?.[LensType.JPP_LENS]?.length ?? 0) > 0;
}

// The columns to draw: the oracle drops out, the J++ Lens too when the server
// has none, and an empty result falls back to the Jacobian lens.
export function resolveLensColumns(columns: LensColumn[], jppAvailable: boolean): LensColumn[] {
  const out = orderLensColumns(columns).filter(
    (c) => c !== LensColumn.ORACLE_LENS && (c !== LensColumn.JPP_LENS || jppAvailable),
  );
  return out.length > 0 ? out : [LensColumn.JACOBIAN_LENS];
}

// The token-ranking lens types among `columns`, in display order.
export function lensTypesOf(columns: LensColumn[]): LensType[] {
  return columns.filter((c): c is LensType => isLensTypeColumn(c));
}

// Hover help for the J++ toggle. A click in the card must not reach the
// toggle: React events bubble through the portal to the button.
function JppLensHelp() {
  return (
    <HoverCard openDelay={100} closeDelay={100}>
      <HoverCardTrigger asChild>
        <span title="" className="flex items-center">
          <QuestionMarkCircledIcon className="h-3 w-3" />
        </span>
      </HoverCardTrigger>
      <HoverCardContent
        side="bottom"
        className="w-80 bg-white p-3 text-left text-[11px] font-normal leading-normal tracking-normal text-slate-600"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-1.5 font-bold text-slate-700">
          <a
            href="https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr"
            className="text-sky-700 underline"
            target="_blank"
            rel="noopener noreferrer"
          >
            J++ Lens (Ayonrinde & Lindsey, 2026)
          </a>
        </div>
        Jacobian Lens sometimes misses expected concepts and has unreliable readouts on the{' '}
        <a
          href="https://transformer-circuits.pub/2026/workspace/#struct-layers"
          className="text-sky-700 underline"
          target="_blank"
          rel="noopener noreferrer"
        >
          first ~1/3 layers
        </a>
        . To address this, Ayonrinde and Lindsey{' '}
        <a
          href="https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr"
          className="text-sky-700 underline"
          target="_blank"
          rel="noopener noreferrer"
        >
          introduce <span className="font-bold">J++ Lens</span>
        </a>
        , an improved, drop-in replacement for J-Lens. In this heart-related prompt, J++ Lens shows 623 references to
        terms like <code className="rounded bg-slate-100 px-1 font-mono">heart</code> and{' '}
        <code className="rounded bg-slate-100 px-1 font-mono">cardiac</code>, compared to 168 in J-Lens. J++ Lens is
        also more coherent on earlier layers, as shown in this (<span className="font-bold">content warning</span>){' '}
        <a
          href="https://www.neuronpedia.org/jlens/cmuyz4f2p0000182x9t3s36vh"
          className="text-sky-700 underline"
          target="_blank"
          rel="noopener noreferrer"
        >
          example
        </a>
        .
      </HoverCardContent>
    </HoverCard>
  );
}

export function LensColumnsToggle({
  columns,
  setColumns,
  jppAvailable,
}: {
  columns: LensColumn[];
  setColumns: (c: LensColumn[]) => void;
  jppAvailable: boolean;
}) {
  // The J++ button shows only where the server has the J++ Lens.
  const order = LENS_COLUMN_ORDER.filter(
    (c) => c !== LensColumn.ORACLE_LENS && (c !== LensColumn.JPP_LENS || jppAvailable),
  );
  const toggle = (c: LensColumn) => {
    if (columns.includes(c)) {
      // At least one column stays on.
      if (columns.length > 1) {
        setColumns(orderLensColumns(columns.filter((x) => x !== c)));
      }
      return;
    }
    setColumns(orderLensColumns([...columns, c]));
  };
  return (
    <div className="flex flex-1 flex-col gap-y-1 pl-0 pr-3 sm:pr-1">
      <div className="mb-0 text-start text-[10px] font-medium uppercase tracking-wide text-slate-400">Lenses</div>
      <div className="mx-0 flex flex-row items-center justify-center overflow-hidden rounded border border-sky-600">
        {order.map((c, i) => {
          const isOn = (x: LensColumn | undefined) => x !== undefined && columns.includes(x);
          const on = isOn(c);
          const next = order[i + 1];
          // Between two selected toggles the fills match, so the separator turns white.
          const separator =
            next === undefined ? '' : on && isOn(next) ? 'border-r border-white' : 'border-r border-sky-600';
          return (
            <button
              key={c}
              type="button"
              aria-pressed={on}
              onClick={() => toggle(c)}
              title={
                on && columns.length === 1
                  ? 'At least one lens stays on.'
                  : `${on ? 'Hide' : 'Show'} the ${LENS_COLUMN_LABELS[c]} column`
              }
              className={`flex h-[23px] max-h-[23px] flex-1 items-center justify-center gap-x-1 whitespace-nowrap px-0 py-1.5 text-[9.5px] font-semibold leading-none tracking-wide transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
                on ? 'bg-sky-600 text-white' : 'bg-white text-sky-600 enabled:hover:bg-sky-100'
              } ${separator}`}
            >
              {LENS_COLUMN_LABELS[c].toUpperCase()}
              {c === LensColumn.JPP_LENS && <JppLensHelp />}
            </button>
          );
        })}
      </div>
    </div>
  );
}
