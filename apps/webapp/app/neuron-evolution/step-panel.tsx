'use client';

/**
 * The per-step view: the table of the AI's description at every training stage, and for the chosen
 * stage (optionally beside a second one) the patterns it found and the strongest-activating passages
 * with a token heatmap.
 */

import { ChevronLeft, ChevronRight } from 'lucide-react';
import { Fragment } from 'react';
import { HoverTip, useHoverTip } from './hover-tip';
import {
  ColorBy,
  Context,
  families,
  fmtStep,
  label,
  lineageColor,
  NeuronMeta,
  Pattern,
  StepRecord,
  tr,
  trItem,
} from './shared';

const COLUMNS: { title: string; tip?: string }[] = [
  { title: 'step' },
  { title: 'loss' },
  {
    title: 'explanation accuracy',
    tip: 'Accuracy at telling strong-response text from random text using only the explanation; 0.5 is chance, — means not tested',
  },
  { title: 'pattern clarity', tip: "The AI's own rating, 1–5" },
  { title: 'in short' },
  { title: 'full description' },
];

export function ExplanationTable({
  rows,
  steps,
  loss,
  step,
  onStep,
}: {
  rows: StepRecord[];
  steps: number[];
  loss: number[];
  step: number | null;
  onStep: (s: number) => void;
}) {
  const lossAt = Object.fromEntries(steps.map((s, i) => [s, loss[i]]));
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <h2 className="mb-1.5 text-sm font-semibold text-slate-800">The AI&apos;s description at each training stage</h2>
      <div className="max-w-full overflow-x-auto">
        <table className="w-full min-w-[640px] border-collapse text-[13px]">
          <thead>
            <tr>
              {COLUMNS.map((c) => (
                <th
                  key={c.title}
                  title={c.tip}
                  className="border-b border-slate-200 px-1.5 py-1 text-left text-xs font-medium text-slate-500"
                >
                  {c.title}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr
                key={r.step}
                tabIndex={0}
                onClick={() => onStep(r.step)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') onStep(r.step);
                }}
                className={`cursor-pointer ${r.step === step ? 'bg-sky-50' : 'hover:bg-slate-50'}`}
              >
                <td className="whitespace-nowrap border-b border-slate-200 px-1.5 py-1 tabular-nums">
                  {r.step.toLocaleString('en')}
                </td>
                <td className="whitespace-nowrap border-b border-slate-200 px-1.5 py-1 tabular-nums">
                  {lossAt[r.step]?.toFixed(2) ?? ''}
                </td>
                <td className="whitespace-nowrap border-b border-slate-200 px-1.5 py-1 tabular-nums">
                  {r.det == null ? (
                    '—'
                  ) : (
                    <>
                      <span
                        className="mr-1 inline-block h-2 rounded-sm align-middle"
                        style={{ width: Math.max(0, (r.det - 0.5) * 2) * 40, background: '#2a78d6' }}
                      />
                      {r.det.toFixed(2)}
                    </>
                  )}
                </td>
                <td className="whitespace-nowrap border-b border-slate-200 px-1.5 py-1 tabular-nums">
                  {r.clarity ?? ''}
                </td>
                <td className="border-b border-slate-200 px-1.5 py-1">{tr(r, 'concept')}</td>
                <td className="border-b border-slate-200 px-1.5 py-1 text-slate-600">{tr(r, 'description')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ContextView({ c, i, top1 }: { c: Context; i: number; top1: number }) {
  return (
    <div className="border-t border-slate-200 py-2">
      <div className="mb-0.5 flex flex-wrap gap-2 text-xs text-slate-500">
        <span>
          <b className="text-slate-800">[{i + 1}]</b>
        </span>
        <span>
          response <b className="text-slate-800">{c.value.toFixed(2)}</b>
        </span>
        <span className="rounded-full bg-slate-100 px-2 text-slate-600">{c.subset}</span>
        <span>
          document {c.doc} · position {c.pos}
        </span>
      </div>
      <div className="whitespace-pre-wrap break-words font-mono text-[13px] leading-[1.75]">
        {c.tokens.map((tok, j) => {
          const a = c.a[j];
          const alpha = a > 0 ? (Math.min(a, 100) / 100) * 0.8 : 0;
          const parts = tok.split('\n');
          return (
            <span
              key={j}
              data-a={a}
              data-v={((a / 100) * top1).toFixed(2)}
              className={`rounded-[3px] ${j === c.hit ? 'font-bold outline outline-2 outline-slate-900' : ''}`}
              style={{ background: `rgba(42, 120, 214, ${alpha.toFixed(2)})` }}
            >
              {parts.map((p, pi) => (
                <Fragment key={pi}>
                  {pi > 0 && (
                    <>
                      <span className="text-slate-400">↵</span>
                      {'\n'}
                    </>
                  )}
                  {p}
                </Fragment>
              ))}
              {tok === '' ? '​' : null}
            </span>
          );
        })}
      </div>
    </div>
  );
}

function StepColumn({
  n,
  r,
  topN,
  colorBy,
  open,
}: {
  n: NeuronMeta;
  r: StepRecord;
  topN: number;
  colorBy: ColorBy;
  open: boolean;
}) {
  const fam = families(n);
  const lineages = n.lineages ?? [];
  const patternText = (p: Pattern, pi: number, field: 'name' | 'context' | 'domain') =>
    trItem(r, 'patterns', pi, field) || (p[field] ?? '');
  const unexplained = r.unexplained && r.unexplained.length ? r.unexplained.map((x) => `[${x}]`).join('') : '';
  return (
    <div className="min-w-0">
      <h2 className="text-sm font-semibold text-slate-800">
        {fmtStep(r.step)}
        {'　'}
        <span className="font-normal text-slate-600">{tr(r, 'concept')}</span>
      </h2>
      <div className="text-xs">{tr(r, 'description')}</div>
      {unexplained && <div className="my-1 text-xs text-slate-500">passages left unexplained {unexplained}</div>}
      <details open={open}>
        <summary className="cursor-pointer text-xs text-slate-500">Patterns the AI found ({r.patterns.length})</summary>
        {r.patterns.map((p, pi) => {
          const gi = p.lineage ? lineages.findIndex((g) => g.id === p.lineage) : -1;
          const g = gi >= 0 ? lineages[gi] : null;
          return (
            <div key={pi} className="my-1 text-[13px]">
              {g && (
                <span
                  className="mr-1 inline-block rounded-full bg-slate-100 px-2 text-xs text-slate-600"
                  title={trItem(n, 'lineages', gi, 'name')}
                >
                  <i
                    className="mr-1 inline-block h-2.5 w-2.5 rounded-sm align-middle"
                    style={{ background: lineageColor(fam, g, colorBy) }}
                  />
                  {label('mech', g.birth)}
                </span>
              )}
              <b>{patternText(p, pi, 'name')}</b>{' '}
              {p.trigger && (
                <code className="break-words rounded bg-slate-100 px-1 font-mono text-xs">{p.trigger}</code>
              )}{' '}
              <span className="text-slate-500">see passages {(p.segments ?? []).map((x) => `[${x}]`).join('')}</span>
              <div className="text-xs text-slate-600">
                {[patternText(p, pi, 'context'), patternText(p, pi, 'domain')].filter(Boolean).join('; ')}
              </div>
            </div>
          );
        })}
      </details>
      {r.contexts.slice(0, topN).map((c, i) => (
        <ContextView key={`${c.doc}-${c.pos}`} c={c} i={i} top1={r.top1} />
      ))}
    </div>
  );
}

export function StepPanel({
  n,
  rows,
  step,
  cmp,
  onCmp,
  onMove,
  topN,
  onTopN,
  colorBy,
}: {
  n: NeuronMeta;
  rows: StepRecord[];
  step: number;
  cmp: number | null;
  onCmp: (s: number | null) => void;
  onMove: (delta: number) => void;
  topN: number;
  onTopN: (k: number) => void;
  colorBy: ColorBy;
}) {
  const { tip, show, hide } = useHoverTip();
  const r = rows.find((x) => x.step === step) ?? rows[rows.length - 1];
  const c = cmp != null ? rows.find((x) => x.step === cmp) : undefined;
  const selectClass = 'rounded-md border border-slate-200 bg-white px-1.5 py-0.5 text-xs text-slate-700';
  const navClass =
    'flex h-6 w-6 items-center justify-center rounded border border-slate-200 text-slate-700 hover:bg-slate-100';
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <div className="mb-1.5 flex flex-wrap items-center gap-2">
        <b className="text-sm">
          Text it responded to most at {fmtStep(r.step)}
          {c ? ` (compared with ${fmtStep(c.step)})` : ''}
        </b>
        <button
          type="button"
          title="Previous stage (←)"
          aria-label="Previous stage"
          className={navClass}
          onClick={() => onMove(-1)}
        >
          <ChevronLeft className="h-3.5 w-3.5" />
        </button>
        <button
          type="button"
          title="Next stage (→)"
          aria-label="Next stage"
          className={navClass}
          onClick={() => onMove(1)}
        >
          <ChevronRight className="h-3.5 w-3.5" />
        </button>
        <label className="flex items-center gap-1.5 text-xs text-slate-600">
          Show
          <select className={selectClass} value={topN} onChange={(e) => onTopN(Number(e.target.value))}>
            <option value={5}>top 5</option>
            <option value={10}>top 10</option>
          </select>
        </label>
        <label className="flex items-center gap-1.5 text-xs text-slate-600">
          Compare side by side with
          <select
            className={selectClass}
            value={cmp == null ? '' : String(cmp)}
            onChange={(e) => onCmp(e.target.value === '' ? null : Number(e.target.value))}
          >
            <option value="">nothing</option>
            {rows.map((x) => (
              <option key={x.step} value={x.step}>
                {fmtStep(x.step)}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="text-xs text-slate-500">
        The bluer the background, the stronger the response; the box marks the strongest spot in each passage.
      </div>
      <div
        className={`grid gap-3.5 ${c ? 'grid-cols-1 md:grid-cols-2' : 'grid-cols-1'}`}
        onMouseMove={(e) => {
          const el = (e.target as Element).closest('span[data-a]');
          if (!el) {
            hide();
            return;
          }
          show(
            e,
            <>
              {JSON.stringify((el.textContent ?? '').replace(/↵/g, ''))}
              {'　'}response ≈ {el.getAttribute('data-v')} ({el.getAttribute('data-a')}% of the strongest)
            </>,
          );
        }}
        onMouseLeave={hide}
      >
        <StepColumn n={n} r={r} topN={topN} colorBy={colorBy} open={cmp == null} />
        {c && <StepColumn n={n} r={c} topN={topN} colorBy={colorBy} open={false} />}
      </div>
      <HoverTip tip={tip} />
    </div>
  );
}
