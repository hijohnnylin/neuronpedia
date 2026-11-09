'use client';

/**
 * How the concepts a neuron responds to appear, extend and fade: the lineage tree, the stacked share
 * chart under it, and the same numbers as a table.
 *
 * Drawing rules for the tree. A branch (recursive generalization) forks off its parent's line and runs
 * below it. A replacement continues on the line of the lineage it replaced, in a new color, after a
 * tick. An unrelated arrival enters its own line from outside, with a hollow circle and an arrow. A
 * sporadic lineage (seen once, no parent, no children) sits on the faint bottom line.
 */

import { Fragment, ReactNode, useEffect, useMemo, useRef, useState } from 'react';
import { HoverTip, useHoverTip } from './hover-tip';
import {
  arrowKey,
  barSeries,
  ColorBy,
  families,
  FAMILY_COLORS,
  fmtAxis,
  fmtStep,
  focusKey,
  label,
  Lineage,
  lineageColor,
  MECHANISMS,
  NeuronMeta,
  SERIES_OTHER,
  treeLayout,
  trItem,
} from './shared';

const toggleClass = (on: boolean) =>
  `px-2.5 py-0.5 text-xs ${on ? 'bg-sky-700 text-white' : 'bg-white text-slate-600 hover:bg-slate-100'}`;

/** The container's width, so the SVGs can lay out in pixels and keep their text legible. */
function useWidth<T extends HTMLElement>(fallback: number) {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const update = () => setWidth(el.clientWidth || fallback);
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, [fallback]);
  return { ref, width };
}

export default function LineagePanel({
  n,
  steps,
  step,
  onStep,
  colorBy,
  onColorBy,
  famFocus,
  onFamFocus,
}: {
  n: NeuronMeta;
  steps: number[];
  step: number | null;
  onStep: (s: number) => void;
  colorBy: ColorBy;
  onColorBy: (c: ColorBy) => void;
  famFocus: string | null;
  onFamFocus: (root: string | null) => void;
}) {
  const fam = useMemo(() => families(n), [n]);
  const layout = useMemo(() => treeLayout(n, steps.length), [n, steps.length]);
  const series = useMemo(() => barSeries(n, fam, colorBy, steps.length), [n, fam, colorBy, steps.length]);
  const fk = colorBy === 'family' ? focusKey(fam, famFocus) : null;
  const { ref, width } = useWidth<HTMLDivElement>(640);
  const { tip, show, hide } = useHoverTip();
  const [hover, setHover] = useState<string | null>(null);

  const lineages = n.lineages ?? [];
  const nameOf = (id: string) => {
    const i = lineages.findIndex((g) => g.id === id);
    return i >= 0 ? trItem(n, 'lineages', i, 'name') : id;
  };

  const { rows, rowOf, laneOf, items, sporadic, k } = layout;
  const byId = Object.fromEntries(items.map((g) => [g.id, g]));

  // ----- tree geometry -----
  const W = width;
  const pl = 14;
  const pr = 14;
  const edge = 14;
  const gap = 13;
  const pb = 4;
  const yOf = (li: number) => edge + 8 + rowOf[li] * gap;
  const spY = edge + 8 + rows.length * gap + 4;
  const H = (sporadic.length ? spY + 6 : edge + 8 + Math.max(rows.length - 1, 0) * gap) + edge + pb;
  const colW = (W - pl - pr) / Math.max(k - 1, 1);
  const X = (j: number) => pl + j * colW;
  const dx = colW * 0.55;
  const top = 2;
  const bottom = H - pb - 2;
  const si = step == null ? -1 : steps.indexOf(step);
  const curve = (x0: number, y0: number, x1: number, y1: number) =>
    `M${x0},${y0} C${(x0 + x1) / 2},${y0} ${(x0 + x1) / 2},${y1} ${x1},${y1}`;
  const widthOf = (core: boolean | undefined, v: number) => (core ? 3 : 1.2) + 6 * Math.min(1, v);
  const color = (g: Pick<Lineage, 'id' | 'birth'>) => lineageColor(fam, g, colorBy);
  const arrows: [string, string][] =
    colorBy === 'family'
      ? FAMILY_COLORS.map((c, i): [string, string] => [`f${i}`, c]).concat([['fo', SERIES_OTHER]])
      : MECHANISMS.map((m): [string, string] => [m.key, m.color]);
  const opacityOf = (id: string) => {
    if (famFocus) return fam.rootOf[id] === famFocus ? 1 : 0.12;
    if (hover) return hover === id ? 1 : 0.15;
    return 1;
  };
  const enter = (x: number, y: number, c: string, marker: string) => (
    <>
      <circle cx={x - dx * 0.85} cy={y} r={2.2} fill="#fff" stroke={c} strokeWidth={1.2} />
      <path
        d={`M${x - dx * 0.85 + 2.2},${y} L${x},${y}`}
        stroke={c}
        strokeWidth={1.4}
        markerEnd={`url(#nevo-arr-${marker})`}
      />
    </>
  );
  const replacedBy: Record<string, boolean> = {};
  for (const g of items) if (g.birth === 'replace' && g.replaces) replacedBy[g.replaces] = true;

  const info = (id: string): ReactNode => {
    const g = byId[id];
    if (!g) return null;
    const i = lineages.findIndex((x) => x.id === id);
    const span =
      g.first === g.last
        ? `appears only at ${fmtStep(steps[g.first])}`
        : g.last === k - 1
          ? `appears at ${fmtStep(steps[g.first])} and lasts to the end of training`
          : `appears at ${fmtStep(steps[g.first])}, gone after ${fmtStep(steps[g.last])}`;
    const how = trItem(n, 'lineages', i, 'how');
    const root = fam.rootOf[id];
    return (
      <>
        <b>{nameOf(id)}</b>
        {g.core ? ' · core concept' : ''}
        <br />
        <span className="text-slate-500">
          {label('mech', g.birth)} · {span}
        </span>
        {g.birth === 'branch' && g.birth_parent ? (
          <>
            <br />
            grew out of &ldquo;{nameOf(g.birth_parent)}&rdquo;
          </>
        ) : null}
        {colorBy === 'family' && root && root !== id ? (
          <>
            <br />
            part of the &ldquo;{nameOf(root)}&rdquo; family
          </>
        ) : null}
        {g.birth === 'replace' && g.replaces ? (
          <>
            <br />
            took the place of &ldquo;{nameOf(g.replaces)}&rdquo;, which disappeared here
          </>
        ) : null}
        {how ? (
          <>
            <br />
            {how}
          </>
        ) : null}
      </>
    );
  };

  const lineGroups = items
    .filter((g) => g.birth !== 'sporadic')
    .map((g) => {
      const y = yOf(laneOf[g.id]);
      const c = color(g);
      const parts: ReactNode[] = [];
      const p = g.birth === 'branch' && g.birth_parent ? byId[g.birth_parent] : undefined;
      if (g.birth === 'branch' && p && laneOf[p.id] != null) {
        const py = yOf(laneOf[p.id]);
        const at = Math.max(0, g.first - 1);
        const xs = g.first > 0 ? X(at) : X(0) - pl + 2;
        // The parent is not among the strongest responses at the fork: extend its line there, dotted.
        if (p.last < at) {
          parts.push(
            <path
              key="pl"
              d={`M${X(p.last)},${py} L${X(at)},${py}`}
              stroke={color(p)}
              strokeWidth={1.2}
              strokeDasharray="2 4"
              opacity={0.7}
            />,
          );
        }
        if (p.first > at) {
          parts.push(
            <path
              key="pf"
              d={`M${xs},${py} L${X(p.first)},${py}`}
              stroke={color(p)}
              strokeWidth={1.2}
              strokeDasharray="2 4"
              opacity={0.7}
            />,
          );
        }
        parts.push(<circle key="pc" cx={xs} cy={py} r={2} fill={c} />);
        parts.push(
          <path
            key="curve"
            d={curve(xs, py, X(g.first), y)}
            fill="none"
            stroke={c}
            strokeWidth={1.8}
            markerEnd={`url(#nevo-arr-${arrowKey(fam, g, colorBy)})`}
          />,
        );
      } else if (g.birth === 'replace' && g.replaces && byId[g.replaces]) {
        parts.push(
          <path
            key="rp"
            d={`M${X(g.first - 1)},${y} L${X(g.first)},${y}`}
            stroke={c}
            strokeWidth={widthOf(g.core, g.pres[g.first])}
            strokeLinecap="round"
          />,
        );
        parts.push(
          <line
            key="rt"
            x1={X(g.first - 1)}
            x2={X(g.first - 1)}
            y1={y - 6}
            y2={y + 6}
            stroke="#0f172a"
            strokeWidth={1.5}
          />,
        );
      } else if (g.first === 0) {
        parts.push(
          <path
            key="e0"
            d={`M${X(0) - pl + 2},${y} L${X(0)},${y}`}
            stroke={c}
            strokeWidth={widthOf(g.core, g.pres[0])}
          />,
        );
      } else {
        parts.push(<Fragment key="en">{enter(X(g.first), y, c, arrowKey(fam, g, colorBy))}</Fragment>);
      }
      // Alive: solid where present at consecutive steps, dotted across a gap.
      for (let j = g.first; j < g.last; j += 1) {
        const present = g.pres[j] > 0 && g.pres[j + 1] > 0;
        parts.push(
          present ? (
            <path
              key={`s${j}`}
              d={`M${X(j)},${y} L${X(j + 1)},${y}`}
              stroke={c}
              strokeWidth={widthOf(g.core, (g.pres[j] + g.pres[j + 1]) / 2)}
              strokeLinecap="round"
            />
          ) : (
            <path
              key={`s${j}`}
              d={`M${X(j)},${y} L${X(j + 1)},${y}`}
              stroke={c}
              strokeWidth={1.2}
              strokeDasharray="2 4"
              opacity={0.7}
            />
          ),
        );
      }
      // Ending: survivors run to the right edge; the replaced hand their line on; the rest fade out.
      if (g.last === k - 1) {
        parts.push(
          <path
            key="end"
            d={`M${X(k - 1)},${y} L${W - 2},${y}`}
            stroke={c}
            strokeWidth={widthOf(g.core, g.pres[k - 1])}
          />,
        );
      } else if (!replacedBy[g.id]) {
        parts.push(
          <path
            key="fade"
            d={`M${X(g.last)},${y} L${X(g.last) + dx * 0.5},${y}`}
            stroke={c}
            strokeWidth={widthOf(g.core, g.pres[g.last])}
            strokeLinecap="round"
            opacity={0.35}
          />,
        );
      }
      g.pres.forEach((v, j) => {
        if (v > 0) {
          parts.push(
            <circle
              key={`d${j}`}
              cx={X(j)}
              cy={y}
              r={2 + 2.5 * Math.min(1, v)}
              fill={c}
              stroke="#fff"
              strokeWidth={1.5}
            />,
          );
        }
      });
      const x0 = g.first > 0 ? X(g.first - 1) : 0;
      const x1 = g.last === k - 1 ? W : X(g.last) + dx * 0.5;
      parts.push(
        <path key="hit" d={`M${x0},${y} L${x1},${y}`} stroke="transparent" strokeWidth={12} pointerEvents="stroke" />,
      );
      return (
        <g key={g.id} data-g={g.id} className="cursor-pointer" opacity={opacityOf(g.id)}>
          {parts}
        </g>
      );
    });

  const sporadicGroups = sporadic.length ? (
    <>
      <line x1={X(0)} x2={X(k - 1)} y1={spY} y2={spY} stroke="#e2e8f0" strokeWidth={1} />
      {sporadic.map((g) => {
        const c = color(g);
        const j = g.first;
        return (
          <g key={g.id} data-g={g.id} className="cursor-pointer" opacity={opacityOf(g.id)}>
            {enter(X(j), spY, c, arrowKey(fam, g, colorBy))}
            <circle cx={X(j)} cy={spY} r={2 + 2.5 * Math.min(1, g.pres[j])} fill={c} stroke="#fff" strokeWidth={1.5} />
            <path
              d={`M${X(j) - dx},${spY} L${X(j) + 4},${spY}`}
              stroke="transparent"
              strokeWidth={10}
              pointerEvents="stroke"
            />
          </g>
        );
      })}
    </>
  ) : null;

  // ----- share chart geometry -----
  const BH = 150;
  const bpt = 4;
  const bpb = 20;
  const bw = Math.min(colW * 0.7, 28);
  const Y = (v: number) => bpt + (BH - bpt - bpb) * (1 - v);
  const every = colW < 26 ? 4 : 2;

  const famFocusName = famFocus ? nameOf(famFocus) : null;

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="mr-auto text-sm font-semibold text-slate-800">How the concepts it responds to evolved</h2>
        <span className="inline-flex overflow-hidden rounded-md border border-slate-200" role="group">
          <button
            type="button"
            aria-pressed={colorBy === 'mech'}
            className={toggleClass(colorBy === 'mech')}
            onClick={() => onColorBy('mech')}
          >
            Color by mechanism
          </button>
          <button
            type="button"
            aria-pressed={colorBy === 'family'}
            className={toggleClass(colorBy === 'family')}
            onClick={() => onColorBy('family')}
          >
            Color by family
          </button>
        </span>
      </div>
      <div className="my-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-600">
        {series.map((x) => (
          <span
            key={x.key}
            title={x.label}
            className="max-w-[260px] truncate"
            style={fk && x.key !== fk ? { opacity: 0.35 } : undefined}
          >
            <i className="mr-1 inline-block h-2.5 w-2.5 rounded-sm align-middle" style={{ background: x.color }} />
            {x.label}
          </span>
        ))}
        {famFocusName && (
          <>
            <span>Only the &ldquo;{famFocusName}&rdquo; family</span>
            <button
              type="button"
              className="rounded border border-slate-200 px-1.5 py-0.5 hover:bg-slate-100"
              onClick={() => onFamFocus(null)}
            >
              Show all families
            </button>
          </>
        )}
      </div>
      <div ref={ref} className="w-full">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          style={{ width: '100%', height: H, display: 'block' }}
          role="img"
          aria-label="How the themes it responds to appear, extend and fade"
          onMouseMove={(e) => {
            const el = (e.target as Element).closest('g[data-g]');
            const id = el?.getAttribute('data-g') ?? null;
            setHover(id);
            show(e, id ? info(id) : null);
          }}
          onMouseLeave={() => {
            setHover(null);
            hide();
          }}
          onClick={(e) => {
            const el = (e.target as Element).closest('g[data-g]');
            if (el) {
              const root = fam.rootOf[el.getAttribute('data-g') ?? ''];
              if (root) onFamFocus(famFocus === root ? null : root);
              return;
            }
            const col = (e.target as Element).closest('rect[data-col]');
            if (col) onStep(steps[Number(col.getAttribute('data-col'))]);
          }}
        >
          <defs>
            {arrows.map(([m, c]) => (
              <marker
                key={m}
                id={`nevo-arr-${m}`}
                viewBox="0 0 6 6"
                refX={5}
                refY={3}
                markerWidth={5}
                markerHeight={5}
                orient="auto-start-reverse"
              >
                <path d="M0,0 L6,3 L0,6 z" fill={c} />
              </marker>
            ))}
          </defs>
          {steps.map((s, j) => (
            <rect
              key={`c${s}`}
              x={X(j) - colW / 2}
              y={0}
              width={colW}
              height={H - pb}
              fill="transparent"
              data-col={j}
            />
          ))}
          {steps.map((s, j) => (
            <line key={`g${s}`} x1={X(j)} x2={X(j)} y1={top} y2={bottom} stroke="#e2e8f0" strokeWidth={1} />
          ))}
          {si >= 0 && <line x1={X(si)} x2={X(si)} y1={top} y2={bottom} stroke="#0f172a" strokeDasharray="4 3" />}
          {lineGroups}
          {sporadicGroups}
        </svg>
      </div>
      <div className="mb-0.5 mt-2.5 text-xs text-slate-600">
        {colorBy === 'family'
          ? 'Share of activation in the 10 strongest responses, by family'
          : 'Share of activation in the 10 strongest responses, by mechanism'}
      </div>
      <svg
        viewBox={`0 0 ${W} ${BH}`}
        style={{ width: '100%', height: BH, display: 'block' }}
        role="img"
        aria-label="Share of activation in the 10 strongest responses"
        onMouseMove={(e) => {
          const r = (e.target as Element).closest('rect[data-i]');
          if (!r) {
            hide();
            return;
          }
          const i = Number(r.getAttribute('data-i'));
          show(
            e,
            <>
              <b>{fmtStep(steps[i])}</b>
              {series
                .filter((x) => x.vals[i] > 0)
                .map((x) => (
                  <Fragment key={x.key}>
                    <br />
                    {x.label}: {Math.round(x.vals[i] * 100)}%
                  </Fragment>
                ))}
            </>,
          );
        }}
        onMouseLeave={hide}
        onClick={(e) => {
          const r = (e.target as Element).closest('rect[data-i]');
          if (r) onStep(steps[Number(r.getAttribute('data-i'))]);
        }}
      >
        <line x1={pl - bw / 2} x2={W - pr + bw / 2} y1={Y(0)} y2={Y(0)} stroke="#e2e8f0" />
        {steps.map((s, i) => {
          let acc = 0;
          const bars = series.map((x) => {
            const v = x.vals[i];
            if (!(v > 0)) return null;
            const rect = (
              <rect
                key={x.key}
                x={X(i) - bw / 2}
                y={Y(acc + v)}
                width={bw}
                height={Math.max(0, Y(acc) - Y(acc + v))}
                fill={x.color}
                stroke="#fff"
                strokeWidth={2}
                opacity={fk && x.key !== fk ? 0.15 : 1}
              />
            );
            acc += v;
            return rect;
          });
          return (
            <Fragment key={s}>
              {bars}
              {s === step && (
                <rect
                  x={X(i) - bw / 2 - 3}
                  y={bpt - 2}
                  width={bw + 6}
                  height={BH - bpt - bpb + 4}
                  fill="none"
                  stroke="#0f172a"
                  strokeWidth={1.5}
                  strokeDasharray="3 2"
                  rx={3}
                />
              )}
              {(i % every === 0 || i === k - 1) && (
                <text x={X(i)} y={BH - 5} textAnchor="middle" fill="#94a3b8" fontSize={11}>
                  {fmtAxis(s)}
                </text>
              )}
            </Fragment>
          );
        })}
        {steps.map((s, i) => (
          <rect
            key={`h${s}`}
            x={X(i) - colW / 2}
            y={bpt}
            width={colW}
            height={BH - bpt - bpb}
            fill="transparent"
            data-i={i}
            className="cursor-pointer"
          />
        ))}
      </svg>
      <details className="mt-1">
        <summary className="cursor-pointer text-xs text-slate-500">Show the numbers</summary>
        <div className="max-w-full overflow-x-auto">
          <table className="mt-1.5 border-collapse text-xs tabular-nums">
            <thead>
              <tr>
                <th className="whitespace-nowrap border-b border-slate-200 px-1.5 py-0.5 text-left font-medium">
                  Share %
                </th>
                {steps.map((s) => (
                  <th key={s} className="border-b border-slate-200 px-1.5 py-0.5 text-right font-medium">
                    {fmtAxis(s)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {series.map((x) => (
                <tr key={x.key}>
                  <td className="whitespace-nowrap border-b border-slate-200 px-1.5 py-0.5 text-left">{x.label}</td>
                  {x.vals.map((v, i) => (
                    <td key={steps[i]} className="border-b border-slate-200 px-1.5 py-0.5 text-right">
                      {Math.round(v * 100)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
      <HoverTip tip={tip} />
    </div>
  );
}
