'use client';

/**
 * The map of neurons by explanation similarity: closer dots have more alike descriptions at the
 * chosen training step. A slider (or play) moves through training; double-clicking a dot anchors the
 * map on it, so distance from the center is distance to that neuron's explanation.
 */

import { Pause, Play } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { HoverTip, useHoverTip } from './hover-tip';
import { DecodedSim, DOT_MUTED, fmtStep, mapTargets, NeuronMeta, neuronName, SERIES } from './shared';

const MW = 360;
const MH = 260;
const MP = 10;
const mX = (v: number) => MP + (MW - 2 * MP) * v;
const mY = (v: number) => MH - MP - (MH - 2 * MP) * v;

export default function SemanticMap({
  neurons,
  steps,
  selected,
  isMatch,
  onSelect,
  loadSim,
}: {
  neurons: NeuronMeta[];
  steps: number[];
  selected: string | null;
  isMatch: (n: NeuronMeta) => boolean;
  onSelect: (id: string) => void;
  loadSim: () => Promise<DecodedSim>;
}) {
  const placed = useMemo(() => neurons.filter((n) => n.sem && n.sem.length === steps.length), [neurons, steps.length]);
  const last = steps.length - 1;
  const [j, setJ] = useState(last);
  const [playing, setPlaying] = useState(false);
  const [anchor, setAnchor] = useState<string | null>(null);
  const [sim, setSim] = useState<DecodedSim | null>(null);
  const { tip, show, hide } = useHoverTip();

  const positions = useMemo(() => mapTargets(placed, j, anchor, sim), [placed, j, anchor, sim]);

  useEffect(() => {
    if (!playing) return undefined;
    const id = setInterval(() => setJ((cur) => Math.min(cur + 1, last)), 600);
    return () => clearInterval(id);
  }, [playing, last]);
  useEffect(() => {
    if (playing && j >= last) setPlaying(false);
  }, [playing, j, last]);

  const togglePlay = () => {
    if (playing) {
      setPlaying(false);
      return;
    }
    if (j >= last) setJ(0);
    setPlaying(true);
  };

  const toggleAnchor = async (id: string | null) => {
    if (id && !sim) {
      try {
        setSim(await loadSim());
      } catch {
        return; // Without the similarity matrix the map stays unanchored.
      }
    }
    setAnchor((prev) => (prev === id ? null : id));
  };

  const anchorNeuron = anchor ? placed.find((n) => n.id === anchor) : undefined;
  const selectedNeuron = selected ? placed.find((n) => n.id === selected) : undefined;
  const on = placed.filter(isMatch);
  const off = placed.filter((n) => !isMatch(n));

  const dot = (n: NeuronMeta, muted: boolean) => {
    const p = positions[n.id];
    if (!p) return null;
    return (
      <circle
        key={n.id}
        cx={mX(p[0]).toFixed(1)}
        cy={mY(p[1]).toFixed(1)}
        r={4}
        fill={muted ? DOT_MUTED : SERIES[0]}
        stroke={muted ? 'none' : '#fff'}
        strokeWidth={1.5}
        className="cursor-pointer transition-[cx,cy] duration-500 ease-in-out motion-reduce:transition-none"
        onMouseMove={(e) => {
          const concept = n.ce?.[j] || n.cz?.[j] || '';
          show(
            e,
            <>
              <b>{neuronName(n)}</b> · {fmtStep(steps[j])}
              {concept ? <br /> : null}
              {concept}
            </>,
          );
        }}
        onMouseLeave={hide}
        onClick={() => {
          if (n.id !== selected) onSelect(n.id);
        }}
        onDoubleClick={() => toggleAnchor(n.id)}
      />
    );
  };

  const ring = (n: NeuronMeta | undefined, color: string, r: number, dashed: boolean) => {
    const p = n ? positions[n.id] : undefined;
    if (!p) return null;
    return (
      <circle
        cx={mX(p[0])}
        cy={mY(p[1])}
        r={r}
        fill="none"
        stroke={color}
        strokeWidth={dashed ? 1.5 : 2.5}
        strokeDasharray={dashed ? '3 2' : undefined}
        pointerEvents="none"
        className="transition-[cx,cy] duration-500 ease-in-out motion-reduce:transition-none"
      />
    );
  };

  return (
    <div>
      <h2 className="text-sm font-semibold text-slate-800">Neurons with similar explanations</h2>
      <div className="text-xs text-slate-500">
        Closer dots have more alike descriptions. Click to switch; double-click to anchor.
      </div>
      <svg
        viewBox={`0 0 ${MW} ${MH}`}
        role="img"
        aria-label="Neurons with similar explanations"
        className="mt-1 block aspect-[36/26] w-full"
      >
        {ring(anchorNeuron, '#0f172a', 9, true)}
        {off.map((n) => dot(n, true))}
        {on.map((n) => dot(n, false))}
        {ring(selectedNeuron, SERIES[1], 7, false)}
        {!placed.length && (
          <text x={MW / 2} y={MH / 2} textAnchor="middle" fill="#94a3b8" fontSize={12}>
            Semantic map is pending full analysis
          </text>
        )}
      </svg>
      {anchorNeuron && (
        <div className="mt-1 flex items-center gap-2 text-xs text-slate-600">
          <span>Anchor: {neuronName(anchorNeuron)}</span>
          <button
            type="button"
            className="rounded border border-slate-200 px-1.5 py-0.5 hover:bg-slate-100"
            onClick={() => toggleAnchor(null)}
          >
            Clear anchor
          </button>
        </div>
      )}
      <div className="mt-2 flex items-center gap-2">
        <button
          type="button"
          aria-label={playing ? 'Pause' : 'Play'}
          title={playing ? 'Pause' : 'Play'}
          disabled={!placed.length || steps.length < 2}
          onClick={togglePlay}
          className="flex h-6 w-6 items-center justify-center rounded border border-slate-200 text-slate-700 hover:bg-slate-100 disabled:opacity-40"
        >
          {playing ? <Pause className="h-3 w-3" /> : <Play className="h-3 w-3" />}
        </button>
        <input
          type="range"
          aria-label="Training stage"
          min={0}
          max={last}
          value={j}
          disabled={!placed.length}
          onChange={(e) => setJ(Number(e.target.value))}
          className="min-w-0 flex-1 accent-sky-700"
        />
        <span className="min-w-[6.5em] text-right text-xs tabular-nums text-slate-600">{fmtStep(steps[j])}</span>
      </div>
      <HoverTip tip={tip} />
    </div>
  );
}
