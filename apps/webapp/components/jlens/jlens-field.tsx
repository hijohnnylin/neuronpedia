'use client';

// The jlens "Field" view: an animated, time-based reading of a run's lens
// read-outs, as an alternative to the token-by-token transcript. Concepts the
// lens surfaces rise as bubbles (size = accumulated decode-probability mass,
// colour = salience vs the strongest concept) and fade as the model moves on,
// so you can watch what is in the model's workspace across the whole generation
// rather than inspecting one token at a time. All the simulation lives in the
// framework-free `jlens-field-sim` module; this component only owns the canvas,
// the playback loop, and the controls.

import { LensTokenMessage, LensType } from '@/lib/utils/lens';
import { List, Orbit, Pause, Play, RotateCcw } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  buildFieldSteps,
  clamp,
  DEFAULT_FIELD_CONFIG,
  FIELD_DT,
  FIELD_RAMP,
  FieldBounds,
  FieldSim,
  hasAnyConcepts,
  rampColor,
  smoothstep,
} from './jlens-field-sim';

// Canvas label font, with CJK fallbacks so non-latin read-out tokens render.
const FIELD_FONT =
  '"Inter","SF Pro Text","Segoe UI",system-ui,"Noto Sans SC","PingFang SC","Microsoft YaHei",sans-serif';
const SPEEDS = [0.5, 1, 2];
const CANVAS_PADDING = 16;

interface HoverInfo {
  label: string;
  salience: number;
  peakLayer: number;
  peakProb: number;
  x: number;
  y: number;
}

function formatTime(s: number): string {
  const clamped = Math.max(0, s);
  const mins = Math.floor(clamped / 60);
  const secs = Math.floor(clamped % 60);
  const tenths = Math.floor((clamped % 1) * 10);
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${tenths}`;
}

const rgba = (c: [number, number, number], a: number) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

// Which reading the left panel shows: the default token transcript, or the
// animated concept field.
export type JlensLeftView = 'default' | 'field';

// Segmented toggle between the transcript and the field, matching the sidebar's
// LensModeToggle so it reads as part of the same control set. `defaultLabel`
// names the transcript option per interface ("Tokens" for completion, "Chat"
// for chat).
export function JlensViewToggle({
  view,
  onChange,
  defaultLabel,
}: {
  view: JlensLeftView;
  onChange: (v: JlensLeftView) => void;
  defaultLabel: string;
}) {
  const options: { value: JlensLeftView; label: string; icon: typeof List }[] = [
    { value: 'default', label: defaultLabel, icon: List },
    { value: 'field', label: 'Field', icon: Orbit },
  ];
  return (
    <div className="flex flex-row items-center overflow-hidden rounded border border-sky-600">
      {options.map((o, i) => {
        const Icon = o.icon;
        return (
          <button
            key={o.value}
            type="button"
            onClick={() => onChange(o.value)}
            aria-pressed={view === o.value}
            className={`flex h-[23px] items-center justify-center gap-x-1 whitespace-nowrap px-2.5 py-1.5 text-[9.5px] font-semibold uppercase leading-none tracking-wide transition-colors ${
              view === o.value ? 'bg-sky-600 text-white' : 'bg-white text-sky-600 hover:bg-sky-500 hover:text-white'
            } ${i > 0 ? 'border-l border-sky-600' : ''}`}
          >
            <Icon className="h-3 w-3" />
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

export default function JlensField({
  tokens,
  lensType,
  layerStartFraction,
  className,
}: {
  tokens: LensTokenMessage[];
  lensType: LensType;
  // Overrides the default layer window (fraction of low layers to drop).
  layerStartFraction?: number;
  className?: string;
}) {
  const config = useMemo(
    () => ({
      ...DEFAULT_FIELD_CONFIG,
      lensType,
      layerStartFraction: layerStartFraction ?? DEFAULT_FIELD_CONFIG.layerStartFraction,
    }),
    [lensType, layerStartFraction],
  );

  const steps = useMemo(() => buildFieldSteps(tokens, config), [tokens, config]);
  const isEmpty = useMemo(() => !hasAnyConcepts(steps), [steps]);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const simRef = useRef<FieldSim | null>(null);
  const rafRef = useRef<number>(0);
  const lastTsRef = useRef<number>(0);
  const playingRef = useRef<boolean>(false);
  const speedRef = useRef<number>(1);
  const boundsRef = useRef<FieldBounds>({ x0: 0, y0: 0, x1: 0, y1: 0 });
  const dprRef = useRef<number>(1);

  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [displayTime, setDisplayTime] = useState(0);
  const [duration, setDuration] = useState(1);
  const [hover, setHover] = useState<HoverInfo | null>(null);

  speedRef.current = speed;

  // ---- rendering ----

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const sim = simRef.current;
    if (!canvas || !sim) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const b = boundsRef.current;
    const w = b.x1 + CANVAS_PADDING;
    const h = b.y1 + CANVAS_PADDING;

    ctx.setTransform(dprRef.current, 0, 0, dprRef.current, 0, 0);
    // Panel-matched light backdrop with a faint centre lift for depth.
    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(0, 0, w, h);
    const bg = ctx.createRadialGradient(w / 2, h * 0.46, 20, w / 2, h / 2, Math.max(w, h) * 0.7);
    bg.addColorStop(0, 'rgba(255,255,255,0.9)');
    bg.addColorStop(1, 'rgba(248,250,252,0)');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);

    const bubbles = [...sim.bubbles.values()].sort((a, z) => a.energy - z.energy);
    for (const bub of bubbles) {
      const f = clamp(bub.energy / sim.energyMax, 0, 1);
      const fade = smoothstep((sim.time - bub.born) / 0.5);
      const alpha = fade * (0.4 + 0.6 * f);
      if (alpha <= 0.01 || bub.radius < 1.5) continue;
      const col = rampColor(f);

      const grad = ctx.createRadialGradient(
        bub.x,
        bub.y - bub.radius * 0.3,
        bub.radius * 0.1,
        bub.x,
        bub.y,
        bub.radius,
      );
      grad.addColorStop(0, rgba(col, 0.28 * alpha + 0.06));
      grad.addColorStop(0.7, rgba(col, 0.16 * alpha));
      grad.addColorStop(1, rgba(col, 0.04 * alpha));
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(bub.x, bub.y, bub.radius, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = rgba(col, (0.45 + 0.4 * f) * fade);
      ctx.lineWidth = 1.5;
      ctx.stroke();

      if (bub.boost > 0.03) {
        ctx.strokeStyle = rgba(col, bub.boost * 0.55);
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(bub.x, bub.y, bub.radius + 3 + 8 * (1 - bub.boost), 0, Math.PI * 2);
        ctx.stroke();
      }

      if (bub.radius >= 13) {
        let fontSize = clamp(bub.radius * 0.42, 10, 24);
        ctx.font = `600 ${fontSize}px ${FIELD_FONT}`;
        let textW = ctx.measureText(bub.label).width;
        const maxIn = bub.radius * 1.85;
        while (textW > maxIn && fontSize > 9) {
          fontSize -= 1;
          ctx.font = `600 ${fontSize}px ${FIELD_FONT}`;
          textW = ctx.measureText(bub.label).width;
        }
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        if (textW <= maxIn) {
          // White halo keeps dark labels legible over the strongest fills.
          ctx.lineWidth = 3;
          ctx.strokeStyle = `rgba(255,255,255,${0.85 * fade})`;
          ctx.strokeText(bub.label, bub.x, bub.y);
          ctx.fillStyle = `rgba(15,23,42,${(0.7 + 0.3 * f) * fade})`;
          ctx.fillText(bub.label, bub.x, bub.y);
        }
        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
      }
    }
  }, []);

  // ---- sizing ----

  const resize = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;
    const rect = container.getBoundingClientRect();
    const cssW = Math.max(120, Math.floor(rect.width));
    const cssH = Math.max(120, Math.floor(rect.height));
    const dpr = Math.min(2, typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1);
    dprRef.current = dpr;
    canvas.width = Math.floor(cssW * dpr);
    canvas.height = Math.floor(cssH * dpr);
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;
    boundsRef.current = {
      x0: CANVAS_PADDING,
      y0: CANVAS_PADDING,
      x1: cssW - CANVAS_PADDING,
      y1: cssH - CANVAS_PADDING,
    };
    simRef.current?.setBounds(boundsRef.current);
    draw();
  }, [draw]);

  // ---- playback loop ----

  const stopLoop = useCallback(() => {
    playingRef.current = false;
    setPlaying(false);
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = 0;
  }, []);

  const startLoop = useCallback(() => {
    const sim = simRef.current;
    if (!sim || isEmpty) return;
    if (sim.time >= sim.duration) sim.reset();
    playingRef.current = true;
    setPlaying(true);
    lastTsRef.current = performance.now();
    let acc = 0;
    const loop = (ts: number) => {
      if (!playingRef.current) return;
      acc += Math.min(0.1, (ts - lastTsRef.current) / 1000) * speedRef.current;
      lastTsRef.current = ts;
      while (acc >= FIELD_DT) {
        sim.tick();
        acc -= FIELD_DT;
      }
      draw();
      setDisplayTime(sim.time);
      if (sim.time >= sim.duration) {
        stopLoop();
        return;
      }
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
  }, [draw, isEmpty, stopLoop]);

  const togglePlay = useCallback(() => {
    if (playingRef.current) stopLoop();
    else startLoop();
  }, [startLoop, stopLoop]);

  const seekToFraction = useCallback(
    (frac: number) => {
      const sim = simRef.current;
      if (!sim) return;
      sim.seek(clamp(frac, 0, 1) * sim.duration);
      draw();
      setDisplayTime(sim.time);
    },
    [draw],
  );

  const replay = useCallback(() => {
    const sim = simRef.current;
    if (!sim) return;
    sim.reset();
    setDisplayTime(0);
    startLoop();
  }, [startLoop]);

  // Build (or rebuild) the sim whenever the run or config changes, then autoplay
  // unless the visitor prefers reduced motion.
  useEffect(() => {
    stopLoop();
    resize();
    const sim = new FieldSim(steps, config, boundsRef.current);
    simRef.current = sim;
    setDuration(sim.duration);
    setDisplayTime(0);
    draw();
    const reduced = typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    if (!isEmpty && !reduced) startLoop();
    return () => stopLoop();
    // startLoop/resize/draw are stable-by-callback; re-running only on data change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [steps, config, isEmpty]);

  // Keep the canvas sized to its container.
  useEffect(() => {
    resize();
    if (typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(() => resize());
    if (containerRef.current) ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, [resize]);

  // ---- scrubber (pointer-driven, styled like the layer slider) ----

  const scrubRef = useRef<HTMLDivElement | null>(null);
  const scrubbingRef = useRef(false);
  const fractionFromClientX = (clientX: number, el: HTMLElement) => {
    const rect = el.getBoundingClientRect();
    return rect.width > 0 ? clamp((clientX - rect.left) / rect.width, 0, 1) : 0;
  };

  const progress = duration > 0 ? clamp(displayTime / duration, 0, 1) : 0;
  const tokenIndex = simRef.current ? simRef.current.tokenIndexAt(displayTime) : 0;
  const tokenCount = simRef.current?.stepCount ?? tokens.length;

  // ---- hover read-out ----

  const onCanvasMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const sim = simRef.current;
    const canvas = canvasRef.current;
    if (!sim || !canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    let best: HoverInfo | null = null;
    let bestDist = Infinity;
    for (const bub of sim.bubbles.values()) {
      const d = Math.hypot(bub.x - mx, bub.y - my);
      if (d < bub.radius + 4 && d < bestDist) {
        bestDist = d;
        best = {
          label: bub.label,
          salience: clamp(bub.energy / sim.energyMax, 0, 1),
          peakLayer: bub.peakLayer,
          peakProb: bub.peakProb,
          x: mx,
          y: my,
        };
      }
    }
    setHover(best);
  };

  if (isEmpty) {
    return (
      <div
        className={`flex min-h-0 flex-1 flex-col items-center justify-center rounded-2xl border border-slate-200 bg-slate-50 text-center text-slate-400 ${className ?? ''}`}
      >
        <div className="text-[13px] font-medium text-slate-500">No concepts to animate</div>
        <div className="mt-1 max-w-xs text-[11px] leading-snug">
          This run has no lens read-outs above the noise floor for the selected lens and layer window.
        </div>
      </div>
    );
  }

  return (
    <div className={`flex min-h-0 flex-1 flex-col gap-y-2 ${className ?? ''}`}>
      <div ref={containerRef} className="relative min-h-0 flex-1 overflow-hidden rounded-2xl border border-slate-200">
        <canvas
          ref={canvasRef}
          className="block h-full w-full"
          onMouseMove={onCanvasMove}
          onMouseLeave={() => setHover(null)}
        />
        {hover && (
          <div
            className="pointer-events-none absolute z-10 max-w-[220px] rounded-md border border-slate-200 bg-white/95 px-2.5 py-1.5 text-[11px] shadow-md backdrop-blur-sm"
            style={{
              left: clamp(hover.x + 12, 8, (boundsRef.current.x1 || 200) - 160),
              top: clamp(hover.y + 12, 8, (boundsRef.current.y1 || 200) - 40),
            }}
          >
            <div className="font-semibold text-slate-800">{hover.label}</div>
            <div className="mt-0.5 font-mono text-[10px] leading-relaxed text-slate-500">
              salience {Math.round(hover.salience * 100)}%
              {hover.peakLayer >= 0 && (
                <>
                  {' · '}
                  peak {Math.round(hover.peakProb * 100)}% @ layer {hover.peakLayer}
                </>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Transport bar, styled to the explorer's light controls. */}
      <div className="flex flex-none flex-row items-center gap-x-2.5 px-1 pb-1">
        <button
          type="button"
          onClick={togglePlay}
          aria-label={playing ? 'Pause' : 'Play'}
          className="flex h-8 w-8 flex-none items-center justify-center rounded-full bg-sky-700 text-white transition-colors hover:bg-sky-800"
        >
          {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4 pl-0.5" />}
        </button>
        <button
          type="button"
          onClick={replay}
          aria-label="Replay from start"
          className="flex h-7 w-7 flex-none items-center justify-center rounded-md border border-slate-200 bg-white text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-600"
        >
          <RotateCcw className="h-3.5 w-3.5" />
        </button>

        <div
          ref={scrubRef}
          className="relative h-[23px] flex-1 cursor-pointer touch-none select-none rounded border border-slate-300 bg-slate-200"
          onPointerDown={(e) => {
            scrubbingRef.current = true;
            if (playingRef.current) stopLoop();
            seekToFraction(fractionFromClientX(e.clientX, e.currentTarget));
            e.currentTarget.setPointerCapture(e.pointerId);
          }}
          onPointerMove={(e) => {
            if (!scrubbingRef.current) return;
            seekToFraction(fractionFromClientX(e.clientX, e.currentTarget));
          }}
          onPointerUp={(e) => {
            scrubbingRef.current = false;
            if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
          }}
        >
          <div
            className="pointer-events-none absolute inset-y-0 left-0 rounded-[3px] bg-sky-600"
            style={{ width: `${progress * 100}%` }}
          />
          <div
            className="pointer-events-none absolute top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border border-sky-700 bg-white shadow"
            style={{ left: `${progress * 100}%` }}
          />
        </div>

        <span className="flex-none whitespace-nowrap font-mono text-[10.5px] tabular-nums text-slate-500">
          <span className="text-slate-700">{formatTime(displayTime)}</span> / {formatTime(duration)}
          {' · '}
          <span className="text-slate-700">{String(Math.max(1, tokenIndex)).padStart(2, '0')}</span>/{tokenCount}
        </span>

        <button
          type="button"
          onClick={() => {
            const next = SPEEDS[(SPEEDS.indexOf(speed) + 1) % SPEEDS.length];
            setSpeed(next);
          }}
          aria-label="Playback speed"
          className="flex-none rounded-md border border-slate-200 bg-white px-2 py-1 font-mono text-[10.5px] font-semibold text-slate-500 transition-colors hover:bg-slate-100"
        >
          {speed}&times;
        </button>
      </div>

      {/* Encoding legend, mirroring the sidebar's compact key. */}
      <div className="flex flex-none flex-row items-center justify-center gap-x-4 pb-1 text-[9px] font-medium uppercase tracking-wide text-slate-400">
        <span className="flex items-center gap-x-1.5">
          <span className="inline-flex items-end gap-x-0.5">
            <span className="inline-block h-1.5 w-1.5 rounded-full border border-slate-400" />
            <span className="inline-block h-3 w-3 rounded-full border border-slate-400" />
          </span>
          size · probability mass
        </span>
        <span className="flex items-center gap-x-1.5">
          <span
            className="inline-block h-2 w-14 rounded-full"
            style={{
              background: `linear-gradient(to right, ${FIELD_RAMP.map(([f, c]) => `${rgba(c, 0.9)} ${Math.round(f * 100)}%`).join(', ')})`,
            }}
          />
          salience · share of strongest
        </span>
      </div>
    </div>
  );
}
