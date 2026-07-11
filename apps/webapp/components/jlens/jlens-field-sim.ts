// Pure simulation + parsing for the jlens "Field" view: a time-based, animated
// reading of a run's lens read-outs. Each concept the lens surfaces becomes a
// bubble whose size tracks the decode-probability mass it accumulates across
// layers and whose colour tracks its salience relative to the strongest concept
// on screen. The module is framework-free (no React, no DOM) and deterministic
// so a scrubber can seek to any time by replaying from the start — this keeps it
// unit-testable and keeps the React component thin.

import type { LensTokenMessage, LensType, LensTypeSlice } from '@/lib/utils/lens';

// Fixed simulation step. The sim is integrated at this rate regardless of the
// display frame rate, so playback and seeks are frame-rate independent.
export const FIELD_DT = 1 / 60;

export interface FieldConfig {
  // Which lens type's read-outs feed the field. DIFF has no single read-out
  // direction of its own, so callers pass the concrete type to render (we fall
  // back to the first available type per token).
  lensType: LensType;
  // Fraction of layers to discard from the bottom before reading concepts.
  // Early layers are mostly token identity; meaning concentrates in the middle
  // and late layers. Matches the explorer's START_LAYER_FRACTION trimming.
  layerStartFraction: number;
  // Seconds each token occupies on the timeline.
  secondsPerToken: number;
  // Time constant (seconds) for a concept's mass to decay without reinforcement.
  decaySeconds: number;
  // Most concurrent bubbles; the weakest are evicted first.
  maxBubbles: number;
  // Drop the concept that matches the token actually emitted, so the field
  // shows context rather than the obvious next word.
  hideEmitted: boolean;
}

export const DEFAULT_FIELD_CONFIG: FieldConfig = {
  lensType: 'JACOBIAN_LENS',
  layerStartFraction: 0.5,
  secondsPerToken: 0.32,
  decaySeconds: 5,
  maxBubbles: 48,
  hideEmitted: true,
};

export interface FieldConcept {
  key: string;
  label: string;
  // Accumulated, layer-weighted decode-probability mass for this token.
  weight: number;
  // Layer index and probability where this concept decodes strongest.
  peakLayer: number;
  peakProb: number;
}

export interface FieldStep {
  position: number;
  isGenerated: boolean;
  concepts: FieldConcept[];
}

export interface FieldBubble {
  key: string;
  label: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  // Current (decayed) mass.
  energy: number;
  // Current rendered radius.
  radius: number;
  // Sim time this bubble was born, for the fade-in.
  born: number;
  // Reinforcement flash (0..1), decays quickly after each new injection.
  boost: number;
  peakLayer: number;
  peakProb: number;
  // Token position that first surfaced this concept (-1 while unseen).
  firstPosition: number;
  // Per-bubble phase offsets for the deterministic drift.
  driftX: number;
  driftY: number;
}

export interface FieldBounds {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

// ---- small deterministic helpers ----

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
export const smoothstep = (t: number) => {
  const c = clamp(t, 0, 1);
  return c * c * (3 - 2 * c);
};

// Cool-to-hot ramp tuned for a light background: slate, then sky, indigo, and a
// warm amber at the top end so the strongest concept reads as "hot" without the
// dark-theme neon of the standalone tool.
export const FIELD_RAMP: [number, [number, number, number]][] = [
  [0.0, [148, 163, 184]], // slate-400
  [0.35, [56, 189, 248]], // sky-400
  [0.6, [79, 70, 229]], // indigo-600
  [0.8, [217, 70, 239]], // fuchsia-500
  [1.0, [245, 158, 11]], // amber-500
];

export function rampColor(f: number): [number, number, number] {
  const x = clamp(f, 0, 1);
  for (let i = 1; i < FIELD_RAMP.length; i++) {
    if (x <= FIELD_RAMP[i][0]) {
      const [f0, c0] = FIELD_RAMP[i - 1];
      const [f1, c1] = FIELD_RAMP[i];
      const t = (x - f0) / (f1 - f0 || 1);
      return [Math.round(lerp(c0[0], c1[0], t)), Math.round(lerp(c0[1], c1[1], t)), Math.round(lerp(c0[2], c1[2], t))];
    }
  }
  return FIELD_RAMP[FIELD_RAMP.length - 1][1];
}

// ---- token cleaning ----

const PUNCT_ONLY = /^[\p{P}\p{S}\p{Z}\p{N}\p{C}\s]+$/u;
const SPECIAL = /^<[^>]*>$/;

// Whether a decoded token is not worth showing as a concept: empty, a special
// marker, punctuation/whitespace only, a lone ascii char, or a decode-failure
// replacement character. The explorer already filters non-word tokens
// server-side by default; this keeps the field clean for exports that did not.
export function isJunkToken(raw: string): boolean {
  const k = raw.trim();
  if (!k) return true;
  if (k.includes('�')) return true;
  if (SPECIAL.test(k)) return true;
  if (PUNCT_ONLY.test(k)) return true;
  if (k.length < 2 && k.charCodeAt(0) < 0x2e80) return true;
  return false;
}

// Strip leading tokenizer word-boundary markers and normalize for de-duping.
export function normalizeKey(raw: string): string {
  return raw
    .replace(/^[ĠĊ▁]+/, '')
    .trim()
    .toLowerCase();
}

// Display form: strip the boundary markers but keep the original case.
export function cleanLabel(raw: string): string {
  return raw.replace(/^[ĠĊ▁]+/, '').trim();
}

// Pick the read-out slice for the configured lens type, falling back to the
// token's first slice (covers DIFF, where there is no single own direction).
function sliceForToken(token: LensTokenMessage, lensType: LensType): LensTypeSlice | undefined {
  const match = token.results.find((r) => r.type === lensType);
  return match ?? token.results[0];
}

// Aggregate one token's per-layer read-outs into a ranked concept list.
export function conceptsForToken(token: LensTokenMessage, config: FieldConfig): FieldConcept[] {
  const slice = sliceForToken(token, config.lensType);
  if (!slice || !Array.isArray(slice.top_tokens)) return [];

  const layers = slice.top_tokens.length;
  const cut = Math.floor(layers * config.layerStartFraction);
  const span = layers - 1 - cut;
  const emittedKey = normalizeKey(token.token);

  const acc = new Map<string, FieldConcept & { bestWeight: number }>();
  for (let l = cut; l < layers; l++) {
    const row = slice.top_tokens[l];
    const probs = slice.top_probs?.[l];
    if (!row) continue;
    // Later layers weigh more: their read-outs are closer to what gets said.
    const ramp = span > 0 ? (l - cut) / span : 1;
    for (let k = 0; k < row.length; k++) {
      const p = probs ? probs[k] : 1 / (k + 2);
      if (p < 0.004) continue;
      const raw = row[k];
      if (isJunkToken(raw)) continue;
      const key = normalizeKey(raw);
      if (config.hideEmitted && key === emittedKey) continue;
      const w = p * ramp;
      let e = acc.get(key);
      if (!e) {
        e = { key, label: cleanLabel(raw), weight: 0, peakLayer: -1, peakProb: 0, bestWeight: 0 };
        acc.set(key, e);
      }
      e.weight += w;
      if (w > e.bestWeight) {
        e.bestWeight = w;
        e.label = cleanLabel(raw);
      }
      if (p > e.peakProb) {
        e.peakProb = p;
        e.peakLayer = l;
      }
    }
  }

  const out: FieldConcept[] = [];
  for (const e of acc.values()) {
    if (e.weight >= 0.04) {
      out.push({ key: e.key, label: e.label, weight: e.weight, peakLayer: e.peakLayer, peakProb: e.peakProb });
    }
  }
  out.sort((a, b) => b.weight - a.weight);
  return out.slice(0, 24);
}

// Build the per-token concept steps for a whole run.
export function buildFieldSteps(tokens: LensTokenMessage[], config: FieldConfig): FieldStep[] {
  return tokens.map((t) => ({
    position: t.position,
    isGenerated: t.is_generated,
    concepts: conceptsForToken(t, config),
  }));
}

// Whether a run has any renderable concepts at all under the given config, so
// the UI can show an empty state instead of a blank canvas.
export function hasAnyConcepts(steps: FieldStep[]): boolean {
  return steps.some((s) => s.concepts.length > 0);
}

interface TimelineEvent {
  time: number;
  concepts: FieldConcept[];
  position: number;
}

// Prompt read-outs count for less than generated ones: the field should be
// driven by what the model is composing, not what it was given.
const PROMPT_WEIGHT = 0.5;
const GENERATED_WEIGHT = 1;
// The concept flares a fraction of a token-step before its token lands.
const LEAD_FRACTION = 0.55;

// The deterministic bubble field. Construct with the run's steps + bounds, then
// drive it with `tick(dt)` or jump to any time with `seek(t)` (which replays
// from the start when moving backwards, keeping playback reproducible).
export class FieldSim {
  readonly duration: number;
  // Number of tokens on the timeline (used for the "token i / N" counter).
  readonly stepCount: number;
  private readonly events: TimelineEvent[];
  private readonly startTime = 0.6;
  bubbles: Map<string, FieldBubble> = new Map();
  time = 0;
  energyMax = 1;

  private eventIdx = 0;
  private spawnCount = 0;
  private rng: () => number = mulberry32(1337);

  constructor(
    steps: FieldStep[],
    private readonly config: FieldConfig,
    private bounds: FieldBounds,
  ) {
    this.events = [];
    steps.forEach((step, i) => {
      if (step.concepts.length === 0) return;
      const time = Math.max(0.05, this.startTime + i * config.secondsPerToken - LEAD_FRACTION * config.secondsPerToken);
      const mult = step.isGenerated ? GENERATED_WEIGHT : PROMPT_WEIGHT;
      const concepts = step.concepts.map((c) => ({ ...c, weight: c.weight * mult }));
      this.events.push({ time, concepts, position: step.position });
    });
    this.events.sort((a, b) => a.time - b.time);
    this.stepCount = steps.length;
    this.duration = this.startTime + steps.length * config.secondsPerToken + 2.5;
    this.reset();
  }

  // 1-based ordinal of the token being read at time t (0 before the first),
  // clamped to the run length. Drives the "token i / N" counter.
  tokenIndexAt(t: number): number {
    const idx = Math.floor((t - this.startTime) / this.config.secondsPerToken) + 1;
    return clamp(idx, 0, this.stepCount);
  }

  setBounds(bounds: FieldBounds) {
    this.bounds = bounds;
  }

  reset() {
    this.time = 0;
    this.eventIdx = 0;
    this.spawnCount = 0;
    this.rng = mulberry32(1337);
    this.bubbles = new Map();
    this.energyMax = 1;
  }

  private spawnPos(): [number, number] {
    const b = this.bounds;
    const cx = (b.x0 + b.x1) / 2;
    const cy = (b.y0 + b.y1) / 2;
    const n = this.spawnCount++;
    // Golden-angle scatter so early bubbles fan out instead of stacking.
    const ang = n * 2.39996 + this.rng() * 0.7;
    const rad = (0.12 + 0.72 * Math.sqrt((n % 60) / 60)) * Math.min(b.x1 - b.x0, b.y1 - b.y0) * 0.5;
    return [
      cx + Math.cos(ang) * rad * 1.35 + (this.rng() - 0.5) * 30,
      cy + Math.sin(ang) * rad + (this.rng() - 0.5) * 30,
    ];
  }

  private inject(concepts: FieldConcept[], position: number) {
    for (const c of concepts) {
      let bub = this.bubbles.get(c.key);
      if (!bub) {
        if (this.bubbles.size >= this.config.maxBubbles) {
          let weakestKey: string | null = null;
          let weakest = Infinity;
          for (const [k, b] of this.bubbles) {
            if (b.energy < weakest) {
              weakest = b.energy;
              weakestKey = k;
            }
          }
          if (weakest > c.weight * 1.5 || weakestKey === null) continue;
          this.bubbles.delete(weakestKey);
        }
        const [x, y] = this.spawnPos();
        bub = {
          key: c.key,
          label: c.label,
          x,
          y,
          vx: 0,
          vy: 0,
          energy: 0,
          radius: 0,
          born: this.time,
          boost: 0,
          peakLayer: -1,
          peakProb: 0,
          firstPosition: position,
          driftX: this.rng() * 1000,
          driftY: this.rng() * 1000,
        };
        this.bubbles.set(c.key, bub);
      }
      bub.energy += c.weight;
      bub.label = c.label;
      if (c.peakProb > bub.peakProb) {
        bub.peakProb = c.peakProb;
        bub.peakLayer = c.peakLayer;
      }
      bub.boost = Math.min(1, bub.boost + clamp(c.weight * 2, 0.25, 1));
    }
  }

  tick(dt: number = FIELD_DT) {
    const next = this.time + dt;
    while (this.eventIdx < this.events.length && this.events[this.eventIdx].time <= next) {
      this.inject(this.events[this.eventIdx].concepts, this.events[this.eventIdx].position);
      this.eventIdx++;
    }
    this.time = next;

    const b = this.bounds;
    const decayK = Math.exp(-dt / this.config.decaySeconds);
    let em = 0.001;
    for (const bub of this.bubbles.values()) if (bub.energy > em) em = bub.energy;
    this.energyMax = Math.max(em, lerp(this.energyMax, em, 0.02), 0.6);

    const arr = [...this.bubbles.values()];
    const cx = (b.x0 + b.x1) / 2;
    const cy = (b.y0 + b.y1) / 2;
    const rMin = 10;
    const rMax = Math.min(b.x1 - b.x0, b.y1 - b.y0) * 0.155;

    let sumR2 = 0;
    for (const bub of arr) {
      bub.energy *= decayK;
      bub.boost *= Math.exp(-dt / 0.45);
      const f = clamp(bub.energy / this.energyMax, 0, 1);
      const target = rMin + (rMax - rMin) * Math.sqrt(f);
      (bub as FieldBubble & { target: number }).target = target;
      sumR2 += target * target;
    }
    const fieldArea = (b.x1 - b.x0) * (b.y1 - b.y0);
    const areaScale = Math.min(1, Math.sqrt((fieldArea * 0.3) / (Math.PI * sumR2 || 1)));
    for (const bub of arr) {
      const target = (bub as FieldBubble & { target: number }).target * areaScale;
      bub.radius += (target - bub.radius) * (1 - Math.exp(-dt * 6));
    }

    const pad = 10;
    for (let i = 0; i < arr.length; i++) {
      const a = arr[i];
      const nx = Math.sin(this.time * 0.35 + a.driftX) * 0.9 + Math.sin(this.time * 0.13 + a.driftY * 1.7) * 0.5;
      const ny = Math.cos(this.time * 0.29 + a.driftY) * 0.9 + Math.sin(this.time * 0.17 + a.driftX * 1.3) * 0.5;
      a.vx += nx * 3.2 * dt;
      a.vy += ny * 3.2 * dt;
      a.vx += (cx - a.x) * 0.13 * dt;
      a.vy += (cy - a.y) * 0.13 * dt;
      for (let j = i + 1; j < arr.length; j++) {
        const c = arr[j];
        let dx = c.x - a.x;
        let dy = c.y - a.y;
        const min = a.radius + c.radius + pad;
        const d2 = dx * dx + dy * dy;
        if (d2 < min * min) {
          const d = Math.sqrt(d2) || 0.001;
          const push = ((min - d) / min) * 52 * dt;
          dx /= d;
          dy /= d;
          a.vx -= dx * push;
          a.vy -= dy * push;
          c.vx += dx * push;
          c.vy += dy * push;
        }
      }
    }

    for (const bub of arr) {
      bub.vx *= 0.93;
      bub.vy *= 0.93;
      const vmax = 60;
      const sp = Math.hypot(bub.vx, bub.vy);
      if (sp > vmax) {
        bub.vx *= vmax / sp;
        bub.vy *= vmax / sp;
      }
      bub.x += bub.vx * dt;
      bub.y += bub.vy * dt;
      const wall = 6;
      if (bub.x - bub.radius < b.x0 + wall) bub.vx += (b.x0 + wall - (bub.x - bub.radius)) * 4 * dt;
      if (bub.x + bub.radius > b.x1 - wall) bub.vx -= (bub.x + bub.radius - (b.x1 - wall)) * 4 * dt;
      if (bub.y - bub.radius < b.y0 + wall) bub.vy += (b.y0 + wall - (bub.y - bub.radius)) * 4 * dt;
      if (bub.y + bub.radius > b.y1 - wall) bub.vy -= (bub.y + bub.radius - (b.y1 - wall)) * 4 * dt;
    }

    // Two relaxation passes resolve overlaps the springs miss, so neighbouring
    // bubbles keep a readable gap.
    for (let pass = 0; pass < 2; pass++) {
      for (let i = 0; i < arr.length; i++) {
        const a = arr[i];
        for (let j = i + 1; j < arr.length; j++) {
          const c = arr[j];
          let dx = c.x - a.x;
          let dy = c.y - a.y;
          const min = a.radius + c.radius + pad;
          const d2 = dx * dx + dy * dy;
          if (d2 >= min * min) continue;
          const d = Math.sqrt(d2) || 0.001;
          const corr = (min - d) * 0.55;
          dx /= d;
          dy /= d;
          const wA = c.radius / (a.radius + c.radius + 0.001);
          a.x -= dx * corr * wA;
          a.y -= dy * corr * wA;
          c.x += dx * corr * (1 - wA);
          c.y += dy * corr * (1 - wA);
        }
      }
    }
    for (const bub of arr) {
      bub.x = clamp(bub.x, b.x0 + bub.radius * 0.55, b.x1 - bub.radius * 0.55);
      bub.y = clamp(bub.y, b.y0 + bub.radius * 0.55, b.y1 - bub.radius * 0.55);
    }

    for (const [k, bub] of this.bubbles) {
      if (bub.energy < 0.012 && this.time - bub.born > 1.5) this.bubbles.delete(k);
    }
  }

  seek(t: number) {
    const target = clamp(t, 0, this.duration);
    if (target < this.time) this.reset();
    const steps = Math.ceil((target - this.time) / FIELD_DT);
    for (let i = 0; i < steps; i++) this.tick();
  }
}
