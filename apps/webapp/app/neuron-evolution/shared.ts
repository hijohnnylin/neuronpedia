/**
 * Neuron Evolution: how MLP neurons of the Pythia suite change across training checkpoints.
 *
 * The data is produced by https://github.com/HeartyHaven/neuron-evolution. For each sampled neuron and
 * each checkpoint an LLM explains the strongest activations; the explanation is scored by detection
 * (telling top-activating text from random text); the per-step patterns are then chained into concept
 * lineages, each born by branching from an older one, replacing a dead one, arriving unrelated, or
 * appearing only once. Two judges summarize the whole trajectory (recursive generalization vs
 * superposition; denoising-like sharpening vs drift).
 *
 * The published bundles are static JSON served from DATA_BASE_URL, one folder per model:
 *
 *   data/models.json                 catalog of models                              (ModelCatalog)
 *   <data_path>/meta.json[.gz]       steps, loss, groups and one entry per neuron    (Meta)
 *   <data_path>/L<layer>.json[.gz]   per-step explanations and top contexts, by layer (LayerData)
 *   <data_path>/sim.json[.gz]        pairwise explanation similarity per step          (SimData)
 *
 * Text fields are in the pipeline's source language, with translations under `en` (and `zh`); this page
 * reads English through `tr`. Everything in this file is pure so it can be unit tested; the components
 * beside it do the rendering.
 */

export const DATA_BASE_URL = 'https://heartyhaven.github.io/pythia-neuron-explorer';
export const NEURON_EVOLUTION_PATH = '/neuron-evolution';
export const PIPELINE_URL = 'https://github.com/HeartyHaven/neuron-evolution';
export const BLOG_URL = 'https://heartyhaven.github.io/blog/neuron-evolution/';

/** The neuron opened when the URL names none: a representative recursive-generalization example. */
export const DEFAULT_NEURON_ID = '9_2943';

/** From this step on, a mechanism's mean share decides the "late-training makeup" badge. */
export const LATE_STEP = 16000;

// ---------- Data types ----------

export type Mechanism = 'branch' | 'coexist' | 'replace' | 'sporadic' | 'unexplained';
export type Birth = 'branch' | 'coexist' | 'replace' | 'sporadic';
export type ColorBy = 'mech' | 'family';

export interface ModelEntry {
  id: string;
  label: string;
  status: string;
  data_version?: string;
  compression?: 'gzip' | string;
  data_path?: string;
  source_language?: string;
  corpus?: string;
  dataset_url?: string;
  annotation_model?: string;
  review_model?: string;
  translation_model?: string;
  counts?: {
    neurons?: number;
    published_steps?: number;
    planned_annotation_steps?: number;
  };
  en?: { message?: string } | null;
  message?: string;
}

export interface ModelCatalog {
  schema_version: number;
  default_model: string;
  models: ModelEntry[];
}

export interface GroupItem {
  name: string;
  desc?: string;
  name_en?: string;
  desc_en?: string;
}

export interface GroupDef {
  title: string;
  title_en?: string;
  order: string[];
  items: Record<string, GroupItem>;
}

export interface Lineage {
  id: string;
  name: string;
  parent?: string | null;
  how?: string;
  first_step?: number | null;
  core?: boolean;
  birth?: Birth | null;
  birth_parent?: string | null;
  replaces?: string | null;
}

/** A record whose text fields may carry an `en` translation object with the same field names. */
export interface Translatable {
  en?: Record<string, unknown> | null;
  [key: string]: unknown;
}

export interface NeuronMeta extends Translatable {
  id: string;
  layer: number;
  neuron: number;
  mech?: Partial<Record<Mechanism, number[]>> | null;
  mech_late?: Partial<Record<Mechanism, number>> | null;
  dominant: string;
  core_summary?: string | null;
  lineage_shares?: Record<string, number>[];
  lineages?: Lineage[];
  category: string;
  label: string;
  rg: string;
  rg_conf?: number | null;
  core?: string | null;
  rg_summary?: string | null;
  rg_evidence?: string | null;
  diff: string;
  diff_conf?: number | null;
  diff_summary?: string | null;
  diff_evidence?: string | null;
  commit?: number | null;
  saturation?: number | null;
  cbf?: string | null;
  trend?: string | null;
  det?: (number | null)[];
  /** Semantic-map position per step, each in [0, 1]. Absent until the semantic map has been computed. */
  sem?: [number, number][] | null;
  /** One-line concept per step, source language and English. */
  cz?: string[];
  ce?: string[];
  cluster?: number;
}

export interface Meta {
  steps: number[];
  loss: number[];
  planned_steps?: number[];
  partial?: boolean;
  groups: Record<string, GroupDef>;
  neurons: NeuronMeta[];
}

export interface Pattern {
  name?: string;
  trigger?: string;
  context?: string;
  domain?: string;
  segments?: number[];
  lineage?: string | null;
}

export interface Context {
  subset: string;
  doc: number;
  pos: number;
  value: number;
  /** Index of the strongest token in `tokens`. */
  hit: number;
  tokens: string[];
  /** Activation per token as a percentage of this step's strongest activation (`top1`). */
  a: number[];
}

export interface StepRecord extends Translatable {
  step: number;
  det?: number | null;
  clarity?: number | null;
  concept?: string;
  description?: string;
  relation?: string;
  patterns: Pattern[];
  unexplained?: number[];
  top1: number;
  contexts: Context[];
}

export type LayerData = Record<string, StepRecord[]>;

export interface SimData {
  lo: number;
  hi: number;
  order: string[];
  /** One base64 N×N uint8 matrix per step; a byte maps linearly onto [lo, hi]. */
  steps: string[];
}

export interface DecodedSim {
  lo: number;
  hi: number;
  N: number;
  /** The largest explanation distance anywhere, so the anchored map keeps one scale across steps. */
  dmax: number;
  idx: Record<string, number>;
  steps: Uint8Array[];
}

// ---------- Labels ----------

export const LABELS: Record<string, Record<string, string>> = {
  rg: {
    recursive_generalization: 'grew out from a core (recursive generalization)',
    superposition: 'unrelated things coexist (superposition)',
    replacement: 'replaced by something else',
    stable_monosemantic: 'always one thing',
    mixed: 'mixed',
    uninterpretable: 'no clear pattern',
  },
  diff: {
    denoising: 'sharpened gradually (like denoising)',
    drift: 'changed content midway',
    phase_transition: 'appeared suddenly',
    early_fixed: 'settled early',
    late_degradation: 'got blurrier late',
    unclear: 'unclear',
  },
  dom: {
    branch: 'mostly branches (recursive generalization)',
    coexist: 'mostly unrelated coexistence',
    replace: 'mostly replacement',
    sporadic: 'mostly sporadic',
    none: 'no clear majority',
    unknown: 'not analyzed',
  },
  mech: {
    branch: 'branch (recursive generalization)',
    coexist: 'unrelated coexistence',
    replace: 'replacement',
    sporadic: 'sporadic',
    unexplained: 'unexplained',
  },
  cbf: { yes: 'yes', no: 'no', unclear: 'unclear' },
  trend: {
    monotonic_up: 'kept getting clearer',
    up_then_plateau: 'got clearer, then leveled off',
    non_monotonic: 'up and down',
    flat: 'stayed about the same',
    down: 'got blurrier',
  },
  rel: {
    single: 'a single pattern',
    related: 'related to each other',
    unrelated: 'unrelated to each other',
    unclear: 'of unclear relation',
  },
};

export function label(kind: string, value: string | null | undefined): string {
  if (value == null) return '—';
  if (value === 'pending') return 'not yet analyzed';
  return LABELS[kind]?.[value] ?? value;
}

/** The series palette, shared by the mechanism and family colorings. */
export const SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'];
export const SERIES_OTHER = '#a8a7a0';
export const DOT_MUTED = '#cfcec8';

export const MECHANISMS: { key: Mechanism; color: string }[] = [
  { key: 'branch', color: SERIES[1] },
  { key: 'coexist', color: SERIES[2] },
  { key: 'replace', color: SERIES[3] },
  { key: 'sporadic', color: SERIES[4] },
  { key: 'unexplained', color: SERIES_OTHER },
];
export const MECHANISM_COLOR: Record<string, string> = Object.fromEntries(MECHANISMS.map((m) => [m.key, m.color]));

/** Families (lineages traced to their root concept) take the whole palette; smaller ones share grey. */
export const FAMILY_COLORS = SERIES;

// ---------- Text helpers ----------

/** An English text field: the `en` translation when present, else the field itself. */
export function tr(obj: Translatable | null | undefined, field: string): string {
  const v = obj?.en?.[field] ?? obj?.[field];
  return typeof v === 'string' ? v : '';
}

/** An English text field of the i-th item of a list field (lineage names, pattern names, ...). */
export function trItem(obj: Translatable | null | undefined, list: string, i: number, field: string): string {
  const en = obj?.en?.[list];
  const enItem = Array.isArray(en) ? (en[i] as Record<string, unknown> | undefined) : undefined;
  const base = obj?.[list];
  const baseItem = Array.isArray(base) ? (base[i] as Record<string, unknown> | undefined) : undefined;
  const v = enItem?.[field] ?? baseItem?.[field];
  return typeof v === 'string' ? v : '';
}

export const groupName = (item: GroupItem | undefined, key: string) => (item ? item.name_en || item.name : key);
export const groupDesc = (item: GroupItem | undefined) => (item ? item.desc_en || item.desc || '' : '');
export const groupTitle = (g: GroupDef) => g.title_en || g.title;

export const fmtStep = (s: number) => `step ${s.toLocaleString('en')}`;
export const fmtAxis = (s: number) => (s >= 1000 ? `${s / 1000}k` : String(s));
export const pct = (v: number | undefined) => `${Math.round((v || 0) * 100)}%`;

export const neuronName = (n: { layer: number; neuron: number }) => `L${n.layer}/N${n.neuron}`;
export const neuronRef = (n: { layer: number; neuron: number }) => `L${n.layer}-N${n.neuron}`;

/** "L6-N201", "L6/N201", "L6N201" or "6_201" → "6_201"; anything else → null. */
export function parseNeuronRef(ref: string | null | undefined): string | null {
  if (!ref) return null;
  const m = /^\s*#?L?(\d+)\s*[-/_]?\s*N?(\d+)\s*$/i.exec(ref);
  return m ? `${Number(m[1])}_${Number(m[2])}` : null;
}

// ---------- Catalog ----------

/** A model has something to show once its bundle has been published. */
export const isModelAvailable = (m: ModelEntry) => !!m.data_path;

export function modelStatus(m: ModelEntry): string {
  if (!m.data_path) return 'pending';
  return m.status === 'ready' ? 'complete' : 'partial';
}

export function dataUrl(model: ModelEntry, name: string): string {
  const root = (model.data_path ?? 'data').replace(/\/$/, '');
  const gz = model.compression === 'gzip' ? '.gz' : '';
  const version = model.data_version ? `?v=${encodeURIComponent(model.data_version)}` : '';
  return `${DATA_BASE_URL}/${root}/${name}.json${gz}${version}`;
}

export async function fetchCatalog(): Promise<ModelCatalog> {
  const res = await fetch(`${DATA_BASE_URL}/data/models.json`, { cache: 'no-store' });
  if (!res.ok) throw new Error(`The model catalog could not be loaded (HTTP ${res.status})`);
  return (await res.json()) as ModelCatalog;
}

export async function fetchModelJson<T>(model: ModelEntry, name: string): Promise<T> {
  const res = await fetch(dataUrl(model, name));
  if (!res.ok) throw new Error(`${name} for ${model.label} could not be loaded (HTTP ${res.status})`);
  if (model.compression !== 'gzip') return (await res.json()) as T;
  const buf = await res.arrayBuffer();
  const bytes = new Uint8Array(buf);
  // The host serves .gz files as they are, but a proxy may already have inflated one: check the magic bytes.
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
    if (typeof DecompressionStream === 'undefined') {
      throw new Error('This browser cannot decompress gzip data');
    }
    const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'));
    return JSON.parse(await new Response(stream).text()) as T;
  }
  return JSON.parse(new TextDecoder().decode(bytes)) as T;
}

// ---------- Selection ----------

export function groupKey(n: NeuronMeta, grouping: string): string {
  switch (grouping) {
    case 'category':
      return n.category;
    case 'cluster':
      return String(n.cluster);
    case 'rg':
      return n.rg;
    case 'diff':
      return n.diff;
    case 'dominant':
      return n.dominant;
    default:
      return String(n.layer);
  }
}

const SEARCH_FIELDS = ['label', 'core', 'rg_summary', 'diff_summary', 'core_summary'];

/** Whether a neuron passes the current group and search filters. */
export function matches(n: NeuronMeta, grouping: string, group: string | null, query: string): boolean {
  if (group != null && groupKey(n, grouping) !== group) return false;
  const q = query.trim().toLowerCase();
  if (!q) return true;
  if (`l${n.layer}/n${n.neuron}`.includes(q.replace(/\s/g, ''))) return true;
  if (neuronRef(n).toLowerCase().includes(q.replace(/\s/g, ''))) return true;
  return SEARCH_FIELDS.some(
    (f) =>
      String(n[f] ?? '')
        .toLowerCase()
        .includes(q) || tr(n, f).toLowerCase().includes(q),
  );
}

// ---------- Families and the lineage tree ----------

export interface Families {
  /** Each lineage's root: the lineage reached by following branch parents to the top. */
  rootOf: Record<string, string>;
  /** Roots sorted by their family's total activation share over training, largest first. */
  roots: string[];
  /** Palette slot per root, or -1 for families beyond the palette ("other"). */
  slot: Record<string, number>;
}

export function families(n: NeuronMeta): Families {
  const lineages = n.lineages ?? [];
  const byId = Object.fromEntries(lineages.map((g) => [g.id, g]));
  const rootOf: Record<string, string> = {};
  const total: Record<string, number> = {};
  for (const g0 of lineages) {
    if (!g0.birth) continue;
    let g = g0;
    const seen = new Set<string>();
    while (g.birth === 'branch' && g.birth_parent && byId[g.birth_parent] && !seen.has(g.id)) {
      seen.add(g.id);
      g = byId[g.birth_parent];
    }
    rootOf[g0.id] = g.id;
  }
  for (const sh of n.lineage_shares ?? []) {
    for (const id in sh) {
      if (rootOf[id]) total[rootOf[id]] = (total[rootOf[id]] || 0) + sh[id];
    }
  }
  const roots = [...new Set(Object.values(rootOf))].sort((a, b) => (total[b] || 0) - (total[a] || 0));
  const slot = Object.fromEntries(roots.map((r, i) => [r, i < FAMILY_COLORS.length ? i : -1]));
  return { rootOf, roots, slot };
}

export const familySlot = (fam: Families, id: string) => fam.slot[fam.rootOf[id]] ?? -1;

/** When a family is focused, the bar-chart series it corresponds to (small families are pooled in "other"). */
export function focusKey(fam: Families, famFocus: string | null): string | null {
  if (!famFocus) return null;
  return fam.slot[famFocus] >= 0 ? famFocus : '_other';
}

export function lineageColor(fam: Families, g: Pick<Lineage, 'id' | 'birth'>, colorBy: ColorBy): string {
  if (colorBy !== 'family') return MECHANISM_COLOR[g.birth ?? ''] || SERIES_OTHER;
  const i = familySlot(fam, g.id);
  return i >= 0 ? FAMILY_COLORS[i] : SERIES_OTHER;
}

/** Which arrowhead marker a lineage's entry uses; markers are defined once per color. */
export function arrowKey(fam: Families, g: Pick<Lineage, 'id' | 'birth'>, colorBy: ColorBy): string {
  if (colorBy !== 'family') return g.birth ?? 'coexist';
  const i = familySlot(fam, g.id);
  return i >= 0 ? `f${i}` : 'fo';
}

export interface BarSeries {
  key: string;
  label: string;
  color: string;
  vals: number[];
}

/**
 * The stacked series of the share chart, legend and table: one per mechanism, or, by family, one per
 * palette-sized family plus "other" and "unexplained".
 */
export function barSeries(n: NeuronMeta, fam: Families, colorBy: ColorBy, steps: number): BarSeries[] {
  const zeros = () => new Array<number>(steps).fill(0);
  if (colorBy !== 'family') {
    return MECHANISMS.map((m) => ({
      key: m.key,
      label: label('mech', m.key),
      color: m.color,
      vals: n.mech?.[m.key] ?? zeros(),
    }));
  }
  const lineages = n.lineages ?? [];
  const shares = n.lineage_shares ?? [];
  const idx = (id: string) => lineages.findIndex((g) => g.id === id);
  const top = fam.roots.filter((r) => fam.slot[r] >= 0);
  const vals = (key: string) =>
    shares.map((sh) => {
      let v = 0;
      for (const id in sh) {
        const r = fam.rootOf[id];
        if (r && (key === '_other' ? fam.slot[r] < 0 : r === key)) v += sh[id];
      }
      return v;
    });
  return top
    .map((r) => ({
      key: r,
      label: trItem(n, 'lineages', idx(r), 'name'),
      color: FAMILY_COLORS[fam.slot[r]],
      vals: vals(r),
    }))
    .concat([
      { key: '_other', label: 'other, smaller families', color: SERIES_OTHER, vals: vals('_other') },
      {
        key: 'unexplained',
        label: label('mech', 'unexplained'),
        color: DOT_MUTED,
        vals: n.mech?.unexplained ?? zeros(),
      },
    ])
    .filter((x) => x.vals.some((v) => v > 0));
}

export interface TreeItem extends Lineage {
  /** Index into `n.lineages`. */
  i: number;
  /** Activation share per step. */
  pres: number[];
  first: number;
  last: number;
}

export interface Lane {
  items: TreeItem[];
  parentLane: number | null;
  children: number[];
}

export interface TreeLayout {
  lanes: Lane[];
  /** Lane indices in drawing order: each lane followed by its branches, depth first. */
  rows: number[];
  rowOf: Record<number, number>;
  laneOf: Record<string, number>;
  items: TreeItem[];
  sporadic: TreeItem[];
  k: number;
}

/**
 * Lays out the lineage tree. Branches get a lane under their parent's; a replacement continues on the
 * lane of the lineage it replaced; an unrelated arrival gets a lane of its own; one-off (sporadic)
 * lineages are gathered on a single bottom line.
 */
export function treeLayout(n: NeuronMeta, k: number): TreeLayout {
  const shares = n.lineage_shares ?? [];
  const items: TreeItem[] = (n.lineages ?? [])
    .map((g, i) => {
      const pres = shares.map((x) => x[g.id] || 0);
      const first = pres.findIndex((v) => v > 0);
      let last = -1;
      pres.forEach((v, j) => {
        if (v > 0) last = j;
      });
      return { ...g, i, pres, first, last };
    })
    .filter((g) => g.first >= 0 && g.birth);
  const sporadic = items.filter((g) => g.birth === 'sporadic');
  const byId = Object.fromEntries(items.map((g) => [g.id, g]));
  const lanes: Lane[] = [];
  const laneOf: Record<string, number> = {};
  // Place in order of first appearance, deferring anything whose parent or predecessor is not placed yet.
  let rest = items.filter((g) => g.birth !== 'sporadic').sort((a, b) => a.first - b.first);
  for (let round = 0; rest.length && round < 100; round += 1) {
    const next: TreeItem[] = [];
    for (const g of rest) {
      if (g.birth === 'replace' && g.replaces && byId[g.replaces]) {
        if (laneOf[g.replaces] == null) {
          next.push(g);
          continue;
        }
        laneOf[g.id] = laneOf[g.replaces];
        lanes[laneOf[g.id]].items.push(g);
      } else if (g.birth === 'branch' && g.birth_parent && byId[g.birth_parent]) {
        const pl = laneOf[g.birth_parent];
        if (pl == null) {
          next.push(g);
          continue;
        }
        laneOf[g.id] = lanes.length;
        lanes.push({ items: [g], parentLane: pl, children: [] });
        lanes[pl].children.push(laneOf[g.id]);
      } else {
        laneOf[g.id] = lanes.length;
        lanes.push({ items: [g], parentLane: null, children: [] });
      }
    }
    if (next.length === rest.length) {
      // A dependency is missing or cyclic; these get lanes of their own.
      for (const g of next) {
        laneOf[g.id] = lanes.length;
        lanes.push({ items: [g], parentLane: null, children: [] });
      }
      break;
    }
    rest = next;
  }
  const rows: number[] = [];
  const walk = (li: number) => {
    rows.push(li);
    for (const c of lanes[li].children) walk(c);
  };
  lanes.forEach((l, li) => {
    if (l.parentLane == null) walk(li);
  });
  const rowOf: Record<number, number> = {};
  rows.forEach((li, r) => {
    rowOf[li] = r;
  });
  return { lanes, rows, rowOf, laneOf, items, sporadic, k };
}

// ---------- Semantic map ----------

export function decodeSim(d: SimData): DecodedSim {
  const steps = d.steps.map((b64) => {
    const bin = atob(b64);
    const a = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i += 1) a[i] = bin.charCodeAt(i);
    return a;
  });
  const N = d.order.length;
  let qmin = 255;
  for (const a of steps) {
    for (let r = 0; r < N; r += 1) {
      for (let c = 0; c < N; c += 1) {
        if (r !== c && a[r * N + c] < qmin) qmin = a[r * N + c];
      }
    }
  }
  const dmax = 1 - (d.lo + (qmin / 255) * (d.hi - d.lo));
  return {
    lo: d.lo,
    hi: d.hi,
    N,
    dmax: dmax > 0 ? dmax : 1,
    idx: Object.fromEntries(d.order.map((id, i) => [id, i])),
    steps,
  };
}

/**
 * Where each neuron sits on the map at step j, in [0, 1]². Without an anchor these are the stored
 * positions. With one, the anchor sits at the center and every other neuron's distance from it is its
 * explanation distance to the anchor, in the direction it had on the plain map.
 */
export function mapTargets(
  neurons: NeuronMeta[],
  j: number,
  anchorId: string | null,
  sim: DecodedSim | null,
): Record<string, [number, number]> {
  const out: Record<string, [number, number]> = {};
  const placed = neurons.filter((n) => n.sem && n.sem[j]);
  const anchor = anchorId && sim ? placed.find((n) => n.id === anchorId) : undefined;
  if (!anchor || !sim || sim.idx[anchor.id] == null || !sim.steps[j]) {
    for (const n of placed) out[n.id] = n.sem![j];
    return out;
  }
  const row = sim.idx[anchor.id] * sim.N;
  const mat = sim.steps[j];
  const distance = (n: NeuronMeta) => 1 - (sim.lo + (mat[row + sim.idx[n.id]] / 255) * (sim.hi - sim.lo));
  const [ax, ay] = anchor.sem![j];
  for (const n of placed) {
    if (n.id === anchor.id) {
      out[n.id] = [0.5, 0.5];
      continue;
    }
    if (sim.idx[n.id] == null) {
      out[n.id] = n.sem![j];
      continue;
    }
    const [px, py] = n.sem![j];
    let angle = Math.atan2(py - ay, px - ax);
    if (!Number.isFinite(angle) || (px === ax && py === ay)) angle = (sim.idx[n.id] * 2.399963) % (2 * Math.PI);
    const r = (0.47 * distance(n)) / sim.dmax;
    out[n.id] = [0.5 + r * Math.cos(angle), 0.5 + r * Math.sin(angle)];
  }
  return out;
}
