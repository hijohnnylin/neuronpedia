// Neuron dashboards stored as files, not database rows. A source set with externalDashboardsUrl serves:
//   neurons/layer_XX.idx      N + 1 byte offsets, little-endian uint64
//   neurons/layer_XX.json.gz  one gzip member per neuron, at bytes offsets[n] .. offsets[n+1]-1
//   corpus/vocab.json         decoded string for each token id
//   corpus/tokens.npy         uint32 tokens of all documents; corpus/offsets.npy gives each document's span
//   ../corpus/docs.txt        UTF-8 text of all documents, shared by the models; ../corpus/docs_offsets.npy
// The browser reads one neuron with two Range requests, and one document only when the user asks for it.
// utils/neuron-dashboards-worker serves the files.

import { linearHistogram } from '@/lib/utils/neuron-dashboard-histogram';
import { ActivationPartialWithRelations, NeuronWithPartialRelations } from '@/prisma/generated/zod';

export const NEURON_DASHBOARD_FORMAT_VERSION = 1;

// Same value as the private constant in components/activation-item.tsx, which draws these in red.
export const NEURON_DASHBOARD_BOTTOM_DATA_SOURCE = 'bottom';
export const NEURON_DASHBOARD_TOP_DATA_SOURCE = 'top';

export type NeuronDashboardExample = {
  doc: number;
  pos: number;
  act: number;
  gate: number;
  up: number;
  start: number;
  tokens: number[];
  acts: (number | null)[];
  // one char per window token, '1' means the gate is closed (gate <= 0)
  gate_closed: string;
};

export type NeuronDashboardBand = {
  q_lo: number;
  q_hi: number;
  lo: number;
  hi: number;
  examples: NeuronDashboardExample[];
};

export type NeuronDashboardLogit = [tokenId: number, token: string, value: number];

export type NeuronDashboardRecord = {
  v: number;
  model: string;
  layer: number;
  neuron: number;
  stats: {
    max: number;
    min: number;
    mean: number;
    std: number;
    sigma_calib: number;
    frac_gate_open: number;
    echo_abs_share: number;
    n_hist_tokens: number;
  };
  quantiles: { q: number[]; v: number[] };
  hist: {
    sigma: number;
    open_pos: number[];
    open_neg: number[];
    closed_pos: number[];
    closed_neg: number[];
  };
  top: NeuronDashboardExample[];
  bottom: NeuronDashboardExample[];
  echo_top: NeuronDashboardExample[];
  echo_bottom: NeuronDashboardExample[];
  bands: NeuronDashboardBand[];
  logits: { pos: NeuronDashboardLogit[]; neg: NeuronDashboardLogit[] };
};

export type NeuronDashboard = {
  // Fields of the Neuron row that the dashboard components read.
  neuron: Pick<
    NeuronWithPartialRelations,
    | 'maxActApprox'
    | 'pos_str'
    | 'pos_values'
    | 'neg_str'
    | 'neg_values'
    | 'frac_nonzero'
    | 'freq_hist_data_bar_values'
    | 'freq_hist_data_bar_heights'
  >;
  activations: ActivationPartialWithRelations[];
  // The full record, for panels the Neuron row has no fields for (gate and up, echoes, histogram).
  record: NeuronDashboardRecord;
  // The source document of each activation row, by activation id, to load the full text on request.
  docRefs: Record<string, NeuronDashboardDocRef>;
};

// Window tokens are document tokens start .. start + length - 1.
export type NeuronDashboardDocRef = { doc: number; start: number; length: number };

export class NeuronDashboardError extends Error {}

export function layerFileStem(layer: number) {
  return `neurons/layer_${String(layer).padStart(2, '0')}`;
}

// Returns [start, end) of neuron n's gzip member from the 16 bytes at 8n .. 8n+15 of the .idx file.
export function neuronByteRange(idxBytes: ArrayBuffer): [number, number] {
  if (idxBytes.byteLength !== 16) {
    throw new NeuronDashboardError(`expected 16 index bytes, got ${idxBytes.byteLength}`);
  }
  const view = new DataView(idxBytes);
  const start = Number(view.getBigUint64(0, true));
  const end = Number(view.getBigUint64(8, true));
  if (!(end > start)) {
    throw new NeuronDashboardError(`bad index range ${start}..${end}`);
  }
  return [start, end];
}

export async function gunzipJson<T>(bytes: ArrayBuffer): Promise<T> {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return (await new Response(stream).json()) as T;
}

function exampleToActivation(
  example: NeuronDashboardExample,
  vocab: string[],
  id: string,
  ids: { modelId: string; layer: string; index: string },
  bin: { binMin: number; binMax: number; binContains: number | null },
): ActivationPartialWithRelations {
  const values = example.acts.map((v) => v ?? 0);
  return {
    id,
    modelId: ids.modelId,
    layer: ids.layer,
    index: ids.index,
    tokens: example.tokens.map((t) => vocab[t] ?? `<${t}>`),
    values,
    // The peak token's value, not the window max: ActivationsList re-bins rows whose maxValue passes binMax.
    maxValue: example.act,
    maxValueTokenIndex: example.pos - example.start,
    // Window values have 3 significant digits and the peak has 4, so the peak can be the lower one.
    minValue: Math.min(...values, example.act),
    dataIndex: String(example.doc),
    dataSource: example.act < 0 ? NEURON_DASHBOARD_BOTTOM_DATA_SOURCE : NEURON_DASHBOARD_TOP_DATA_SOURCE,
    binMin: bin.binMin,
    binMax: bin.binMax,
    binContains: bin.binContains,
    dfaValues: [],
    dfaTargetIndex: null,
    dfaMaxValue: null,
    zQIndices: [],
    zKIndices: [],
    zValues: [],
    lossValues: [],
  };
}

// Maps a record to the Neuron fields and Activation rows that FeatureStats and ActivationsList read.
// dataSource puts a row in the positive or negative column. Bands carry binContains; the top and bottom
// lists do not, so their headers read "Top" and "Bottom". Top rows keep binMin -1 for the one-column view.
export function recordToNeuronDashboard(
  record: NeuronDashboardRecord,
  vocab: string[],
  ids: { modelId: string; layer: string; index: string },
): NeuronDashboard {
  if (record.v !== NEURON_DASHBOARD_FORMAT_VERSION) {
    throw new NeuronDashboardError(`unsupported dashboard format v${record.v}`);
  }
  const prefix = `${ids.modelId}-${ids.layer}-${ids.index}`;
  const activations: ActivationPartialWithRelations[] = [];
  const docRefs: Record<string, NeuronDashboardDocRef> = {};
  const add = (
    example: NeuronDashboardExample,
    id: string,
    bin: { binMin: number; binMax: number; binContains: number | null },
  ) => {
    activations.push(exampleToActivation(example, vocab, id, ids, bin));
    docRefs[id] = { doc: example.doc, start: example.start, length: example.tokens.length };
  };
  record.top.forEach((example, i) => {
    add(example, `${prefix}-top-${i}`, { binMin: -1, binMax: record.stats.max, binContains: null });
  });
  record.bands.forEach((band, b) => {
    band.examples.forEach((example, i) => {
      add(example, `${prefix}-band${b}-${i}`, {
        binMin: band.lo,
        binMax: band.hi,
        binContains: band.q_hi - band.q_lo,
      });
    });
  });
  const lowestBand = record.bands.reduce<NeuronDashboardBand | undefined>(
    (low, band) => (low === undefined || band.q_lo < low.q_lo ? band : low),
    undefined,
  );
  record.bottom.forEach((example, i) => {
    add(example, `${prefix}-bottom-${i}`, {
      binMin: record.stats.min,
      binMax: lowestBand ? lowestBand.lo : example.act,
      binContains: null,
    });
  });

  const { hist } = record;
  const total = record.stats.n_hist_tokens;
  const positive = [...hist.open_pos, ...hist.closed_pos].reduce((sum, count) => sum + count, 0);
  const linear = linearHistogram(record);

  return {
    neuron: {
      maxActApprox: record.stats.max,
      pos_str: record.logits.pos.map(([, token]) => token),
      pos_values: record.logits.pos.map(([, , value]) => value),
      neg_str: record.logits.neg.map(([, token]) => token),
      neg_values: record.logits.neg.map(([, , value]) => value),
      // FeatureStats labels this as the share of tokens with activation > 0.
      frac_nonzero: total > 0 ? positive / total : 0,
      // approximate; FeatureStats can also draw the exact counts with signedLogHistogram
      freq_hist_data_bar_values: linear.values,
      freq_hist_data_bar_heights: linear.heights,
    },
    activations,
    record,
    docRefs,
  };
}

// Puts dashboard data over a neuron. Database-only fields (explanations, comments, lists) stay.
export function applyNeuronDashboard(
  neuron: NeuronWithPartialRelations,
  dashboard: NeuronDashboard,
): NeuronWithPartialRelations {
  return { ...neuron, ...dashboard.neuron, activations: dashboard.activations };
}

function trimSlash(url: string) {
  return url.replace(/\/+$/, '');
}

async function fetchRange(url: string, start: number, endExclusive: number, signal?: AbortSignal) {
  const response = await fetch(url, { headers: { Range: `bytes=${start}-${endExclusive - 1}` }, signal });
  if (response.status === 206) {
    return response.arrayBuffer();
  }
  if (response.status === 200) {
    // The server ignored the Range header and sent the whole file.
    return (await response.arrayBuffer()).slice(start, endExclusive);
  }
  if (response.status === 416) {
    throw new NeuronDashboardError(`no dashboard at ${url} bytes ${start}-${endExclusive - 1}`);
  }
  throw new NeuronDashboardError(`${url}: HTTP ${response.status}`);
}

const vocabCache = new Map<string, Promise<string[]>>();

export function fetchNeuronDashboardVocab(baseUrl: string): Promise<string[]> {
  const url = `${trimSlash(baseUrl)}/corpus/vocab.json`;
  let vocab = vocabCache.get(url);
  if (!vocab) {
    vocab = fetch(url).then(async (response) => {
      if (!response.ok) {
        throw new NeuronDashboardError(`${url}: HTTP ${response.status}`);
      }
      return (await response.json()) as string[];
    });
    // A failed fetch is not cached, so the next feature tries again.
    vocab.catch(() => vocabCache.delete(url));
    vocabCache.set(url, vocab);
  }
  return vocab;
}

export async function fetchNeuronDashboardRecord(
  baseUrl: string,
  layer: number,
  index: number,
  signal?: AbortSignal,
): Promise<NeuronDashboardRecord> {
  if (!Number.isInteger(layer) || layer < 0 || !Number.isInteger(index) || index < 0) {
    throw new NeuronDashboardError(`bad neuron ${layer}/${index}`);
  }
  const stem = `${trimSlash(baseUrl)}/${layerFileStem(layer)}`;
  const [start, end] = neuronByteRange(await fetchRange(`${stem}.idx`, 8 * index, 8 * index + 16, signal));
  const record = await gunzipJson<NeuronDashboardRecord>(await fetchRange(`${stem}.json.gz`, start, end, signal));
  if (record.layer !== layer || record.neuron !== index) {
    throw new NeuronDashboardError(`asked for ${layer}/${index}, got ${record.layer}/${record.neuron}`);
  }
  return record;
}

export async function fetchNeuronDashboard(
  baseUrl: string,
  ids: { modelId: string; layer: string; index: string },
  signal?: AbortSignal,
): Promise<NeuronDashboard> {
  if (!/^\d+$/.test(ids.layer) || !/^\d+$/.test(ids.index)) {
    throw new NeuronDashboardError(`bad neuron ${ids.layer}/${ids.index}`);
  }
  const [record, vocab] = await Promise.all([
    fetchNeuronDashboardRecord(baseUrl, Number(ids.layer), Number(ids.index), signal),
    fetchNeuronDashboardVocab(baseUrl),
  ]);
  return recordToNeuronDashboard(record, vocab, ids);
}

// Returns the byte offset of the data in a .npy file, from the magic string and header length.
export function npyDataOffset(head: ArrayBuffer): number {
  const bytes = new Uint8Array(head);
  const magic = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];
  if (bytes.length < 12 || magic.some((b, i) => bytes[i] !== b)) {
    throw new NeuronDashboardError('not a .npy file');
  }
  const view = new DataView(head);
  return bytes[6] === 1 ? 10 + view.getUint16(8, true) : 12 + view.getUint32(8, true);
}

const npyOffsetCache = new Map<string, Promise<number>>();

function fetchNpyDataOffset(url: string): Promise<number> {
  let offset = npyOffsetCache.get(url);
  if (!offset) {
    offset = fetchRange(url, 0, 12).then(npyDataOffset);
    offset.catch(() => npyOffsetCache.delete(url));
    npyOffsetCache.set(url, offset);
  }
  return offset;
}

// Reads items from .. to - 1 of a 1-D .npy array with itemBytes bytes per item.
async function fetchNpyItems(url: string, itemBytes: number, from: number, to: number): Promise<ArrayBuffer> {
  if (to <= from) {
    return new ArrayBuffer(0);
  }
  const offset = await fetchNpyDataOffset(url);
  return fetchRange(url, offset + from * itemBytes, offset + to * itemBytes);
}

// Returns [start, end) of document doc from an int64 offsets file.
async function fetchDocSpan(offsetsUrl: string, doc: number): Promise<[number, number]> {
  if (!Number.isInteger(doc) || doc < 0) {
    throw new NeuronDashboardError(`bad document ${doc}`);
  }
  const view = new DataView(await fetchNpyItems(offsetsUrl, 8, doc, doc + 2));
  return [Number(view.getBigInt64(0, true)), Number(view.getBigInt64(8, true))];
}

// All token ids of one document, in this model's tokenizer.
export async function fetchNeuronDashboardDocTokens(baseUrl: string, doc: number): Promise<number[]> {
  const base = trimSlash(baseUrl);
  const [start, end] = await fetchDocSpan(`${base}/corpus/offsets.npy`, doc);
  return Array.from(new Uint32Array(await fetchNpyItems(`${base}/corpus/tokens.npy`, 4, start, end)));
}

// The exact text of one document, without the BOS token. The text file is shared by all models.
export async function fetchNeuronDashboardDocText(baseUrl: string, doc: number): Promise<string> {
  const corpus = trimSlash(baseUrl).replace(/\/[^/]+$/, '/corpus');
  const [start, end] = await fetchDocSpan(`${corpus}/docs_offsets.npy`, doc);
  if (end <= start) {
    return '';
  }
  return new TextDecoder('utf-8').decode(await fetchRange(`${corpus}/docs.txt`, start, end));
}
