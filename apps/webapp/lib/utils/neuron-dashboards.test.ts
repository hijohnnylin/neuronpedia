import { gzipSync } from 'node:zlib';
import { afterEach, describe, expect, it, vi } from 'vitest';
import fixture from './__fixtures__/neuron-dashboard-qwen3-32b-31-12345.json';
import {
  applyNeuronDashboard,
  fetchNeuronDashboard,
  fetchNeuronDashboardDocText,
  fetchNeuronDashboardDocTokens,
  gunzipJson,
  layerFileStem,
  NEURON_DASHBOARD_BOTTOM_DATA_SOURCE,
  neuronByteRange,
  NeuronDashboardError,
  NeuronDashboardRecord,
  npyDataOffset,
  recordToNeuronDashboard,
} from './neuron-dashboards';

const record = fixture.record as NeuronDashboardRecord;
const vocabArray: string[] = [];
Object.entries(fixture.vocab).forEach(([id, token]) => {
  vocabArray[Number(id)] = token;
});
const ids = { modelId: 'qwen3-32b', layer: '31', index: '12345' };

function uint64Offsets(offsets: number[]) {
  const view = new DataView(new ArrayBuffer(8 * offsets.length));
  offsets.forEach((offset, i) => view.setBigUint64(8 * i, BigInt(offset), true));
  return new Uint8Array(view.buffer);
}

describe('neuronByteRange', () => {
  it('reads two little-endian uint64 offsets', () => {
    const bytes = uint64Offsets([396275988, 396309573]);
    expect(neuronByteRange(bytes.buffer)).toEqual([396275988, 396309573]);
  });

  it('refuses a short read and an empty range', () => {
    expect(() => neuronByteRange(new ArrayBuffer(8))).toThrow(NeuronDashboardError);
    expect(() => neuronByteRange(uint64Offsets([5, 5]).buffer)).toThrow(NeuronDashboardError);
  });
});

describe('layerFileStem', () => {
  it('pads to two digits like the Python writer', () => {
    expect(layerFileStem(3)).toBe('neurons/layer_03');
    expect(layerFileStem(31)).toBe('neurons/layer_31');
    expect(layerFileStem(120)).toBe('neurons/layer_120');
  });
});

describe('recordToNeuronDashboard', () => {
  const dashboard = recordToNeuronDashboard(record, vocabArray, ids);

  it('decodes tokens and puts the peak at maxValueTokenIndex', () => {
    const top = dashboard.activations[0];
    const example = record.top[0];
    expect(top.tokens?.join('')).toContain('EHEC infection');
    expect(top.maxValueTokenIndex).toBe(example.pos - example.start);
    expect(top.values?.[top.maxValueTokenIndex ?? -1]).toBeCloseTo(example.act, 1);
    expect(top.maxValue).toBe(example.act);
    expect(top.tokens?.length).toBe(top.values?.length);
  });

  it('gives top examples binMin -1 and each band its interval', () => {
    const tops = dashboard.activations.filter((a) => a.id?.includes('-top-'));
    expect(tops).toHaveLength(record.top.length);
    tops.forEach((a) => {
      expect(a.binMin).toBe(-1);
      expect(a.binMax).toBe(record.stats.max);
    });
    record.bands.forEach((band, b) => {
      const rows = dashboard.activations.filter((a) => a.id?.includes(`-band${b}-`));
      expect(rows).toHaveLength(band.examples.length);
      rows.forEach((a) => {
        expect(a.binMin).toBe(band.lo);
        expect(a.binMax).toBe(band.hi);
        expect(a.binContains).toBeCloseTo(band.q_hi - band.q_lo);
        // A row whose maxValue reaches binMax would be moved to another interval by ActivationsList.
        expect(a.maxValue).toBeLessThan(band.hi);
      });
    });
  });

  it('marks negative-peak examples as bottom and puts the bottom list below every band', () => {
    const bottoms = dashboard.activations.filter((a) => a.id?.includes('-bottom-'));
    expect(bottoms).toHaveLength(record.bottom.length);
    const lowestEdge = Math.min(...record.bands.map((b) => b.lo));
    bottoms.forEach((a) => {
      expect(a.dataSource).toBe(NEURON_DASHBOARD_BOTTOM_DATA_SOURCE);
      expect(a.binMin).toBe(record.stats.min);
      expect(a.binMax).toBe(lowestEdge);
      expect(a.binContains).toBeNull();
      expect(a.minValue).toBeLessThanOrEqual(a.maxValue ?? 0);
    });
    dashboard.activations
      .filter((a) => (a.maxValue ?? 0) < 0)
      .forEach((a) => expect(a.dataSource).toBe(NEURON_DASHBOARD_BOTTOM_DATA_SOURCE));
  });

  it('fills the Neuron stats fields', () => {
    expect(dashboard.neuron.maxActApprox).toBe(record.stats.max);
    expect(dashboard.neuron.pos_str).toEqual(record.logits.pos.map((l) => l[1]));
    expect(dashboard.neuron.neg_values).toEqual(record.logits.neg.map((l) => l[2]));
    expect(dashboard.neuron.frac_nonzero).toBeCloseTo(0.4719, 3);
  });

  it('keeps the source document and window of every row', () => {
    const example = record.top[0];
    expect(dashboard.docRefs[dashboard.activations[0].id ?? '']).toEqual({
      doc: example.doc,
      start: example.start,
      length: example.tokens.length,
    });
    expect(Object.keys(dashboard.docRefs).length).toBe(dashboard.activations.length);
  });

  it('gives every row a unique id', () => {
    const rowIds = dashboard.activations.map((a) => a.id);
    expect(new Set(rowIds).size).toBe(rowIds.length);
  });

  it('refuses an unknown format version', () => {
    expect(() => recordToNeuronDashboard({ ...record, v: 2 }, vocabArray, ids)).toThrow(NeuronDashboardError);
  });

  it('keeps database fields when applied over a neuron', () => {
    const neuron = { modelId: 'qwen3-32b', layer: '31', index: '12345', explanations: [{ id: 'e1' }] };
    const applied = applyNeuronDashboard(neuron as never, dashboard);
    expect(applied.explanations).toEqual([{ id: 'e1' }]);
    expect(applied.activations).toBe(dashboard.activations);
    expect(applied.maxActApprox).toBe(record.stats.max);
  });
});

describe('fetchNeuronDashboard', () => {
  const base = 'https://dashboards.test/v1/qwen3-32b/';
  const other = { ...record, neuron: 12344 };
  const members = [gzipSync(JSON.stringify(other)), gzipSync(JSON.stringify(record))];
  const shard = Buffer.concat(members);
  // A file whose neurons 12344 and 12345 are its two members.
  const offsets = new Array<number>(12345).fill(0);
  offsets.push(members[0].length, shard.length);
  const idx = uint64Offsets(offsets);
  const files: Record<string, Uint8Array> = {
    'https://dashboards.test/v1/qwen3-32b/neurons/layer_31.idx': idx,
    'https://dashboards.test/v1/qwen3-32b/neurons/layer_31.json.gz': shard,
    'https://dashboards.test/v1/qwen3-32b/corpus/vocab.json': new TextEncoder().encode(JSON.stringify(vocabArray)),
  };

  function serve(honorRange: boolean, overrides: Record<string, Uint8Array> = {}) {
    const ranges: string[] = [];
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      const file = overrides[url] ?? files[url];
      const body = file ? Uint8Array.from(file) : undefined;
      if (!body) {
        return new Response('not found', { status: 404 });
      }
      const range = new Headers(init?.headers).get('Range');
      if (!range || !honorRange) {
        return new Response(body, { status: 200 });
      }
      ranges.push(`${url.split('/').pop()} ${range}`);
      const [, a, b] = /bytes=(\d+)-(\d+)/.exec(range) ?? [];
      return new Response(body.slice(Number(a), Number(b) + 1), { status: 206 });
    });
    return ranges;
  }

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('reads one neuron with two Range requests', async () => {
    const ranges = serve(true);
    const dashboard = await fetchNeuronDashboard(base, ids);
    expect(dashboard.record.neuron).toBe(12345);
    expect(dashboard.activations[0].tokens?.join('')).toContain('EHEC');
    expect(ranges).toEqual([
      `layer_31.idx bytes=${8 * 12345}-${8 * 12345 + 15}`,
      `layer_31.json.gz bytes=${members[0].length}-${shard.length - 1}`,
    ]);
  });

  it('still works when the server ignores Range', async () => {
    serve(false);
    const dashboard = await fetchNeuronDashboard(base, ids);
    expect(dashboard.record.neuron).toBe(12345);
  });

  it('reads the neighbouring member for the neighbouring neuron', async () => {
    serve(true);
    const dashboard = await fetchNeuronDashboard(base, { ...ids, index: '12344' });
    expect(dashboard.record.neuron).toBe(12344);
  });

  it('refuses a record for a different neuron', async () => {
    const swapped = Buffer.concat([members[1], members[0]]);
    const swappedIdx = uint64Offsets([...offsets.slice(0, 12345), members[1].length, swapped.length]);
    serve(true, {
      'https://dashboards.test/v1/qwen3-32b/neurons/layer_31.idx': swappedIdx,
      'https://dashboards.test/v1/qwen3-32b/neurons/layer_31.json.gz': swapped,
    });
    await expect(fetchNeuronDashboard(base, ids)).rejects.toThrow(/got 31\/12344/);
  });

  it('refuses ids that are not numbers', async () => {
    serve(true);
    await expect(fetchNeuronDashboard(base, { ...ids, layer: '31-res-jb' })).rejects.toThrow(NeuronDashboardError);
  });
});

describe('gunzipJson', () => {
  it('round-trips a gzip member', async () => {
    const bytes = gzipSync(JSON.stringify({ a: [1, 2] }));
    expect(await gunzipJson(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length))).toEqual({
      a: [1, 2],
    });
  });
});

// A 1-D little-endian .npy file, laid out like numpy writes it (version 1, data aligned to 64 bytes; 128 here, as in the real files).
function npy(descr: string, data: ArrayBufferView) {
  let header = `{'descr': '${descr}', 'fortran_order': False, 'shape': (${data.byteLength / (descr === '<u4' ? 4 : 8)},), }`;
  header += ' '.repeat(64 - ((10 + header.length + 1) % 64)) + '\n';
  const out = new Uint8Array(10 + header.length + data.byteLength);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0], 0);
  new DataView(out.buffer).setUint16(8, header.length, true);
  out.set(new TextEncoder().encode(header), 10);
  out.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), 10 + header.length);
  return out;
}

describe('npyDataOffset', () => {
  it('reads the header length', () => {
    const file = npy('<u4', new Uint32Array([1, 2, 3]));
    expect(npyDataOffset(file.buffer.slice(0, 12))).toBe(128);
  });

  it('refuses a file that is not .npy', () => {
    expect(() => npyDataOffset(new Uint8Array(12).buffer)).toThrow(NeuronDashboardError);
  });
});

describe('document loaders', () => {
  const base = 'https://dashboards.test/v2/qwen3-32b';
  const texts = ['first doc', '第二个文件 — second', ''];
  const encoded = texts.map((t) => new TextEncoder().encode(t));
  const byteOffsets = [0];
  encoded.forEach((e) => byteOffsets.push(byteOffsets[byteOffsets.length - 1] + e.length));
  const files: Record<string, Uint8Array> = {
    [`${base}/corpus/tokens.npy`]: npy('<u4', new Uint32Array([5, 6, 7, 8, 9, 10])),
    [`${base}/corpus/offsets.npy`]: npy('<i8', new BigInt64Array([0n, 2n, 6n, 6n])),
    'https://dashboards.test/v2/corpus/docs.txt': Uint8Array.from(encoded.flatMap((e) => Array.from(e))),
    'https://dashboards.test/v2/corpus/docs_offsets.npy': npy('<i8', new BigInt64Array(byteOffsets.map(BigInt))),
  };
  const ranges: string[] = [];

  afterEach(() => {
    vi.unstubAllGlobals();
    ranges.length = 0;
  });

  function serve() {
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      const file = files[url];
      if (!file) {
        return new Response('not found', { status: 404 });
      }
      const range = new Headers(init?.headers).get('Range') ?? '';
      ranges.push(`${url.split('/').pop()} ${range}`);
      const [, a, b] = /bytes=(\d+)-(\d+)/.exec(range) ?? [];
      return new Response(Uint8Array.from(file).slice(Number(a), Number(b) + 1), { status: 206 });
    });
  }

  it("reads one document's tokens with small Range requests", async () => {
    serve();
    expect(await fetchNeuronDashboardDocTokens(base, 1)).toEqual([7, 8, 9, 10]);
    expect(ranges).toContain(`offsets.npy bytes=${128 + 8}-${128 + 23}`);
    expect(ranges).toContain(`tokens.npy bytes=${128 + 8}-${128 + 23}`);
  });

  it('decodes multibyte text exactly and reads it from the shared corpus', async () => {
    serve();
    expect(await fetchNeuronDashboardDocText(base, 1)).toBe(texts[1]);
    expect(await fetchNeuronDashboardDocText(base, 2)).toBe('');
  });

  it('refuses a bad document id', async () => {
    serve();
    await expect(fetchNeuronDashboardDocTokens(base, -1)).rejects.toThrow(NeuronDashboardError);
  });
});
