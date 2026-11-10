import { describe, expect, it } from 'vitest';
import {
  barSeries,
  dataUrl,
  decodeSim,
  families,
  mapTargets,
  matches,
  ModelEntry,
  NeuronMeta,
  parseNeuronRef,
  tr,
  treeLayout,
  trItem,
} from './shared';

/** Four steps; A arrives at step 0, B branches from A at step 1, C replaces A at step 2, D is a one-off. */
function neuron(): NeuronMeta {
  return {
    id: '3_7',
    layer: 3,
    neuron: 7,
    dominant: 'branch',
    category: 'cat',
    label: 'base label',
    rg: 'recursive_generalization',
    diff: 'denoising',
    en: { label: 'english label', lineages: [{ name: 'A (en)' }, { name: 'B (en)', how: 'from A' }] },
    lineages: [
      { id: 'A', name: 'A', birth: 'coexist', parent: null, first_step: 0 },
      { id: 'B', name: 'B', birth: 'branch', birth_parent: 'A', parent: 'A', first_step: 1 },
      { id: 'C', name: 'C', birth: 'replace', replaces: 'A', first_step: 2 },
      { id: 'D', name: 'D', birth: 'sporadic', first_step: 3 },
    ],
    lineage_shares: [{ A: 1 }, { A: 0.6, B: 0.4 }, { B: 0.5, C: 0.5 }, { B: 0.3, C: 0.5, D: 0.2 }],
    mech: {
      branch: [0, 0.4, 0.5, 0.3],
      coexist: [1, 0.6, 0, 0],
      replace: [0, 0, 0.5, 0.5],
      sporadic: [0, 0, 0, 0.2],
      unexplained: [0, 0, 0, 0],
    },
  };
}

describe('parseNeuronRef', () => {
  it('accepts the url, display and storage spellings', () => {
    expect(parseNeuronRef('L6-N201')).toBe('6_201');
    expect(parseNeuronRef('L6/N201')).toBe('6_201');
    expect(parseNeuronRef('l6n201')).toBe('6_201');
    expect(parseNeuronRef('6_201')).toBe('6_201');
    expect(parseNeuronRef('#L0-N5')).toBe('0_5');
  });
  it('rejects anything else', () => {
    expect(parseNeuronRef('')).toBeNull();
    expect(parseNeuronRef(undefined)).toBeNull();
    expect(parseNeuronRef('layer six')).toBeNull();
  });
});

describe('translations', () => {
  it('prefers the English field and falls back to the source text', () => {
    const n = neuron();
    expect(tr(n, 'label')).toBe('english label');
    expect(tr(n, 'category')).toBe('cat');
    expect(tr(n, 'missing')).toBe('');
  });
  it('reads list items the same way', () => {
    const n = neuron();
    expect(trItem(n, 'lineages', 0, 'name')).toBe('A (en)');
    expect(trItem(n, 'lineages', 1, 'how')).toBe('from A');
    expect(trItem(n, 'lineages', 2, 'name')).toBe('C');
  });
});

describe('families', () => {
  it('traces branches to their root and ranks roots by total share', () => {
    const fam = families(neuron());
    expect(fam.rootOf).toEqual({ A: 'A', B: 'A', C: 'C', D: 'D' });
    // A's family holds A (1 + 0.6) and B (0.4 + 0.5 + 0.3) = 2.8; C holds 1.0; D holds 0.2.
    expect(fam.roots).toEqual(['A', 'C', 'D']);
    expect(fam.slot).toEqual({ A: 0, C: 1, D: 2 });
  });
});

describe('barSeries', () => {
  it('by mechanism reads the stored shares', () => {
    const n = neuron();
    const s = barSeries(n, families(n), 'mech', 4);
    expect(s.map((x) => x.key)).toEqual(['branch', 'coexist', 'replace', 'sporadic', 'unexplained']);
    expect(s[0].vals).toEqual([0, 0.4, 0.5, 0.3]);
  });
  it('by family sums each root family and drops empty series', () => {
    const n = neuron();
    const s = barSeries(n, families(n), 'family', 4);
    expect(s.map((x) => x.key)).toEqual(['A', 'C', 'D']);
    expect(s[0].label).toBe('A (en)');
    expect(s[0].vals).toEqual([1, 1, 0.5, 0.3]);
    expect(s[1].vals).toEqual([0, 0, 0.5, 0.5]);
  });
});

describe('treeLayout', () => {
  it('puts a branch under its parent, a replacement on the replaced lane, and one-offs aside', () => {
    const t = treeLayout(neuron(), 4);
    expect(t.laneOf.B).not.toBe(t.laneOf.A);
    expect(t.lanes[t.laneOf.B].parentLane).toBe(t.laneOf.A);
    expect(t.laneOf.C).toBe(t.laneOf.A);
    expect(t.sporadic.map((g) => g.id)).toEqual(['D']);
    expect(t.laneOf.D).toBeUndefined();
    // Drawing order: A's lane first, then its branch B.
    expect(t.rows).toEqual([t.laneOf.A, t.laneOf.B]);
    const b = t.items.find((g) => g.id === 'B')!;
    expect([b.first, b.last]).toEqual([1, 3]);
  });
  it('gives a lineage with a missing parent its own lane instead of looping', () => {
    const n = neuron();
    n.lineages![1].birth_parent = 'nope';
    const t = treeLayout(n, 4);
    expect(t.laneOf.B).toBeDefined();
    expect(t.lanes[t.laneOf.B].parentLane).toBeNull();
  });
});

describe('semantic map', () => {
  const sims = () => {
    // Two steps, three neurons. Byte 255 = identical, 0 = distance (1 - lo).
    const order = ['0_1', '0_2', '0_3'];
    const m0 = Uint8Array.from([255, 0, 128, 0, 255, 128, 128, 128, 255]);
    const m1 = Uint8Array.from([255, 255, 0, 255, 255, 0, 0, 0, 255]);
    const b64 = (a: Uint8Array) => Buffer.from(a).toString('base64');
    return decodeSim({ lo: 0.4, hi: 1, order, steps: [b64(m0), b64(m1)] });
  };
  const neurons = (): NeuronMeta[] =>
    ['0_1', '0_2', '0_3'].map((id, i) => ({
      ...neuron(),
      id,
      neuron: i + 1,
      sem: [
        [0.1 * (i + 1), 0.5],
        [0.5, 0.1 * (i + 1)],
      ],
    }));

  it('decodes the matrices and the global distance scale', () => {
    const sim = sims();
    expect(sim.N).toBe(3);
    expect(sim.steps[0][1]).toBe(0);
    expect(sim.dmax).toBeCloseTo(0.6);
  });
  it('uses stored positions without an anchor and centers the anchor with one', () => {
    const ns = neurons();
    const plain = mapTargets(ns, 1, null, null);
    expect(plain['0_2']).toEqual([0.5, 0.2]);
    const anchored = mapTargets(ns, 0, '0_1', sims());
    expect(anchored['0_1']).toEqual([0.5, 0.5]);
    // 0_2 is at full distance from the anchor at step 0, 0_3 at half.
    const d = (p: [number, number]) => Math.hypot(p[0] - 0.5, p[1] - 0.5);
    expect(d(anchored['0_2'])).toBeCloseTo(0.47);
    expect(d(anchored['0_3'])).toBeCloseTo((0.47 * (1 - (0.4 + (128 / 255) * 0.6))) / 0.6, 6);
  });
});

describe('matches', () => {
  it('filters by group and searches ids and English text', () => {
    const n = neuron();
    expect(matches(n, 'category', 'cat', '')).toBe(true);
    expect(matches(n, 'category', 'other', '')).toBe(false);
    expect(matches(n, 'layer', null, 'L3/N7')).toBe(true);
    expect(matches(n, 'layer', null, 'l3-n7')).toBe(true);
    expect(matches(n, 'layer', null, 'english')).toBe(true);
    expect(matches(n, 'layer', null, 'nothing here')).toBe(false);
  });
});

describe('dataUrl', () => {
  it('follows the bundle layout, with gzip and a cache-busting version when declared', () => {
    const plain: ModelEntry = { id: 'pythia-160m', label: 'Pythia-160M', status: 'ready', data_path: 'data' };
    expect(dataUrl(plain, 'meta')).toBe('https://heartyhaven.github.io/pythia-neuron-explorer/data/meta.json');
    const gz: ModelEntry = {
      id: 'pythia-12b',
      label: 'Pythia-12B',
      status: 'partial',
      data_path: 'data/models/pythia-12b/',
      compression: 'gzip',
      data_version: 'abc',
    };
    expect(dataUrl(gz, 'L3')).toBe(
      'https://heartyhaven.github.io/pythia-neuron-explorer/data/models/pythia-12b/L3.json.gz?v=abc',
    );
  });
});
