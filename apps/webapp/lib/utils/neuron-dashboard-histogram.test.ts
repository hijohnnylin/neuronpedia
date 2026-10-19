import { describe, expect, it } from 'vitest';
import fixture from './__fixtures__/neuron-dashboard-qwen3-32b-31-12345.json';
import { HIST_BINS_PER_SIGN, histBinRange, linearHistogram, signedLogHistogram } from './neuron-dashboard-histogram';
import { NeuronDashboardRecord } from './neuron-dashboards';

const record = fixture.record as NeuronDashboardRecord;
const sum = (values: number[]) => values.reduce((total, v) => total + v, 0);

// A record with one count in one bin, on a sigma of 1.
function oneBin(side: 'open_pos' | 'open_neg' | 'closed_pos', k: number, count: number, min = -1000, max = 1000) {
  const empty = () => new Array<number>(HIST_BINS_PER_SIGN).fill(0);
  const hist = { sigma: 1, open_pos: empty(), open_neg: empty(), closed_pos: empty(), closed_neg: empty() };
  hist[side][k] = count;
  return { hist, stats: { ...record.stats, min, max } };
}

describe('histBinRange', () => {
  it('matches the Python bins: zero bin, 64 log bins over six decades, overflow bin', () => {
    expect(histBinRange(0, 2)).toEqual([0, 2e-3]);
    expect(histBinRange(1, 2)[0]).toBeCloseTo(2e-3, 12);
    expect(histBinRange(64, 2)[1]).toBeCloseTo(2e3, 9);
    expect(histBinRange(65, 2)).toEqual([histBinRange(64, 2)[1], Infinity]);
    for (let k = 1; k < HIST_BINS_PER_SIGN; k += 1) {
      expect(histBinRange(k, 2)[0]).toBe(histBinRange(k - 1, 2)[1]);
    }
  });
});

describe('linearHistogram', () => {
  it('keeps the total count of a real record, to within rounding', () => {
    const { hist } = record;
    const total = sum([...hist.open_pos, ...hist.open_neg, ...hist.closed_pos, ...hist.closed_neg]);
    const { values, heights } = linearHistogram(record, 50);
    expect(values).toHaveLength(50);
    expect(Math.abs(sum(heights) - total)).toBeLessThanOrEqual(25);
    expect(values[0]).toBeGreaterThan(record.stats.min);
    expect(values[49]).toBeLessThan(record.stats.max);
  });

  it('puts a log bin only in the linear bins it overlaps', () => {
    // bin 54 is |act| from about 93 to 115, which spans two linear bins of width 20
    const [lo, hi] = histBinRange(54, 1);
    const { values, heights } = linearHistogram(oneBin('open_pos', 54, 1000), 100);
    const width = values[1] - values[0];
    values.forEach((v, j) => {
      if (heights[j] > 0) {
        expect(v + width / 2).toBeGreaterThan(lo);
        expect(v - width / 2).toBeLessThan(hi);
      }
    });
    expect(heights.filter((h) => h > 0)).toHaveLength(2);
    expect(sum(heights)).toBe(1000);
  });

  it('mirrors negative counts below zero', () => {
    const { values, heights } = linearHistogram(oneBin('open_neg', 54, 1000), 100);
    values.forEach((v, j) => {
      if (heights[j] > 0) {
        expect(v).toBeLessThan(-10);
      }
    });
  });

  it('puts a count past a rounded max in the last bin', () => {
    const { heights } = linearHistogram(oneBin('open_pos', 60, 7, -1, 5), 10);
    expect(heights[9]).toBe(7);
  });
});

describe('signedLogHistogram', () => {
  it('keeps every count of a real record, by gate state', () => {
    const { hist } = record;
    const h = signedLogHistogram(record);
    expect(sum(h.gateOpen)).toBe(sum([...hist.open_pos, ...hist.open_neg]));
    expect(sum(h.gateClosed)).toBe(sum([...hist.closed_pos, ...hist.closed_neg]));
    expect(h.x).toEqual([...h.x].sort((a, b) => a - b));
    expect(h.labels).toHaveLength(h.x.length);
  });

  it('places bins at k + 0.5, negative bins mirrored, with ticks at powers of ten', () => {
    const h = signedLogHistogram(oneBin('closed_pos', 33, 5));
    expect(h.x).toEqual([33.5]);
    expect(h.gateClosed).toEqual([5]);
    expect(h.gateOpen).toEqual([0]);
    // |act| = 1 with sigma 1 is the edge between bins 32 and 33, at x = 33
    expect(h.tickvals).toEqual([33]);
    expect(h.ticktext).toEqual(['1']);
    expect(h.labels[0]).toBe(`1 to ${Number((10 ** (6 / 64)).toPrecision(3))}`);

    const neg = signedLogHistogram(oneBin('open_neg', 33, 5));
    expect(neg.x).toEqual([-33.5]);
    expect(neg.ticktext).toEqual(['-1']);
  });

  it('labels the zero tick and keeps at most four ticks per side', () => {
    const h = signedLogHistogram(record);
    expect(h.ticktext).toContain('0');
    expect(h.tickvals.filter((x) => x > 0).length).toBeLessThanOrEqual(4);
    expect(h.tickvals.filter((x) => x < 0).length).toBeLessThanOrEqual(4);
    expect(h.tickvals.every((x) => x >= h.x[0] - 0.5 && x <= h.x[h.x.length - 1] + 0.5)).toBe(true);
  });
});
