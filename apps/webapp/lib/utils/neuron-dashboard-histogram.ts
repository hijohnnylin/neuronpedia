// Activation histograms from a neuron dashboard record. Per sign, the record counts |act| / sigma in
// 66 bins: bin 0 is below 1e-3, bins 1..64 are log-spaced up to 1e3, and bin 65 is above.

import type { NeuronDashboardRecord } from '@/lib/utils/neuron-dashboards';

export const HIST_BINS_PER_SIGN = 66;
const LOG_BINS = 64;
const LOG_MIN = -3;
const LOG_DECADES = 6;
const MAX_TICKS_PER_SIDE = 4;

type HistRecord = Pick<NeuronDashboardRecord, 'hist' | 'stats'>;

// Magnitude range [lo, hi) of bin k, in activation units. The last bin has no upper edge.
export function histBinRange(k: number, sigma: number): [number, number] {
  const edge = (i: number) => sigma * 10 ** (LOG_MIN + (i * LOG_DECADES) / LOG_BINS);
  if (k === 0) {
    return [0, edge(0)];
  }
  if (k > LOG_BINS) {
    return [edge(LOG_BINS), Infinity];
  }
  return [edge(k - 1), edge(k)];
}

function sumBins(a: number[], b: number[]) {
  return a.map((count, k) => count + (b[k] ?? 0));
}

export type LinearHistogram = { values: number[]; heights: number[] };

// Equal-width bins from min to max, like the feature dashboard plot, with gate states summed.
// A log bin's count is spread over the linear bins it overlaps, uniform in log |act|. This keeps
// the total, but the shape inside one log bin is a guess. Heights are rounded to whole tokens.
export function linearHistogram(record: HistRecord, numBins = 50): LinearHistogram {
  const { hist } = record;
  const lo = Math.min(record.stats.min, 0);
  const hi = Math.max(record.stats.max, 0);
  if (!(hi > lo) || !(hist.sigma > 0)) {
    return { values: [], heights: [] };
  }
  const width = (hi - lo) / numBins;
  const heights = new Array<number>(numBins).fill(0);
  const binOf = (x: number) => Math.min(numBins - 1, Math.max(0, Math.floor((x - lo) / width)));

  [1, -1].forEach((sign) => {
    const limit = sign > 0 ? hi : -lo;
    const counts = sign > 0 ? sumBins(hist.open_pos, hist.closed_pos) : sumBins(hist.open_neg, hist.closed_neg);
    counts.forEach((count, k) => {
      if (!(count > 0)) {
        return;
      }
      const [binLo, rawHi] = histBinRange(k, hist.sigma);
      const binHi = Math.min(rawHi, limit);
      if (!(binHi > binLo)) {
        // max and min are rounded, so a count can sit just past them
        heights[binOf(sign * limit)] += count;
        return;
      }
      for (let j = 0; j < numBins; j += 1) {
        const a = lo + j * width;
        const b = a + width;
        // this linear bin as a magnitude range on this side of zero
        const magLo = sign > 0 ? Math.max(a, 0) : Math.max(-b, 0);
        const magHi = sign > 0 ? b : -a;
        const ovLo = Math.max(magLo, binLo);
        const ovHi = Math.min(magHi, binHi);
        if (ovHi > ovLo) {
          const frac = binLo === 0 ? (ovHi - ovLo) / (binHi - binLo) : Math.log(ovHi / ovLo) / Math.log(binHi / binLo);
          heights[j] += count * frac;
        }
      }
    });
  });

  return {
    values: heights.map((_, j) => lo + (j + 0.5) * width),
    heights: heights.map((h) => Math.round(h)),
  };
}

export type SignedLogHistogram = {
  // Bin k is at k + 0.5 on the positive side and at -(k + 0.5) on the negative side.
  x: number[];
  gateOpen: number[];
  gateClosed: number[];
  // signed activation range of each bar, for hover text
  labels: string[];
  tickvals: number[];
  ticktext: string[];
};

function formatValue(v: number) {
  const m = Math.abs(v);
  if (m !== 0 && (m >= 1e4 || m < 1e-3)) {
    return v.toExponential(0).replace('e+', 'e');
  }
  return String(Number(v.toPrecision(3)));
}

function barLabel(sign: number, k: number, sigma: number) {
  const [lo, hi] = histBinRange(k, sigma);
  if (hi === Infinity) {
    return sign > 0 ? `above ${formatValue(lo)}` : `below ${formatValue(-lo)}`;
  }
  return sign > 0 ? `${formatValue(lo)} to ${formatValue(hi)}` : `${formatValue(-hi)} to ${formatValue(-lo)}`;
}

// The exact counts, on a log scale of |act| that is mirrored for negative values. Gate states are
// kept apart so they can be stacked. Bars run from the outermost non-empty bin on each side.
export function signedLogHistogram(record: HistRecord): SignedLogHistogram {
  const { hist } = record;
  const sigma = hist.sigma;
  const bars: { x: number; gateOpen: number; gateClosed: number; label: string }[] = [];
  for (let k = HIST_BINS_PER_SIGN - 1; k >= 0; k -= 1) {
    bars.push({
      x: -(k + 0.5),
      gateOpen: hist.open_neg[k] ?? 0,
      gateClosed: hist.closed_neg[k] ?? 0,
      label: barLabel(-1, k, sigma),
    });
  }
  for (let k = 0; k < HIST_BINS_PER_SIGN; k += 1) {
    bars.push({
      x: k + 0.5,
      gateOpen: hist.open_pos[k] ?? 0,
      gateClosed: hist.closed_pos[k] ?? 0,
      label: barLabel(1, k, sigma),
    });
  }
  const filled = bars.map((bar) => bar.gateOpen + bar.gateClosed > 0);
  const first = filled.indexOf(true);
  const last = filled.lastIndexOf(true);
  if (first === -1 || !(sigma > 0)) {
    return { x: [], gateOpen: [], gateClosed: [], labels: [], tickvals: [], ticktext: [] };
  }
  const shown = bars.slice(first, last + 1);
  const xMin = shown[0].x - 0.5;
  const xMax = shown[shown.length - 1].x + 0.5;

  // Ticks at powers of ten in activation units, not in sigma units, so the labels are round.
  const xOf = (m: number) => 1 + ((Math.log10(m / sigma) - LOG_MIN) * LOG_BINS) / LOG_DECADES;
  const tickvals: number[] = [];
  const ticktext: string[] = [];
  if (xMin <= 0 && xMax >= 0) {
    tickvals.push(0);
    ticktext.push('0');
  }
  [1, -1].forEach((sign) => {
    const powers: number[] = [];
    const pLo = Math.ceil(Math.log10(sigma) + LOG_MIN);
    const pHi = Math.floor(Math.log10(sigma) + LOG_MIN + LOG_DECADES);
    for (let p = pLo; p <= pHi; p += 1) {
      const x = sign * xOf(10 ** p);
      if (x >= xMin && x <= xMax) {
        powers.push(p);
      }
    }
    const step = Math.ceil(powers.length / MAX_TICKS_PER_SIDE);
    const outer = powers[powers.length - 1];
    powers
      .filter((p) => (outer - p) % step === 0)
      .forEach((p) => {
        tickvals.push(sign * xOf(10 ** p));
        ticktext.push(formatValue(sign * 10 ** p));
      });
  });

  return {
    x: shown.map((bar) => bar.x),
    gateOpen: shown.map((bar) => bar.gateOpen),
    gateClosed: shown.map((bar) => bar.gateClosed),
    labels: shown.map((bar) => bar.label),
    tickvals,
    ticktext,
  };
}
