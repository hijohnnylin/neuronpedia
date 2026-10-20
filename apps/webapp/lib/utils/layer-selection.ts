import { getAdditionalInfoFromSource, getLayerNumAsStringFromSource } from '@/lib/utils/source';

// An empty selection means all layers.

// Sets every layer from `from` to `to` (indexes into `layers`, either order) on or off, starting from `base`.
// The result keeps the order of `layers`. Selecting nothing or everything gives [].
export function selectLayerRange(layers: string[], base: string[], from: number, to: number, add: boolean) {
  const lo = Math.min(from, to);
  const hi = Math.max(from, to);
  const selected = new Set(base);
  layers.slice(lo, hi + 1).forEach((layer) => (add ? selected.add(layer) : selected.delete(layer)));
  const result = layers.filter((layer) => selected.has(layer));
  return result.length === layers.length ? [] : result;
}

export function layerLabel(layer: string) {
  const info = getAdditionalInfoFromSource(layer);
  return `${getLayerNumAsStringFromSource(layer)}${info.length > 0 ? ` ${info}` : ''}`;
}

// A short label for a selection, with runs of neighbouring layers as ranges, e.g. "Layers 1–3, 7".
export function formatLayerSelection(layers: string[], selected: string[]) {
  if (selected.length === 0) {
    return 'All Layers';
  }
  const positions = selected
    .map((layer) => layers.indexOf(layer))
    .filter((i) => i >= 0)
    .sort((a, b) => a - b);
  const runs: string[] = [];
  let start = 0;
  positions.forEach((pos, i) => {
    if (i + 1 < positions.length && positions[i + 1] === pos + 1) {
      return;
    }
    const first = positions[start];
    runs.push(first === pos ? layerLabel(layers[pos]) : `${layerLabel(layers[first])}–${layerLabel(layers[pos])}`);
    start = i + 1;
  });
  if (runs.length === 0) {
    return 'No Layers Selected';
  }
  return `${selected.length > 1 ? 'Layers' : 'Layer'} ${runs.join(', ')}`;
}
