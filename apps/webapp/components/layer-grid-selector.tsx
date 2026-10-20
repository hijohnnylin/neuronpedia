'use client';

import { formatLayerSelection, layerLabel, selectLayerRange } from '@/lib/utils/layer-selection';
import * as Popover from '@radix-ui/react-popover';
import { ChevronDownIcon } from 'lucide-react';
import { useEffect, useState } from 'react';

function gridStyle(count: number) {
  const columns = Math.max(1, Math.min(count, count > 48 ? 16 : 12));
  return { gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` };
}

const CELL_CLASS = 'h-6 min-w-[28px] rounded-sm px-0.5 font-mono text-[10px] leading-none';
const CELL_ON = 'bg-sky-700 text-white';
const CELL_OFF = 'bg-slate-100 text-slate-500 hover:bg-sky-100 hover:text-sky-700';

// Picks layers on a grid of small numbers. Click a layer to toggle it, or drag across layers to set a range.
// An empty selection means all layers.
export default function LayerGridSelector({
  layers,
  selectedLayers,
  onChange,
}: {
  layers: string[];
  selectedLayers: string[];
  onChange: (selectedLayers: string[]) => void;
}) {
  // The layer where the drag started, whether it adds or removes, and the selection before it.
  const [drag, setDrag] = useState<{ anchor: number; add: boolean; base: string[] } | null>(null);
  const allSelected = selectedLayers.length === 0;

  useEffect(() => {
    if (!drag) {
      return undefined;
    }
    const end = () => setDrag(null);
    window.addEventListener('pointerup', end);
    window.addEventListener('pointercancel', end);
    return () => {
      window.removeEventListener('pointerup', end);
      window.removeEventListener('pointercancel', end);
    };
  }, [drag]);

  // From "all layers", the first click starts a new selection instead of removing one layer.
  function start(index: number) {
    const add = allSelected || !selectedLayers.includes(layers[index]);
    const base = allSelected ? [] : selectedLayers;
    setDrag({ anchor: index, add, base });
    onChange(selectLayerRange(layers, base, index, index, add));
  }

  return (
    <Popover.Root>
      {/* A fixed width, so the grid does not move under the pointer while the label changes. */}
      <Popover.Trigger className="flex h-10 max-h-[40px] min-h-[40px] w-[220px] flex-row items-center justify-center gap-x-1 whitespace-pre rounded border border-slate-300 bg-white px-2 font-mono text-[10px] font-medium leading-tight text-sky-700 hover:bg-slate-50 sm:pl-4 sm:pr-2 sm:text-xs">
        <span className="truncate" title={formatLayerSelection(layers, selectedLayers)}>
          {formatLayerSelection(layers, selectedLayers)}
        </span>
        <ChevronDownIcon className="-mr-1 ml-0 w-4 shrink-0 leading-none" />
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          className="z-30 flex select-none flex-col gap-y-2 rounded-md border border-slate-300 bg-white p-2 shadow"
          sideOffset={5}
        >
          <div className="flex flex-row items-center justify-between gap-x-3">
            <button
              type="button"
              onClick={() => onChange([])}
              className={`rounded px-2.5 py-1 font-mono text-[11px] font-medium ${
                allSelected
                  ? 'bg-sky-700 text-white'
                  : 'bg-slate-100 text-slate-600 hover:bg-sky-100 hover:text-sky-700'
              }`}
            >
              All Layers
            </button>
            <div className="text-[10px] text-slate-400">Click or drag to select layers</div>
          </div>
          <div className="grid touch-none gap-0.5" style={gridStyle(layers.length)}>
            {layers.map((layer, i) => {
              const selected = selectedLayers.includes(layer);
              return (
                <button
                  key={layer}
                  type="button"
                  title={layer}
                  aria-pressed={allSelected || selected}
                  onPointerDown={(e) => {
                    e.preventDefault();
                    // A touch keeps capture on the first button, so later buttons would not get pointerenter.
                    e.currentTarget.releasePointerCapture(e.pointerId);
                    start(i);
                  }}
                  onPointerEnter={() => {
                    if (drag) {
                      onChange(selectLayerRange(layers, drag.base, drag.anchor, i, drag.add));
                    }
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      start(i);
                      setDrag(null);
                    }
                  }}
                  className={`${CELL_CLASS} ${
                    selected ? CELL_ON : allSelected ? 'bg-sky-100 text-sky-700 hover:bg-sky-200' : CELL_OFF
                  }`}
                >
                  {layerLabel(layer)}
                </button>
              );
            })}
          </div>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}

// Picks one layer on a grid of small numbers.
export function LayerGridPicker({
  layers,
  layer,
  onChange,
}: {
  layers: string[];
  layer: string;
  onChange: (layer: string) => void;
}) {
  const [open, setOpen] = useState(false);

  return (
    <Popover.Root open={open} onOpenChange={setOpen}>
      <Popover.Trigger className="flex h-10 max-h-[40px] min-h-[40px] flex-row items-center justify-center gap-x-1 whitespace-pre rounded border border-slate-300 bg-white px-2 font-mono text-sm font-medium text-sky-700 hover:bg-slate-50 focus:outline-none sm:pl-5 sm:pr-2">
        <div className="flex flex-col">
          <span className="uppercase">{layerLabel(layer)}</span>
          <div className="mt-0.5 text-center text-[8px] font-medium leading-none text-slate-400">LAYER</div>
        </div>
        <ChevronDownIcon className="-mr-1 ml-0 w-4 shrink-0 leading-none" />
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          className="z-30 select-none rounded-md border border-slate-300 bg-white p-2 shadow"
          sideOffset={5}
        >
          <div className="grid gap-0.5" style={gridStyle(layers.length)}>
            {layers.map((l) => (
              <button
                key={l}
                type="button"
                title={l}
                aria-pressed={l === layer}
                onClick={() => {
                  onChange(l);
                  setOpen(false);
                }}
                className={`${CELL_CLASS} ${l === layer ? CELL_ON : CELL_OFF}`}
              >
                {layerLabel(l)}
              </button>
            ))}
          </div>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
