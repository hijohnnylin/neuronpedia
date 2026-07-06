'use client';

// Layer-range selector for the chat sidebar. Renders one slot per layer and
// lets you click a layer (selects just that one) or click-and-drag to select a
// [start, end] range. Hovering a layer highlights its slot and shows a small
// layer-number tooltip. This replaced the old dual-thumb slider: the thumbs
// were an imprecise way to pick individual layers, and the click/drag slots are
// both more accurate and cheaper to render.

import { ReactNode, useState } from 'react';

export type SliderRange = [number, number];
export type LayerWeight = { layer: number; weight: number };

export function LayerRangeSlider({
  bounds,
  value,
  onChange,
  onLayerHover,
  defaultValue,
  onReset,
  label = 'Prominence of Selected Tokens by Layer',
}: {
  bounds: SliderRange;
  value: SliderRange;
  onChange: (r: SliderRange) => void;
  // Called with the layer under the cursor (or null on leave) so the sidebar
  // list can preview the top readouts/logits for that single layer.
  onLayerHover?: (layer: number | null) => void;
  // The default range; when `value` differs from it, a Reset button appears.
  defaultValue?: SliderRange;
  onReset?: () => void;
  label?: ReactNode;
}) {
  const [min, max] = bounds;
  // One slot per layer (min..max inclusive).
  const layers = Array.from({ length: max - min + 1 }, (_, i) => min + i);
  const n = layers.length;
  const step = 100 / n;
  // Slot-based geometry so the selected band lines up exactly with the dividers
  // and hover slots (each layer L occupies the slot [(L-min)·step, +step]).
  const rangeLeft = (value[0] - min) * step;
  const rangeRight = (value[1] - min + 1) * step;

  // Layer where a click/drag started, so dragging selects a [start, end] range.
  const [dragStart, setDragStart] = useState<number | null>(null);
  // Layer currently hovered, for the highlight ring + layer-number tooltip.
  const [hoverLayer, setHoverLayer] = useState<number | null>(null);
  const layerFromClientX = (clientX: number, el: HTMLElement): number => {
    const rect = el.getBoundingClientRect();
    const ratio = rect.width > 0 ? (clientX - rect.left) / rect.width : 0;
    const idx = Math.min(n - 1, Math.max(0, Math.floor(ratio * n)));
    return min + idx;
  };
  const showReset =
    onReset != null && defaultValue != null && (value[0] !== defaultValue[0] || value[1] !== defaultValue[1]);

  return (
    <div className="flex w-full max-w-[30%] flex-1 flex-col gap-x-4 gap-y-0 border-slate-200 bg-transparent sm:w-[50%] sm:min-w-[45%] sm:max-w-[45%]">
      <div className="mb-0 mb-2 flex flex-row items-center justify-between gap-x-1 text-[10px] font-medium uppercase tracking-wide text-slate-400">
        {label}
        {showReset && (
          <button
            type="button"
            onClick={onReset}
            className="-mt-[4px] flex items-center justify-center rounded-[3px] bg-slate-300 px-1.5 py-[2.5px] text-[7px] font-bold uppercase leading-none tracking-wide text-slate-500 transition-colors hover:bg-slate-300 hover:text-slate-700"
          >
            Reset
          </button>
        )}
      </div>
      <div className="relative -mt-[2px] flex w-full items-center">
        <div
          className="relative h-[23px] w-full cursor-pointer touch-none select-none rounded border border-slate-300 bg-slate-200"
          onMouseLeave={() => {
            onLayerHover?.(null);
            setHoverLayer(null);
          }}
          onPointerDown={(e) => {
            const layer = layerFromClientX(e.clientX, e.currentTarget);
            setDragStart(layer);
            setHoverLayer(layer);
            onChange([layer, layer]);
            e.currentTarget.setPointerCapture(e.pointerId);
            e.preventDefault();
          }}
          onPointerMove={(e) => {
            const layer = layerFromClientX(e.clientX, e.currentTarget);
            if (layer !== hoverLayer) {
              setHoverLayer(layer);
              onLayerHover?.(layer);
            }
            if (dragStart != null) {
              onChange([Math.min(dragStart, layer), Math.max(dragStart, layer)]);
            }
          }}
          onPointerUp={(e) => {
            setDragStart(null);
            if (e.currentTarget.hasPointerCapture(e.pointerId)) {
              e.currentTarget.releasePointerCapture(e.pointerId);
            }
          }}
        >
          {/* Darker underlay for the out-of-range (unselected) layers. */}
          <div
            className="pointer-events-none absolute inset-y-0 left-0 rounded-l-[3px] bg-slate-100"
            style={{ width: `${rangeLeft}%` }}
          />
          <div
            className="pointer-events-none absolute inset-y-0 right-0 rounded-r-[3px] bg-slate-100"
            style={{ width: `${100 - rangeRight}%` }}
          />
          {/* Selected range. */}
          <div
            className="pointer-events-none absolute inset-y-0 rounded-[3px] bg-sky-600"
            style={{ left: `${rangeLeft}%`, width: `${rangeRight - rangeLeft}%` }}
          />
          {/* Per-layer divider lines. */}
          <div className="pointer-events-none absolute inset-0 flex flex-row">
            {layers.map((layer, i) => (
              <span key={layer} className={`h-full flex-1 ${i > 0 ? 'border-l border-slate-400/20' : ''}`} />
            ))}
          </div>
          {/* Hovered-layer highlight slot. */}
          {hoverLayer != null && (
            <span
              className="pointer-events-none absolute inset-y-0 bg-slate-500/20 ring-2 ring-slate-500 ring-offset-1"
              style={{ left: `${(hoverLayer - min) * step}%`, width: `${step}%` }}
            />
          )}
          {/* Selected min/max layer labels above the band edges. */}
          <span
            className="pointer-events-none absolute -top-2.5 -translate-x-1/2 whitespace-nowrap text-[8px] font-bold uppercase tabular-nums leading-none text-slate-400"
            style={{ left: `${(value[0] - min + 0.5) * step}%` }}
          >
            {value[0]}
          </span>
          {value[1] !== value[0] && (
            <span
              className="pointer-events-none absolute -top-2.5 -translate-x-1/2 whitespace-nowrap text-[8px] font-bold uppercase tabular-nums leading-none text-slate-400"
              style={{ left: `${(value[1] - min + 0.5) * step}%` }}
            >
              {value[1]}
            </span>
          )}
          {/* Lightweight layer-number tooltip for the hovered slot. */}
          {hoverLayer != null && (
            <div
              className="pointer-events-none absolute -top-5 z-10 -translate-x-1/2 whitespace-nowrap rounded border border-slate-400 bg-white px-1.5 py-1 text-[8px] font-bold tabular-nums leading-none text-slate-700 shadow-sm"
              style={{ left: `${(hoverLayer - min + 0.5) * step}%` }}
            >
              Layer {hoverLayer}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
