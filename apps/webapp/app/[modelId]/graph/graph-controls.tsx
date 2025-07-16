import CustomTooltip from '@/components/custom-tooltip';
import { useGraphModalContext } from '@/components/provider/graph-modal-provider';
import { Input } from '@/components/shadcn/input';
import { Label } from '@/components/shadcn/label';
import { QuestionMarkCircledIcon } from '@radix-ui/react-icons';
import * as RadixSlider from '@radix-ui/react-slider';
import { debounce } from 'lodash';
import { useCallback, useEffect, useState } from 'react';
import { clientCheckIsEmbed, CLTGraphExtended, CltVisState, graphModelHasNpDashboards } from './utils';

export default function GraphControls({
  selectedGraph,
  visState,
  updateVisStateField,
}: {
  selectedGraph: CLTGraphExtended | null;
  visState: {
    pruningThreshold?: number;
    densityThreshold?: number;
  };
  updateVisStateField: <K extends keyof CltVisState>(key: K, value: CltVisState[K]) => void;
}) {
  // Check if we're in embed mode
  const isEmbed = clientCheckIsEmbed();
  const [localPruningThreshold, setLocalPruningThreshold] = useState(
    visState.pruningThreshold || selectedGraph?.metadata.node_threshold || 0.5,
  );
  const [localDensityThreshold, setLocalDensityThreshold] = useState(visState.densityThreshold || 0.99);
  const { openWelcomeModalToStep } = useGraphModalContext();

  // Debounced update functions
  const debouncedUpdatePruningThreshold = useCallback(
    debounce((value: number) => updateVisStateField('pruningThreshold', value), 500),
    [updateVisStateField],
  );

  const debouncedUpdateDensityThreshold = useCallback(
    debounce((value: number) => updateVisStateField('densityThreshold', value), 500),
    [updateVisStateField],
  );

  // Update local state when visState changes (e.g., from external sources)
  useEffect(() => {
    if (visState.pruningThreshold !== undefined) {
      setLocalPruningThreshold(visState.pruningThreshold);
    }
  }, [visState.pruningThreshold]);

  useEffect(() => {
    if (visState.densityThreshold !== undefined) {
      setLocalDensityThreshold(visState.densityThreshold);
    }
  }, [visState.densityThreshold]);

  // Update local state when selectedGraph changes (for initial loading)
  useEffect(() => {
    if (selectedGraph?.metadata.node_threshold !== undefined && visState.pruningThreshold === undefined) {
      setLocalPruningThreshold(selectedGraph.metadata.node_threshold);
    }
  }, [selectedGraph?.metadata.node_threshold, visState.pruningThreshold]);

  return (
    <div className="absolute -top-2 left-5 z-10 flex items-center space-x-2.5 sm:left-1">
      {!isEmbed && (
        <button
          type="button"
          onClick={() => openWelcomeModalToStep(2)}
          className="hidden h-[24px] w-[24px] items-center justify-center gap-x-1 rounded-full bg-slate-200 py-0.5 text-[12px] font-medium transition-colors hover:bg-slate-300 sm:flex"
          aria-label="Open User Guide"
        >
          ?
        </button>
      )}

      {selectedGraph?.metadata.node_threshold !== undefined && selectedGraph?.metadata.node_threshold && (
        <div className="flex h-[24px] flex-row items-center rounded bg-slate-200 px-1 py-0.5 sm:px-2">
          <Label
            htmlFor="pruningThreshold"
            className="mr-1 hidden text-center text-[9px] font-medium leading-[10px] text-slate-600 sm:block"
          >
            Show Nodes Accounting for
          </Label>
          <Input
            id="pruningThreshold"
            name="pruningThreshold"
            type="number"
            value={Math.round(localPruningThreshold * 100)
              .toString()
              .padStart(2, '0')}
            onChange={(e) => {
              const newValue = Number(e.target.value) / 100;
              setLocalPruningThreshold(newValue);
              debouncedUpdatePruningThreshold(newValue);
            }}
            className="mx-0.5 mr-2 h-[18px] w-9 rounded border-slate-300 bg-white px-2 py-0 text-left font-mono text-[10px] leading-none sm:text-[10px] md:text-[10px]"
            min={0}
            max={99}
            step={1}
          />
          <div className="-ml-[20px] mr-2.5 font-mono text-[10px] leading-none text-slate-400">%</div>
          <div className="mr-1 cursor-default select-none text-center text-[9px] font-medium leading-[10px] text-slate-600">
            <span className="hidden sm:inline">of the</span> Influence
          </div>
          <RadixSlider.Root
            name="pruningThreshold"
            value={[localPruningThreshold]}
            onValueChange={(newVal) => {
              setLocalPruningThreshold(newVal[0]);
              debouncedUpdatePruningThreshold(newVal[0]);
            }}
            min={0.2}
            max={0.99}
            step={0.01}
            className="relative flex h-4 w-14 min-w-14 flex-1 touch-none select-none items-center"
          >
            <RadixSlider.Track className="relative h-1 w-full flex-grow overflow-hidden rounded-full bg-slate-300">
              <RadixSlider.Range className="absolute h-full rounded-full bg-sky-600" />
            </RadixSlider.Track>
            <RadixSlider.Thumb className="block h-3 w-3 rounded-full border border-sky-600 bg-white shadow transition-colors focus:outline-none focus:ring-0 disabled:pointer-events-none disabled:opacity-50" />
          </RadixSlider.Root>
          <CustomTooltip
            trigger={
              <QuestionMarkCircledIcon className="ml-1.5 hidden h-3.5 w-3.5 cursor-pointer text-slate-500 sm:block" />
            }
          >
            <div className="pb-1 font-bold">How This Works</div>
            The nodes shown are always the most influential with respect to the output token. Increasing this score will
            gradually include more nodes until their cumulative contribution is 100%.
          </CustomTooltip>
        </div>
      )}
      {selectedGraph?.metadata.scan && graphModelHasNpDashboards(selectedGraph) && (
        <div className="hidden flex-row items-center rounded bg-slate-200 px-2 py-0.5 sm:flex">
          <div className="flex flex-row items-center">
            <Label htmlFor="densityThreshold" className="mr-1 text-center text-[9px] leading-[10px] text-slate-600">
              <span className="hidden sm:inline">Show Nodes with</span>
              <br className="hidden sm:inline" />
              Feature Density
            </Label>
            <div className="z-10 -mr-[16px] ml-3 font-mono text-[10px] leading-none text-slate-400">{`<`}</div>

            <Input
              id="densityThreshold"
              name="densityThreshold"
              type="number"
              value={`${(localDensityThreshold * 100).toFixed(0)}`}
              onChange={(e) => {
                const newValue = Number(e.target.value) / 100;
                setLocalDensityThreshold(newValue);
                debouncedUpdateDensityThreshold(newValue);
              }}
              className="ml-0.5 h-[18px] w-11 rounded border-slate-300 bg-white px-1 py-0 text-center font-mono text-[10px] leading-none sm:text-[10px] md:text-[10px]"
              min={0}
              max={100}
              step={1}
            />
            <div className="-ml-[11px] mr-2.5 font-mono text-[10px] leading-none text-slate-400">%</div>
            <RadixSlider.Root
              name="densityThreshold"
              value={[Math.log10(localDensityThreshold * 100 + 1)]}
              onValueChange={(newVal) => {
                const logValue = newVal[0];
                const linearValue = (10 ** logValue - 1) / 100;
                setLocalDensityThreshold(linearValue);
                debouncedUpdateDensityThreshold(linearValue);
              }}
              min={0}
              max={Math.log10(101)}
              step={0.01}
              className="relative flex h-4 w-14 min-w-14 flex-1 touch-none select-none items-center"
            >
              <RadixSlider.Track className="relative h-1 w-full flex-grow overflow-hidden rounded-full bg-slate-300">
                <RadixSlider.Range className="absolute h-full rounded-full bg-sky-600" />
              </RadixSlider.Track>
              <RadixSlider.Thumb className="block h-3 w-3 rounded-full border border-sky-600 bg-white shadow transition-colors focus:outline-none focus:ring-0 disabled:pointer-events-none disabled:opacity-50" />
            </RadixSlider.Root>
          </div>
        </div>
      )}
    </div>
  );
}
