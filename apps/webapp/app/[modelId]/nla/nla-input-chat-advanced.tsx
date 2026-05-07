'use client';

import { MAX_COMPLETION_TOKENS } from '@/lib/nla-constants';
import * as Slider from '@radix-ui/react-slider';

type Props = {
  isBusy: boolean;
  temperature: number;
  setTemperature: (t: number) => void;
  maxNewTokens: number;
  setMaxNewTokens: (n: number) => void;
};

export default function NLAInputChatAdvanced({
  isBusy,
  temperature,
  setTemperature,
  maxNewTokens,
  setMaxNewTokens,
}: Props) {
  return (
    <div className="mt-2.5 flex w-full flex-col gap-y-3 px-2.5">
      <div className="flex w-full flex-row gap-x-5">
        <div className="flex flex-1 flex-col gap-y-0.5">
          <span className="text-[10px] font-medium uppercase text-slate-500">Temperature</span>
          <div className="flex items-center gap-x-3">
            <Slider.Root
              value={[temperature]}
              min={0}
              max={1}
              step={0.1}
              disabled={isBusy}
              onValueChange={(value) => setTemperature(Math.round(value[0] * 10) / 10)}
              className="relative flex h-5 w-full flex-1 cursor-pointer items-center data-[disabled]:cursor-not-allowed data-[disabled]:opacity-50"
            >
              <Slider.Track className="relative h-[6px] grow rounded-full border border-slate-300 bg-white">
                <Slider.Range className="absolute h-full rounded-full bg-sky-600" />
              </Slider.Track>
              <Slider.Thumb className="flex h-[18px] w-8 items-center justify-center rounded-full border border-sky-600 bg-white text-[9px] font-medium text-sky-700 shadow hover:bg-sky-50 focus:outline-none">
                {temperature.toFixed(1)}
              </Slider.Thumb>
            </Slider.Root>
          </div>
        </div>
        <div className="flex flex-1 flex-col gap-y-0">
          <span className="text-[10px] font-medium uppercase text-slate-500">Generated Tokens</span>
          <Slider.Root
            value={[maxNewTokens]}
            min={1}
            max={MAX_COMPLETION_TOKENS}
            step={1}
            disabled={isBusy}
            onValueChange={(value) => setMaxNewTokens(value[0])}
            className="relative flex h-5 w-full flex-1 cursor-pointer items-center data-[disabled]:cursor-not-allowed data-[disabled]:opacity-50"
          >
            <Slider.Track className="relative h-[6px] grow rounded-full border border-slate-300 bg-white">
              <Slider.Range className="absolute h-full rounded-full bg-sky-600" />
            </Slider.Track>
            <Slider.Thumb className="flex h-[18px] w-8 items-center justify-center rounded-full border border-sky-600 bg-white text-[9px] font-medium text-sky-700 shadow hover:bg-sky-50 focus:outline-none">
              {maxNewTokens}
            </Slider.Thumb>
          </Slider.Root>
        </div>
      </div>
    </div>
  );
}
