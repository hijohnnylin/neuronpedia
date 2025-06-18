'use client';

import {
  STEER_N_COMPLETION_TOKENS_MAX,
  STEER_STRENGTH_MULTIPLIER_MAX,
  STEER_TEMPERATURE_MAX,
} from '@/lib/utils/steer';

interface SteererMoreOptionsContentProps {
  strMultiple: number;
  setStrMultiple: (value: number) => void;
  steerSpecialTokens: boolean;
  setSteerSpecialTokens: (value: boolean) => void;
  steerTokens: number;
  setSteerTokens: (value: number) => void;
  temperature: number;
  setTemperature: (value: number) => void;
  freqPenalty: number;
  setFreqPenalty: (value: number) => void;
  seed: number;
  setSeed: (value: number) => void;
  randomSeed: boolean;
}

export default function SteererMoreOptionsContent({
  strMultiple,
  setStrMultiple,
  steerSpecialTokens,
  setSteerSpecialTokens,
  steerTokens,
  setSteerTokens,
  temperature,
  setTemperature,
  freqPenalty,
  setFreqPenalty,
  seed,
  setSeed,
  randomSeed,
}: SteererMoreOptionsContentProps) {
  return (
    <>
      <div className="mt-1 flex flex-col items-center rounded-xl bg-green-100 px-0 pb-0 pt-3.5">
        <div className="mb-1.5 hidden text-center text-[10px] font-bold uppercase leading-none text-green-800 sm:block">
          Steering Method
        </div>
        {/* <ToggleGroup.Root
          className="mx-2 inline-flex h-8 w-[90%] overflow-hidden rounded-full border-2 border-green-600 py-0"
          type="single"
          defaultValue={SteeringMethod.SAE}
          value={steeringMethod}
          onValueChange={(value) => {
            alert('Activation not available yet.');
            return;
            setSteeringMethod(value as SteeringMethod);
          }}
          aria-label="steering method"
        >
          <ToggleGroup.Item
            key={SteeringMethod.Activation}
            className="flex-auto items-center rounded-r-full px-1 py-1 text-[10px] font-medium text-green-600 transition-all hover:bg-green-100 data-[state=on]:bg-green-600 data-[state=on]:text-white sm:px-4 sm:text-[11px]"
            value={SteeringMethod.Activation}
            aria-label={SteeringMethod.Activation}
          >
            Activation
          </ToggleGroup.Item>
          <ToggleGroup.Item
            key={SteeringMethod.SAE}
            className="flex-auto items-center rounded-l-full px-1 py-1 text-[10px] font-medium text-green-600  transition-all hover:bg-green-100 data-[state=on]:bg-green-600 data-[state=on]:text-white sm:px-4  sm:text-[11px]"
            value={SteeringMethod.SAE}
            aria-label={SteeringMethod.SAE}
          >
            SAE
          </ToggleGroup.Item>
        </ToggleGroup.Root> */}

        <div className="mb-1.5 mt-5 hidden text-center text-[10px] font-bold uppercase leading-none text-green-800 sm:block">
          Advanced
        </div>
        <div className="flex w-full flex-col gap-y-2 rounded-xl bg-green-200 px-2 py-3">
          <div className="flex w-full flex-row items-center justify-center gap-x-3">
            <div className="w-[100px] text-center text-[12px] font-medium leading-tight text-green-800">
              Strength Multiple
            </div>
            <input
              type="number"
              onChange={(e) => {
                if (
                  parseFloat(e.target.value) < 0 ||
                  parseFloat(e.target.value) > STEER_STRENGTH_MULTIPLIER_MAX
                ) {
                  alert(`Strength multiplier must be >= 0 and <= ${STEER_STRENGTH_MULTIPLIER_MAX}`);
                } else {
                  setStrMultiple(parseFloat(e.target.value));
                }
              }}
              className="max-w-[80px] flex-1 rounded-md border-green-400 py-1 text-center text-xs text-green-800"
              value={strMultiple}
            />
          </div>
          <div className="flex w-full flex-row items-center justify-center gap-x-1.5">
            <div className="w-[150px] text-center text-[12px] font-medium leading-tight text-green-800">
              Steer Special Tokens
            </div>
            <div className="flex w-[80px] flex-row items-center justify-start py-1">
              <input
                onChange={(e) => {
                  setSteerSpecialTokens(e.target.checked);
                }}
                type="checkbox"
                checked={steerSpecialTokens}
                className="h-5 w-5 cursor-pointer rounded border-green-400 bg-green-100 py-1 text-center text-xs text-green-800 checked:bg-green-600 checked:text-white"
              />
            </div>
          </div>
        </div>
      </div>
      <div className="mt-3 flex h-full flex-col">
        <div className="flex flex-col rounded-xl bg-amber-100 px-2 pb-3 pt-3.5">
          <div className="hidden text-center text-[10px] font-bold uppercase leading-none text-amber-800 sm:block">
            Generation Settings
          </div>
          <div className="mt-1.5 flex flex-col gap-y-2 rounded-xl bg-amber-200 px-2 py-3">
            <div className="flex w-full flex-row items-center justify-center gap-x-3">
              <div className="w-[100px] text-center text-[12px] font-medium leading-tight text-amber-800">
                Num Tokens
              </div>
              <input
                type="number"
                onChange={(e) => {
                  if (parseInt(e.target.value, 10) > STEER_N_COMPLETION_TOKENS_MAX) {
                    alert(
                      `Due to compute constraints, the current allowed max tokens is: ${STEER_N_COMPLETION_TOKENS_MAX}`,
                    );
                  } else {
                    setSteerTokens(parseInt(e.target.value, 10));
                  }
                }}
                className="max-w-[80px] flex-1 rounded-md border-amber-400 py-1 text-center text-xs text-amber-800"
                value={steerTokens}
              />
            </div>
            <div className="flex w-full flex-row items-center justify-center gap-x-3">
              <div className="w-[100px] text-center text-[12px] font-medium leading-tight text-amber-800">
                Temperature
              </div>
              <input
                type="number"
                onChange={(e) => {
                  if (parseFloat(e.target.value) > STEER_TEMPERATURE_MAX || parseFloat(e.target.value) < 0) {
                    alert(`Temperature must be >= 0 and <= ${STEER_TEMPERATURE_MAX}`);
                  } else {
                    setTemperature(parseFloat(e.target.value));
                  }
                }}
                className="max-w-[80px] flex-1 rounded-md border-amber-400 py-1 text-center text-xs text-amber-800"
                value={temperature}
              />
            </div>
            <div className="flex w-full flex-row items-center justify-center gap-x-3">
              <div className="w-[100px] text-center text-[12px] font-medium leading-tight text-amber-800">
                Freq Penalty
              </div>
              <input
                type="number"
                onChange={(e) => {
                  if (parseFloat(e.target.value) > 2 || parseFloat(e.target.value) < -2) {
                    alert('Freq penalty must be >= -2 and <= 2');
                  } else {
                    setFreqPenalty(parseFloat(e.target.value));
                  }
                }}
                className="max-w-[80px] flex-1 rounded-md border-amber-400 py-1 text-center text-xs text-amber-800"
                value={freqPenalty}
              />
            </div>
            <div className="flex w-full flex-row items-center justify-center gap-x-3">
              <div className="w-[100px] text-center text-[12px] font-medium leading-tight text-amber-800">Seed</div>
              <input
                type="number"
                disabled={randomSeed}
                onChange={(e) => {
                  if (parseInt(e.target.value, 10) > 100000000 || parseInt(e.target.value, 10) < -100000000) {
                    alert('Seed must be >= -100000000 and <= 100000000');
                  } else {
                    setSeed(parseInt(e.target.value, 10));
                  }
                }}
                className="max-w-[80px] flex-1 rounded-md border-amber-400 py-1 text-center text-xs text-amber-800 disabled:bg-amber-200 disabled:text-amber-400"
                value={seed}
              />
            </div>
          </div>
        </div>
      </div>
    </>
  );
}