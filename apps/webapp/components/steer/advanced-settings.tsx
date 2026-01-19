import { Button } from '@/components/shadcn/button';
import {
  STEER_FREQUENCY_PENALTY,
  STEER_FREQUENCY_PENALTY_MAX,
  STEER_FREQUENCY_PENALTY_MIN,
  STEER_N_COMPLETION_TOKENS,
  STEER_N_COMPLETION_TOKENS_MAX,
  STEER_N_COMPLETION_TOKENS_MAX_LARGE_LLM,
  STEER_N_COMPLETION_TOKENS_MAX_THINKING,
  STEER_SEED,
  STEER_SPECIAL_TOKENS,
  STEER_STRENGTH_MULTIPLIER,
  STEER_STRENGTH_MULTIPLIER_MAX,
  STEER_TEMPERATURE,
  STEER_TEMPERATURE_MAX,
} from '@/lib/utils/steer';
import * as Popover from '@radix-ui/react-popover';
import * as Select from '@radix-ui/react-select';
import { ChevronDown, HelpCircle } from 'lucide-react';
import { NPSteerMethod } from 'neuronpedia-inference-client';

export const STEER_METHODS_ALLOWED = [
  NPSteerMethod.SimpleAdditive,
  NPSteerMethod.OrthogonalDecomp,
  NPSteerMethod.ProjectionCap,
];

export default function SteerAdvancedSettings({
  isNnSightModel = false,
  thinking,
  inCompletionMode,
  steerTokens,
  setSteerTokens,
  temperature,
  setTemperature,
  freqPenalty,
  setFreqPenalty,
  strMultiple,
  setStrMultiple,
  seed,
  setSeed,
  randomSeed,
  setRandomSeed,
  steerSpecialTokens,
  setSteerSpecialTokens,
  steerMethod,
  setSteerMethod,
  isPersonaSpace = false,
}: {
  isNnSightModel: boolean;
  thinking: boolean;
  inCompletionMode: boolean;
  steerTokens: number;
  setSteerTokens: (tokens: number) => void;
  temperature: number;
  setTemperature: (temperature: number) => void;
  freqPenalty: number;
  setFreqPenalty: (freqPenalty: number) => void;
  strMultiple: number;
  setStrMultiple: (strMultiple: number) => void;
  seed: number;
  setSeed: (seed: number) => void;
  randomSeed: boolean;
  setRandomSeed: (randomSeed: boolean) => void;
  steerSpecialTokens: boolean;
  setSteerSpecialTokens: (steerSpecialTokens: boolean) => void;
  steerMethod: NPSteerMethod;
  setSteerMethod: (steerMethod: NPSteerMethod) => void;
  isPersonaSpace?: boolean;
}) {
  return (
    <div className="flex w-full flex-col pb-5 pt-3 sm:pt-0">
      <div className="mb-1 mt-2 text-left text-[10px] font-medium uppercase text-slate-500">Advanced Settings</div>
      <div className="grid w-full grid-cols-2 items-center justify-center gap-x-1 gap-y-1.5">
        <div className="flex w-full flex-row items-center justify-start gap-x-3">
          <div className="w-[70px] text-right text-[10px] font-medium uppercase leading-tight text-slate-400">
            Tokens
          </div>
          <input
            type="number"
            onChange={(e) => {
              if (
                parseInt(e.target.value, 10) >
                (thinking
                  ? STEER_N_COMPLETION_TOKENS_MAX_THINKING
                  : isNnSightModel
                    ? STEER_N_COMPLETION_TOKENS_MAX_LARGE_LLM
                    : STEER_N_COMPLETION_TOKENS_MAX)
              ) {
                alert(
                  `Due to compute constraints, the current allowed max tokens is: ${
                    thinking
                      ? STEER_N_COMPLETION_TOKENS_MAX_THINKING
                      : isNnSightModel
                        ? STEER_N_COMPLETION_TOKENS_MAX_LARGE_LLM
                        : STEER_N_COMPLETION_TOKENS_MAX
                  }`,
                );
              } else {
                setSteerTokens(parseInt(e.target.value, 10));
              }
            }}
            className="max-w-[80px] flex-1 rounded-md border-slate-300 py-1 text-center text-xs text-slate-700"
            value={steerTokens}
          />
        </div>
        <div className="flex w-full flex-row items-center justify-start gap-x-3">
          <div className="w-[70px] text-right text-[10px] font-medium uppercase leading-tight text-slate-400">Temp</div>
          <input
            type="number"
            onChange={(e) => {
              if (parseFloat(e.target.value) > STEER_TEMPERATURE_MAX || parseFloat(e.target.value) < 0) {
                alert(`Temperature must be >= 0 and <= ${STEER_TEMPERATURE_MAX}`);
              } else {
                setTemperature(parseFloat(e.target.value));
              }
            }}
            className="max-w-[80px] flex-1 rounded-md border-slate-300 py-1 text-center text-xs text-slate-700"
            value={temperature}
          />
        </div>
        <div className="flex w-full flex-row items-center justify-start gap-x-3">
          <div className="w-[70px] text-right text-[10px] font-medium uppercase leading-tight text-slate-400">
            Freq Penalty
          </div>
          <input
            type="number"
            onChange={(e) => {
              if (
                parseFloat(e.target.value) > STEER_FREQUENCY_PENALTY_MAX ||
                parseFloat(e.target.value) < STEER_FREQUENCY_PENALTY_MIN
              ) {
                alert(`Freq penalty must be >= ${STEER_FREQUENCY_PENALTY_MIN} and <= ${STEER_FREQUENCY_PENALTY_MAX}`);
              } else {
                setFreqPenalty(parseFloat(e.target.value));
              }
            }}
            className="max-w-[80px] flex-1 rounded-md border-slate-300 py-1 text-center text-xs text-slate-700 disabled:bg-slate-200 disabled:text-slate-400"
            value={freqPenalty}
            disabled={isNnSightModel}
          />
        </div>
        <div className="flex w-full flex-row items-center justify-start gap-x-3">
          <div className="w-[70px] text-right text-[10px] font-medium uppercase leading-tight text-slate-400">
            Strength Multiple
          </div>
          <input
            type="number"
            onChange={(e) => {
              if (parseFloat(e.target.value) < 0 || parseFloat(e.target.value) > STEER_STRENGTH_MULTIPLIER_MAX) {
                alert(`Strength multiplier must be >= 0 and <= ${STEER_STRENGTH_MULTIPLIER_MAX}`);
              } else {
                setStrMultiple(parseFloat(e.target.value));
              }
            }}
            className="max-w-[80px] flex-1 rounded-md border-slate-300 py-1 text-center text-xs text-slate-700"
            value={strMultiple}
          />
        </div>
        <div className="col-span-1 flex w-full flex-row items-center justify-start gap-x-3">
          <div className="w-[70px] text-right text-[10px] font-medium uppercase leading-tight text-slate-400">
            Manual Seed
          </div>
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
            className="max-w-[80px] flex-1 rounded-md border-slate-300 py-1 text-center text-xs text-slate-700 disabled:bg-slate-200 disabled:text-slate-400"
            value={seed}
          />
        </div>
        <div className="col-span-1 flex w-full flex-row items-center justify-start gap-x-3">
          <div className="w-[70px] text-right text-[10px] font-medium uppercase leading-tight text-slate-400">
            Random Seed
          </div>
          <input
            onChange={(e) => {
              setRandomSeed(e.target.checked);
              setSeed(STEER_SEED);
            }}
            type="checkbox"
            checked={randomSeed}
            className="h-5 w-5 cursor-pointer rounded border-slate-300 bg-slate-100 py-1 text-center text-xs text-slate-700 checked:bg-slate-500"
          />
        </div>
        <div className="col-span-1 flex w-full flex-row items-center justify-start gap-x-3">
          <Select.Root value={steerMethod.toString()} onValueChange={setSteerMethod}>
            <Select.Trigger className="flex w-[162px] items-center justify-between rounded-md border border-slate-300 bg-white px-3 py-1 text-xs text-slate-700">
              <Select.Value className="flex-1 text-center" />
              <Select.Icon>
                <ChevronDown className="h-4 w-4" />
              </Select.Icon>
            </Select.Trigger>
            <Select.Portal>
              <Select.Content className="overflow-hidden rounded-md border bg-white shadow-lg">
                <Select.Viewport>
                  {Object.entries(NPSteerMethod)
                    // @ts-ignore
                    .filter(([_, value]) => STEER_METHODS_ALLOWED.includes(value))
                    .map(([key, value]) => (
                      <Select.Item
                        key={key}
                        value={value}
                        className="relative flex cursor-pointer items-center px-3 py-2 text-xs text-slate-700 hover:bg-slate-100 focus:bg-slate-100 focus:outline-none"
                      >
                        <Select.ItemText className="flex-1 text-center">
                          {key.replace(/([A-Z])/g, ' $1').trim()}
                        </Select.ItemText>
                      </Select.Item>
                    ))}
                </Select.Viewport>
              </Select.Content>
            </Select.Portal>
          </Select.Root>
        </div>
        {!inCompletionMode && (
          <div className="col-span-1 flex w-full flex-row items-center justify-start gap-x-1.5">
            <Popover.Root>
              <Popover.Trigger asChild>
                <div className="flex flex-row items-center justify-center gap-x-1 text-center text-[10px] font-medium uppercase leading-tight text-slate-400">
                  Steer Special Tokens <HelpCircle className="h-3.5 w-3.5 cursor-pointer" />
                </div>
              </Popover.Trigger>
              <Popover.Portal>
                <Popover.Content
                  className="z-50 w-[360px] rounded-lg border bg-slate-100 p-3 px-4 shadow-lg"
                  sideOffset={-5}
                >
                  <div className="flex flex-col gap-2">
                    <div className="flex flex-col text-xs font-medium text-slate-500">
                      <div className="mt-0">
                        If checked, will steer special tokens like {`<bos> and <start_of_turn>`}.
                      </div>
                    </div>
                    <div className="flex flex-row gap-x-2">
                      <Popover.Close
                        className="flex-1 rounded-lg bg-slate-300 py-2 text-center text-xs font-bold text-slate-600 outline-none hover:bg-slate-200 hover:text-slate-700"
                        aria-label="Close"
                      >
                        Close
                      </Popover.Close>
                    </div>
                  </div>
                </Popover.Content>
              </Popover.Portal>
            </Popover.Root>

            <input
              onChange={(e) => {
                setSteerSpecialTokens(e.target.checked);
              }}
              type="checkbox"
              checked={steerSpecialTokens}
              className="h-5 w-5 cursor-pointer rounded border-slate-300 bg-slate-100 py-1 text-center text-xs text-slate-700 checked:bg-slate-500"
            />
          </div>
        )}
        <div className="col-span-1 flex w-full flex-row items-center justify-center gap-x-3">
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              setSteerTokens(STEER_N_COMPLETION_TOKENS);
              setTemperature(STEER_TEMPERATURE);
              setFreqPenalty(STEER_FREQUENCY_PENALTY);
              setStrMultiple(STEER_STRENGTH_MULTIPLIER);
              setSeed(STEER_SEED);
              setRandomSeed(false);
              setSteerSpecialTokens(STEER_SPECIAL_TOKENS);
            }}
            className="h-7 text-right text-[10px] font-medium uppercase leading-tight text-slate-500"
          >
            Reset Settings
          </Button>
        </div>
      </div>
    </div>
  );
}
