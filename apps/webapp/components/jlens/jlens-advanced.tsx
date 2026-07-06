'use client';

import * as Slider from '@radix-ui/react-slider';
import { HelpCircle } from 'lucide-react';
import CustomTooltip from '../custom-tooltip';

const DEFAULT_MAX_COMPLETION_TOKENS = 256;

type Props = {
  isBusy: boolean;
  temperature: number;
  setTemperature: (t: number) => void;
  numCompletionTokens: number;
  setNumCompletionTokens: (n: number) => void;
  topN: number;
  setTopN: (n: number) => void;
  // Upper bound for the "Generated Tokens" slider. Defaults to 256 (chat);
  // the completion interface passes 128.
  maxCompletionTokens?: number;
  // Completion runs always generate; the chat interface also always generates.
  // `showCompletionTokens` lets a caller hide the slider if generation is
  // fixed for that interface.
  showCompletionTokens?: boolean;
  // Optional "show non-word tokens" toggle (sidebar list filter). Only rendered
  // when both the value and setter are provided.
  hideNonWordTokens?: boolean;
  setHideNonWordTokens?: (b: boolean) => void;
};

function LabeledSlider({
  label,
  value,
  min,
  max,
  step,
  disabled,
  onChange,
  format,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  disabled: boolean;
  onChange: (v: number) => void;
  format: (v: number) => string;
}) {
  return (
    <div className="flex flex-1 flex-col gap-y-2">
      <span className="text-[10px] font-medium uppercase text-slate-500">{label}</span>
      <Slider.Root
        value={[value]}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onValueChange={(v) => onChange(v[0])}
        className="relative flex h-5 w-full flex-1 cursor-pointer items-center data-[disabled]:cursor-not-allowed data-[disabled]:opacity-50"
      >
        <Slider.Track className="relative h-[6px] grow rounded-full border border-slate-300 bg-white">
          <Slider.Range className="absolute h-full rounded-full bg-sky-600" />
        </Slider.Track>
        <Slider.Thumb className="flex h-[18px] w-9 items-center justify-center rounded-full border border-sky-600 bg-white text-[9px] font-medium text-sky-700 shadow hover:bg-sky-50 focus:outline-none">
          {format(value)}
        </Slider.Thumb>
      </Slider.Root>
    </div>
  );
}

export default function JlensAdvanced({
  isBusy,
  temperature,
  setTemperature,
  numCompletionTokens,
  setNumCompletionTokens,
  topN,
  setTopN,
  maxCompletionTokens = DEFAULT_MAX_COMPLETION_TOKENS,
  showCompletionTokens = true,
  hideNonWordTokens,
  setHideNonWordTokens,
}: Props) {
  const showNonWordToggle = hideNonWordTokens !== undefined && setHideNonWordTokens !== undefined;
  return (
    <div className="flex w-full flex-col gap-y-3 rounded-lg border border-slate-200 bg-slate-50 px-2 py-1.5 pb-5 sm:mt-2 sm:px-3 sm:py-3">
      <div
        className="grid w-full grid-cols-1 gap-x-5 gap-y-3 sm:flex sm:grid-cols-none sm:flex-row sm:flex-wrap"
        style={{
          gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
        }}
      >
        <LabeledSlider
          label="Temperature"
          value={temperature}
          min={0}
          max={2}
          step={0.1}
          disabled={isBusy}
          onChange={(v) => setTemperature(Math.round(v * 10) / 10)}
          format={(v) => v.toFixed(1)}
        />
        {showCompletionTokens && (
          <LabeledSlider
            label="Generated Tokens"
            value={numCompletionTokens}
            min={0}
            max={maxCompletionTokens}
            step={1}
            disabled={isBusy}
            onChange={setNumCompletionTokens}
            format={(v) => String(v)}
          />
        )}
        <LabeledSlider
          label="Readouts Per Layer+Pos"
          value={topN}
          min={1}
          max={8}
          step={1}
          disabled={isBusy}
          onChange={setTopN}
          format={(v) => String(v)}
        />
        {showNonWordToggle && (
          <div className="flex flex-1 flex-col gap-y-2">
            <div className="mt-1.5 flex h-5 items-center gap-x-1 text-[10px] font-medium uppercase text-slate-500">
              <label className="flex cursor-pointer items-center justify-center gap-x-1 leading-none">
                <input
                  type="checkbox"
                  checked={!hideNonWordTokens}
                  onChange={(e) => setHideNonWordTokens?.(!e.target.checked)}
                  className="h-3.5 w-3.5 cursor-pointer rounded border-slate-400 text-sky-600 focus:ring-sky-500"
                />
                Show Non-Words
              </label>
              <CustomTooltip
                minMargin
                trigger={
                  <button
                    type="button"
                    aria-label="About non-word token filtering"
                    className="mt-1 text-slate-400 hover:text-slate-600"
                  >
                    <HelpCircle className="h-3 w-3" />
                  </button>
                }
              >
                When off, punctuation, whitespace, and symbol tokens are hidden from the top predictions so word tokens
                surface. The model&apos;s actual output token (the final-layer top prediction) is always shown, even
                when it&apos;s a non-word token.
              </CustomTooltip>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
