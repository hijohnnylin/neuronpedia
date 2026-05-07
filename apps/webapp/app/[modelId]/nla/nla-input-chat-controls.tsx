'use client';

import * as ToggleGroup from '@radix-ui/react-toggle-group';
import { Settings, Trash2 } from 'lucide-react';
import { Dispatch, SetStateAction } from 'react';

type TopLevelMode = 'auto' | 'manual';

type Props = {
  isBusy: boolean;
  hasMessages: boolean;
  hasTokens: boolean;
  onClearChat: () => void;
  showAdvanced: boolean;
  setShowAdvanced: Dispatch<SetStateAction<boolean>>;
  topLevelMode: TopLevelMode;
  onTopLevelModeChange: (mode: TopLevelMode) => void;
  showChatTokens: boolean;
  setShowChatTokens: (show: boolean) => void;
};

export default function NLAInputChatControls({
  isBusy,
  hasMessages,
  hasTokens,
  onClearChat,
  showAdvanced,
  setShowAdvanced,
  topLevelMode,
  onTopLevelModeChange,
  showChatTokens,
  setShowChatTokens,
}: Props) {
  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-2">
      <button
        type="button"
        onClick={onClearChat}
        disabled={isBusy || (!hasMessages && !hasTokens)}
        title="Clear chat"
        className="flex h-8 w-8 items-center justify-center rounded-md border-rose-200 bg-rose-100 text-rose-400 transition-colors hover:border-rose-600 hover:bg-rose-200 hover:text-rose-600 disabled:cursor-not-allowed disabled:border-slate-200 disabled:bg-slate-100 disabled:text-slate-400 disabled:opacity-50 disabled:hover:border-slate-200 disabled:hover:bg-slate-100"
      >
        <Trash2 className="h-3.5 w-3.5" />
      </button>
      <button
        type="button"
        onClick={() => setShowAdvanced((v) => !v)}
        disabled={isBusy}
        title="Settings"
        className={`flex h-8 w-8 items-center justify-center rounded-md border-slate-200 bg-slate-100 transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${
          showAdvanced
            ? 'border-sky-500 bg-sky-50 text-sky-700'
            : 'border-slate-200 text-slate-400 hover:border-sky-300 hover:bg-sky-50'
        }`}
      >
        <Settings className="h-3.5 w-3.5" />
      </button>
      <div className="flex flex-col rounded-md">
        <div className="text-[7px] font-medium uppercase text-slate-400">Token Selection</div>
        <ToggleGroup.Root
          className="inline-flex overflow-hidden rounded border-slate-200 bg-slate-100 px-0 py-0 data-[disabled]:cursor-not-allowed data-[disabled]:opacity-50"
          type="single"
          value={topLevelMode}
          disabled={isBusy}
          onValueChange={(value) => {
            if (value === 'auto' || value === 'manual') onTopLevelModeChange(value);
          }}
          aria-label="Top-level chat mode"
        >
          <ToggleGroup.Item
            key="auto"
            className="items-center rounded-sm px-2 py-1 text-[10px] font-medium text-slate-400 transition-all hover:bg-slate-100 data-[state=on]:bg-slate-200 data-[state=on]:text-slate-500 sm:rounded-sm sm:px-3 sm:py-[4px] sm:text-[9px]"
            value="auto"
            aria-label="Auto"
          >
            AUTO
          </ToggleGroup.Item>
          <ToggleGroup.Item
            key="manual"
            className="items-center rounded-sm px-2 py-1 text-[10px] font-medium text-slate-400 transition-all hover:bg-slate-100 data-[state=on]:bg-slate-200 data-[state=on]:text-slate-500 sm:rounded-sm sm:px-3 sm:py-[4px] sm:text-[9px]"
            value="manual"
            aria-label="Manual"
          >
            MANUAL
          </ToggleGroup.Item>
        </ToggleGroup.Root>
      </div>
      <div className="flex flex-col rounded-md">
        <div className="text-[7px] font-medium uppercase text-slate-400">Special Tokens</div>
        <ToggleGroup.Root
          className="inline-flex overflow-hidden rounded border-slate-200 bg-slate-100 px-0 py-0 data-[disabled]:cursor-not-allowed data-[disabled]:opacity-50"
          type="single"
          value={showChatTokens ? 'show' : 'hide'}
          disabled={isBusy}
          onValueChange={(value) => {
            if (value) setShowChatTokens(value === 'show');
          }}
          aria-label="Show or hide chat-template special tokens"
        >
          <ToggleGroup.Item
            key="hide"
            className="items-center rounded-sm px-2 py-1 text-[10px] font-medium text-slate-400 transition-all hover:bg-slate-100 data-[state=on]:bg-slate-200 data-[state=on]:text-slate-500 sm:rounded-sm sm:px-3 sm:py-[4px] sm:text-[9px]"
            value="hide"
            aria-label="Hide special tokens"
          >
            HIDE
          </ToggleGroup.Item>
          <ToggleGroup.Item
            key="show"
            className="items-center rounded-sm px-2 py-1 text-[10px] font-medium text-slate-400 transition-all hover:bg-slate-100 data-[state=on]:bg-slate-200 data-[state=on]:text-slate-500 sm:rounded-sm sm:px-3 sm:py-[4px] sm:text-[9px]"
            value="show"
            aria-label="Show special tokens"
          >
            SHOW
          </ToggleGroup.Item>
        </ToggleGroup.Root>
      </div>
    </div>
  );
}
