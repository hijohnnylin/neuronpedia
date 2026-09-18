'use client';

import WarningTooltip from '@/app/[modelId]/gemmascope/tooltip-warning';
import SteererSimple from '@/components/steer/steerer-simple';
import { Rocket } from 'lucide-react';
import { useEffect, useRef } from 'react';
import FeatureTooltip from './feature-tooltip';
import DeepDiveTooltip from './tooltip-deepdive';

export default function TabSteer({
  initialModelId,
  tabUpdater,
}: {
  initialModelId: string;
  tabUpdater: (tab: string) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (ref.current) {
      ref.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, []);

  return (
    <div className="mt-0 flex w-full max-w-screen-xl flex-col items-center justify-center pb-24 pt-1">
      <div ref={ref} className="pt-20 sm:pt-0" />
      <div className="mb-10 mt-5 flex w-full flex-col items-start justify-start gap-x-4 gap-y-1.5 rounded px-3 py-1 sm:mb-4 sm:flex-row sm:px-7">
        <span className="w-[105px] min-w-[105px] max-w-[105px] whitespace-nowrap rounded-full bg-slate-100 px-3 py-1 text-center text-[10px] font-bold uppercase text-slate-600">
          💡 Purpose
        </span>
        <div className="text-sm font-medium leading-normal text-slate-500">
          {"Let's "} put these <FeatureTooltip /> to use! {`We'll make `} Gemma give different responses by{' '}
          <strong>steering or amplifying</strong> specific features. You can think of this, roughly, as surgically
          changing the way Gemma thinks, instead of just telling Gemma what to do.
        </div>
      </div>

      <div className="flex w-full flex-col items-start justify-start gap-x-4 gap-y-1.5 rounded px-2 py-1 sm:flex-row sm:px-7">
        <div className="mb-2 flex w-full flex-row items-center justify-between gap-x-2 sm:mb-0 sm:w-auto sm:flex-col sm:justify-start">
          <span className="w-[105px] min-w-[105px] max-w-[105px] whitespace-nowrap rounded-full bg-slate-100 px-0 py-1 text-center text-[10px] font-bold uppercase text-slate-600">
            🕹️ Demo
          </span>
          <div className="flex flex-row gap-x-1.5 sm:flex-auto sm:flex-col">
            <WarningTooltip>
              <div className="font-medium text-slate-600">
                {`Steering using features is a relatively new method. We haven't settled on best practices, and often times steering can make the model become incomprehensible.`}
                <br />
                <br />
                {`Yet again - an unsolved interpretability problem that's looking for a clever solver of problems - could it be you?`}
              </div>
            </WarningTooltip>
            <DeepDiveTooltip>
              <div className="font-medium text-slate-600">
                <div className="text-center">Technical Deep Dive</div>
                <br />
                We’ve cherry picked features and selected prompts that we think will create a fun experience. However,
                the ease of model steering models with features is variable and depends on many factors.
                <br />
                <br />
                These steering features are from Gemma 2 2B IT.
                <br />
                <br />
                <strong>Settings</strong>
                <br />
                Temperature = 0.5
                <br />
                Presence Penalty = 0
                <br />
                Max Response Tokens = 48
                <br />
                Seed = 16
                <br />
                <br />
                <a
                  href="https://www.alignmentforum.org/tag/activation-engineering"
                  target="_blank"
                  rel="noreferrer"
                  className="text-gBlue"
                >
                  Click here
                </a>{' '}
                to read more about the technical details involved in steering.
              </div>
            </DeepDiveTooltip>
          </div>
        </div>
        <div className="flex w-full flex-col text-sm font-medium leading-normal text-slate-500">
          Pick a feature that you want to steer with and the steering strength, then just chat!
          <div className="mb-3 mt-0.5 hidden w-full flex-row items-center justify-start gap-x-1 py-1 pb-0.5 text-center text-[11px] text-gBlue sm:flex">
            <span className="mr-1 w-[60px] min-w-[60px] max-w-[60px] whitespace-nowrap rounded bg-gBlue px-3 py-1 text-center text-[8px] font-bold uppercase leading-none text-white">
              TRY THIS{' '}
            </span>
            <div className="flex flex-col items-start justify-start gap-y-0.5 text-left leading-none">
              Click a gray preset chat message to get started quickly.
            </div>
          </div>
          <SteererSimple initialModelId={initialModelId} cappedHeight showOptionsButton={false} />
        </div>
      </div>

      <div className="mb-5 mt-5 flex w-full flex-row items-center justify-start px-2 sm:px-5">
        <div className="flex w-full flex-col items-start justify-start gap-x-4 gap-y-1.5 rounded px-2 py-1 sm:flex-row">
          <span className="w-[105px] min-w-[105px] max-w-[105px] whitespace-nowrap rounded-full bg-slate-100 px-3 py-1 text-center text-[10px] font-bold uppercase text-slate-600">
            🎁 Next
          </span>
          <div className="flex w-full flex-col items-start justify-start text-sm font-medium text-slate-500">
            <div className="mb-3 text-sm leading-normal">
              {`Steering with features still often doesn't work reliably. But it's one of many tools that interpretability
              researchers hope will one day be useful for steering AI models toward more desirable behaviors , such as`}
              <a
                href="https://techcrunch.com/2024/01/13/anthropic-researchers-find-that-ai-models-can-be-trained-to-deceive/"
                target="_blank"
                rel="noreferrer"
                className="ml-1 text-gBlue"
              >
                being honest
              </a>
              .
            </div>
            <button
              type="button"
              onClick={() => {
                tabUpdater('learn');
              }}
              className="flex min-w-[160px] cursor-pointer flex-row gap-x-2 rounded-full border border-gGreen bg-white px-5 py-2.5 text-sm font-medium text-gGreen shadow transition-all hover:scale-105 hover:bg-gGreen/20"
            >
              <Rocket className="h-5 w-5 text-gGreen" /> Next - Do More
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
