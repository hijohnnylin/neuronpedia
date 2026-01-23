'use client';

import { ArrowUpRight, CircuitBoard, HelpCircle, Notebook, ScrollText, Smile } from 'lucide-react';
import Link from 'next/link';
import { useEffect, useRef } from 'react';
import { blogPostLink, codingTutorialLink, hfLink } from './tab-main';

export default function TabSafety({ tabUpdater }: { tabUpdater: (tab: string) => void }) {
  const featureTuples = [
    [40, 13029, 'AI safety & existential risk'],
    [53, 57326, 'AI safety & inner alignment'],
    [53, 25001, 'Meta (discussion of RLHF & model reasoning)'],
    [53, 10620, 'Emotional manipulation'],
    [53, 448, 'Power seizing / AI takeover'],
    [40, 432, 'Power seizing / AI takeover'],
    [53, 2878, 'Giving caveats / analysis after maybe dangerous responses'],
    [53, 24084, 'adopting a different persona'],
    [40, 26035, 'meta (talking about its own context)'],
    [53, 167558, 'climate change skepticism'],
    [53, 62359, 'sarcasm'],
    [40, 43644, 'sarcasm #2'],
    [31, 7282, 'irony'],
    [53, 145701, 'AI singularity'],
    [31, 23266, 'AI warfare / cyberattacks'],
    [53, 50705, 'conspiracy theories'],
  ];

  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (ref.current) {
      ref.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, []);

  return (
    <div className="mt-0 flex w-full max-w-screen-xl flex-col items-center justify-center pb-24 pt-1">
      <div ref={ref} className="pt-20 sm:pt-0" />
      <div className="mb-10 mt-5 flex w-full flex-row items-center justify-start px-2 sm:mb-3 sm:px-5 sm:pt-0">
        <div className="flex w-full flex-col items-start justify-start gap-x-4 gap-y-1.5 rounded px-2 py-1 sm:flex-row">
          <span className="w-[105px] min-w-[105px] max-w-[105px] whitespace-nowrap rounded-full bg-slate-100 px-3 py-1 text-center text-[10px] font-bold uppercase text-slate-600">
            💡 GOAL
          </span>
          <div className="text-sm font-medium leading-normal text-slate-500">
            <span>
              The following are some safety- and alignment-relevant features found in{' '}
              <Link
                href="https://huggingface.co/google/gemma-scope-2-27b-it/tree/main/resid_post"
                target="_blank"
                rel="noreferrer"
                className="font-bold text-gBlue hover:underline"
              >
                Gemma 3 27B IT
              </Link>
              .<br />
              You can review what a {`'feature'`} is in the original{' '}
              <Link href="/gemma-scope#microscope" className="font-bold text-gBlue hover:underline">
                Exploring Gemma Scope
              </Link>{' '}
              demo.
            </span>
          </div>
        </div>
      </div>

      <div className="mb-10 flex w-full flex-row items-center justify-start px-2 sm:mb-5 sm:px-5">
        <div className="flex w-full flex-col items-start justify-start gap-x-4 gap-y-1.5 rounded px-2 py-1 sm:flex-row">
          <div className="mb-2 flex w-full flex-row items-center justify-between gap-x-2 sm:mb-0 sm:w-auto sm:flex-col sm:justify-start">
            <span className="w-[105px] min-w-[105px] max-w-[105px] whitespace-nowrap rounded-full bg-slate-100 px-0 py-1 text-center text-[10px] font-bold uppercase text-slate-600">
              🎨 Examples
            </span>
          </div>
          <div className="flex w-full flex-col text-sm font-medium text-slate-500">
            You can see top positive and negative logits as well as the max activating features for each example feature
            . These features were labeled by a human annotator based on their top activations and logits. Activation
            examples are truncated by default - click an activation example to view full context, or click the top right
            feature ID to view full feature details.
            <div className="mt-5 grid grid-cols-2 gap-x-6 gap-y-6">
              {featureTuples.map((featureTuple) => (
                <iframe
                  key={`${featureTuple[0]}-${featureTuple[1]}`}
                  title={`Gemma 3 27B IT feature ${featureTuple[1]}`}
                  src={`/gemma-3-27b-it/${featureTuple[0]}-gemmascope-2-res-262k/${featureTuple[1]}?embed=true`}
                  className="col-span-2 h-[540px] w-full max-w-[540px] rounded-lg border bg-slate-50 px-2 sm:col-span-1"
                  scrolling={typeof window !== 'undefined' && window.innerWidth < 640 ? 'no' : 'yes'}
                />
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="mb-5 flex w-full flex-row items-center justify-start px-2 sm:px-5">
        <div className="flex w-full flex-col items-start justify-start gap-x-4 gap-y-1.5 rounded px-2 py-1 sm:flex-row">
          <span className="w-[105px] min-w-[105px] max-w-[105px] whitespace-nowrap rounded-full bg-slate-100 px-3 py-1 text-center text-[10px] font-bold uppercase text-slate-600">
            🎁 Next
          </span>

          <div className="flex w-full flex-col items-start justify-start gap-y-4 text-sm font-medium text-slate-500">
            <button
              type="button"
              onClick={() => {
                tabUpdater('circuit');
              }}
              className="mt-0 flex min-w-[160px] cursor-pointer flex-row justify-center gap-x-2 rounded-full border border-slate-600 bg-white px-5 py-2 text-sm font-medium text-slate-600 shadow transition-all hover:scale-105 hover:bg-slate-300"
            >
              <CircuitBoard className="h-5 w-5 text-slate-600" /> Circuit Tracing
            </button>
            <button
              type="button"
              onClick={() => {
                tabUpdater('browse');
              }}
              className="mt-0 flex min-w-[160px] cursor-pointer flex-row justify-center gap-x-2 rounded-full border border-slate-600 bg-white px-5 py-2 text-sm font-medium text-slate-600 shadow transition-all hover:scale-105 hover:bg-slate-300"
            >
              <HelpCircle className="h-5 w-5 text-slate-600" /> Dashboards + Inference
            </button>
            <a
              href={codingTutorialLink}
              target="_blank"
              rel="noreferrer"
              className="mt-0 flex min-w-[160px] cursor-pointer flex-row justify-center gap-x-2 rounded-full border border-slate-600 bg-white px-5 py-2 text-sm font-medium text-slate-600 shadow transition-all hover:scale-105 hover:bg-slate-300"
            >
              <Notebook className="h-5 w-5 text-slate-600" /> Tutorial Notebook{' '}
              <ArrowUpRight className="h-5 w-5 text-slate-600" />
            </a>
            <a
              href={hfLink}
              target="_blank"
              rel="noreferrer"
              className="mt-0 flex min-w-[160px] cursor-pointer flex-row justify-center gap-x-2 rounded-full border border-slate-600 bg-white px-5 py-2 text-sm font-medium text-slate-600 shadow transition-all hover:scale-105 hover:bg-slate-300"
            >
              <Smile className="h-5 w-5 text-slate-600" /> HuggingFace{' '}
              <ArrowUpRight className="h-5 w-5 text-slate-600" />
            </a>
            <a
              href={blogPostLink}
              target="_blank"
              rel="noreferrer"
              className="mt-0 flex min-w-[160px] cursor-pointer flex-row justify-center gap-x-2 rounded-full border border-slate-600 bg-white px-5 py-2 text-sm font-medium text-slate-600 shadow transition-all hover:scale-105 hover:bg-slate-300"
            >
              <ScrollText className="h-5 w-5 text-slate-600" /> DeepMind Blog{' '}
              <ArrowUpRight className="h-5 w-5 text-slate-600" />
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
