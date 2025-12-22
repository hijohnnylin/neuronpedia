import { useGlobalContext } from '@/components/provider/global-provider';
import { ArrowUpRight, HelpCircle, Notebook, ScrollText, Smile } from 'lucide-react';
import { useSession } from 'next-auth/react';
import { useEffect, useRef } from 'react';
import { blogPostLink, codingTutorialLink, hfLink } from './tab-main';

export default function TabCircuit({ tabUpdater }: { tabUpdater: (tab: string) => void }) {
  const session = useSession();
  const { setSignInModalOpen } = useGlobalContext();
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (ref.current) {
      ref.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, []);

  return (
    <div className="relative mt-0 flex h-full w-full max-w-screen-xl flex-col items-center justify-start bg-white pb-24 pt-1">
      <div ref={ref} className="pt-20 sm:pt-0" />

      <div className="mb-5 mt-5 flex w-full flex-col items-start justify-start gap-x-4 gap-y-1.5 rounded px-3 py-1 sm:flex-row sm:px-7">
        <span className="w-[105px] min-w-[105px] max-w-[105px] whitespace-nowrap rounded-full bg-slate-100 px-3 py-1 text-center text-[10px] font-bold uppercase text-slate-600">
          🔌 Circuits
        </span>

        <div className="flex w-full flex-col items-start justify-start text-left text-sm font-medium text-slate-500">
          Circuit tracing for Gemma 3 is coming soon as part of the December 2025 rolling release. In the meantime,
          check out other Gemma Scope 2 resources below.
        </div>
      </div>

      <div className="mb-5 flex w-full flex-row items-center justify-start px-2 pb-24 sm:px-5">
        <div className="flex w-full flex-col items-start justify-start gap-x-4 gap-y-1.5 rounded px-2 py-1 sm:flex-row">
          <span className="w-[105px] min-w-[105px] max-w-[105px] whitespace-nowrap rounded-full bg-slate-100 px-3 py-1 text-center text-[10px] font-bold uppercase text-slate-600">
            🎁 Next
          </span>
          <div className="flex w-full flex-col items-start justify-start gap-y-4 text-sm font-medium text-slate-500">
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
