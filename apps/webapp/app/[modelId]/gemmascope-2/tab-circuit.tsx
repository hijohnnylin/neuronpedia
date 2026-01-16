import { useGlobalContext } from '@/components/provider/global-provider';
import { ArrowUpRight, HelpCircle, Notebook, RocketIcon, ScrollText, Smile, YoutubeIcon } from 'lucide-react';
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

      <div className="mb-5 mt-5 flex w-full flex-col items-start justify-start gap-x-4 gap-y-1.5 rounded px-3 py-1 sm:flex-row sm:px-3">
        <span className="w-[105px] min-w-[105px] max-w-[105px] whitespace-nowrap rounded-full bg-slate-100 px-3 py-1 text-center text-[10px] font-bold uppercase text-slate-600">
          🔌 Circuits
        </span>

        <div className="flex w-full flex-col items-start justify-start gap-y-2.5 text-left text-sm font-medium text-slate-500">
          <div>Gemma Scope 2 supports a new way to analyze a model{`'`}s internals: circuit tracing.</div>
          <div>
            Circuit tracing allows you to interactively{' '}
            <a
              href="https://www.anthropic.com/research/tracing-thoughts-language-model"
              target="_blank"
              rel="noreferrer"
              className="text-gBlue hover:underline"
            >
              trace a model&apos;s internal reasoning steps (Lindsey et al)
            </a>{' '}
            as it decides how to respond. Given a prompt, we can generate an{' '}
            <a
              href="https://transformer-circuits.pub/2025/attribution-graphs/methods.html"
              target="_blank"
              rel="noreferrer"
              className="text-gBlue hover:underline"
            >
              attribution graph (Ameisen et al)
            </a>{' '}
            that shows how the model's arrives at its final response. Gemma Scope 2 contains transcoders for all layers
            of all Gemma 3 sizes, allowing us to generate our own graphs to analyze any prompt, including for instruct
            models.
          </div>
          <div>
            Below, we've generated a graph to trace <code>Gemma 3 4B IT</code>
            {`'`}s internal reasoning steps when we ask it:{' '}
            <code>What is the capital of the state containing Dallas? Answer immediately.</code>
          </div>
          <div>
            You can hover and click on nodes (features) in the graph to see details, and you can build a subgraph with
            grouped nodes to {`"organize"`} them to understand what{`'`}s happening inside the model. This is a
            simplified, embedded version of the circuit tracing interface.
          </div>
          <div>
            For a full guide on circuit tracing, check out the fully featured{' '}
            <a href="/graph" target="_blank" rel="noreferrer" className="text-gBlue hover:underline">
              circuit tracing interface
            </a>{' '}
            on Neuronpedia, and watch the{' '}
            <a
              href="https://youtu.be/ruLcDtr_cGo"
              target="_blank"
              rel="noreferrer"
              className="text-gBlue hover:underline"
            >
              Youtube guided mini-series
            </a>{' '}
            on circuit tracing.
          </div>
          <div className="w-full">
            <iframe
              src="/gemma-3-4b-it/graph?slug=dallas-austin&pruningThreshold=0.55&densityThreshold=0.03&embed=true"
              className="h-[720px] w-full rounded-lg border border-slate-200"
            />
          </div>
        </div>
      </div>

      <div className="mb-5 flex w-full flex-row items-center justify-start px-2 pb-24 sm:px-1">
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
              href="https://youtu.be/ruLcDtr_cGo"
              target="_blank"
              rel="noreferrer"
              className="mt-0 flex min-w-[160px] cursor-pointer flex-row justify-center gap-x-2 rounded-full border border-slate-600 bg-white px-5 py-2 text-sm font-medium text-slate-600 shadow transition-all hover:scale-105 hover:bg-slate-300"
            >
              <YoutubeIcon className="h-5 w-5 text-slate-600" /> Youtube Demo{' '}
              <ArrowUpRight className="h-5 w-5 text-slate-600" />
            </a>
            <a
              href="https://neuronpedia.org/graph"
              target="_blank"
              rel="noreferrer"
              className="mt-0 flex min-w-[160px] cursor-pointer flex-row justify-center gap-x-2 rounded-full border border-slate-600 bg-white px-5 py-2 text-sm font-medium text-slate-600 shadow transition-all hover:scale-105 hover:bg-slate-300"
            >
              <RocketIcon className="h-5 w-5 text-slate-600" /> Circuit Tracer{' '}
              <ArrowUpRight className="h-5 w-5 text-slate-600" />
            </a>
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
