import Link from 'next/link';

export const blogPostLink =
  'https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/';
export const codingTutorialLink = 'https://colab.research.google.com/drive/1NhWjg7n0nhfW--CjtsOdw5A5J_-Bzn4r';
export const hfLink = 'https://huggingface.co/google/gemma-scope-2';
export const techReportLink = 'https://storage.googleapis.com/gemma-scope/gemma-scope-report.pdf';

export default function TabMain({ tabUpdater }: { tabUpdater: (tab: string) => void }) {
  return (
    <div className="relative mt-0 flex h-full w-full max-w-screen-xl flex-col items-start justify-start bg-white pb-24 pt-1 sm:items-center">
      <div className="mt-0 flex w-full flex-col gap-y-6 px-5">
        <div className="flex items-start gap-x-3 rounded-lg border border-amber-200 bg-amber-50 px-5 py-4">
          <div className="flex w-full flex-col gap-y-1 text-center text-[13px]">
            <div className="mb-1.5 text-center text-sm font-bold text-slate-800">⚠️ Rolling Release</div>
            <div>
              An initial release of artifacts and datasets are available now, with all artifacts expected to be
              finalized by <strong>January 26, 2026</strong>.
            </div>
            <div>
              Some data may be replaced or updated during this final verification and fine-tuning process. Please check
              the{' '}
              <a
                href="https://huggingface.co/google/gemma-scope-2"
                target="_blank"
                rel="noreferrer noopener"
                className="text-sky-700"
              >
                HuggingFace
              </a>{' '}
              for details.
            </div>
          </div>
        </div>
      </div>
      <div className="mt-5 flex w-full flex-row items-center justify-start gap-x-2 px-3 text-2xl font-bold text-slate-600 sm:mt-5 sm:justify-center sm:text-3xl">
        <div className="inline-block bg-clip-text text-gBlue">Gemma Scope 2</div> Demo
      </div>
      <div className="mb-8 mt-1.5 px-3 text-left text-sm font-medium text-slate-500 sm:mb-4 sm:px-0 sm:text-center">
        Examining Safety-Relevant Features and Circuits in Gemma 3
      </div>
      <div className="mb-10 flex w-full flex-col items-start justify-start gap-x-4 gap-y-1.5 rounded px-3 py-1 sm:flex-row sm:px-7">
        <div className="flex w-[105px] min-w-[105px] max-w-[105px] flex-row items-center gap-x-2 sm:flex-col">
          <span className="max-w-[105px]whitespace-nowrap w-[105px] min-w-[105px] rounded-full bg-slate-100 px-3 py-1 text-center text-[10px] font-bold uppercase text-slate-600">
            👋 New Here?
          </span>
        </div>

        <div className="flex w-full flex-col items-start justify-start text-left text-sm font-medium text-slate-600">
          <div className="leading-normal">
            {`If you're new to interpretability (the science of understanding what happens inside AI), we recommend you start with the`}{' '}
            <Link href="/gemma-scope" target="_blank" rel="noreferrer" className="font-bold text-gBlue hover:underline">
              original {`"Exploring Gemma Scope"`}
            </Link>
            , which has more beginner-friendly interactive demos and content.
          </div>
          <div className="mt-3 leading-normal">
            This Gemma Scope 2 demo focuses on exploring safety-relevant features in{' '}
            <Link
              href="https://deepmind.google/models/gemma/gemma-3/"
              target="_blank"
              rel="noreferrer"
              className="font-bold text-gBlue hover:underline"
            >
              Gemma 3
            </Link>{' '}
            27B-IT, the largest model in the new Gemma 3 model series. Since the{' '}
            <Link
              href="https://huggingface.co/google/gemma-scope-2"
              target="_blank"
              rel="noreferrer"
              className="font-bold text-gBlue hover:underline"
            >
              Gemma Scope 2
            </Link>{' '}
            release also includes transcoders, cross-layer transcoders, and crosscoders, Neuronpedia is also adding
            support for{' '}
            <Link href="/graph" target="_blank" rel="noreferrer" className="font-bold text-gBlue hover:underline">
              circuit tracing
            </Link>{' '}
            with those new artifacts.
          </div>
        </div>
      </div>

      <div className="mb-8 flex w-full flex-col items-start justify-start gap-x-4 gap-y-1.5 rounded px-3 py-1 sm:flex-row sm:px-7">
        <div className="mb-2 flex w-full flex-row items-center justify-between gap-x-2 sm:mb-0 sm:w-auto sm:flex-col sm:justify-start">
          <span className="w-[105px] min-w-[105px] max-w-[105px] whitespace-nowrap rounded-full bg-slate-100 px-0 py-1 text-center text-[10px] font-bold uppercase text-slate-600">
            🔢 Sections
          </span>
        </div>

        <div className="grid w-full grid-cols-1 flex-row items-start justify-start gap-x-3 gap-y-3 text-left text-sm font-medium text-slate-500 sm:grid-cols-2 lg:grid-cols-3 xl:gap-x-5">
          <button
            type="button"
            onClick={() => {
              tabUpdater('safety');
            }}
            className="relative flex min-h-[200px] flex-1 cursor-pointer flex-col items-center justify-center rounded-3xl border border-gGreen bg-gGreen/5 px-5 text-gGreen shadow-md transition-all hover:scale-105 hover:border-2 hover:bg-white hover:shadow-xl xl:min-h-[230px]"
          >
            <div className="mb-2 text-6xl leading-none">🛡️</div>
            <div className="text-md font-bold xl:text-lg">Safety & Alignment</div>
            <div className="mt-1.5 text-center text-xs leading-normal text-slate-600 xl:text-[13px]">
              Explore safety and alignment relevant features in Gemma 3.
            </div>
          </button>
          <button
            type="button"
            onClick={() => {
              tabUpdater('circuit');
            }}
            className="relative flex min-h-[200px] flex-1 cursor-default flex-col items-center justify-center rounded-3xl border border-gYellow bg-gYellow/5 px-5 text-gYellow opacity-40 transition-all hover:opacity-60 xl:min-h-[230px]"
          >
            <div className="absolute top-2 mx-auto flex w-full justify-center">
              <div className="whitespace-nowrap rounded-full bg-gYellow px-3 py-1.5 text-[9px] font-bold uppercase leading-none text-white lg:text-[11px]">
                Coming Soon
              </div>
            </div>
            <div className="mb-2 text-6xl">🔌</div>
            <div className="text-md font-bold xl:text-lg">Circuit Tracing</div>
            <div className="mt-1.5 text-center text-xs leading-normal text-slate-600 xl:text-[13px]">
              Using prompts to activate and trace Gemma {`3's`} internal reasoning steps.
            </div>
          </button>
          <button
            type="button"
            onClick={() => {
              tabUpdater('browse');
            }}
            className="relative flex min-h-[200px] flex-1 cursor-pointer flex-col items-center justify-center rounded-3xl border border-gBlue bg-gBlue/5 px-5 text-gBlue shadow-md transition-all hover:scale-105 hover:border-2 hover:bg-white hover:shadow-xl xl:min-h-[230px]"
          >
            <div className="mb-2 text-6xl">📖</div>
            <div className="text-md font-bold xl:text-lg">Dashboards + Inference</div>
            <div className="mt-1.5 text-center text-xs leading-normal text-slate-600 xl:text-[13px]">
              See top activating examples, search, and test with inference.
            </div>
          </button>
        </div>
      </div>

      <div className="mb-8 flex w-full flex-col items-start justify-start gap-x-4 gap-y-1.5 rounded px-3 py-1 pb-8 sm:flex-row sm:px-7">
        <div className="mb-2 flex w-full flex-row items-center justify-between gap-x-2 sm:mb-0 sm:w-auto sm:flex-col sm:justify-start">
          <span className="w-[105px] min-w-[105px] max-w-[105px] whitespace-nowrap rounded-full bg-slate-100 px-0 py-1 text-center text-[10px] font-bold uppercase text-slate-600">
            Do More
          </span>
        </div>

        <div className="grid w-full grid-cols-2 flex-col items-start justify-start gap-x-3 gap-y-3 text-left text-sm font-medium text-slate-500 sm:grid-cols-3">
          <Link
            href={codingTutorialLink}
            target="_blank"
            rel="noreferrer"
            className="flex flex-1 cursor-pointer flex-col items-center justify-center rounded-3xl border border-slate-600 bg-slate-600/5 px-5 py-5 text-slate-600 transition-all hover:bg-slate-600/20"
          >
            <div className="text-md font-bold xl:text-base">Tutorial Notebook</div>
            <div className="mt-1.5 text-center text-xs leading-normal text-slate-600 xl:text-[13px]">
              A Colab notebook for loading and using Gemma Scope 2 artifacts.
            </div>
          </Link>
          <Link
            href={hfLink}
            target="_blank"
            rel="noreferrer"
            className="flex flex-1 cursor-pointer flex-col items-center justify-center rounded-3xl border border-slate-600 bg-slate-600/5 px-5 py-5 text-slate-600 transition-all hover:bg-slate-600/20"
          >
            <div className="text-md font-bold xl:text-base">Hugging Face</div>
            <div className="mt-1.5 text-center text-xs leading-normal text-slate-600 xl:text-[13px]">
              Download the Gemma Scope 2 SAEs and transcoders.
            </div>
          </Link>
          <Link
            href={blogPostLink}
            target="_blank"
            rel="noreferrer"
            className="flex flex-1 cursor-pointer flex-col items-center justify-center rounded-3xl border border-slate-600 bg-slate-600/5 px-5 py-5 text-slate-600 transition-all hover:bg-slate-600/20"
          >
            <div className="text-md font-bold xl:text-base">DeepMind Blog</div>
            <div className="mt-1.5 text-center text-xs leading-normal text-slate-600 xl:text-[13px]">
              Read the official blog post about Gemma Scope 2.
            </div>
          </Link>
        </div>
      </div>
    </div>
  );
}
