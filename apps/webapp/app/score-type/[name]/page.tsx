import { getExplanationScoreType } from '@/lib/db/explanation-type';
import { notFound } from 'next/navigation';
import RecentScores from './recent-scores';

export default async function Page(props: { params: Promise<{ name: string }> }) {
  const params = await props.params;
  const explanationScoreType = await getExplanationScoreType(params.name);

  if (!explanationScoreType) {
    notFound();
  }

  return (
    <div className="flex max-h-[calc(100vh-100px)] w-full max-w-screen-xl flex-row gap-x-3 overflow-scroll pt-3 text-slate-600">
      <div className="flex basis-1/3 flex-col items-start gap-y-0 rounded-xl border bg-white px-5 py-5 text-sm">
        <div className="mb-2 rounded-full bg-slate-400 px-3 py-1 text-[10px] font-bold uppercase leading-none text-white">
          SCORE TYPE
        </div>
        <div className="mb-5 ml-5 font-bold">{explanationScoreType.name}</div>
        <div className="mb-2 rounded-full bg-slate-400 px-3 py-1 text-[10px] font-bold uppercase leading-none text-white">
          Description
        </div>
        <div className="mb-5 ml-5">{explanationScoreType.description}</div>
        <div className="mb-2 rounded-full bg-slate-400 px-3 py-1 text-[10px] font-bold uppercase leading-none text-white">
          Author
        </div>
        <div className="mb-5 ml-5">{explanationScoreType.creatorName}</div>
        {explanationScoreType.url && (
          <>
            <div className="mb-2 rounded-full bg-slate-400 px-3 py-1 text-[10px] font-bold uppercase leading-none text-white">
              URL
            </div>
            <a
              href={explanationScoreType.url}
              target="_blank"
              rel="noreferrer"
              className="mb-4 ml-5 text-xs font-medium text-sky-800"
            >
              {explanationScoreType.url}
            </a>
          </>
        )}
        <div className="mb-2 rounded-full bg-slate-400 px-3 py-1 text-[10px] font-bold uppercase leading-none text-white">
          Score Calculation
        </div>
        <div className="mb-5 ml-5">{explanationScoreType.scoreDescription}</div>
        <div className="mb-2 rounded-full bg-slate-400 px-3 py-1 text-[10px] font-bold uppercase leading-none text-white">
          Settings
        </div>
        <div className="mb-5 ml-5">{explanationScoreType.settings}</div>
      </div>
      <div className="flex max-h-[100%] basis-2/3 flex-col items-center overflow-y-scroll rounded-xl border bg-white px-5 pb-5">
        <div className="sticky top-0 mb-1 mt-4 rounded-full bg-slate-400 px-3 py-1 text-center text-[10px] font-bold uppercase leading-none text-white">
          Recent Scores
        </div>
        <RecentScores explanationScoreType={explanationScoreType} />
      </div>
    </div>
  );
}
