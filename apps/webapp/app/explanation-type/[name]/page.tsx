import ActivationItem from '@/components/activation-item';
import { getExplanationType } from '@/lib/db/explanation-type';
import Image from 'next/image';
import { notFound } from 'next/navigation';

export default async function Page(props: { params: Promise<{ name: string }> }) {
  const params = await props.params;
  const explanationType = await getExplanationType(params.name);

  if (!explanationType) {
    notFound();
  }

  return (
    <div className="flex max-h-[calc(100vh-100px)] w-full max-w-screen-xl flex-row gap-x-3 overflow-scroll pt-3 text-slate-600">
      <div className="flex basis-1/3 flex-col items-start gap-y-0 rounded-xl border bg-white px-5 py-5 text-sm">
        <div className="mb-2 rounded-full bg-slate-400 px-3 py-1 text-[10px] font-bold uppercase leading-none text-white">
          EXPLANATION TYPE
        </div>
        <div className="mb-5 ml-5 font-bold">{explanationType.name}</div>
        <div className="mb-2 rounded-full bg-slate-400 px-3 py-1 text-[10px] font-bold uppercase leading-none text-white">
          Description
        </div>
        <div className="mb-5 ml-5">{explanationType.description}</div>
        <div className="mb-2 rounded-full bg-slate-400 px-3 py-1 text-[10px] font-bold uppercase leading-none text-white">
          Author
        </div>
        <div className="mb-5 ml-5">{explanationType.creatorName}</div>
        {explanationType.url && (
          <>
            <div className="mb-2 rounded-full bg-slate-400 px-3 py-1 text-[10px] font-bold uppercase leading-none text-white">
              URL
            </div>
            <a
              href={explanationType.url}
              target="_blank"
              rel="noreferrer"
              className="mb-4 ml-5 text-xs font-medium text-sky-800"
            >
              {explanationType.url}
            </a>
          </>
        )}
        <div className="mb-2 rounded-full bg-slate-400 px-3 py-1 text-[10px] font-bold uppercase leading-none text-white">
          Settings
        </div>
        <div className="mb-5 ml-5">{explanationType.settings}</div>
      </div>
      <div className="flex max-h-[100%] basis-2/3 flex-col items-center overflow-y-scroll rounded-xl border bg-white px-5 pb-5">
        <div className="sticky top-0 mb-1 mt-4 rounded-full bg-slate-400 px-3 py-1 text-center text-[10px] font-bold uppercase leading-none text-white">
          Recent Explanations
        </div>
        <div className="mt-2 flex w-full flex-col gap-y-5 text-sm">
          {explanationType.explanations?.map((explanation) => (
            <div
              key={explanation.id}
              className="flex flex-row items-center justify-between gap-x-3 px-3 py-2 odd:bg-slate-50"
            >
              <div className="flex flex-1 flex-col gap-y-0 text-sm">
                <div className="text-[12px] leading-snug">{explanation.description}</div>
                <div className="mb-2 mt-1 text-[10px] leading-none text-slate-400">
                  {explanation.explanationModelName}
                </div>
                {explanation.neuron?.activations?.length && explanation.neuron.activations.length > 0 && (
                  <ActivationItem
                    activation={explanation.neuron.activations[0]}
                    tokensToDisplayAroundMaxActToken={6}
                    overrideTextSize="text-[10px]"
                    overrideLeading="leading-none"
                  />
                )}
              </div>
              <a
                className="flex shrink-0 flex-row items-center gap-x-1 whitespace-nowrap rounded-md bg-slate-200 px-[8px] py-[6px] text-[9px] font-medium leading-none text-slate-700 hover:bg-sky-200 hover:text-sky-700 sm:px-3 sm:py-2 sm:text-[10px]"
                href={`/${explanation.modelId}/${explanation.layer}/${explanation.index}`}
                target="_blank"
                rel="noreferrer"
              >
                <Image src="/logo.png" alt="Neuronpedia logo" width="24" height="24" className="mr-1" />
                <div className="flex flex-col gap-y-[3px]">
                  <div className="">{explanation?.modelId?.toUpperCase()}</div>
                  <div className="">{explanation?.layer?.toUpperCase()}</div>
                  <div className="">INDEX {explanation?.index?.toUpperCase()}</div>
                </div>
              </a>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
