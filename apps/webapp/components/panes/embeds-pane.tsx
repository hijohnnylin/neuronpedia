import { NEXT_PUBLIC_URL } from '@/lib/env';
import * as Checkbox from '@radix-ui/react-checkbox';
import copy from 'copy-to-clipboard';
import { Check, Copy, HelpCircle } from 'lucide-react';
import Link from 'next/link';
import { Activation, NeuronWithPartialRelations } from 'prisma/generated/zod';
import { useState } from 'react';

export default function EmbedsPane({
  currentNeuron,
  testTextResult,
}: {
  currentNeuron: NeuronWithPartialRelations | undefined;
  testTextResult: Activation | undefined;
}) {
  const [embedExplanationChecked, setEmbedExplanationChecked] = useState(true);
  const [embedPlotsChecked, setEmbedPlotsChecked] = useState(true);
  const [embedTestChecked, setEmbedTestChecked] = useState(true);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [embedDefaultTestTextChecked, setEmbedDefaultTestTextChecked] = useState(false);
  const [embedActivationsChecked, setEmbedActivationsChecked] = useState(true);
  const [embedLinkChecked, setEmbedLinkChecked] = useState(true);
  const [embedSteerChecked, setEmbedSteerChecked] = useState(true);

  return (
    <div className="relative mb-0 mt-2 hidden flex-col gap-x-2 overflow-hidden rounded-lg border bg-white px-3 pb-5 pt-2 text-xs shadow transition-all sm:mt-3 sm:flex">
      <div className="mb-2 flex w-full flex-row items-center justify-center gap-x-1 text-[10px] font-normal uppercase text-slate-400">
        Embeds
        <Link href="https://docs.neuronpedia.org/embed-iframe" target="_blank" rel="noreferrer">
          <HelpCircle className="h-3 w-3" />
        </Link>
      </div>
      <div className="mt-1 flex w-full flex-col gap-x-2 gap-y-1.5">
        <div className="grid grid-cols-3 items-center gap-y-1.5 px-1">
          <label className="flex flex-row items-center gap-x-1 leading-none text-slate-400" htmlFor="c2">
            <Checkbox.Root
              className="flex h-4 w-4 appearance-none items-center justify-center rounded-[3px] border border-slate-300 bg-white outline-none"
              defaultChecked
              checked={embedPlotsChecked}
              onCheckedChange={(e) => {
                if (e === true) {
                  setEmbedPlotsChecked(true);
                } else {
                  setEmbedPlotsChecked(false);
                }
              }}
              id="c2"
            >
              <Checkbox.Indicator className="text-sky-700">
                <Check className="h-4 w-4" />
              </Checkbox.Indicator>
            </Checkbox.Root>
            Show Plots
          </label>
          <label className="flex flex-row items-center gap-x-1 leading-none text-slate-400" htmlFor="c1">
            <Checkbox.Root
              className="flex h-4 w-4 appearance-none items-center justify-center rounded-[3px] border border-slate-300 bg-white outline-none"
              defaultChecked
              checked={embedExplanationChecked}
              onCheckedChange={(e) => {
                if (e === true) {
                  setEmbedExplanationChecked(true);
                } else {
                  setEmbedExplanationChecked(false);
                }
              }}
              id="c1"
            >
              <Checkbox.Indicator className="text-sky-700">
                <Check className="h-4 w-4" />
              </Checkbox.Indicator>
            </Checkbox.Root>
            Show Explanation
          </label>
          <label className="flex flex-row items-center gap-x-1 leading-none text-slate-400" htmlFor="c5">
            <Checkbox.Root
              className="flex h-4 w-4 appearance-none items-center justify-center rounded-[3px] border border-slate-300 bg-white outline-none"
              defaultChecked
              checked={embedActivationsChecked}
              onCheckedChange={(e) => {
                if (e === true) {
                  setEmbedActivationsChecked(true);
                } else {
                  setEmbedActivationsChecked(false);
                }
              }}
              id="c5"
            >
              <Checkbox.Indicator className="text-sky-700">
                <Check className="h-4 w-4" />
              </Checkbox.Indicator>
            </Checkbox.Root>
            Show Activations
          </label>
          <label className="flex flex-row items-center gap-x-1 leading-none text-slate-400" htmlFor="c3">
            <Checkbox.Root
              className="flex h-4 w-4 appearance-none items-center justify-center rounded-[3px] border border-slate-300 bg-white outline-none"
              defaultChecked
              onCheckedChange={(e) => {
                if (e === true) {
                  setEmbedTestChecked(true);
                } else {
                  setEmbedTestChecked(false);
                }
              }}
              id="c3"
            >
              <Checkbox.Indicator className="text-sky-700">
                <Check className="h-4 w-4" />
              </Checkbox.Indicator>
            </Checkbox.Root>
            Show Test Field
          </label>

          {/* <label className="flex flex-row items-center gap-x-1 leading-none text-slate-400" htmlFor="c4">
            <Checkbox.Root
              className="flex h-4 w-4 appearance-none items-center justify-center rounded-[3px] border border-slate-300 bg-white outline-none"
              defaultChecked={false}
              onCheckedChange={(e) => {
                if (e === true) {
                  if (!testTextResult) {
                    alert(
                      "To set a default test text in your share/embed, first set an activation text using the Test Activation field, then click 'Test'.",
                    );
                  } else {
                    setEmbedDefaultTestTextChecked(true);
                  }
                } else {
                  setEmbedDefaultTestTextChecked(false);
                }
              }}
              id="c4"
            >
              <Checkbox.Indicator className="text-sky-700">
                <Check className="h-4 w-4" />
              </Checkbox.Indicator>
            </Checkbox.Root>
            Default Test Text
          </label> */}
          <label className="flex flex-row items-center gap-x-1 leading-none text-slate-400" htmlFor="c7">
            <Checkbox.Root
              className="flex h-4 w-4 appearance-none items-center justify-center rounded-[3px] border border-slate-300 bg-white outline-none"
              defaultChecked
              checked={embedSteerChecked}
              onCheckedChange={(e) => {
                if (e === true) {
                  setEmbedSteerChecked(true);
                } else {
                  setEmbedSteerChecked(false);
                }
              }}
              id="c7"
            >
              <Checkbox.Indicator className="text-sky-700">
                <Check className="h-4 w-4" />
              </Checkbox.Indicator>
            </Checkbox.Root>
            Show Steer
          </label>
          <label className="flex flex-row items-center gap-x-1 leading-none text-slate-400" htmlFor="c6">
            <Checkbox.Root
              className="flex h-4 w-4 appearance-none items-center justify-center rounded-[3px] border border-slate-300 bg-white outline-none"
              defaultChecked
              checked={embedLinkChecked}
              onCheckedChange={(e) => {
                if (e === true) {
                  setEmbedLinkChecked(true);
                } else {
                  setEmbedLinkChecked(false);
                }
              }}
              id="c6"
            >
              <Checkbox.Indicator className="text-sky-700">
                <Check className="h-4 w-4" />
              </Checkbox.Indicator>
            </Checkbox.Root>
            Show Link
          </label>
        </div>
        <div className="mt-0.5 flex flex-1 flex-col">
          <div className="text-left text-[10px] font-bold uppercase text-slate-400">IFrame</div>
          <div className="mt-0.5 flex flex-row items-center justify-center gap-x-1.5">
            <textarea
              disabled
              className="disabled form-input block h-8 w-full resize-none whitespace-pre-wrap break-words rounded-md border-0 bg-slate-100 px-2.5 py-[5px] font-mono text-[8px] leading-normal text-slate-500 outline-none focus:outline-none focus:ring-0"
              value={`<iframe src="${NEXT_PUBLIC_URL}/${currentNeuron?.modelId}/${currentNeuron?.layer}/${
                currentNeuron?.index
              }?embed=true&embedexplanation=${embedExplanationChecked ? 'true' : 'false'}&embedplots=${
                embedPlotsChecked ? 'true' : 'false'
              }&embedsteer=${embedSteerChecked ? 'true' : 'false'}&embedactivations=${embedActivationsChecked ? 'true' : 'false'}&embedlink=${embedLinkChecked ? 'true' : 'false'}&embedtest=${embedTestChecked ? 'true' : 'false'}${
                embedDefaultTestTextChecked
                  ? `&defaulttesttext=${encodeURIComponent(testTextResult?.tokens?.join('') || '')}`
                  : ''
              }" title="Neuronpedia" style="height: 300px; width: 540px;"></iframe>`}
            />
            <button
              type="button"
              onClick={(e) => {
                e.preventDefault();
                copy(
                  `<iframe src="${NEXT_PUBLIC_URL}/${currentNeuron?.modelId}/${currentNeuron?.layer}/${
                    currentNeuron?.index
                  }?embed=true&embedexplanation=${embedExplanationChecked ? 'true' : 'false'}&embedplots=${
                    embedPlotsChecked ? 'true' : 'false'
                  }&embedsteer=${embedSteerChecked ? 'true' : 'false'}&embedactivations=${embedActivationsChecked ? 'true' : 'false'}&embedlink=${embedLinkChecked ? 'true' : 'false'}&embedtest=${embedTestChecked ? 'true' : 'false'}${
                    embedDefaultTestTextChecked
                      ? `&defaulttesttext=${encodeURIComponent(testTextResult?.tokens?.join('') || '')}`
                      : ''
                  }" title="Neuronpedia" style="height: 300px; width: 540px;"></iframe>`,
                );
              }}
            >
              <Copy className="h-8 w-8 rounded-md bg-slate-200 px-2 py-2 text-slate-400 hover:bg-slate-300 active:bg-sky-300" />
            </button>
          </div>
        </div>
        <div className="flex flex-1 flex-col">
          <div className="flex flex-row justify-start">
            <div className="text-left text-[10px] font-bold uppercase text-slate-400">Link</div>
          </div>
          <div className="mt-0.5 flex flex-row items-center justify-center gap-x-1.5">
            <textarea
              disabled
              className="disabled form-input block h-8 w-full resize-none whitespace-pre-wrap break-words rounded-md border-0 bg-slate-100 px-2.5 py-[5px] font-mono text-[8px] leading-normal text-slate-500 outline-none focus:outline-none focus:ring-0"
              value={`${NEXT_PUBLIC_URL}/${currentNeuron?.modelId}/${currentNeuron?.layer}/${
                currentNeuron?.index
              }?embed=true&embedexplanation=${embedExplanationChecked ? 'true' : 'false'}&embedplots=${
                embedPlotsChecked ? 'true' : 'false'
              }&embedsteer=${embedSteerChecked ? 'true' : 'false'}&embedactivations=${embedActivationsChecked ? 'true' : 'false'}&embedlink=${embedLinkChecked ? 'true' : 'false'}&embedtest=${embedTestChecked ? 'true' : 'false'}${
                embedDefaultTestTextChecked
                  ? `&defaulttesttext=${encodeURIComponent(testTextResult?.tokens?.join('') || '')}`
                  : ''
              }`}
            />
            <button
              type="button"
              onClick={(e) => {
                e.preventDefault();
                copy(
                  `${NEXT_PUBLIC_URL}/${currentNeuron?.modelId}/${currentNeuron?.layer}/${
                    currentNeuron?.index
                  }?embed=true&embedexplanation=${embedExplanationChecked ? 'true' : 'false'}&embedplots=${
                    embedPlotsChecked ? 'true' : 'false'
                  }&embedsteer=${embedSteerChecked ? 'true' : 'false'}&embedactivations=${embedActivationsChecked ? 'true' : 'false'}&embedlink=${embedLinkChecked ? 'true' : 'false'}&embedtest=${embedTestChecked ? 'true' : 'false'}${
                    embedDefaultTestTextChecked
                      ? `&defaulttesttext=${encodeURIComponent(testTextResult?.tokens?.join('') || '')}`
                      : ''
                  }`,
                );
              }}
            >
              <Copy className="h-8 w-8 rounded-md bg-slate-200 px-2 py-2 text-slate-400 hover:bg-slate-300 active:bg-sky-300" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
