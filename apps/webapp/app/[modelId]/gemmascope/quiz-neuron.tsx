'use client';

import ActivationSingleForm from '@/components/activation-single-form';
import ActivationsList from '@/components/activations-list';
import { Card } from '@/components/shadcn/card';
import { ExternalLink } from 'lucide-react';
import { NeuronWithPartialRelations } from 'prisma/generated/zod';
import { useState } from 'react';

export default function QuizNeuron({
  name,
  neuron,
  answer,
  setFeatureModalFeature,
  setFeatureModalOpen,
}: {
  name: string;
  neuron: NeuronWithPartialRelations;
  answer: string;
  setFeatureModalFeature: (feature: NeuronWithPartialRelations) => void;
  setFeatureModalOpen: (open: boolean) => void;
}) {
  const [showAnswer, setShowAnswer] = useState(false);
  return (
    <div className="flex w-full flex-1 flex-col items-center justify-center">
      <div className="mb-1.5 text-xs">
        <code>{name}</code>
      </div>
      <Card className="overflow-hidden">
        <ActivationsList
          feature={neuron}
          defaultRange={0}
          defaultShowLineBreaks={false}
          showControls={false}
          showTest={false}
          showDatasource={false}
          showTopActivationToken={false}
          activations={neuron?.activations}
          activationItemClassName="py-2"
        />
      </Card>
      <ActivationSingleForm
        neuron={neuron}
        overallMaxValue={100}
        hideBos
        hideSteer
        enterSubmits
        formValue=""
        placeholder={`Test ${name}`}
      />
      <div className="flex w-full flex-row items-center justify-start gap-x-3">
        <button
          type="button"
          onClick={() => {
            setShowAnswer(true);
          }}
          className={`mt-0.5 rounded-full ${
            showAnswer ? 'bg-gRed text-slate-800' : 'bg-white hover:bg-gRed/20'
          } border border-gRed px-3 py-1.5 text-[11px] font-medium text-slate-800 transition-all`}
        >
          🫣 Reveal Our Label
        </button>
        {showAnswer && (
          <>
            <div className="flex-1 text-center text-sm font-medium text-slate-500">{answer}</div>
            <div className="flex max-w-7 flex-col items-center justify-center gap-y-1 overflow-visible">
              <ExternalLink
                className="h-7 w-7 cursor-pointer rounded bg-slate-200 p-1.5 hover:bg-slate-300"
                onClick={() => {
                  // make a copy of neuron to force activations loading
                  const newNeuron: NeuronWithPartialRelations = JSON.parse(JSON.stringify(neuron));
                  newNeuron.activations = undefined;
                  setFeatureModalFeature(newNeuron);
                  setFeatureModalOpen(true);
                }}
              />
            </div>
          </>
        )}
      </div>
    </div>
  );
}
