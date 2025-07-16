'use client';

import InferenceSearcher from '@/components/inference-searcher/inference-searcher';
import InferenceActivationAllProvider from '@/components/provider/inference-activation-all-provider';
import { Button } from '@/components/shadcn/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/shadcn/card';
import { ModelWithPartialRelations } from '@/prisma/generated/zod';

export default function SearchInferenceSourcePane({
  model,
  sourceSetName,
  initialSelectedLayers = [],
}: {
  model: ModelWithPartialRelations;
  sourceSetName: string;
  initialSelectedLayers?: string[];
}) {
  return (
    <div className="flex w-full flex-col items-center">
      <Card className="mt-0 w-full max-w-screen-lg bg-white">
        <CardHeader className="w-full pb-3 pt-5">
          <div className="flex w-full flex-row items-center justify-between">
            <CardTitle>Search via Inference</CardTitle>
            <a href="https://docs.neuronpedia.org/search" target="_blank" rel="noreferrer">
              <Button
                variant="outline"
                size="sm"
                className="flex flex-row gap-x-2 rounded-full text-sm font-semibold text-slate-400 shadow-sm"
              >
                ?
              </Button>
            </a>
          </div>
        </CardHeader>
        <CardContent className="">
          {model.inferenceEnabled ? (
            // set a separate provider for each instance of searcher since we don't want them to share state
            <InferenceActivationAllProvider>
              <InferenceSearcher
                showModels={false}
                initialModelId={model.id}
                initialSourceSet={sourceSetName}
                initialSelectedLayers={initialSelectedLayers}
              />
            </InferenceActivationAllProvider>
          ) : (
            <div className="flex flex-col items-center justify-center">
              <div className="pb-3 pt-2 text-base font-bold text-slate-400">
                Inference is not currently not enabled for this model.
              </div>
              <a
                href={`mailto:team@neuronpedia.org?subject=Request%20Enabling%20Model%20${model.id}&body=I'd%20like%20to%20request%20that%20Neuronpedia%20enables%20search%20for%20model%20${model.id}`}
                target="_blank"
                rel="noreferrer"
              >
                <Button variant="outline">Request</Button>
              </a>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
