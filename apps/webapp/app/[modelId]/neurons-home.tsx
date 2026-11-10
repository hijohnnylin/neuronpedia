'use client';

import FeatureSelector from '@/components/feature-selector/feature-selector';
import ModelSelector from '@/components/feature-selector/model-selector';
import InferenceSearcher from '@/components/inference-searcher/inference-searcher';
import JumpToSAE from '@/components/jump-to-sae';
import BrowserPane from '@/components/panes/browser-pane/browser-pane';
import SearchExplanationsPane from '@/components/panes/search-explanations-pane';
import InferenceActivationAllProvider from '@/components/provider/inference-activation-all-provider';
import { Button } from '@/components/shadcn/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/shadcn/card';
import { SearchExplanationsType } from '@/lib/utils/general';
import { NEURONS_RELEASE, NEURONS_SOURCESET } from '@/lib/utils/source';
import { SourceReleaseWithPartialRelations } from '@/prisma/generated/zod';
import { useRouter } from '@bprogress/next';
import { Shuffle } from 'lucide-react';
import { useState } from 'react';

export type NeuronModelRow = {
  modelId: string;
  layers: string[];
  neuronsPerLayer: number;
  searchable: boolean;
};

const LABEL_CLASS = 'mb-0.5 font-sans text-[10px] font-medium uppercase text-slate-500';
const DEFAULT_MODEL_ID = 'qwen3-8b';

export default function NeuronsHome({
  release,
  models,
}: {
  release: SourceReleaseWithPartialRelations;
  models: NeuronModelRow[];
}) {
  const router = useRouter();
  const defaultModel =
    models.find((m) => m.modelId === DEFAULT_MODEL_ID) || models.find((m) => m.searchable) || models[0];
  const searchModel = defaultModel?.searchable ? defaultModel : models.find((m) => m.searchable);
  const [jumpModelId, setJumpModelId] = useState(defaultModel?.modelId || '');
  // Search by release runs on the server, so it needs the neurons source sets to be in this release.
  const linkedRelease = (release.sourceSets?.length || 0) > 0 ? release : undefined;

  function goToRandomNeuron() {
    const model = models[Math.floor(Math.random() * models.length)];
    if (!model || model.layers.length === 0) {
      return;
    }
    const layer = model.layers[Math.floor(Math.random() * model.layers.length)];
    const index = Math.floor(Math.random() * model.neuronsPerLayer);
    router.push(`/${model.modelId}/${layer}/${index}`);
  }

  return (
    <div className="flex w-full max-w-screen-lg flex-col items-center gap-y-6 px-2 pt-6 text-slate-700 sm:px-0 xl:max-w-screen-xl 2xl:max-w-screen-2xl">
      <div className="flex w-full max-w-screen-lg flex-row items-center justify-between">
        <div className="text-2xl font-bold text-slate-900 sm:text-3xl">MLP Neurons</div>
        <Button onClick={goToRandomNeuron} className="flex flex-row gap-x-1.5" disabled={models.length === 0}>
          <Shuffle className="h-4 w-4" />
          Random Neuron
        </Button>
      </div>

      {defaultModel && (
        <Card className="w-full max-w-screen-lg bg-white">
          <CardHeader className="w-full pb-0 pt-5">
            <CardTitle>Jump To</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col items-start gap-y-3 pb-6 pt-4">
            <div className="flex flex-col items-start">
              <div className={LABEL_CLASS}>Jump to Model</div>
              <div className="flex flex-row gap-x-2">
                <ModelSelector
                  modelId={jumpModelId}
                  modelIdChangedCallback={setJumpModelId}
                  overrideModels={models.map((m) => m.modelId)}
                />
                <button
                  type="button"
                  onClick={() => router.push(`/${jumpModelId}`)}
                  className="flex h-10 max-h-[40px] min-h-[40px] select-none items-center justify-center rounded bg-slate-200 px-3 text-[11px] font-medium uppercase text-slate-500 hover:bg-sky-700 hover:text-white"
                >
                  Go
                </button>
              </div>
            </div>
            <JumpToSAE
              modelId={defaultModel.modelId}
              layer={defaultModel.layers[0]}
              filterToRelease={NEURONS_RELEASE}
              filterToFeaturedReleases={false}
            />
            <div className="flex flex-col items-start">
              <div className={LABEL_CLASS}>Jump to Feature</div>
              <FeatureSelector
                showModel
                filterToRelease={NEURONS_RELEASE}
                defaultModelId={defaultModel.modelId}
                defaultSourceSet={NEURONS_SOURCESET}
                defaultSource={defaultModel.layers[0]}
                filterToPublic
                filterToFeaturedReleases={false}
                openInNewTab={false}
              />
            </div>
          </CardContent>
        </Card>
      )}

      {searchModel && (
        <Card className="w-full max-w-screen-lg bg-white">
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
          <CardContent>
            <InferenceActivationAllProvider>
              <InferenceSearcher
                initialModelId={searchModel.modelId}
                initialSourceSet={NEURONS_SOURCESET}
                filterToRelease={NEURONS_RELEASE}
              />
            </InferenceActivationAllProvider>
          </CardContent>
        </Card>
      )}

      {defaultModel && (
        <div className="w-full">
          <BrowserPane
            modelId={defaultModel.modelId}
            sourceSet={NEURONS_SOURCESET}
            layer={defaultModel.layers[0]}
            showModel
            filterToRelease={NEURONS_RELEASE}
          />
        </div>
      )}

      {defaultModel && (
        <div className="w-full max-w-screen-lg">
          <SearchExplanationsPane
            filterToRelease={release}
            initialModelId={defaultModel.modelId}
            initialSourceSetName={NEURONS_SOURCESET}
            defaultTab={linkedRelease ? SearchExplanationsType.BY_RELEASE : SearchExplanationsType.BY_MODEL}
            showTabs
          />
        </div>
      )}
    </div>
  );
}
