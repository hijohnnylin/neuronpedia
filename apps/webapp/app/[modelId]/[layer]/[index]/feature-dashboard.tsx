'use client';

import ActivationsList from '@/components/activations-list';
import BreadcrumbsComponent from '@/components/breadcrumbs-component';
import ExplanationScoreDetailDialog from '@/components/explanation-score-detail-dialog';
import FeatureSelector from '@/components/feature-selector/feature-selector';
import FeatureStats from '@/components/feature-stats';
import CommentsPane from '@/components/panes/comments-pane';
import CosSimPane from '@/components/panes/cossim-pane';
import EmbedsPane from '@/components/panes/embeds-pane';
import ExplanationsPane from '@/components/panes/explanations-pane';
import ListsPane from '@/components/panes/lists-pane';
import SaeLensConfigPane from '@/components/panes/saelens-config-pane';
import { useGlobalContext } from '@/components/provider/global-provider';
import { BreadcrumbLink, BreadcrumbPage } from '@/components/shadcn/breadcrumbs';
import VectorItem from '@/components/vector-item';
import { neuronHasVectorInDatabase, shouldHideBreadcrumbsAndSelectorForNeuronVector } from '@/lib/utils/neuron-vector';
import { getSourceSetNameFromSource, NEURONS_SOURCESET } from '@/lib/utils/source';
import { Download } from 'lucide-react';
import Image from 'next/image';
import { usePathname } from 'next/navigation';
import {
  Activation,
  CommentWithPartialRelations,
  NeuronWithPartialRelations,
  SourceWithPartialRelations,
} from 'prisma/generated/zod';
import { useEffect, useMemo, useState } from 'react';
import BookmarkButton from './bookmark-button';

export default function FeatureDashboard({
  initialNeuron,
  embed = false,
  embedPlots = true,
  embedTest = true,
  defaultTestText,
  embedTestField = true,
  embedExplanation = true,
  embedActivations = true,
  embedLink = true,
  embedSteer = true,
  forceMiniStats = false,
  activationMarkerValue,
}: {
  initialNeuron?: NeuronWithPartialRelations | undefined;
  embed?: boolean;
  embedPlots?: boolean;
  embedTest?: boolean;
  defaultTestText?: string;
  embedTestField?: boolean;
  embedExplanation?: boolean;
  embedActivations?: boolean;
  embedLink?: boolean;
  embedSteer?: boolean;
  forceMiniStats?: boolean;
  activationMarkerValue?: number;
}) {
  const [currentNeuron, setCurrentNeuron] = useState<NeuronWithPartialRelations | undefined>(initialNeuron);
  const { getReleaseForSourceSet } = useGlobalContext();
  const [hideDeadWarning, setHideDeadWarning] = useState(false);
  const [testTextResult, setTestTextResult] = useState<Activation | undefined>();
  const pathname = usePathname();
  const [neuronLoaded, setNeuronLoaded] = useState(false);

  const isAssistantAxis = useMemo(
    () =>
      currentNeuron?.modelId === 'llama3.3-70b-it' &&
      currentNeuron?.layer === '40-neuronpedia-resid-post' &&
      currentNeuron?.index === '101874252',
    [currentNeuron],
  );

  const allowTest = () => {
    if (isAssistantAxis) {
      return false;
    }
    if (currentNeuron?.hasVector && currentNeuron?.model?.inferenceEnabled) {
      if (embed) {
        return embedTest;
      }
      return true;
    }
    if (currentNeuron?.source?.inferenceEnabled === true) {
      if (embed) {
        return embedTest;
      }
      return true;
    }
    return false;
  };

  async function reloadNeuron() {
    if (currentNeuron) {
      await fetch(`/api/feature/${currentNeuron.modelId}/${currentNeuron.layer}/${currentNeuron.index}`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      })
        .then((response) => response.json())
        .then((n: NeuronWithPartialRelations) => {
          setCurrentNeuron(n);
          setNeuronLoaded(true);
        })
        .catch((error) => {
          console.error(`error submitting getting rest of neuron: ${error}`);
        });
    }
  }

  useEffect(() => {
    if (currentNeuron) {
      if (currentNeuron.activations && currentNeuron.activations.length < 2 && !neuronLoaded) {
        reloadNeuron();
      }
    }
  }, [currentNeuron, neuronLoaded, reloadNeuron]);

  return (
    <div className="relative mx-auto flex w-full flex-col gap-x-3 pt-0 sm:mt-0">
      <ExplanationScoreDetailDialog />

      {!embed &&
        (!neuronHasVectorInDatabase(currentNeuron) ||
          (neuronHasVectorInDatabase(currentNeuron) &&
            !shouldHideBreadcrumbsAndSelectorForNeuronVector(currentNeuron))) && (
          <BreadcrumbsComponent
            crumbsArray={[
              [
                ...(getReleaseForSourceSet(currentNeuron?.modelId || '', currentNeuron?.sourceSetName || '')
                  ? [
                      <BreadcrumbLink
                        key={0}
                        href={`/${
                          getReleaseForSourceSet(currentNeuron?.modelId || '', currentNeuron?.sourceSetName || '')?.name
                        }`}
                      >
                        {`${
                          getReleaseForSourceSet(currentNeuron?.modelId || '', currentNeuron?.sourceSetName || '')
                            ?.creatorNameShort
                        } · ${
                          getReleaseForSourceSet(currentNeuron?.modelId || '', currentNeuron?.sourceSetName || '')
                            ?.descriptionShort
                        }`}
                      </BreadcrumbLink>,
                    ]
                  : []),
              ],
              <BreadcrumbLink href={`/${currentNeuron?.modelId}`} key={1}>
                {currentNeuron?.model?.displayName}
              </BreadcrumbLink>,
              ...(currentNeuron?.sourceSet
                ? [
                    <BreadcrumbLink href={`/${currentNeuron?.modelId}/${currentNeuron?.sourceSet?.name}`} key={2.5}>
                      {currentNeuron?.sourceSet?.type}
                    </BreadcrumbLink>,
                  ]
                : []),
              <BreadcrumbLink href={`/${currentNeuron?.modelId}/${currentNeuron?.layer}`} key={2}>
                {currentNeuron?.layer.toUpperCase()}
              </BreadcrumbLink>,
              <BreadcrumbPage key={3}>{currentNeuron?.index}</BreadcrumbPage>,
            ]}
          />
        )}

      {/* === VECTOR */}
      {!embed &&
        (!neuronHasVectorInDatabase(currentNeuron) ||
          (neuronHasVectorInDatabase(currentNeuron) &&
            !shouldHideBreadcrumbsAndSelectorForNeuronVector(currentNeuron))) && (
          <div className="fixed left-0 right-0 top-12 z-20 mb-0 flex justify-center gap-y-2 border-b bg-white px-2 pb-2 pt-1 shadow-[rgba(0,0,0,0.2)_0px_4px_3px_-3px] sm:static sm:top-16 sm:z-0 sm:mb-2 sm:flex-row sm:gap-y-0 sm:border-b-0 sm:border-t-0 sm:bg-transparent sm:px-2 sm:pb-0 sm:pt-3 sm:shadow-[0]">
            <div className="flex w-full flex-1 select-none flex-row justify-center gap-x-3 overflow-hidden px-0 pb-1 text-center font-sans text-xs font-bold leading-none text-slate-700 sm:gap-x-5 sm:px-0 sm:pt-0 sm:text-base">
              <FeatureSelector
                filterToPublic
                showNextPrev
                defaultModelId={currentNeuron?.modelId || pathname.split('/')[1]}
                defaultSourceSet={
                  currentNeuron?.sourceSetName ||
                  getSourceSetNameFromSource(pathname.split('/')[2] || '') ||
                  NEURONS_SOURCESET
                }
                defaultSource={currentNeuron?.layer || pathname.split('/')[2]}
                defaultIndex={currentNeuron?.index || pathname.split('/')[3]}
                openInNewTab={false}
                filterToFeaturedReleases={false}
              />
            </div>
            <div className="hidden flex-row items-start justify-start gap-x-2 px-0 pt-1 sm:flex">
              {allowTest() && (
                <button
                  type="button"
                  onClick={async () => {
                    const response = await fetch('/api/vector/get', {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({
                        modelId: currentNeuron?.modelId,
                        source: currentNeuron?.layer,
                        index: currentNeuron?.index,
                      }),
                    });
                    const data = await response.json();
                    const blob = new Blob([JSON.stringify(data.vector, null, 2)], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `${currentNeuron?.modelId}_${currentNeuron?.layer}_${currentNeuron?.index}.json`;
                    a.click();
                    URL.revokeObjectURL(url);
                  }}
                  aria-label="Download vector as JSON"
                  className="h-9 w-9 overflow-hidden rounded-full bg-slate-200 px-0 py-1.5 text-center text-xs font-bold uppercase text-slate-500/80 transition-all hover:enabled:bg-sky-300 hover:enabled:text-sky-800 disabled:bg-slate-400 disabled:opacity-60"
                >
                  <div className="-mt-0.5 flex flex-row items-center justify-center">
                    <Download className="h-5 w-5" />
                  </div>
                </button>
              )}
              <BookmarkButton mini={false} currentNeuron={currentNeuron} />
            </div>
          </div>
        )}

      {neuronHasVectorInDatabase(currentNeuron) && currentNeuron && (
        <div className="mt-2 flex w-full flex-row justify-center px-2">
          <div className="w-full rounded-md border bg-white py-2">
            <VectorItem vector={currentNeuron} isInNeuronDetails />
          </div>
        </div>
      )}

      {/* === MAIN CONTENT */}
      <div
        className={`relative ${
          embed || neuronHasVectorInDatabase(currentNeuron) ? 'sm:pt-2' : 'mt-16 sm:pt-0'
        } flex h-full flex-col sm:mt-0 sm:flex-row sm:gap-x-2 sm:gap-y-0 ${
          forceMiniStats ? 'sm:px-0' : 'pt-2 sm:px-2'
        }`}
      >
        {/* === LEFT PANE */}
        <div
          className={`${
            embed
              ? 'hidden'
              : 'mb-3 flex flex-col sm:mb-0 sm:basis-1/2 sm:overflow-auto sm:overflow-y-scroll lg:basis-1/3'
          }`}
        >
          {/* === EXPLANATIONS */}
          <ExplanationsPane currentNeuron={currentNeuron} setCurrentNeuron={setCurrentNeuron} />

          {/* === COS SIM */}
          <CosSimPane currentNeuron={currentNeuron} />

          {/* === SAELENS CONFIG */}
          {currentNeuron?.source &&
            (currentNeuron?.source?.saelensConfig || currentNeuron?.source?.num_tokens_in_prompt) && (
              <div className="mt-2 hidden sm:mt-3 sm:block">
                <SaeLensConfigPane
                  sae={currentNeuron.source as SourceWithPartialRelations}
                  numPrompts={currentNeuron?.source?.num_prompts || undefined}
                  numTokensInPrompt={currentNeuron?.source?.num_tokens_in_prompt || undefined}
                  dashboardDataset={currentNeuron?.source?.dataset || ''}
                />
              </div>
            )}

          {/* === EMBEDS */}
          <EmbedsPane currentNeuron={currentNeuron} testTextResult={testTextResult} />

          {/* === LISTS */}
          {/* TODO: fix this */}
          {/* eslint-disable-next-line react/jsx-no-bind */}
          <ListsPane currentNeuron={currentNeuron} reloadNeuron={reloadNeuron} />

          {/* === COMMENTS */}
          <div className="relative mb-0 mt-2 hidden flex-col overflow-hidden rounded-lg border bg-white text-xs shadow-md transition-all sm:mt-4 sm:flex">
            <CommentsPane
              initialComments={(currentNeuron?.comments as CommentWithPartialRelations[]) || []}
              neuron={currentNeuron}
            />
          </div>
        </div>

        {/* === RIGHT PANE */}
        <div
          className={`flex flex-col overflow-y-scroll px-0 ${
            embed ? 'flex-1' : 'sm:basis-1/2 lg:basis-2/3'
          } sm:overflow-auto sm:px-0`}
        >
          {embed && !currentNeuron?.hasVector && !forceMiniStats && embedLink && (
            <div
              className={`mb-2 flex w-full flex-row items-center sm:mb-2 ${
                embedExplanation ? 'justify-between gap-x-2 pl-2' : 'justify-center'
              } rounded-md py-0.5 pr-0 text-right font-mono text-[10px] font-semibold leading-[1.25] text-slate-600 sm:py-1 sm:text-[11px]`}
            >
              {embedExplanation && (
                <div className="text-left font-sans text-[11.5px] sm:text-[13px]">
                  {currentNeuron &&
                    currentNeuron.explanations &&
                    currentNeuron.explanations.length > 0 &&
                    currentNeuron.explanations[0]?.description}
                </div>
              )}
              <a
                className="mr-1.5 flex shrink-0 flex-row items-center gap-x-1 whitespace-nowrap rounded-md bg-slate-200 px-[8px] py-[6px] text-[9px] font-medium leading-none text-slate-700 hover:bg-sky-200 hover:text-sky-700 sm:mr-0 sm:px-2.5 sm:py-1.5 sm:text-[11px]"
                href={`/${currentNeuron?.modelId}/${currentNeuron?.layer}/${currentNeuron?.index}`}
                target="_blank"
                rel="noreferrer"
              >
                <Image src="/logo.png" alt="Neuronpedia logo" width="24" height="24" className="mr-0.5" />
                <div className="flex flex-col gap-y-[2px]">
                  <div className="">{currentNeuron?.modelId.toUpperCase()}</div>
                  <div className="">{currentNeuron?.layer.toUpperCase()}</div>
                  <div className="">INDEX {currentNeuron?.index.toUpperCase()}</div>
                </div>
              </a>
            </div>
          )}
          {!embedLink && (
            <div className="mb-1.5 flex flex-row justify-end px-1">
              <Image src="/logo.png" alt="Neuronpedia logo" width="18" height="18" />
            </div>
          )}

          <div
            className={`relative ${forceMiniStats ? 'mb-0' : 'mb-5'} flex h-full w-full flex-1 cursor-default flex-col overflow-hidden bg-white px-0 pt-0 shadow sm:overflow-visible sm:rounded-lg`}
            id="activationScrollDiv"
          >
            <div className="flex-1 overflow-y-scroll border border-slate-200 sm:rounded-lg">
              {currentNeuron?.pos_str && currentNeuron?.pos_str?.length > 0 && embedPlots && (
                <div className="border-b px-2.5 pb-1.5 pt-1.5 sm:px-4 sm:pb-2 sm:pt-2">
                  <FeatureStats
                    currentNeuron={currentNeuron}
                    embed={embed}
                    embedPlots={embedPlots}
                    activationMarkerValue={activationMarkerValue}
                    forceMiniStats={forceMiniStats}
                  />
                </div>
              )}
              <ActivationsList
                feature={currentNeuron}
                defaultRange={forceMiniStats ? 1 : currentNeuron?.sourceSet?.defaultRange}
                defaultShowLineBreaks={forceMiniStats ? false : currentNeuron?.sourceSet?.defaultShowBreaks}
                showCopy={allowTest()}
                showTest={allowTest()}
                defaultActivationTestText={defaultTestText}
                showControls={!embed}
                overrideLeading={forceMiniStats ? 'leading-none sm:leading-none' : 'leading-tight sm:leading-tight'}
                overrideTextSize={embed ? (forceMiniStats ? 'text-[10px]' : 'text-[10px] sm:text-[12px]') : undefined}
                embed={embed}
                activations={currentNeuron?.activations}
                activationTestTextCallback={(activation) => {
                  setTestTextResult(activation);
                }}
                // by default we try to show the chat/formatted tokens instead of the raw tokens
                defaultShowRawTokens={false}
                // if false, only show the activation test
                showActivations={embedActivations}
                showSteerButton={embedSteer}
                showTestField={embedTestField}
              />
            </div>
          </div>
        </div>

        {/* === DEAD FEATURE */}
        {currentNeuron?.maxActApprox === 0 && !hideDeadWarning && !currentNeuron?.hasVector && (
          <div className="absolute left-0 top-0 flex h-full w-full flex-col items-center bg-slate-50 pt-12">
            <div className="font-bold text-slate-900">This feature has no known activations.</div>
            <button
              type="button"
              onClick={() => {
                setHideDeadWarning(true);
              }}
              className="mt-3 rounded-lg bg-slate-600 px-5 py-2 text-slate-200 hover:bg-slate-500"
            >
              Show Anyway
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
